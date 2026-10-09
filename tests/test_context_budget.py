import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

from agent import Agent, AgentModelOutputError, ContextManager, ContextWindowError
from tooling.executor import ToolExecutor


# 用可预测的字符数测试预算，消息与 Schema 都进入同一总量
# messages：候选消息；attrs：LangChain 计数接口的工具及估算参数
def token_counter(messages, **attrs):
    return sum(len(str(message.content)) for message in messages) + len(str(attrs.get("tools") or []))


class BudgetSummarizer:
    # 固定返回短摘要，测试不会访问真实模型
    # attrs：摘要器标准参数
    def summarize(self, **attrs):
        return "预算测试摘要"


# 创建不受用户环境预算配置影响的测试 Agent
# attrs：当前用例需要覆盖的配置
def make_agent(**attrs):
    # defaults：每个预算都显式传入，避免测试读取或展示真实密钥
    defaults = dict(model_context_tokens=1000000, max_context_tokens=256 * 1024,
                    max_output_tokens=16384, system_context_tokens=8192,
                    tool_schema_context_tokens=32768, memory_recall_max_tokens=4096,
                    tool_executor=ToolExecutor(registry={}))
    return Agent(**{**defaults, **attrs})


class ContextPartitionTests(unittest.TestCase):
    # 创建使用字符计数的上下文管理器
    # attrs：当前测试覆盖的预算参数
    def manager(self, **attrs):
        return ContextManager(BudgetSummarizer(), token_counter=token_counter, **attrs)

    # 默认配置保留输入上限、压缩线和目标线，输出预算从物理窗口另行扣除
    def test_default_input_and_output_budget(self):
        # manager：默认分区预算
        manager = self.manager()
        self.assertEqual(manager.max_context_tokens, 262144)
        self.assertEqual(manager.max_output_tokens, 16384)
        self.assertEqual(manager.compression_trigger_tokens, 196608)
        self.assertEqual(manager.compression_target_tokens, 131072)

    # 更换小窗口模型后，有效输入上限会扣除输出预留而不是只检查配置上限
    def test_small_model_window_reduces_effective_input(self):
        # manager：配置输入 8192，但物理窗口只能同时容纳 3072 输入和 1024 输出
        manager = self.manager(max_context_tokens=8192, model_context_tokens=4096,
                               max_output_tokens=1024)
        self.assertEqual(manager.configured_input_tokens, 8192)
        self.assertEqual(manager.max_context_tokens, 3072)
        self.assertEqual(manager.compression_trigger_tokens, 2304)
        with self.assertRaises(ContextWindowError):
            manager.prepare_messages([HumanMessage("需求" * 1600)], "需求")

    # 非法预算与过大的输出预留在初始化时失败，不等到模型 API 报错
    def test_invalid_budget_configuration(self):
        # invalid：当前不合法的参数组合
        for invalid in ({"max_output_tokens": True}, {"memory_max_tokens": 0},
                        {"system_soft_tokens": -1}, {"tool_schema_soft_tokens": 1.5},
                        {"model_context_tokens": 4096, "max_output_tokens": 4096}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.manager(**invalid)

    # 系统与工具软预算只能告警，不会截断系统规则或静默删掉工具
    def test_soft_limits_preserve_system_and_schemas(self):
        # manager、messages、schemas：系统和 Schema 均超过软预算但不超过总输入预算
        manager = self.manager(max_context_tokens=4096, system_soft_tokens=50,
                               tool_schema_soft_tokens=50)
        messages = [SystemMessage("系统规则" * 100), HumanMessage("当前问题")]
        schemas = [{"description": "工具定义" * 100}]
        with self.assertLogs("agent.context", level="INFO") as logs:
            # preparation：低于总压缩线也需返回分区统计并输出告警
            preparation = manager.prepare_session_context(messages, "当前问题", schemas)
        self.assertIs(preparation.messages[0], messages[0])
        self.assertEqual(preparation.usage.system_tokens, len(messages[0].content) + 2)
        self.assertGreater(preparation.usage.tool_schema_tokens, 50)
        self.assertEqual(preparation.usage.total_tokens, manager.count_tokens(messages, schemas))
        self.assertTrue(any("系统提示超过软预算" in line for line in logs.output))
        self.assertTrue(any("工具 Schema 超过软预算" in line for line in logs.output))

    # 未实际使用的分区预算可以借给历史，不按软预算预先扣光空间
    def test_working_budget_uses_actual_fixed_occupancy(self):
        # manager、messages：配置较大的软预算，实际仅有极短系统提示
        manager = self.manager(max_context_tokens=4096, system_soft_tokens=8192,
                               tool_schema_soft_tokens=32768)
        messages = [SystemMessage("规则"), HumanMessage("当前任务")]
        # usage：工作记忆按实际固定占用获得剩余空间
        usage = manager.measure_usage(messages, [])
        self.assertEqual(usage.working_budget_tokens, 4096 - manager.count_tokens(messages[:1], []))
        self.assertGreater(usage.working_budget_tokens, 4000)

    # 分区统计正确区分主系统、长期记忆、滚动摘要和工作消息
    def test_partition_classification_includes_wrappers(self):
        # manager、memory、summary：已加完整包装提示的两类记忆
        manager = self.manager()
        memory = manager.memory_message("品牌偏好", ["品牌偏好"])
        summary = manager._session_summary_message("历史任务摘要")
        # messages：所有分区的完整输入
        messages = [SystemMessage("规则"), memory, summary, HumanMessage("当前问题"), AIMessage("进度")]
        # usage：分区各自估算，总量使用完整请求
        usage = manager.measure_usage(messages, [{"name": "tool"}])
        self.assertEqual(usage.memory_tokens, manager.count_tokens([memory]))
        self.assertEqual(usage.summary_tokens, manager.count_tokens([summary]))
        self.assertEqual(usage.working_tokens, manager.count_tokens(messages[-2:]))
        self.assertEqual(usage.total_tokens, manager.count_tokens(messages, [{"name": "tool"}]))

    # 按相关度选择完整条目，跳过放不下的大条目后还可保留后续短条目
    def test_memory_selection_keeps_complete_multiline_entries(self):
        # entries：高相关偏好、多行超长记录及较短补充偏好
        entries = ["高相关完整偏好", "多行事实\n" + "长" * 800, "补充偏好"]
        # budget：刚好容纳第一条、第三条及包装提示
        budget = token_counter([ContextManager.memory_message(entries[0] + "\n" + entries[2])])
        # manager、selected：按 Token 上限完整选择的结果
        manager = self.manager(memory_max_tokens=budget)
        selected = manager.select_memory_entries(entries)
        self.assertEqual(selected, [entries[0], entries[2]])
        self.assertEqual(manager.select_memory_entries(entries, max_chars=len(entries[0])), [entries[0]])
        self.assertEqual(entries[1], "多行事实\n" + "长" * 800)

    # 最终上下文还会检查 Token 预算，不能只依赖第一次召回时的筛选
    def test_final_memory_injection_rechecks_budget_without_mutating_state(self):
        # manager、entries、memory：模拟旧状态携带超过新预算的完整条目列表
        manager = self.manager(memory_max_tokens=200)
        entries = ["短偏好", "长" * 500]
        memory = manager.memory_message("\n".join(entries), entries)
        # messages、preparation：原始记录和最终模型视图
        messages = [SystemMessage("规则"), memory, HumanMessage("当前问题")]
        preparation = manager.prepare_session_context(messages, "当前问题")
        self.assertEqual(preparation.messages[1].additional_kwargs["memory_entries"], ["短偏好"])
        self.assertLessEqual(preparation.usage.memory_tokens, 200)
        self.assertIn("长" * 500, messages[1].content)
        self.assertEqual(memory.additional_kwargs["memory_entries"], entries)

    # 多条记忆系统消息共享 4K 预算，不能每条都独占一份预算
    def test_multiple_memory_messages_share_one_budget(self):
        # first、second、budget：预算只足够保留一个完整记忆块
        first = ContextManager.memory_message("甲" * 100, ["甲" * 100])
        second = ContextManager.memory_message("乙" * 100, ["乙" * 100])
        budget = token_counter([first]) + 10
        # manager、preparation：检查最终输入内全部长期记忆占用
        manager = self.manager(memory_max_tokens=budget)
        preparation = manager.prepare_session_context([first, second, HumanMessage("当前问题")], "当前问题")
        self.assertLessEqual(preparation.usage.memory_tokens, budget)
        self.assertEqual(sum(isinstance(message, SystemMessage) for message in preparation.messages), 1)

    # 旧记忆没有可靠条目边界时，超限整体舍弃，不按换行猜测或截断半条事实
    def test_legacy_memory_block_is_dropped_whole_when_oversized(self):
        # manager、memory：没有 memory_entries 元数据的超长旧消息
        manager = self.manager(memory_max_tokens=200)
        memory = manager.memory_message("旧事实\n" + "长" * 500)
        # preparation：旧消息不进入模型，但原始对象保持不变
        preparation = manager.prepare_session_context([memory, HumanMessage("当前问题")], "当前问题")
        self.assertEqual(len(preparation.messages), 1)
        self.assertEqual(preparation.usage.memory_tokens, 0)
        self.assertIn("长" * 500, memory.content)

    # 删除临时记忆视图不能改变兼容 index 游标的原始下标
    def test_memory_selection_does_not_shift_summary_cursor(self):
        # manager、messages：index:3 是旧回答，不应因删除临时记忆而指向当前问题
        manager = self.manager(memory_max_tokens=20)
        messages = [SystemMessage("规则"), manager.memory_message("超长旧记忆"),
                    HumanMessage("旧问题"), AIMessage("旧回答"), HumanMessage("当前问题")]
        # preparation：游标仍按完整原始列表解释
        preparation = manager.prepare_session_context(messages, "当前问题", [], "旧摘要", "index:3")
        self.assertEqual(preparation.summary_cursor, "index:3")
        self.assertIs(preparation.messages[-1], messages[-1])
        self.assertNotIn(messages[2], preparation.messages)

    # 已持久化的大摘要在恢复时也受硬预算约束，包装提示同样占用摘要预算
    def test_restored_summary_budget_includes_wrapper(self):
        # manager、messages：旧历史已摘要，只需要治理恢复的摘要文本
        manager = self.manager(max_context_tokens=4096, session_summary_target_tokens=128,
                               session_summary_max_tokens=256)
        messages = [HumanMessage("旧问题"), AIMessage("旧回答", id="old-answer"), HumanMessage("当前问题")]
        # preparation：超长旧摘要缩到包含包装后的 256 Token 以内
        preparation = manager.prepare_session_context(messages, "当前问题", [], "摘要" * 1000, "old-answer")
        self.assertLessEqual(preparation.usage.summary_tokens, 256)
        self.assertEqual(preparation.summary_cursor, "old-answer")
        self.assertIs(preparation.messages[-1], messages[-1])

    # 摘要包装本身超过预算时明确失败，不能在反复截断与追加省略号之间死循环
    def test_summary_budget_too_small_for_wrapper_fails_explicitly(self):
        # 模拟一种包装开销特别大的计数口径
        # messages：计数消息；attrs：标准计数器附加参数
        def expensive_counter(messages, **attrs):
            return token_counter(messages, **attrs) + 1000

        # manager：摘要硬上限不足以覆盖包装，调用必须有限次结束
        manager = ContextManager(BudgetSummarizer(), token_counter=expensive_counter,
                                 session_summary_target_tokens=128, session_summary_max_tokens=256)
        with self.assertRaisesRegex(ContextWindowError, "包装提示"):
            manager._limit_session_summary("旧摘要")

    # 固定上下文不可压缩时，异常提供分区诊断，不暗中删除系统或工具定义
    def test_hard_limit_error_reports_partitions(self):
        # manager：输入上限极小以验证可解释的超限终止
        manager = self.manager(max_context_tokens=1024)
        with self.assertRaisesRegex(ContextWindowError, "系统=.*工具=.*输出预留=16384"):
            manager.prepare_messages([SystemMessage("规则" * 600), HumanMessage("当前问题")], "当前问题")


class AgentOutputBudgetTests(unittest.TestCase):
    # Agent 和 ContextManager 对有效输入与输出上限使用一致的配置
    def test_agent_effective_input_budget(self):
        # agent：用户输入上限高于模型窗口，程序自动降低实际可发送上限
        agent = make_agent(max_context_tokens=8192, model_context_tokens=4096, max_output_tokens=1024)
        self.assertEqual(agent.max_context_tokens, 8192)
        self.assertEqual(agent.effective_input_tokens, 3072)
        self.assertEqual(agent.context_manager.max_context_tokens, 3072)

    # 延迟创建的真实 ChatDeepSeek 会获得明确 max_tokens，不依赖供应商默认值
    def test_lazy_model_receives_output_limit(self):
        # agent：初始化无需真实密钥，模型创建时注入测试凭据
        agent = make_agent(max_output_tokens=512)
        agent.api_key = "test-key"
        with patch("agent.runtime.ChatDeepSeek") as constructor:
            agent._get_chat_model()
        self.assertEqual(constructor.call_args.kwargs["max_tokens"], 512)

    # 用本地 HTTP 替身验证主模型请求真实携带 max_tokens，且不修改注入模型默认值
    def test_injected_model_wire_request_has_output_limit(self):
        # requests：只保存本地模拟 HTTP 请求正文，不读取真实环境密钥
        requests = []

        # 回应一次模拟 DeepSeek 请求，不建立任何网络连接
        # request：HTTP MockTransport 收到的请求
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "model": "deepseek-chat",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "正常答案"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            # model、agent：注入默认输出较大的模型，但主调用必须受 Agent 预算限制
            model = ChatDeepSeek(model="deepseek-chat", api_key="test-key", max_tokens=999,
                                 http_client=client, max_retries=0)
            agent = make_agent(chat_model=model, max_output_tokens=512)
            self.assertEqual(agent.run("你好"), "正常答案")
        self.assertEqual(requests[0]["max_tokens"], 512)
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertEqual(model.max_tokens, 999)

    # 被截断的工具计划不会执行，也不会经过最终答案路径提取长期记忆
    def test_truncated_response_does_not_execute_or_extract(self):
        # bound_model、model：返回一个看似完整但 finish_reason=length 的工具计划
        bound_model = MagicMock()
        bound_model.invoke.return_value = AIMessage(content="部分答案", tool_calls=[{
            "id": "partial", "name": "write_file", "args": {"path": "output/x.txt"}, "type": "tool_call",
        }], response_metadata={"finish_reason": "length"})
        model = MagicMock()
        model.bind_tools.return_value = bound_model
        # memory：真实图流程启用记忆，但所有服务为替身
        memory = MagicMock()
        memory.repository.forget_events.return_value = []
        memory.repository.register_memory_turn.return_value = 1
        memory.recall.return_value = []
        # agent：截断检查必须发生在 tools 或 extract_memory 节点之前
        agent = make_agent(chat_model=model, memory_service=memory, forget_judge=MagicMock())
        with patch.object(agent.tool_batch_executor, "execute_batch") as execute:
            with self.assertRaises(AgentModelOutputError):
                agent.run("写文件")
            execute.assert_not_called()
        memory.begin_extraction.assert_not_called()
        self.assertIsNone(agent.memory_turn)

    # 截断回答不持久化为完整答案，同一会话后续可以继续正常提问
    def test_truncated_answer_is_not_checkpointed_and_next_turn_works(self):
        # bound_model、model：第一轮被截断，第二轮正常完成
        bound_model = MagicMock()
        bound_model.invoke.side_effect = [
            AIMessage("不完整回答", response_metadata={"finish_reason": "length"}),
            AIMessage("完整答案", response_metadata={"finish_reason": "stop"}),
        ]
        model = MagicMock()
        model.bind_tools.return_value = bound_model
        # agent：使用内存检查点验证会话状态，而不是读写用户 SQLite 文件
        agent = make_agent(chat_model=model, checkpointer=InMemorySaver())
        with self.assertRaises(AgentModelOutputError):
            agent.run("第一轮", thread_id="budget-test")
        # snapshot：失败轮次没有追加模型的截断消息
        snapshot = agent.graph.get_state({"configurable": {"thread_id": "budget-test"}})
        self.assertFalse(any(message.content == "不完整回答" for message in snapshot.values["messages"]))
        self.assertEqual(agent.run("第二轮", thread_id="budget-test"), "完整答案")

    # 供应商明确正常结束时不能只根据输出 Token 数猜测截断
    def test_stop_reason_is_authoritative(self):
        # agent：直接检查多供应商完成原因的兼容逻辑
        agent = make_agent()
        agent._check_model_output(AIMessage("完成", response_metadata={"finish_reason": "stop"},
                                            usage_metadata={"input_tokens": 1, "output_tokens": 16384,
                                                            "total_tokens": 16385}))
        # reason：各供应商可能用来报告长度截断的标识
        for reason in ("length", "max_tokens", "max_output_tokens", "model_length"):
            with self.subTest(reason=reason), self.assertRaises(AgentModelOutputError):
                agent._check_model_output(AIMessage("部分", response_metadata={"stop_reason": reason}))

    # 自定义上下文管理器不得绕过模型输出预留或配置非法请求上限
    def test_custom_context_manager_must_match_output_budget(self):
        # manager：测试注入的管理器仍使用默认 16K 输出预留
        manager = ContextManager(BudgetSummarizer())
        with self.assertRaisesRegex(ValueError, "配置不一致"):
            make_agent(context_manager=manager, max_output_tokens=1024)
        with self.assertRaises(ValueError):
            make_agent(context_manager=manager, max_output_tokens=True)

    # 召回层同时采用字符与 Token 预算，保留全文边界而不是截断最后一条
    def test_recall_node_retains_full_entries_under_both_limits(self):
        # recalled：长条目放不下，但后面的短条目还能进入预算
        recalled = [(SimpleNamespace(content="高相关事实", memory_type=SimpleNamespace(value="fact")), 0.9),
                    (SimpleNamespace(content="长" * 1000, memory_type=SimpleNamespace(value="fact")), 0.8),
                    (SimpleNamespace(content="补充事实", memory_type=SimpleNamespace(value="fact")), 0.7)]
        # memory、agent：只测试召回节点，不触发真实模型或 SQLite
        memory = MagicMock()
        memory.recall.return_value = recalled
        memory.repository.forget_events.return_value = []
        agent = make_agent(memory_service=memory, memory_recall_max_chars=512,
                           memory_recall_max_tokens=300)
        # state、result：召回需要的最小身份状态和返回的完整条目
        state = dict(goal="查询偏好", tenant_id="tenant", user_id="user", messages=[])
        result = agent._recall_memory_node(state)
        self.assertEqual(len(result["recalled_memory_entries"]), 2)
        self.assertIn("高相关事实", result["recalled_memory_context"])
        self.assertIn("补充事实", result["recalled_memory_context"])
        self.assertNotIn("长", result["recalled_memory_context"])
        self.assertLessEqual(agent.context_manager.count_tokens([
            agent.context_manager.memory_message(result["recalled_memory_context"])]), 300)


if __name__ == "__main__":
    unittest.main()
