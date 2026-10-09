import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.context import ContextManager, ContextProtocolError, ContextWindowError


class FakeContextSummarizer:
    # 初始化记录上下文摘要调用的测试替身
    # summary：每次摘要返回的固定短文本
    # error：需要模拟的摘要服务异常
    def __init__(self, summary="历史工具事实摘要", error=None):
        self.summary = summary
        self.error = error
        self.calls = []
        self.session_calls = []

    # 记录历史上下文摘要参数并返回固定结果
    # tool_name：上下文管理器使用的内部摘要类型
    # value：待压缩的历史消息记录
    # goal：当前用户任务目标
    # max_chars：摘要允许的最大字符数
    def summarize(self, tool_name, value, goal, max_chars):
        self.calls.append((tool_name, value, goal, max_chars))
        if self.error is not None:
            raise self.error
        return self.summary

    # 记录滚动会话摘要参数并返回固定结果
    # existing_summary：上次持久化的会话摘要
    # records：本次新归档轮次的消息记录
    # goal：当前用户任务目标
    # max_chars：更新后摘要允许的最大字符数
    def summarize_session(self, existing_summary, records, goal, max_chars):
        self.session_calls.append((existing_summary, records, goal, max_chars))
        if self.error is not None:
            raise self.error
        return self.summary


# 按消息内容字符数模拟可预测的 Token 计数
# messages：本次待计算的 LangChain 消息
# attrs：LangChain 计数器接收的字符比例、usage scaling 和工具 Schema 参数
def fake_token_counter(messages, **attrs):
    # message_tokens：测试中将一个内容字符视为一个 Token
    message_tokens = sum(len(str(message.content)) for message in messages)
    # tool_tokens：验证工具 Schema 也会计入上下文预算
    tool_tokens = len(str(attrs.get("tools") or []))
    return message_tokens + tool_tokens


# 创建一轮完整的 Function Calling 历史
# call_id：工具调用和结果关联标识
# content：模拟工具返回的 Observation 内容
def tool_round(call_id, content):
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "id": call_id,
                "name": "fake_tool",
                "args": {"query": call_id},
                "type": "tool_call",
            }],
        ),
        ToolMessage(
            content=content,
            tool_call_id=call_id,
            name="fake_tool",
            status="success",
        ),
    ]


# 创建一个已完成且带稳定消息 id 的用户对话轮次
# turn_index：用于生成唯一消息 id 的轮次序号
# answer_chars：模拟最终回答的字符数
def conversation_turn(turn_index, answer_chars):
    return [
        HumanMessage(
            content=f"第 {turn_index} 轮问题",
            id=f"human-{turn_index}",
        ),
        AIMessage(
            content=str(turn_index) * answer_chars,
            id=f"assistant-{turn_index}",
        ),
    ]


class ContextManagerTests(unittest.TestCase):
    # 验证达到 75% 时归档较早轮次并保留最近 6 个已完成轮次原文
    def test_session_summary_keeps_six_recent_turns(self):
        summarizer = FakeContextSummarizer(summary="持久化会话摘要")
        manager = ContextManager(
            summarizer=summarizer,
            max_context_tokens=4096,
            compression_trigger_ratio=0.75,
            compression_target_ratio=0.5,
            recent_turns_to_keep=6,
            minimum_recent_turns=2,
            session_summary_target_tokens=128,
            session_summary_max_tokens=256,
            token_counter=fake_token_counter,
        )
        # messages：前两轮较长，归档后保留最近 6 轮和当前轮次
        messages = [SystemMessage("系统提示", id="system")]
        messages.extend(conversation_turn(1, 800))
        messages.extend(conversation_turn(2, 800))
        for turn_index in range(3, 9):
            messages.extend(conversation_turn(turn_index, 250))
        messages.append(HumanMessage("当前问题", id="human-current"))

        preparation = manager.prepare_session_context(
            messages,
            "当前问题",
            [],
        )

        # retained_humans：摘要视图中保留的最近 6 轮和当前用户消息
        retained_humans = [
            message.content
            for message in preparation.messages
            if isinstance(message, HumanMessage)
        ]
        self.assertEqual(len(retained_humans), 7)
        self.assertNotIn("第 1 轮问题", retained_humans)
        self.assertNotIn("第 2 轮问题", retained_humans)
        self.assertIn("第 3 轮问题", retained_humans)
        self.assertEqual(preparation.session_summary, "持久化会话摘要")
        self.assertEqual(preparation.summary_cursor, "assistant-2")
        self.assertTrue(preparation.messages[1].additional_kwargs["session_summary"])
        self.assertEqual(len(summarizer.session_calls), 1)

        # restored：模拟程序重启后使用同一摘要和游标恢复的视图
        restored = manager.prepare_session_context(
            messages,
            "当前问题",
            [],
            preparation.session_summary,
            preparation.summary_cursor,
        )
        self.assertEqual(restored.session_summary, "持久化会话摘要")
        self.assertEqual(len(summarizer.session_calls), 1)

    # 验证最近 6 轮仍过长时会动态缩小，但至少保留最近 2 轮原文
    def test_session_summary_can_reduce_to_two_recent_turns(self):
        summarizer = FakeContextSummarizer(summary="极端长会话摘要")
        manager = ContextManager(
            summarizer=summarizer,
            max_context_tokens=4096,
            compression_trigger_ratio=0.75,
            compression_target_ratio=0.5,
            recent_turns_to_keep=6,
            minimum_recent_turns=2,
            session_summary_target_tokens=128,
            session_summary_max_tokens=256,
            token_counter=fake_token_counter,
        )
        # messages：4 个较长已完成轮次和 1 个当前轮次
        messages = [SystemMessage("系统提示", id="system")]
        for turn_index in range(1, 5):
            messages.extend(conversation_turn(turn_index, 1000))
        messages.append(HumanMessage("当前问题", id="human-current"))

        preparation = manager.prepare_session_context(messages, "当前问题", [])

        # retained_humans：仅保留最近 2 个已完成轮次和当前轮次
        retained_humans = [
            message.content
            for message in preparation.messages
            if isinstance(message, HumanMessage)
        ]
        self.assertEqual(
            retained_humans,
            ["第 3 轮问题", "第 4 轮问题", "当前问题"],
        )
        self.assertEqual(preparation.summary_cursor, "assistant-2")
        self.assertEqual(len(summarizer.session_calls), 1)

    # 验证低于 75% 压缩线时完整消息原样进入模型
    def test_context_below_trigger_is_not_compressed(self):
        summarizer = FakeContextSummarizer()
        manager = ContextManager(
            summarizer=summarizer,
            max_context_tokens=1024,
            compression_trigger_ratio=0.75,
            token_counter=fake_token_counter,
        )
        messages = [SystemMessage("system"), HumanMessage("short goal")]

        prepared = manager.prepare_messages(messages, "short goal", [])

        self.assertEqual(prepared, messages)
        self.assertEqual(summarizer.calls, [])

    # 验证超过压缩线后旧工具轮次被摘要且最近轮次保持完整
    def test_old_tool_rounds_are_compressed_as_paired_messages(self):
        summarizer = FakeContextSummarizer()
        manager = ContextManager(
            summarizer=summarizer,
            max_context_tokens=1024,
            compression_trigger_ratio=0.75,
            token_counter=fake_token_counter,
        )
        # messages：旧轮次占用 700 Token，最近轮次仅占用 100 Token
        messages = [SystemMessage("system"), HumanMessage("goal")]
        messages.extend(tool_round("call_old", "旧" * 700))
        messages.extend(tool_round("call_recent", "新" * 100))
        # reasoning_content：模拟 DeepSeek 思考模型要求原样回传的模型元数据
        messages[2].additional_kwargs["reasoning_content"] = "真实模型思考元数据"

        prepared = manager.prepare_messages(messages, "goal", [])

        # preserved_ai：保留真实工具调用及 reasoning 元数据的旧轮次 AIMessage
        preserved_ai = prepared[2]
        # compacted_tool：使用原始 tool_call_id 携带历史摘要的 ToolMessage
        compacted_tool = prepared[3]
        self.assertIsInstance(preserved_ai, AIMessage)
        self.assertIs(preserved_ai, messages[2])
        self.assertEqual(
            preserved_ai.additional_kwargs["reasoning_content"],
            "真实模型思考元数据",
        )
        self.assertEqual(preserved_ai.tool_calls[0]["name"], "fake_tool")
        self.assertIsInstance(compacted_tool, ToolMessage)
        self.assertEqual(
            compacted_tool.tool_call_id,
            preserved_ai.tool_calls[0]["id"],
        )
        self.assertIn("历史工具事实摘要", compacted_tool.content)
        self.assertIn('"context_summary":true', compacted_tool.content)
        self.assertEqual(prepared[-1].tool_call_id, "call_recent")
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(messages[3].content, "旧" * 700)

        # 再次处理同一历史时应直接使用摘要缓存
        manager.prepare_messages(messages, "goal", [])
        self.assertEqual(len(summarizer.calls), 1)

    # 验证跨用户轮次压缩时，不会把中间的问题、普通回答或系统消息一起替换
    def test_tool_compression_preserves_intervening_messages(self):
        # summarizer、manager：用小预算触发多轮工具结果独立压缩
        summarizer = FakeContextSummarizer()
        manager = ContextManager(summarizer, max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # messages：两轮用户对话之间穿插普通回答和额外系统消息
        messages = [SystemMessage("系统规则"), HumanMessage("历史问题")]
        messages.extend(tool_round("old", "旧" * 700))
        messages.extend([AIMessage("历史回答"), SystemMessage("有效系统规则"),
                         HumanMessage("当前问题")])
        messages.extend(tool_round("current_early", "中" * 700))
        messages.extend([AIMessage("当前任务进度")])
        messages.extend(tool_round("latest", "新" * 100))

        # prepared：保留全部消息位置，只替换过长工具结果的模型视图
        prepared = manager.prepare_messages(messages, "当前问题", [])

        self.assertEqual(len(prepared), len(messages))
        # index：所有不应被工具压缩替换的消息下标
        for index in (0, 1, 2, 4, 5, 6, 7, 9, 10, 11):
            self.assertIs(prepared[index], messages[index])
        self.assertIn('"context_summary":true', prepared[3].content)
        self.assertIn('"context_summary":true', prepared[8].content)
        self.assertEqual(len(summarizer.calls), 2)
        self.assertTrue(all(record["type"] in ("ai", "tool")
                            for _, records, _, _ in summarizer.calls
                            for record in records))
        self.assertEqual(messages[3].content, "旧" * 700)
        self.assertEqual(messages[8].content, "中" * 700)

    # 验证第二层工具压缩不会删除第一层保留的最近两轮用户问题和普通回答
    def test_session_and_tool_compression_preserve_recent_turn_boundaries(self):
        # summarizer、manager：强制先归档旧会话，再处理保留轮次中的长工具结果
        summarizer = FakeContextSummarizer(summary="会话和工具事实摘要")
        manager = ContextManager(summarizer, max_context_tokens=4096,
                                 session_summary_target_tokens=128,
                                 session_summary_max_tokens=256,
                                 token_counter=fake_token_counter)
        # messages：四个已完成轮次及一个正在执行的当前任务
        messages = [SystemMessage("系统规则", id="system")]
        # turn_index：具有工具结果的历史用户轮次序号
        for turn_index in range(1, 5):
            messages.append(HumanMessage(f"问题{turn_index}", id=f"human-{turn_index}"))
            messages.extend(tool_round(f"call-{turn_index}", "旧" * 1200))
            messages.append(AIMessage(f"回答{turn_index}", id=f"answer-{turn_index}"))
        messages.append(HumanMessage("当前问题", id="current"))
        messages.extend(tool_round("latest", "新" * 1200))

        # preparation：同时经过会话摘要与逐轮工具摘要的最终视图
        preparation = manager.prepare_session_context(messages, "当前问题", [])

        self.assertEqual(preparation.summary_cursor, "answer-2")
        self.assertEqual([message.content for message in preparation.messages
                          if isinstance(message, HumanMessage)],
                         ["问题3", "问题4", "当前问题"])
        self.assertTrue(any(message.content == "回答3" for message in preparation.messages))
        self.assertTrue(any(message.content == "回答4" for message in preparation.messages))
        self.assertIs(preparation.messages[-1], messages[-1])
        self.assertEqual(len(summarizer.session_calls), 1)
        self.assertEqual(len(summarizer.calls), 2)

    # 验证批量调用全部配对，保留调用参数、思考内容、状态和结果元数据
    def test_batch_compression_preserves_all_call_ids_and_metadata(self):
        # summarizer、manager：摘要器及预算管理器
        summarizer = FakeContextSummarizer()
        manager = ContextManager(summarizer, max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # first、second：同一批次中的两次真实调用
        first = tool_round("first", "甲" * 700)
        second = tool_round("second", "乙" * 700)
        first[0].tool_calls.extend(second[0].tool_calls)
        first[0].additional_kwargs["reasoning_content"] = "真实思考内容"
        first[1].id = "result-first"
        first[1].additional_kwargs["memory_source_seq"] = 12
        second[1].status = "error"
        second[1].artifact = {"source": "original"}
        # messages：故意交换结果顺序，校验按 ID 配对而不是按数组下标猜测
        messages = [SystemMessage("system"), HumanMessage("goal"),
                    first[0], second[1], first[1]]

        # prepared、payload：最终消息及该批次的摘要载荷
        prepared = manager.prepare_messages(messages, "goal", [])
        payload = json.loads(prepared[3].content)

        self.assertIs(prepared[2], first[0])
        self.assertEqual(prepared[2].tool_calls[1]["args"], {"query": "second"})
        self.assertEqual(prepared[2].additional_kwargs["reasoning_content"], "真实思考内容")
        self.assertEqual([message.tool_call_id for message in prepared[3:]], ["second", "first"])
        self.assertEqual(prepared[3].status, "error")
        self.assertEqual(prepared[3].artifact, {"source": "original"})
        self.assertEqual(prepared[4].id, "result-first")
        self.assertEqual(prepared[4].additional_kwargs["memory_source_seq"], 12)
        self.assertEqual(payload["compacted_rounds"], 1)
        self.assertEqual(json.loads(prepared[4].content)["summary_in_tool_call_id"], "second")
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(first[1].content, "甲" * 700)

    # 验证当前任务的早期工具结果可以摘要，但达到目标后最新一批保持原文
    def test_compression_stops_before_latest_round_when_target_met(self):
        # summarizer、manager：记录逐轮摘要顺序的测试对象
        summarizer = FakeContextSummarizer()
        manager = ContextManager(summarizer, max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # messages：三个工具轮次，只有最早一轮需要压缩
        messages = [SystemMessage("system"), HumanMessage("goal")]
        messages.extend(tool_round("oldest", "旧" * 700))
        messages.extend(tool_round("middle", "中" * 100))
        messages.extend(tool_round("latest", "新" * 100))

        # prepared：达到目标后停止摘要的视图
        prepared = manager.prepare_messages(messages, "goal", [])

        self.assertEqual(len(summarizer.calls), 1)
        self.assertIs(prepared[-3], messages[-3])
        self.assertIs(prepared[-1], messages[-1])
        self.assertLessEqual(manager.count_tokens(prepared), manager.compression_target_tokens)

    # 验证没有足够旧结果可压缩时，才允许压缩最新完整批次
    def test_latest_completed_round_is_compressed_only_if_needed(self):
        # summarizer、manager：小预算下只包含一个长工具批次
        summarizer = FakeContextSummarizer()
        manager = ContextManager(summarizer, max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # messages：当前问题与最新的一轮完整工具调用
        messages = [SystemMessage("system"), HumanMessage("当前问题")]
        messages.extend(tool_round("latest", "新" * 900))

        # prepared：最新结果摘要后，问题与调用仍保留原始对象
        prepared = manager.prepare_messages(messages, "当前问题", [])

        self.assertIs(prepared[1], messages[1])
        self.assertIs(prepared[2], messages[2])
        self.assertIn('"context_summary":true', prepared[-1].content)

    # 验证不完整批次及已返回的部分结果不参与压缩，也不能被发送给模型
    def test_incomplete_tail_is_preserved_and_model_input_is_rejected(self):
        # summarizer、manager：只摘要早期完整轮次的测试对象
        summarizer = FakeContextSummarizer()
        manager = ContextManager(summarizer, max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # pending：两次调用仅收到第一条结果的未完成批次
        pending = tool_round("pending-first", "部分结果")
        pending[0].tool_calls.extend(tool_round("pending-second", "未返回")[0].tool_calls)
        # messages：完整旧轮次后接未完成批次
        messages = [SystemMessage("system"), HumanMessage("goal")]
        messages.extend(tool_round("old", "旧" * 700))
        messages.extend(pending)

        # prepared：仅测试压缩视图，未完成尾部原样保留
        prepared = manager._compress_tool_history(messages, "goal", [], 512)

        self.assertIs(prepared[-2], pending[0])
        self.assertIs(prepared[-1], pending[1])
        self.assertEqual(len(summarizer.calls), 1)
        with self.assertRaises(ContextProtocolError):
            manager.prepare_messages(prepared, "goal", [])

    # 验证错误配对在低于压缩线时也会被拦截，不能等模型 API 报错
    def test_invalid_tool_protocol_is_rejected_before_model_request(self):
        # manager：无需触发摘要即可检查协议
        manager = ContextManager(FakeContextSummarizer(), token_counter=fake_token_counter)
        # complete：合法的一轮调用，用于构造缺失、重复和错误关联的消息
        complete = tool_round("valid", "结果")
        # duplicate_call：同批次两个调用错误地复用相同 ID
        duplicate_call = complete[0].model_copy(deep=True)
        duplicate_call.tool_calls.extend(complete[0].tool_calls)
        # missing_id_call：模型生成了没有关联标识的调用
        missing_id_call = complete[0].model_copy(deep=True)
        missing_id_call.tool_calls[0]["id"] = None
        # invalid_histories：不应通过协议校验的各类消息
        invalid_histories = [
            [complete[1]],
            [complete[0]],
            [complete[0], HumanMessage("新问题")],
            [complete[0], ToolMessage("结果", tool_call_id="unknown")],
            [*complete, complete[1]],
            [duplicate_call, complete[1]],
            [missing_id_call],
        ]
        # history：当前验证的不合法历史
        for history in invalid_histories:
            with self.subTest(history=history), self.assertRaises(ContextProtocolError):
                manager.prepare_messages([SystemMessage("system"), *history], "goal", [])

    # 验证参数解析失败的原生调用仍按 ID 与错误结果正常配对并参与摘要
    def test_invalid_native_arguments_still_have_valid_message_pairing(self):
        # manager：摘要长错误结果的上下文管理器
        manager = ContextManager(FakeContextSummarizer(), max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # message：LangChain 的无效参数调用，仍然带有合法关联 ID
        message = AIMessage(content="", invalid_tool_calls=[{
            "name": "fake_tool", "args": "{", "id": "invalid-args",
            "error": "参数格式错误", "type": "invalid_tool_call",
        }])
        # messages：参数解析错误不等于工具调用协议错误
        messages = [HumanMessage("goal"), message,
                    ToolMessage("错误" * 500, tool_call_id="invalid-args", status="error")]

        # prepared：保留无效调用及其错误状态的摘要视图
        prepared = manager.prepare_messages(messages, "goal", [])

        self.assertIs(prepared[1], message)
        self.assertEqual(prepared[-1].tool_call_id, "invalid-args")
        self.assertEqual(prepared[-1].status, "error")

    # 验证摘要游标不能落在当前用户问题、当前工具结果或历史调用链中间
    def test_summary_cursor_cannot_cross_protected_boundaries(self):
        # manager：低于触发线时验证摘要游标的保护边界
        manager = ContextManager(FakeContextSummarizer(), token_counter=fake_token_counter)
        # messages：一轮历史工具对话和当前工具对话
        messages = [SystemMessage("system"), HumanMessage("旧问题", id="old-human")]
        messages.extend(tool_round("old-call", "旧结果"))
        messages[2].id = "old-call-ai"
        messages.extend([AIMessage("旧回答", id="old-answer"),
                         HumanMessage("当前问题", id="current-human")])
        messages.extend(tool_round("current-call", "当前结果"))
        messages[-1].id = "current-result"

        # cursor：游标存在但不在完整历史用户轮次末尾的各种情况
        for cursor in ("old-call-ai", "old-human", "current-human", "current-result"):
            with self.subTest(cursor=cursor):
                # preparation：不可信游标必须被清空，不能隐藏当前问题或拆开调用
                preparation = manager.prepare_session_context(
                    messages, "当前问题", [], "不应应用的摘要", cursor,
                )
                self.assertEqual(preparation.messages, messages)
                self.assertEqual(preparation.session_summary, "")
                self.assertIsNone(preparation.summary_cursor)

    # 验证按轮次摘要失败时的降级仍保留当前问题和所有真实调用
    def test_cross_turn_summary_failure_does_not_remove_user_messages(self):
        # manager：模拟摘要超时并使用较小确定性摘要
        manager = ContextManager(FakeContextSummarizer(error=TimeoutError("摘要超时")),
                                 max_context_tokens=4096, summary_max_chars=256,
                                 token_counter=fake_token_counter)
        # messages：两个用户轮次中的工具结果都需要压缩
        messages = [SystemMessage("system"), HumanMessage("旧问题")]
        messages.extend(tool_round("old", "旧" * 1600))
        messages.extend([AIMessage("旧回答"), HumanMessage("当前问题")])
        messages.extend(tool_round("current", "新" * 1600))

        # prepared：两轮分别确定性摘要后的视图
        prepared = manager.prepare_messages(messages, "当前问题", [])

        self.assertEqual(len(prepared), len(messages))
        # index：降级摘要也必须原样保留的非结果消息下标
        for index in (0, 1, 2, 4, 5, 6):
            self.assertIs(prepared[index], messages[index])
        self.assertEqual([prepared[3].tool_call_id, prepared[7].tool_call_id], ["old", "current"])
        self.assertLess(manager.count_tokens(prepared), manager.max_context_tokens)

    # 验证摘要包装比原结果更长时保留原文，避免为了压缩反而扩大上下文
    def test_short_results_are_not_replaced_with_larger_summaries(self):
        # manager：固定消息很长，但工具结果很短，摘要不能改善预算
        manager = ContextManager(FakeContextSummarizer(), max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # messages：系统消息占主要预算，工具只有少量原文
        messages = [SystemMessage("S" * 800), HumanMessage("goal")]
        messages.extend(tool_round("short", "短结果"))

        # prepared：仍高于目标但不超过硬上限，短结果不做无效替换
        prepared = manager.prepare_messages(messages, "goal", [])

        self.assertIs(prepared[-1], messages[-1])

    # 验证空或非字符串摘要触发确定性降级，不把无效结果注入工具消息
    def test_invalid_summary_output_uses_fallback(self):
        # summary：各种不符合摘要器文本约定的返回值
        for summary in ("", "   ", None, {"unexpected": "object"}):
            with self.subTest(summary=summary):
                # manager：测试摘要输出类型的管理器
                manager = ContextManager(FakeContextSummarizer(summary=summary),
                                         max_context_tokens=1024, summary_max_chars=256,
                                         token_counter=fake_token_counter)
                # messages：足够长的结果保证会触发摘要
                messages = [HumanMessage("goal"), *tool_round("call", "长" * 900)]
                # prepared、payload：降级后的工具摘要及其结构化载荷
                prepared = manager.prepare_messages(messages, "goal", [])
                payload = json.loads(prepared[-1].content)
                self.assertIn("fake_tool", payload["summary"])
                self.assertEqual(prepared[-1].tool_call_id, "call")

    # 验证同时选择多个不连续轮次时，每轮独立摘要且中间用户消息保持原文
    def test_selected_noncontiguous_rounds_are_replaced_independently(self):
        # manager：直接验证轮次替换方法的边界契约
        manager = ContextManager(FakeContextSummarizer(), token_counter=fake_token_counter)
        # messages：两个工具轮次之间夹着普通回答、用户问题和系统消息
        messages = [HumanMessage("旧问题"), *tool_round("old", "旧" * 700),
                    AIMessage("旧回答"), HumanMessage("当前问题"), SystemMessage("系统规则"),
                    *tool_round("current", "新" * 700)]

        # prepared：不是从第一个轮次起点到最后一个轮次终点的大切片替换
        prepared = manager._replace_rounds_with_summary(messages, [(1, 3), (6, 8)], "当前问题")

        self.assertEqual(len(prepared), len(messages))
        # index：两个工具轮次外的全部受保护消息下标
        for index in (0, 1, 3, 4, 5, 6):
            self.assertIs(prepared[index], messages[index])
        self.assertIn('"context_summary":true', prepared[2].content)
        self.assertIn('"context_summary":true', prepared[7].content)
        with self.assertRaises(ContextProtocolError):
            manager._replace_rounds_with_summary(messages, [(1, 8)], "当前问题")

    # 验证当前问题自身超过硬上限时明确报错，而不是暗中截断用户需求
    def test_oversized_current_user_message_is_never_truncated(self):
        # summarizer、manager：没有可压缩历史时仍然保护用户原文
        summarizer = FakeContextSummarizer()
        manager = ContextManager(summarizer, max_context_tokens=1024,
                                 token_counter=fake_token_counter)
        # current、messages：超限的当前问题和一个较短的历史完整工具轮次
        current = HumanMessage("当前需求" * 300)
        messages = [HumanMessage("旧问题"), *tool_round("old", "短结果"),
                    AIMessage("旧回答"), current]

        with self.assertRaises(ContextWindowError):
            manager.prepare_messages(messages, current.content, [])

        self.assertIs(messages[-1], current)
        self.assertEqual(current.content, "当前需求" * 300)
        self.assertEqual(summarizer.session_calls, [])

    # 验证摘要模型失败时仍可使用确定性压缩保护上下文窗口
    def test_summary_failure_uses_deterministic_fallback(self):
        summarizer = FakeContextSummarizer(error=TimeoutError("摘要超时"))
        manager = ContextManager(
            summarizer=summarizer,
            max_context_tokens=1024,
            compression_trigger_ratio=0.75,
            summary_max_chars=256,
            token_counter=fake_token_counter,
        )
        messages = [SystemMessage("system"), HumanMessage("goal")]
        messages.extend(tool_round("call_1", "结果" * 500))

        prepared = manager.prepare_messages(messages, "goal", [])

        self.assertLess(manager.count_tokens(prepared, []), 1024)
        self.assertIn('"compacted":true', prepared[-1].content)
        self.assertEqual(len(summarizer.calls), 1)

    # 验证不可压缩的固定消息超过 256K 类硬上限时明确终止
    def test_uncompressible_context_over_hard_limit_raises(self):
        manager = ContextManager(
            summarizer=FakeContextSummarizer(),
            max_context_tokens=1024,
            compression_trigger_ratio=0.75,
            token_counter=fake_token_counter,
        )
        messages = [SystemMessage("S" * 1100), HumanMessage("goal")]

        with self.assertRaises(ContextWindowError):
            manager.prepare_messages(messages, "goal", [])

    # 验证工具 Schema 本身也包含在压缩触发判断中
    def test_tool_schemas_are_counted_in_context_budget(self):
        manager = ContextManager(
            summarizer=FakeContextSummarizer(),
            max_context_tokens=1024,
            compression_trigger_ratio=0.75,
            token_counter=fake_token_counter,
        )
        messages = [SystemMessage("system"), HumanMessage("goal")]
        # tool_schemas：模拟随每次模型请求重复发送的大型工具定义
        tool_schemas = [{"description": "D" * 1100}]

        with self.assertRaises(ContextWindowError):
            manager.prepare_messages(messages, "goal", tool_schemas)


if __name__ == "__main__":
    unittest.main()
