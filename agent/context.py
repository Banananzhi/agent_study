import hashlib
import json
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately


logger = logging.getLogger(__name__)


class ContextWindowError(RuntimeError):
    """上下文经过压缩后仍然超过 Agent 硬上限。"""


class ContextManager:
    # 初始化全局上下文预算管理器
    # summarizer：用于压缩历史执行轮次的无工具摘要器
    # max_context_tokens：Agent 允许发送给模型的最大上下文 Token
    # compression_trigger_ratio：达到上下文上限的该比例时开始自动压缩
    # summary_max_chars：单份历史执行摘要允许的最大字符数
    # token_counter：兼容 LangChain count_tokens_approximately 签名的计数函数
    def __init__(
        self,
        summarizer,
        max_context_tokens=256 * 1024,
        compression_trigger_ratio=0.75,
        summary_max_chars=4000,
        token_counter=count_tokens_approximately,
    ):
        if type(max_context_tokens) is not int or max_context_tokens < 1024:
            raise ValueError("max_context_tokens 必须是大于等于 1024 的整数")
        if not isinstance(compression_trigger_ratio, (int, float)):
            raise TypeError("compression_trigger_ratio 必须是数字")
        if not 0 < compression_trigger_ratio < 1:
            raise ValueError("compression_trigger_ratio 必须在 0 和 1 之间")
        if type(summary_max_chars) is not int or summary_max_chars < 256:
            raise ValueError("summary_max_chars 必须是大于等于 256 的整数")
        if not callable(token_counter):
            raise TypeError("token_counter 必须可调用")

        self.summarizer = summarizer
        self.max_context_tokens = max_context_tokens
        self.compression_trigger_ratio = float(compression_trigger_ratio)
        self.compression_trigger_tokens = int(
            max_context_tokens * self.compression_trigger_ratio
        )
        self.summary_max_chars = summary_max_chars
        self.token_counter = token_counter
        # summary_cache：按目标和历史内容指纹缓存的上下文摘要，避免重复调用模型
        self.summary_cache = {}

    # 计算消息与工具 Schema 合计占用的近似 Token 数
    # messages：本次准备发送给模型的消息列表
    # tool_schemas：本次模型调用同时绑定的完整工具定义
    def count_tokens(self, messages, tool_schemas=None):
        return self.token_counter(
            messages,
            chars_per_token=2.0,
            use_usage_metadata_scaling=True,
            tools=tool_schemas,
        )

    # 根据 256K 硬上限和触发比例生成本次模型调用的上下文视图
    # messages：LangGraph State 中保留的完整消息记录
    # goal：当前用户任务目标
    # tool_schemas：每次请求都会发送给模型的工具 Schema
    def prepare_messages(self, messages, goal, tool_schemas=None):
        # full_messages：与 State 隔离的消息列表容器，不修改完整历史
        full_messages = list(messages)
        # original_tokens：压缩前消息和工具 Schema 的近似 Token 总量
        original_tokens = self.count_tokens(full_messages, tool_schemas)
        logger.info(
            "📏 当前上下文: %d / %d tokens，压缩线=%d",
            original_tokens,
            self.max_context_tokens,
            self.compression_trigger_tokens,
        )
        if original_tokens < self.compression_trigger_tokens:
            return full_messages

        logger.info("📦 上下文达到 %.0f%%，开始压缩历史执行记录", self.compression_trigger_ratio * 100)
        # rounds：完整 AIMessage(tool_calls) 与对应 ToolMessage 的位置范围
        rounds = self._find_completed_tool_rounds(full_messages)
        # candidate_messages：优先保留最近一轮工具调用后的模型上下文
        candidate_messages = full_messages
        # compacted_rounds：当前已经被历史摘要替代的工具执行轮次数
        compacted_rounds = 0

        if len(rounds) > 1:
            candidate_messages = self._replace_rounds_with_summary(
                full_messages,
                rounds[:-1],
                goal,
            )
            compacted_rounds = len(rounds) - 1

        # candidate_tokens：第一次压缩后模型实际将接收的 Token 数
        candidate_tokens = self.count_tokens(candidate_messages, tool_schemas)
        if candidate_tokens >= self.compression_trigger_tokens and rounds:
            # 最近一轮本身仍导致超限时，将全部已完成轮次合并为一份摘要
            candidate_messages = self._replace_rounds_with_summary(
                full_messages,
                rounds,
                goal,
            )
            compacted_rounds = len(rounds)
            candidate_tokens = self.count_tokens(candidate_messages, tool_schemas)

        if compacted_rounds:
            logger.info("📝 已压缩 %d 个历史工具执行轮次", compacted_rounds)
        if candidate_tokens > self.max_context_tokens:
            raise ContextWindowError(
                "上下文压缩后仍超过 Agent 上限："
                f"{candidate_tokens} > {self.max_context_tokens} tokens"
            )
        if candidate_tokens >= self.compression_trigger_tokens:
            logger.warning(
                "⚠️ 可压缩历史已处理，但固定上下文仍高于压缩线: %d tokens",
                candidate_tokens,
            )
        else:
            logger.info(
                "✂️ 上下文压缩完成: %d → %d tokens",
                original_tokens,
                candidate_tokens,
            )
        return candidate_messages

    # 查找工具调用和全部对应工具结果组成的完整执行轮次
    # messages：按实际对话顺序排列的 LangChain 消息
    @staticmethod
    def _find_completed_tool_rounds(messages):
        # rounds：每项为左闭右开的完整工具执行轮次下标范围
        rounds = []
        # index：当前扫描的消息下标
        index = 0
        while index < len(messages):
            # message：当前待判断是否发起工具调用的消息
            message = messages[index]
            if not isinstance(message, AIMessage):
                index += 1
                continue
            # tool_calls：有效与无效工具调用的完整集合
            tool_calls = [*message.tool_calls, *message.invalid_tool_calls]
            if not tool_calls:
                index += 1
                continue
            # expected_ids：当前 AIMessage 要求返回结果的全部调用标识
            expected_ids = {
                call.get("id")
                for call in tool_calls
                if isinstance(call, dict) and call.get("id")
            }
            # next_index：连续读取当前调用之后的 ToolMessage
            next_index = index + 1
            # observed_ids：实际找到结果消息的工具调用标识
            observed_ids = set()
            while next_index < len(messages) and isinstance(
                messages[next_index], ToolMessage
            ):
                observed_ids.add(messages[next_index].tool_call_id)
                next_index += 1
            if expected_ids and expected_ids.issubset(observed_ids):
                rounds.append((index, next_index))
                index = next_index
                continue
            index += 1
        return rounds

    # 将选中的完整工具执行轮次替换为一对合规的摘要工具消息
    # messages：压缩前的完整消息列表
    # rounds：需要合并压缩的工具执行轮次下标范围
    # goal：摘要时需要保留关键信息的用户目标
    def _replace_rounds_with_summary(self, messages, rounds, goal):
        # start：本次压缩历史在完整消息中的起始下标
        start = rounds[0][0]
        # end：本次压缩历史在完整消息中的结束下标
        end = rounds[-1][1]
        # history_messages：包含工具调用和结果的连续历史消息
        history_messages = messages[start:end]
        # records：移除运行时对象后可稳定摘要和生成指纹的历史记录
        records = [self._message_record(message) for message in history_messages]
        # fingerprint_source：摘要缓存使用的稳定 JSON 原文
        fingerprint_source = json.dumps(
            {"goal": goal, "records": records},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        # fingerprint：不暴露原文且可稳定复用摘要的历史指纹
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        # summary：优先读取缓存，否则调用无工具摘要器压缩历史
        summary = self.summary_cache.get(fingerprint)
        if summary is None:
            try:
                summary = self.summarizer.summarize(
                    tool_name="context_history",
                    value=records,
                    goal=goal,
                    max_chars=self.summary_max_chars,
                )
            except Exception as error:
                logger.warning("⚠️ 上下文摘要失败，回退到确定性压缩：%s", error)
                summary = self._fallback_summary(records)
            self.summary_cache[fingerprint] = summary

        # anchor_start：保留真实 reasoning_content 和工具调用的最后一个压缩轮次起点
        anchor_start = rounds[-1][0]
        # anchor_end：最后一个压缩轮次全部 ToolMessage 之后的下标
        anchor_end = rounds[-1][1]
        # anchor_ai_message：由真实模型生成且不能伪造 reasoning_content 的原始 AIMessage
        anchor_ai_message = messages[anchor_start]
        # anchor_tool_messages：与保留的真实工具调用逐一配对的原始结果消息
        anchor_tool_messages = messages[anchor_start + 1:anchor_end]
        # summary_payload：明确标记为不可信历史数据的压缩结果
        summary_payload = json.dumps(
            {
                "compacted": True,
                "context_summary": True,
                "untrusted_data": True,
                "compacted_rounds": len(rounds),
                "summary": summary,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        # compacted_tool_messages：保留全部原始 tool_call_id 的受限结果消息
        compacted_tool_messages = []
        for index, tool_message in enumerate(anchor_tool_messages):
            # compacted_content：首条携带摘要，其余结果只保留指向摘要的结构标记
            compacted_content = (
                summary_payload
                if index == 0
                else json.dumps(
                    {
                        "compacted": True,
                        "context_summary": True,
                        "untrusted_data": True,
                        "summary_in_tool_call_id": anchor_tool_messages[0].tool_call_id,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            compacted_tool_messages.append(
                ToolMessage(
                    content=compacted_content,
                    tool_call_id=tool_message.tool_call_id,
                    name=tool_message.name,
                    status=tool_message.status,
                )
            )
        return [
            *messages[:start],
            anchor_ai_message,
            *compacted_tool_messages,
            *messages[end:],
        ]

    # 将 LangChain 消息转换成仅包含摘要所需字段的普通字典
    # message：历史工具执行轮次中的一条消息
    @staticmethod
    def _message_record(message):
        # record：用于摘要和缓存指纹的稳定消息记录
        record = {
            "type": message.type,
            "content": message.content,
        }
        if isinstance(message, AIMessage):
            record["tool_calls"] = message.tool_calls
            record["invalid_tool_calls"] = message.invalid_tool_calls
        if isinstance(message, ToolMessage):
            record["name"] = message.name
            record["tool_call_id"] = message.tool_call_id
            record["status"] = message.status
        return record

    # 在摘要模型不可用时生成长度受限且确定性的历史记录
    # records：已经转换为普通字典的历史消息记录
    def _fallback_summary(self, records):
        # source_text：保留工具名、参数和结果结构的紧凑 JSON
        source_text = json.dumps(
            records,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        if len(source_text) <= self.summary_max_chars:
            return source_text
        return source_text[: self.summary_max_chars - 1] + "…"
