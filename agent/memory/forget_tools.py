from pydantic import BaseModel, Field

from tooling.policy import SideEffectLevel
from tooling.registry import Tool, ObservationPolicy


class MemorySearchTarget(BaseModel):
    """当前可信范围内可以核对的记忆目标。"""
    # id：稳定记忆标识；version：搜索时版本；content：完整正文
    id: str
    version: int = Field(ge=1)
    content: str


class MemorySearchOutput(BaseModel):
    """遗忘目标搜索结果，票据为空时必须澄清，不能执行删除。"""
    # matches：当前范围匹配记忆；ticket：同轮授权票据；message：执行说明
    matches: list[MemorySearchTarget] = Field(default_factory=list)
    ticket: str | None = None
    message: str


class MemoryForgetOutput(BaseModel):
    """遗忘事件已持久化的状态，不返回被遗忘的原始正文。"""
    # event_seq：遗忘事件序号；message：生效与异步同步状态
    event_seq: int
    message: str


# 构建当前 Agent 独有的记忆工具，身份始终来自运行时
# get_service：获取当前轮次的 MemoryForgetService，不接收模型提供的身份
def build_forget_tools(get_service):
    # query：用户希望遗忘的主题，搜索不会执行删除
    def search_memories(query):
        return get_service().search(query)

    # ticket：同轮搜索返回的凭据，执行可恢复的 SQLite 软删除
    def forget_memories(ticket):
        return get_service().forget(ticket)

    # result：独立注册表，不修改全局 TOOLS
    result = {}
    for name, function, parameter, output, description, level in [
        ("search_memories", search_memories, "query", MemorySearchOutput,
         "用户要求遗忘时先搜索目标。包含未索引记忆；票据为空先澄清。无长期记录也可申请清理上下文。",
         SideEffectLevel.NONE),
        ("forget_memories", forget_memories, "ticket", MemoryForgetOutput,
         "仅使用本轮 search_memories 返回的票据遗忘已确认范围。先搜索后执行，不能猜票据。"
         "可恢复软删除而非物理擦除；本轮不保存其他新记忆。请单独调用此工具，不与其他工具并行。",
         SideEffectLevel.LOCAL_WRITE),
    ]:
        result[name] = Tool(
            function=function, display_name=name, output_model=output,
            side_effect_level=level, observation_policy=ObservationPolicy.RAW,
            schema={"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "properties": {
                    parameter: {"type": "string", "minLength": 1, "maxLength": 1000}},
                    "required": [parameter], "additionalProperties": False}}},
        )
    return result
