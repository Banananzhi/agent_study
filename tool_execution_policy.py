from dataclasses import dataclass
from enum import Enum


class SideEffectLevel(str, Enum):
    # NONE：只读取或计算，不改变本地及外部状态
    NONE = "none"
    # LOCAL_WRITE：创建或修改当前工作区内的数据
    LOCAL_WRITE = "local_write"
    # EXTERNAL_WRITE：改变远程系统、账号或其他外部状态
    EXTERNAL_WRITE = "external_write"
    # DESTRUCTIVE：删除、支付或执行其他高风险且难恢复的操作
    DESTRUCTIVE = "destructive"


class PolicyAction(str, Enum):
    # ALLOW：允许工具继续进入资源调度和执行阶段
    ALLOW = "allow"
    # REQUIRE_APPROVAL：当前不能执行，需要先获得用户审批
    REQUIRE_APPROVAL = "require_approval"
    # DENY：无论是否审批，当前策略都禁止执行
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    # action：策略对本次工具调用作出的处理动作
    action: PolicyAction
    # reason：供日志或稳定错误结果使用的中文原因
    reason: str

    # 校验策略决定包含有效动作和非空原因
    def __post_init__(self):
        if not isinstance(self.action, PolicyAction):
            raise TypeError("action 必须是 PolicyAction 枚举")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason 不能为空")


class ToolExecutionPolicy:
    # 初始化不同副作用等级对应的执行动作
    # level_actions：覆盖默认副作用等级策略的映射
    def __init__(self, level_actions=None):
        # default_actions：当前阶段默认允许无副作用和本地写入的策略表
        default_actions = {
            SideEffectLevel.NONE: PolicyAction.ALLOW,
            SideEffectLevel.LOCAL_WRITE: PolicyAction.ALLOW,
            SideEffectLevel.EXTERNAL_WRITE: PolicyAction.REQUIRE_APPROVAL,
            SideEffectLevel.DESTRUCTIVE: PolicyAction.REQUIRE_APPROVAL,
        }
        # configured_actions：调用方传入并准备覆盖默认值的策略映射
        configured_actions = {} if level_actions is None else dict(level_actions)
        for level, action in configured_actions.items():
            if not isinstance(level, SideEffectLevel):
                raise TypeError("level_actions 的键必须是 SideEffectLevel")
            if not isinstance(action, PolicyAction):
                raise TypeError("level_actions 的值必须是 PolicyAction")
        default_actions.update(configured_actions)
        self.level_actions = default_actions

    # 判断一次工具调用是否允许进入资源调度和执行阶段
    # tool：已经从注册表找到的工具定义
    # args：已经通过输入 Schema 校验的工具参数
    def evaluate(self, tool, args):
        # level：工具注册时声明的副作用等级
        level = tool.side_effect_level
        # action：当前执行策略为该副作用等级配置的处理动作
        action = self.level_actions[level]
        if action == PolicyAction.ALLOW:
            reason = f"允许执行 {level.value} 级工具"
        elif action == PolicyAction.REQUIRE_APPROVAL:
            reason = f"{level.value} 级工具需要用户审批"
        else:
            reason = f"执行策略禁止 {level.value} 级工具"
        return PolicyDecision(action, reason)
