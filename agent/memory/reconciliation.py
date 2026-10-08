import logging

from agent.memory.models import IndexStatus, MemoryDecision


logger = logging.getLogger(__name__)


class MemoryReconciler:
    # 组合精确规则、语义判断和事务写入，不让判断模型直接访问数据库
    # service：长期记忆服务；resolver：实现 resolve 的供应商判断适配器
    def __init__(self, service, resolver):
        self.service = service
        self.resolver = resolver

    # 处理一条候选并返回最终执行决策
    # value：可信范围与候选正文；candidate：提取证据；user_message：用户原文
    # tool_observations：本轮受限工具结果，供证据校验使用
    def reconcile(self, value, candidate, user_message, tool_observations):
        # snapshot：包括未索引记忆的完整范围快照，事务提交时再次核对版本
        snapshot = self.service.repository.conflict_snapshot(value)
        try:
            # decision：模型建议经过程序审核后的存储动作
            decision = self._decide(value, candidate, snapshot, user_message, tool_observations)
        except Exception as error:
            # 判断或检索失败不能视为没有旧记忆，也不能阻断其他候选和主任务
            logger.warning("⚠️ 记忆冲突判断失败，暂缓候选：%s", type(error).__name__)
            decision = MemoryDecision(action="DEFER", reason="检索或判断失败，本次未修改记忆")
        decision = self.service.repository.apply_decision(
            value, candidate, decision, snapshot, self.service.embedder.model_name,
        )
        logger.info("🧩 记忆决策: %s，%s", decision.action, decision.reason)
        return decision

    # 先做确定性检查，再调用供应商无关的判断接口
    # value：写入范围；candidate：候选；snapshot：旧记忆；user_message：用户原文
    # tool_observations：本轮工具证据列表
    def _decide(self, value, candidate, snapshot, user_message, tool_observations):
        if candidate.scope_kind == "temporary":
            return MemoryDecision(action="DEFER", reason="仅限当前任务，不更新长期默认值")
        # duplicate：完全相同正文及有效期无需调用判断模型或再次生成向量
        # 重新授权后优先复用较新的相同事实，避免多条旧残留导致再次新增
        duplicate = next((item for item in sorted(snapshot, key=lambda memory: memory.source_seq, reverse=True)
                          if item.content == value.content and item.expires_at == value.expires_at), None)
        if duplicate is not None:
            return MemoryDecision(action="NOOP", target_id=duplicate.id, reason="已有相同正文与有效期")
        # evidence_valid：程序验证来源片段确实存在，避免只有助手回答就更新事实
        evidence_valid = bool(candidate.evidence.strip()) and (
            (candidate.source_kind == "user" and candidate.evidence in user_message)
            or (candidate.source_kind == "tool" and any(
                candidate.evidence in observation for observation in tool_observations
            ))
        )
        if not evidence_valid:
            return MemoryDecision(action="DEFER", reason="缺少可核对的用户或工具原文依据")
        if not snapshot:
            return MemoryDecision(action="ADD", reason="当前范围无有效旧记忆，保存有依据的新事实")

        # related：小范围直接比较全部记录，避免索引延迟漏掉本轮刚写入的记忆
        related = snapshot
        if len(snapshot) > 16:
            # selected：精确键与所有未索引记录优先，不依赖向量同步是否已经完成
            selected = {item.id: item for item in snapshot
                        if item.index_status != IndexStatus.INDEXED
                        or (value.memory_key is not None and item.memory_key == value.memory_key)}
            # recalled：跨键查找语义相关旧记忆，失败会转成 DEFER
            recalled = self.service.recall(
                value.content, value.tenant_id, value.user_id,
                project_id=value.project_id, limit=8, exact_project=True,
            )
            # allowed：召回允许包含用户全局记忆，冲突判断只接受当前精确项目范围
            allowed = {item.id: item for item in snapshot}
            for memory, _score in recalled:
                if memory.id in allowed:
                    selected[memory.id] = allowed[memory.id]
            related = list(selected.values())
        if len(related) > 32 or sum(len(item.content) for item in related) > 24000:
            return MemoryDecision(action="DEFER", reason="待比较记忆超出判断预算，暂不自动覆盖")
        if not related:
            return MemoryDecision(action="ADD", reason="未找到相关旧记忆，保存新事实")

        # decision：适配器必须返回稳定结构，替换 Jev 时无需改变后续存储流程
        decision = MemoryDecision.model_validate(self.resolver.resolve(candidate, related, user_message))
        if decision.action in {"UPDATE", "NOOP"}:
            # target：模型不能引用未提供的 ID 或其他租户、用户、项目的记忆
            target = next((item for item in related if item.id == decision.target_id), None)
            if target is None:
                return MemoryDecision(action="DEFER", reason="判断模型返回了未提供的记忆目标")
        elif decision.target_id is not None:
            return MemoryDecision(action="DEFER", reason="新增或暂缓决策不应指定更新目标")
        if decision.action == "UPDATE" and (
            candidate.source_kind != "user" or candidate.change_intent != "update"
        ):
            return MemoryDecision(action="DEFER", reason="缺少用户明确修改长期事实的依据")
        return decision
