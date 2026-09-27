import os
import threading
from dataclasses import dataclass
from enum import Enum


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class ResourceAccess:
    namespace: str
    key: str
    mode: AccessMode

    # 校验并规范化资源访问声明
    def __post_init__(self):
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("resource namespace 不能为空")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("resource key 不能为空")
        if not isinstance(self.mode, AccessMode):
            raise ValueError("resource mode 必须是 AccessMode 枚举")


# 判断两个资源访问声明是否必须互斥执行
# left：第一个资源访问声明
# right：第二个资源访问声明
def resource_accesses_conflict(left, right):
    if left.namespace != right.namespace:
        return False

    if left.namespace == "file":
        # left_path：用于比较的规范化左侧文件路径
        left_path = os.path.normcase(os.path.normpath(left.key))
        # right_path：用于比较的规范化右侧文件路径
        right_path = os.path.normcase(os.path.normpath(right.key))
        try:
            # common_path：两个文件路径共有的最长父路径
            common_path = os.path.commonpath((left_path, right_path))
        except ValueError:
            return False
        # same_resource：相同路径或父子路径都视为同一文件资源范围
        same_resource = common_path in {left_path, right_path}
    else:
        # same_resource：非文件资源使用完整 key 判断是否相同
        same_resource = left.key == right.key

    if not same_resource:
        return False
    return not (left.mode == AccessMode.READ and right.mode == AccessMode.READ)


# 判断两组资源访问声明之间是否存在任意冲突
# left_accesses：第一个工具调用需要的资源集合
# right_accesses：第二个工具调用需要的资源集合
def resource_sets_conflict(left_accesses, right_accesses):
    return any(
        resource_accesses_conflict(left, right)
        for left in left_accesses
        for right in right_accesses
    )


class ResourceLease:
    # 初始化一次已成功预留的原子资源租约
    # manager：创建并负责释放租约的锁管理器
    # owner：本次工具调用的唯一所有者标识
    # accesses：租约一次性占用的全部资源
    def __init__(self, manager, owner, accesses):
        self.manager = manager
        self.owner = owner
        self.accesses = tuple(accesses)
        self.released = False

    # 幂等释放本次租约占用的全部资源
    def release(self):
        if not self.released:
            self.manager.release(self)

    # 进入资源租约上下文
    def __enter__(self):
        return self

    # 离开上下文时保证释放资源租约
    # error_type：上下文中异常的类型
    # error：上下文中的异常对象
    # traceback：上下文中异常的调用栈
    def __exit__(self, error_type, error, traceback):
        self.release()


class ResourceLockManager:
    # 初始化进程内共享的资源锁管理器
    def __init__(self):
        self.condition = threading.Condition()
        self.active_leases = []

    # 将资源访问序列固定为经过类型校验的元组
    # accesses：工具调用声明的资源访问序列
    @staticmethod
    def _normalize_accesses(accesses):
        # normalized_accesses：固定为元组的资源访问集合
        normalized_accesses = tuple(accesses)
        if not all(isinstance(access, ResourceAccess) for access in normalized_accesses):
            raise TypeError("accesses 必须全部是 ResourceAccess")
        return normalized_accesses

    # 在已持有 condition 的前提下判断本次资源是否与活动租约冲突
    # accesses：已经规范化的本次资源访问元组
    def _has_conflict(self, accesses):
        return any(
            resource_sets_conflict(accesses, lease.accesses)
            for lease in self.active_leases
        )

    # 不阻塞地尝试一次性预留全部资源
    # owner：本次资源租约的唯一所有者标识
    # accesses：工具调用需要的全部资源
    def try_acquire(self, owner, accesses):
        # normalized_accesses：固定为元组的本次资源访问集合
        normalized_accesses = self._normalize_accesses(accesses)

        with self.condition:
            if self._has_conflict(normalized_accesses):
                return None
            # new_lease：在同一临界区内检查并预留成功的新租约
            new_lease = ResourceLease(self, owner, normalized_accesses)
            self.active_leases.append(new_lease)
            return new_lease

    # 阻塞等待直到可以一次性预留全部资源
    # owner：本次资源租约的唯一所有者标识
    # accesses：工具调用需要的全部资源
    def acquire(self, owner, accesses):
        # normalized_accesses：固定为元组的本次资源访问集合
        normalized_accesses = self._normalize_accesses(accesses)
        with self.condition:
            while self._has_conflict(normalized_accesses):
                self.condition.wait()
            # lease：等待冲突消失后在同一临界区内创建的原子租约
            lease = ResourceLease(self, owner, normalized_accesses)
            self.active_leases.append(lease)
            return lease

    # 释放租约并唤醒所有等待资源变化的调度器
    # lease：需要释放的资源租约
    def release(self, lease):
        with self.condition:
            if lease.released:
                return
            try:
                self.active_leases.remove(lease)
            except ValueError as error:
                raise RuntimeError("资源租约不属于当前锁管理器") from error
            lease.released = True
            self.condition.notify_all()

    # 等待其他批次释放资源，避免无任务运行时忙轮询
    # timeout：最长等待秒数
    def wait_for_change(self, timeout=0.1):
        with self.condition:
            self.condition.wait(timeout)


# DEFAULT_RESOURCE_LOCK_MANAGER：同一 Python 进程内多个 Agent 和 ToolExecutor 共享的默认锁管理器
DEFAULT_RESOURCE_LOCK_MANAGER = ResourceLockManager()
