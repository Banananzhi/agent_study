# 阶段 1A：业务契约与 SQLite 存储交付

日期：2026-10-09。技术路线为同步 Agent + 单个后台 Worker + SQLite；本阶段完成业务存储基础，Worker 与 Agent 业务接入在后续实施。

## 已完成

- 独立 `media_operations/` 包，不导入 Agent、LangGraph、MCP 或网络 SDK；现有核心源码、记忆与检查点数据库保持原样。
- Pydantic 账号、目标、策略、Run/Task、依赖、预算、结果信封和事件模型。禁止未知字段、空白必填值、错误列表/范围、非法依赖、非 JSON/非有限数值和无时区时间；时间保存为 UTC。
- 账号修改保留版本历史，策略修改新增不可变版本。乐观版本检查阻止旧配置覆盖新配置。每个 Run 保存创建时的账号/策略快照，下一次 Run 使用最新配置。
- 独立 SQLite 库，WAL、外键、30 秒锁等待、每操作独立连接及短事务。
- 两份连续版本迁移、checksum 校验、重复执行检查、历史兼容校验与失败回滚。后续研究/内容表随功能增加，不提前创建未使用表。
- 幂等创建 Run：同账号 + 同 key + 同请求返回原 Run；同 key 不同请求抛出 `ConflictError`。任务依赖和输入一起入库。
- 原子 `claim_next_run()` 供后续 Worker 使用，按创建时间领取当前可信身份的 ACTIVE 账号待执行 Run。
- 阶段开始/完成/失败/取消及事件同事务保存。依赖未完成禁止开始，第一版每个 Run 一次执行一个阶段；完成阶段后进入 WAITING_APPROVAL，不自动批准或发布。
- 失败/取消保留已经完成的结果，未执行任务标记 SKIPPED；运行任务标记 FAILED 并保存原因；取消后拒绝迟到结果。
- 重新打开库可查询账号、快照、任务结果和事件；事件序号支持分页补拉。`mark_interrupted_runs()` 为未来独占 Worker 启动提供显式中断标记，不会在普通查询/打开库时自动修改状态。
- 可信 tenant/user 归属由 Repository 构造参数绑定；所有 Run/Task 操作还校验传入 account_id 和 run_id，避免跨账号引用。

## 文件与表

主要文件：`media_operations/schemas.py`、`config.py`、`ports.py`、`persistence/repository.py`、`persistence/migrate.py`、`demo.py`、`data/samples/media_account.json`、三份 `tests/test_media_*.py`。

迁移：

1. `0001_accounts_strategy.sql`：media_account、account_revision、content_strategy、account_goal。
2. `0002_runs_tasks.sql`：agent_run、agent_task、task_dependency、run_event。

schema_migrations 保存版本/文件名/checksum/时间。task_dependency 通过复合外键阻止跨 Run 引用。业务表主键为 account_id、strategy_id、goal_id、run_id、task_id、event_id。账号与策略完整配置、Run 快照及阶段结果以经过模型校验的 JSON 保存；状态、版本、归属、索引和时间为独立列。

## 运行和验证

项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m media_operations.demo
```

默认使用临时数据库并自动清理，无需模型密钥、Qdrant 或网络服务。演示使用明确带 `simulation: true` 的六阶段结果，不生成文章。输出应包含：

```json
{
  "simulation": true,
  "duplicate_returns_same_run": true,
  "restored_status": "WAITING_APPROVAL",
  "events": 21
}
```

完整输出还包括程序生成的账号/Run ID 及六个阶段状态。需要保留演示数据时：

```powershell
.\.venv\Scripts\python.exe -m media_operations.demo --database .agent_data/media_demo.sqlite3
```

每次演示创建一个新示例账号。不要把演示库当作真实运营数据。正式业务库默认 `.agent_data/media_operations.sqlite3`，路径和归属可通过 `MediaSettings.from_env()` 提供；该方法只读取进程环境，不加载 `.env`，也不初始化模型。

最小编程接口：

```python
from media_operations.config import MediaSettings
from media_operations.persistence.repository import MediaRepository
from media_operations.schemas import AccountCreate, RunCreate

settings = MediaSettings.from_env()
store = MediaRepository(settings.database_path, owner=settings.owner)
account = store.create_account(AccountCreate(
    account_name="AI 技术分享",
    positioning="AI Agent 开发实战",
    target_audience="Java 开发者",
    tone="准确、清晰",
    content_pillars=["基础教程", "项目实战"],
    publishing_frequency=3,
))
run = store.create_run(account.account_id, RunCreate(
    idempotency_key="week-2026-10-12",
    goal="调研并制作 MCP 入门图文",
))
# 此处仅创建待执行记录；没有自动调用 Agent。
tasks = store.list_tasks(account.account_id, run.run_id)
```

查询/更新异常：`NotFoundError` 表示不存在或不可访问；`ConflictError` 表示版本/幂等键/状态冲突；Pydantic `ValidationError` 表示无效输入/存储数据。Repository 无常驻连接，不需要 close；每次数据库操作都会关闭连接。

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_media_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

完整回归结果：**177 项全部通过，9.758 秒**，其中阶段 1A 新增 32 项。测试验证迁移失败回滚、版本历史、Run 快照、并发重复提交/领取、并发配置冲突、跨账号/身份隔离、依赖执行、取消后迟到结果、结果与事件事务一致，以及重开库读取。离线演示已验证。没有真实模型/搜索/发布服务验证。

已移除 `.gitignore` 中忽略测试源码的规则，保留缓存、运行数据和输出目录忽略；原本被忽略的测试源码现在可被 Git 跟踪。未执行提交，保留工作区原有修改。干净检出/CI 验证留待后续。

## 限制与下一步

- 当前结果信封中的 data 是严格 JSON 映射，代表存储层边界；具体 ResearchFinding/TopicCandidate/ContentDraft 等输出模型和真实来源校验将在 1B/1C 加入。存入一个成功结果不证明对应业务已经被执行或审核。
- 预算目前保存配置，尚未计量/限制实际模型与工具调用。Worker、任务超时/进程隔离、API、Web、真实生产与产物导出未实现。
- 取消目前是数据库状态和结果提交门禁，不能停止尚未接入的执行进程。中断标记只能由未来唯一 Worker 明确确认旧进程退出后调用；不支持自动断点续跑/重试。
- READY 在 start_task 事务中转入 RUNNING，事件可查；本阶段没有常驻调度循环或跨 Worker 租约。
- WAITING_APPROVAL 仅是生产链完成后的业务状态；人工审核实体、通过/驳回接口及内容版本将随 1C/1D 实现，当前没有发布入口。
- 本地可信身份绑定不是 Web 鉴权；后续应用入口负责身份来源，远程部署需要鉴权与权限控制。

下一工作单元是 **1B：结构化搜索/网页抽取、受控账号知识检索、真实来源快照与工具轨迹**。相关迁移与具体领域模型按实际功能增补，继续复用既有工具执行器及 MCP 接入。
