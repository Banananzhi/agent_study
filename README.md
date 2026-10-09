# Agent Study

当前项目使用 LangChain 适配 DeepSeek 与消息协议，使用 LangGraph 管理 Agent 状态和循环；自定义执行层继续负责批量调度、资源锁、重试、副作用策略和 Observation 管理。

迁移过程以及手写实现与框架实现的逐项对比见 [LANGCHAIN_MIGRATION.md](LANGCHAIN_MIGRATION.md)。

## 自媒体业务：阶段 1A / 1B

已新增独立 `media_operations/` 业务包，支持账号与策略版本、SQLite 事务迁移、Run/Task 输入与结果存储、幂等提交、任务依赖、事件查询、失败与取消记录。保留现有 Agent、工具、记忆数据库和命令行入口。

阶段 1B 新增可注入现有 Agent 的三个工具：`search_web`、`extract_web_page`、`search_account_knowledge`，支持结构化来源、来源快照和工具执行轨迹。工具绑定当前账号及正在运行的 research Task；正文和摘要标为不可信资料，取得内容不表示事实已核实。未修改 Agent 核心或原 CLI 的工具配置。

运行离线研究演示（脚本模型 + 模拟 HTTP，通过现有 Agent 图实际执行工具）：

```powershell
.\.venv\Scripts\python.exe -m media_operations.research_demo
.\.venv\Scripts\python.exe -m media_operations.research_demo --database .agent_data/research_demo.sqlite3
```

输出 `simulation: true`、三种来源及三条执行轨迹。真实搜索使用 `BOCHA_API_KEY`，本地知识放在 `MEDIA_KNOWLEDGE_ROOT/<account_id>/**/*.md`，默认根目录为 `.agent_data/media_knowledge`。配置加载由入口负责；接入方法与限制见 [阶段 1B 交付说明](docs/PHASE_1B_RESEARCH.md)。

运行离线存储演示（不需要模型/API Key，默认临时数据库自动清理）：

```powershell
.\.venv\Scripts\python.exe -m media_operations.demo
```

演示创建示例账号，保存六个**模拟**阶段结果，并重新打开数据库验证读取；输出 `simulation: true`、幂等检查和 `WAITING_APPROVAL`。这不代表真实调研、文章生成或人工审批已经完成。

需要保留演示数据库时，显式指定路径；每次演示会创建新示例账号：

```powershell
.\.venv\Scripts\python.exe -m media_operations.demo --database .agent_data/media_demo.sqlite3
```

业务库默认路径为 `.agent_data/media_operations.sqlite3`；`MediaSettings.from_env()` 支持 `MEDIA_DATABASE_PATH` 及可信身份 `AGENT_TENANT_ID` / `AGENT_USER_ID`。配置加载与依赖组装由后续应用入口负责，业务包不隐式读取 `.env`。

运行业务测试或全部回归：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_media_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前完整回归：222 项用例，221 项通过，1 项因 Windows 符号链接权限不足跳过；其中业务用例 77 项。阶段 1A 的存储基线为 177 项通过。尚未进行真实外部搜索验证，也未实现内容生产闭环、Worker、API 或 Web；取消目前在调用/提交边界检查。存储说明见 [阶段 1A 交付说明](docs/PHASE_1A_STORAGE.md)，下一步为 [实施计划](docs/IMPLEMENTATION_PLAN.md) 的 1C 生产闭环。

启动时还会使用 FastMCP Client 连接只读的 DeepWiki 公共 MCP Server，通过 `list_tools()` 动态注册以下工具：

- `deepwiki__ask_wiki_question`
- `deepwiki__read_wiki_contents`
- `deepwiki__read_wiki_structure`

连接失败时自动降级为只使用本地工具。可通过 `.env` 中的 `DEEPWIKI_MCP_URL` 覆盖默认地址 `https://mcp.deepwiki.com/mcp`。

## 项目结构

```text
agent-study/
├─ main.py                  # 命令行入口和依赖组装
├─ agent/                   # Agent 编排层
│  ├─ runtime.py            # LangGraph、AgentState 和模型/工具节点
│  ├─ config.py             # .env 配置加载
│  ├─ context.py            # 全局上下文预算、计数和压缩
│  └─ summarizer.py         # 长工具结果摘要
├─ tooling/                 # 工具领域与执行基础设施
│  ├─ registry.py           # 本地工具定义、Schema 和注册表
│  ├─ models.py             # Pydantic 工具返回模型
│  ├─ result.py             # ToolResult 和稳定错误码
│  ├─ errors.py             # 已分类工具异常契约
│  ├─ policy.py             # 副作用等级与执行策略
│  ├─ executor.py           # 单工具校验、执行和重试
│  ├─ scheduler.py          # 批量资源感知调度
│  └─ resources.py          # 资源声明与进程内资源锁
├─ integrations/            # 外部协议与服务适配
│  └─ mcp.py                # FastMCP Client 和 MCP Tool 适配
└─ tests/                   # Agent、工具和执行层测试
```

依赖方向保持单向：`main → agent/integrations → tooling`。工具基础设施不依赖 Agent 编排层，后续增加其他模型、MCP Server 或调用入口时不需要改动底层执行策略。

## 工具

工具统一定义和注册在 `tooling/registry.py`：

- `calculator`：基础算术计算
- `web_search`：通过博查搜索实时网页信息
- `read_webpage`：读取公开网页正文
- `get_current_time`：获取指定时区的当前时间
- `create_file`：在项目工作区内创建新文件，拒绝覆盖已有文件
- `write_file`：使用字符偏移覆盖首段或追加写入后续分段
- `read_file`：使用字符偏移分页读取工作区内的 UTF-8 文本文件

每个工具都定义输入 Function Tool Schema 和 Pydantic 业务返回模型，再通过 `register_tool()` 与实际 Python 函数绑定。输入 Schema 会通过 DeepSeek API 的 `tools` 字段发送给模型；Pydantic 模型会自动生成 Output Schema，其语义说明会追加到工具 `description`。ToolExecutor 会在执行前校验输入参数，在执行后使用 `model_validate()` 校验业务返回值，再用 `model_dump(mode="json")` 规范化 `ToolResult.value`；不符合约定的返回值会转换为 `invalid_output` 失败。

文件分页使用 Unicode 字符下标。`read_file()` 在 `has_more=true` 时返回下一页的 `next_offset`；`create_file()` 和 `write_file()` 同样返回下一段写入位置。`write_file(offset=0)` 覆盖首段，后续调用只能使用当前文件长度作为 `offset` 追加，偏移不匹配时拒绝写入，避免分段乱序或产生内容空洞。

工具通过 `observation_policy` 声明长结果处理策略：`web_search` 和 `read_webpage` 使用 `summarize`，`read_file` 使用 `paginate`，短结构化工具使用 `raw`。公共 `ResultSummarizer` 会先按块摘要完整业务结果，再汇总各块摘要；摘要请求不携带工具权限。摘要服务失败时回退到统一截断，错误类 ToolResult 始终保留原有错误码和恢复字段，不交给模型改写。

全局 `ContextManager` 将 Agent 主动使用的上下文限制为 256K Token，并在达到 75%（196608 Token）时自动压缩较早的完整工具执行轮次。Token 估算使用 LangChain `count_tokens_approximately()`，同时计算消息、工具调用参数和每轮请求都会携带的 Tool Schema。LangGraph State 始终保留完整消息，只有本次发给模型的上下文视图会被压缩；压缩结果仍以成对的 `AIMessage(tool_calls)` 和 `ToolMessage` 表达，避免破坏 Function Calling 消息协议。

多轮会话使用持久化滚动摘要管理较早历史。当模型输入达到 196608 Token 时触发，尝试压回 131072 Token；会话级摘要默认保留最近 6 个已完成用户轮次，仍过长时逐轮缩小到最近 2 轮，不归档当前用户轮次。`session_summary` 目标上限为 12288 Token、硬上限为 16384 Token，`summary_cursor` 只能位于已完成历史用户轮次的末尾，避免重复摘要或隐藏当前问题；两者都随 LangGraph Checkpoint 持久化。

如果会话摘要后仍超过目标线，第二层按从旧到新的顺序逐个压缩完整工具批次，达到目标即停止；必要时也可摘要当前任务较早的结果，最新完整批次最后才处理。每个批次独立摘要，只替换 `ToolMessage.content`，保留原始 `AIMessage`（含参数与思考元数据）、全部 `tool_call_id`、结果状态和其他元数据；批次之间的用户问题、普通回答、系统消息不会被整段替换。因此最近轮次的工具结果在预算紧张时仍可能变成摘要，但用户问题与普通回答不会被第二层删除。摘要失败时按批次确定性降级，摘要包装反而更长则保留原文。未完成工具批次不参与压缩，发送模型前会检查协议并以 `ContextProtocolError` 拦截缺失结果、孤立结果、错误关联或重复 ID，不伪造工具结果。原始 State 不受视图压缩影响，256K 硬上限和现有触发比例保持不变。

上下文预算采用“输入上限 + 输出预留”的口径：`AGENT_CONTEXT_TOKENS` 仍是输入上限，主模型默认预留 16384 Token，并在创建及绑定模型时实际设置请求的 `max_tokens`。有效输入上限为 `min(AGENT_CONTEXT_TOKENS, DEEPSEEK_CONTEXT_TOKENS - AGENT_MAX_OUTPUT_TOKENS)`；默认模型窗口为 1000000，因而 262144 输入上限、196608 压缩线、131072 目标线不变。换成小窗口模型后，输入上限与两条压缩线会自动下调。输出预留是最大生成额度，不是每次必定使用；thinking 内容是否计入该额度遵循供应商 API 的口径。

| 输入分区 | 默认预算 | 超限行为 |
| --- | --- | --- |
| 主系统及其他非记忆系统提示 | 8192 Token，软预算 | 只告警，保留完整规则 |
| 全部工具 Schema | 32768 Token，软预算 | 只告警，不静默删除工具或截断定义 |
| 召回长期记忆及包装提示 | 4096 Token，硬上限 | 按相关度选择能完整放入预算的条目；同时遵守原有字符上限 |
| 会话摘要及包装提示 | 12288 Token 目标、16384 Token 硬上限 | 增量摘要与确定性降级，恢复旧摘要时同样校验 |
| 历史与当前任务 | 扣除实际固定占用后的共享剩余空间 | 使用会话摘要与安全工具压缩，不截断当前问题 |

分区不是固定切块：没有使用的系统或 Schema 额度不会预先扣除；超过软预算的固定信息仍占用共享输入空间，只能压缩其他允许压缩的历史。每次模型调用记录各分区估算占用、共享剩余预算和输出预留；各分区独立估算可能存在非加性开销，硬上限始终以完整消息和完整 Schema 的统一估算为准。无可靠条目边界的旧记忆块超限时整体舍弃，而不是按换行猜测或截断半条事实。辅助模型暂不纳入这套统一输入治理。

以下为新增预算配置示例，未配置时上述默认值自动生效：

```dotenv
AGENT_CONTEXT_TOKENS=262144
AGENT_MAX_OUTPUT_TOKENS=16384
AGENT_SYSTEM_CONTEXT_TOKENS=8192
AGENT_TOOL_SCHEMA_CONTEXT_TOKENS=32768
AGENT_MEMORY_RECALL_MAX_TOKENS=4096
```

供应商报告 `finish_reason=length`（或同义长度结束原因）时，Agent 抛出 `AgentModelOutputError`，不把该响应持久化为完整答案、不执行其中的工具计划、不提取本轮记忆。命令行显示明确原因并允许继续提问，不自动重放可能具有副作用的操作。全部可压缩内容处理后仍超过输入硬上限时，异常会列出各分区占用以便诊断，不擅自截断系统或用户需求。

同一轮的多个原生 `tool_calls` 由 `ToolBatchExecutor` 使用线程池动态调度，默认最大并行数为 4，可通过 `Agent(max_parallel_tools=...)` 调整。调度器只将已经一次性获得全部资源 Lease 的调用提交 Worker，等待锁的调用保留在调度队列中，不占用线程；后续无冲突调用可以先执行，但同资源冲突调用始终保持模型生成时的顺序。`read_file` 申请文件 READ 资源，`create_file` 和 `write_file` 申请 WRITE 资源，因此同文件读读可以并行、读写和写写互斥、不同文件可以并行。最终 ToolResult 与 tool message 始终按原始 `tool_calls` 顺序返回。

工具还通过 `side_effect_level` 独立声明副作用等级：`none`、`local_write`、`external_write` 或 `destructive`。`ToolExecutionPolicy` 在工具进入资源调度和线程池前进行程序化判断；当前默认允许无副作用和本地写入工具，外部写入及破坏性工具返回 `approval_required`，不会实际执行。该字段只负责权限和风险控制，并行关系仍由具体资源声明决定。

MCP 适配层会在协议边界将底层 SDK、HTTP 和远程工具错误转换为稳定错误码：参数错误与远程业务错误返回模型修正；连接、超时、限流和服务端错误仅允许幂等工具按策略自动重试；认证和协议错误不能通过修改调用恢复，Agent 会立即终止当前任务。`ToolExecutor` 只读取统一的 `ClassifiedToolError`，不依赖具体 MCP SDK。

使用搜索前，需要在 `.env` 中设置博查 API Key：

```dotenv
BOCHA_API_KEY=你的博查API-Key
BOCHA_BASE_URL=https://api.bochaai.com/v1/web-search
```

## 运行

### 启动 Qdrant

项目根目录提供了 `compose.yaml`，安装并启动 Docker Desktop 后执行：

```bash
docker compose up -d
```

Qdrant 默认地址：

```text
HTTP API: http://localhost:6333
gRPC API: localhost:6334
Dashboard: http://localhost:6333/dashboard
```

查看运行状态或停止服务：

```bash
docker compose ps
docker compose down
```

向量数据持久化在 `.agent_data/qdrant`。本地开发默认不需要配置
`QDRANT_API_KEY`；如果端口被占用，可以在 `.env` 中覆盖：

```dotenv
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=agent_memories
```

### 长期记忆存储

长期记忆使用 SQLite 保存完整文本与生命周期信息，Qdrant 只保存向量和
租户、用户等过滤字段。记忆与索引任务通过 SQLite Outbox 在同一事务中写入，
索引 Worker 负责异步生成向量并幂等同步到 Qdrant。

在 `.env` 中配置：

```dotenv
AGENT_MEMORY_DB_PATH=.agent_data/memories.sqlite3
AGENT_TENANT_ID=local
AGENT_USER_ID=local-user
AGENT_PROJECT_ID=agent-study

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=agent_memories

AGENT_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
AGENT_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_EMBEDDING_API_KEY=你的百炼API-Key
AGENT_EMBEDDING_TIMEOUT=30
# 可选；留空时使用模型默认向量维度
AGENT_EMBEDDING_DIMENSIONS=
AGENT_MEMORY_WORKER_BATCH_SIZE=16
AGENT_MEMORY_WORKER_LEASE_SECONDS=120
AGENT_MEMORY_WORKER_MAX_ATTEMPTS=5
AGENT_MEMORY_WORKER_POLL_SECONDS=2
AGENT_MEMORY_RECALL_LIMIT=8
AGENT_MEMORY_RECALL_MAX_CHARS=8192
AGENT_MEMORY_EXTRACTION_MAX_CHARS=24000
AGENT_MEMORY_MIN_IMPORTANCE=0.5
AGENT_MEMORY_MIN_CONFIDENCE=0.7
```

独立启动索引 Worker：

```bash
uv run python -m agent.memory.worker
```

索引 Worker 通过百炼 OpenAI 兼容接口调用 `qwen3.7-text-embedding-flash`，
不会在本地下载模型。也可以用 `DASHSCOPE_API_KEY` 代替
`AGENT_EMBEDDING_API_KEY`。每个新用户轮次开始时只召回一次长期记忆，最终答案
生成后只提取一次候选记忆；召回和提取失败不会终止主任务。

### 记忆冲突与更新

自动提取的候选先经过 `MemoryReconciler`，不再直接按键覆盖：

1. 提取原子事实、原文证据、来源、长期/临时范围和修改意图。
2. 完全重复直接 `NOOP`；临时要求、缺少用户或工具原文依据的候选 `DEFER`。
3. 同租户、用户、精确项目范围内匹配旧记忆。最多 16 条时直接比较全部；更多时合并
   精确键、全部未索引记录和 Qdrant 前 8 条语义结果。超过 32 条或正文总量 24000 字符时暂缓。
4. 独立的 DeepSeek `MemoryConflictResolver` 返回 `ADD / NOOP / UPDATE / DEFER`。
   提取器和判断器各自关闭 thinking，主 Agent 配置不变；判断器失败或检索失败会暂缓保存。
5. 程序核对目标 ID、证据和明确修改意图；提交事务时再核对范围快照的 ID 与版本。
   并发变化时暂缓，避免旧决策覆盖新事实。ADD 不允许隐式覆盖已占用的稳定键。
6. UPDATE 保持旧 ID、类型和稳定键，旧记录归档到 `memory_versions`，正文、历史、
   `memory_decisions` 审计与 Outbox 一起提交。Worker 复用现有版本化索引同步。

`MemoryConflictResolver.resolve(candidate, memories, user_message)` 是供应商适配边界；
后续 Jev 可返回同一个 `MemoryDecision`，不必改数据库事务与权限检查。
`LongTermMemoryService.remember()` 保留为可信程序写入接口；自动提取必须走上述判断管线。

数据库启动时自动创建新增表，不改写已有记忆，也不会补齐更早版本的历史。
`DEFER` 仅保存候选与原因，不进入召回、不自动重放。新的明确用户表达可在后续轮次重新判断。
语义判断仍取决于模型和检索质量，证据片段检查只验证来源存在，不等于证明内容理解绝对正确。
当前范围快照从 SQLite 读取完整有效记录，适合学习项目规模；后续大量记忆时可优化为范围修订号与分页候选查询。

在 DBeaver 中可以查看决策和旧版本：

```sql
SELECT action, reason, target_id, candidate_json, created_at
FROM memory_decisions ORDER BY created_at DESC LIMIT 20;

SELECT memory_id, version, snapshot_json, archived_at
FROM memory_versions ORDER BY archived_at DESC LIMIT 20;
```

手工验证：先记住“文案最多45字”，再表达“以后改为80字”，检查原记录 ID 不变、版本递增、
历史表保留45字、Worker 索引新版本；重复表达80字不应新增版本；“仅本次允许100字”不应覆盖长期默认值。

### 统一记忆遗忘

用户可以说“忘记我的文案字数限制”。Agent 先调用 `search_memories`，目标明确后使用
本轮搜索票据调用 `forget_memories`。身份与项目从可信运行时注入；不接受任意用户 ID、
猜测的记忆 ID 或跨轮次票据。目标有歧义时先澄清，第一版不提供全量清空和跨项目删除。
遗忘工具属于可恢复的本地软删除，必须单独成批执行。

遗忘包含以下处理：

- SQLite 同事务保存遗忘事件、旧版本历史、软删除和 Qdrant DELETE Outbox；版本变化会拒绝整批操作。
- 当前临时召回清空；其他未精确命中 ID 但属于被遗忘主题的旧记忆，也在召回与搜索阶段过滤。
- 最近原始消息由专用 DeepSeek 请求定位并删除片段，随后复核；无法可靠定位、超出 24000 字符预算
  或涉及工具调用配对时，屏蔽整个用户轮次，避免破坏 Function Calling 协议。
- 第一版保守清空旧会话摘要和摘要游标，而不是直接修补摘要；后续只用清理后的消息重建。
- 成功遗忘的本轮跳过自动提取。同轮还要求新增偏好时，请在下一轮重新提供。
- 旧会话恢复、主模型调用前、工具实际执行前和答案交付前检查遗忘版本。模型请求期间发生遗忘，
  丢弃过期结果并请用户重新提问，不自动重放可能产生副作用的任务。已经开始的外部操作无法撤回。

新增 `memory_events` 表保存 `turn` 和 `forget` 事件。数据库单调序号用于区分原始来源，
不是用摘要生成时间或提取时间判断信息新旧。`memories.source_seq` 保存原始轮次序号；
迁移前记录默认是 0，按未知旧来源处理。当前 Checkpoint 保存 `applied_forget_seq`，旧会话惰性应用
尚未处理的事件。同一持久化 thread_id 绑定租户、用户与项目，不能用切换身份读取旧会话。

遗忘主题不是永久黑名单。`MemoryWriteGate` 只允许遗忘后的用户明确新长期要求重新进入写入管线：
例如“以后文案最多80字”可以建立新 ID；“以前是不是45字”、助手复述、工具返回或旧摘要不能恢复。
即便出现新记忆，旧遗忘事件也继续生效，不恢复旧 ID。检索、判断失败时保守过滤或暂缓。
正文语义分类仍依赖模型，原文片段验证不等于绝对语义正确。

这是**逻辑遗忘，不是物理擦除**：旧 Checkpoint、历史版本、已有审计和备份不会被批量删除。
清理专用模型仍需处理旧文本以定位片段；普通任务模型只接收清理后的有效上下文。
主系统提示词和外部文件不属于本次记忆清理范围，不应将用户秘密硬编码在系统提示词中。
会话来源按完整用户轮次追踪；摘要通过清空重建避免混合来源误判。

新增表和字段在启动时自动迁移，请重启 Agent 与 Worker。在 DBeaver 可执行：

```sql
SELECT seq, tenant_id, user_id, project_id, turn_id, payload
FROM memory_events WHERE kind = 'forget' ORDER BY seq DESC;

SELECT id, content, status, version, source_seq, index_status
FROM memories ORDER BY updated_at DESC;
```

验证步骤：记住45字限制 → 要求遗忘 → 检查记录为 `deleted`、事件已保存 → 当前及旧会话恢复后
不再使用45字 → 下一轮明确“以后最多80字” → 出现新 ID、新来源序号，旧45字仍失效。
Worker 未同步成功时，SQLite 也会立即阻止旧正文回填；晚到的旧 Worker 会重新安排最新版本同步。

先编辑项目根目录的 `.env`，将占位符替换成自己的真实 DeepSeek API Key：

```dotenv
DEEPSEEK_API_KEY=sk-你的真实API-Key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CONTEXT_TOKENS=1000000
AGENT_CONTEXT_TOKENS=262144
AGENT_CONTEXT_COMPRESSION_RATIO=0.75
AGENT_CONTEXT_TARGET_RATIO=0.5
AGENT_RECENT_TURNS=6
AGENT_MIN_RECENT_TURNS=2
AGENT_SESSION_SUMMARY_TARGET_TOKENS=12288
AGENT_SESSION_SUMMARY_MAX_TOKENS=16384
AGENT_CHECKPOINT_PATH=.agent_data/checkpoints.sqlite3
AGENT_THREAD_ID=default
```

然后运行：

```powershell
python main.py
```

启动后会进入多轮命令行对话，输入 `exit`、`quit` 或“退出”结束程序。LangGraph SQLite Checkpointer 按 `AGENT_THREAD_ID` 保存完整 State；使用相同 `thread_id` 重启程序后，Agent 仍能恢复之前的 `messages`。更换 `AGENT_THREAD_ID` 可以开始一个独立会话。检查点文件默认保存在 `.agent_data/checkpoints.sqlite3` 并被 Git 忽略。

每次用户输入仍是一个独立任务，因此 `step`、工具纠错次数和重复 Action 指纹都会在新一轮重置；之前的对话消息则会作为会话上下文继续传给模型。

运行期间会输出中文日志，包括步骤、模型名、模型是否决定调用工具、Action、工具执行、Observation 和最终答案；不会伪造模型的内部 Thought，API Key 也不会写入日志。

`.env` 已被 Git 忽略。如果你使用的第三方服务确实提供 `deepseek-flash`，请在 `.env` 中同时修改模型名和服务地址。

`agent/runtime.py` 包含 `Agent`、`AgentState` 和 LangGraph 节点：`think` 返回标准 `AIMessage`，自定义 `tools` 节点解析并执行 `tool_calls`，`run` 启动编译后的图。工具结果会通过包含 `tool_call_id` 的 `ToolMessage` 回传模型；模型不再返回 `tool_calls` 时，其 `content` 就是最终答案。`main.py` 只负责连接 MCP、组装依赖、创建 Agent 和调用 `run`。

Agent 会使用“工具名称 + 按 Schema 补齐默认值后的标准参数”生成 Action 指纹。如果上一个 Action 已经成功，模型又立即生成完全相同的调用，Agent 会拦截重复执行并通过 `repeated_action` Observation 要求模型使用已有结果或调整调用。

单条 Observation 默认限制为 8000 个字符，可通过 `Agent(max_observation_chars=...)` 调整。超限时会保留合法 JSON 结构，优先截断成功结果的 `value` 或失败结果的 `error.message`，并在 `truncation` 中返回原始长度和实际保留长度。

默认中文系统提示词定义在 `agent/runtime.py` 的 `SYSTEM` 常量中，也可以通过 `Agent(system="...")` 覆盖。
