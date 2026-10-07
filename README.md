# Agent Study

当前项目使用 LangChain 适配 DeepSeek 与消息协议，使用 LangGraph 管理 Agent 状态和循环；自定义执行层继续负责批量调度、资源锁、重试、副作用策略和 Observation 管理。

迁移过程以及手写实现与框架实现的逐项对比见 [LANGCHAIN_MIGRATION.md](LANGCHAIN_MIGRATION.md)。

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

多轮会话使用持久化滚动摘要管理较早历史。当模型输入达到 196608 Token 时触发，尝试压回 131072 Token；默认保留最近 6 个已完成用户轮次的原文，仍过长时逐轮缩小到最近 2 轮，当前未完成轮次始终完整保留。`session_summary` 目标上限为 12288 Token、硬上限为 16384 Token，`summary_cursor` 用于避免重复摘要同一段历史；两者都随 LangGraph Checkpoint 持久化。如果会话摘要后仍超过目标线，再使用原有工具轮次压缩作为第二层保护。

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
