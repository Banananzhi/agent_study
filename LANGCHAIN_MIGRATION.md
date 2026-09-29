# LangChain / LangGraph 迁移说明

## 迁移目标

本次迁移直接替换原有手写 Agent 运行时，不保留两套实现。迁移后的职责划分如下：

```text
LangChain
├─ DeepSeek 模型适配
├─ bind_tools 工具绑定
└─ AIMessage / ToolMessage 消息协议

LangGraph
├─ AgentState 状态传递
├─ model → tools → model 循环
├─ 条件路由
└─ 消息 reducer

项目自定义执行层
├─ ToolExecutor / ToolBatchExecutor
├─ ToolResult 和错误恢复
├─ Action 去重与自动重试
├─ 参数、返回值校验与资源锁
├─ 副作用策略
└─ Observation 摘要、分页与截断
```

主要版本：`langchain 1.4.2`、`langchain-core 1.6.5`、`langchain-deepseek 1.1.1`、`langgraph 1.2.12`、`fastmcp 4.0.10`。

## 修改内容

### `pyproject.toml` 和 `uv.lock`

新增 LangChain、LangGraph 和 DeepSeek 模型适配依赖，并由 uv 更新锁文件。

迁移时验证了 `langchain-mcp-adapters 0.3.1`，但它与当前 FastMCP 使用的 MCP SDK 存在 `RequestContext` 导入冲突，因此没有保留。后续 MCP Client 直接基于 FastMCP 实现，再适配到项目工具注册表。

### `agent/runtime.py`

- 删除 `urllib.request` 和手工 DeepSeek HTTP 请求。
- 使用 `ChatDeepSeek` 和 `bind_tools()`。
- 使用 `AIMessage`、`HumanMessage`、`SystemMessage`、`ToolMessage`。
- 新增显式 `AgentState`。
- 使用 `StateGraph` 构建 `model`、`tools` 节点和条件边。
- 使用 `add_messages` reducer 维护消息历史。
- 删除手写 `for step in range(...)` 循环。
- 保留自定义工具批量调度、安全策略和 Observation 管理。

### `tests/test_agent.py`

- 原始消息字典断言改为 LangChain Message 属性断言。
- 原始 HTTP Mock 改为 ChatModel 和 `bind_tools()` Mock。
- 非法参数使用 `AIMessage.invalid_tool_calls` 模拟。
- 原有批量调用、错误恢复、Action 去重和长结果测试继续保留。

## Agent Loop 对比

迁移前由 `Agent.run()` 手动维护循环：

```python
for step in range(1, self.max_steps + 1):
    message = self.think(messages)
    messages.append(message)
    if not tool_calls:
        return answer
    results = execute_tools(tool_calls)
    messages.extend(tool_messages)
```

程序需要自行负责循环、消息追加、工具消息关联、结束判断和状态变量。

迁移后由 LangGraph 表达流程：

```text
START
  ↓
model
  ├─ 有 tool_calls → tools ─┐
  └─ 无 tool_calls → END    │
                            └→ model
```

LangGraph 负责执行节点、沿条件边路由、传递 State 和调用 reducer。`max_steps` 是项目业务限制，仍在 `model` 节点检查。

## 模型调用对比

迁移前手动构造请求体、认证头和 URL，并解析供应商 JSON：

```python
body = {"model": model, "messages": messages, "tools": schemas}
urllib.request.urlopen(...)
```

迁移后使用：

```python
chat_model = ChatDeepSeek(...)
bound_model = chat_model.bind_tools(tool_schemas, tool_choice="auto")
message = bound_model.invoke(messages)
```

LangChain 负责 DeepSeek 请求格式、消息序列化、Function Calling 响应解析和 `AIMessage` 转换。项目仍决定模型名称、超时、重试设置和工具集合。

## 消息协议对比

迁移前使用普通字典：

```python
{"role": "assistant", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "call_1", "content": "..."}
```

迁移后使用：

```python
AIMessage(tool_calls=[...])
ToolMessage(tool_call_id="call_1", content="...")
```

LangChain 负责标准消息对象和供应商格式转换。无法解析的工具参数通过 `AIMessage.invalid_tool_calls` 进入原有模型纠错流程。

## 状态管理对比

迁移前，`messages`、`step`、`consecutive_recoveries` 和 `last_successful_signatures` 都是 `run()` 的局部变量。

迁移后统一定义在 `AgentState`。每个节点只返回自己修改的字段；`messages` 使用 `add_messages` 追加，其他字段覆盖更新。

当前 Graph 尚未配置 Checkpointer，因此任务结束后仍不保存状态。以后增加短期记忆或审批恢复时，可在编译 Graph 时接入 Checkpointer，无需再次改写 Agent Loop。

## 工具调用对比

LangChain 已接管：

- 工具 Schema 与模型绑定。
- 模型调用解析为 `AIMessage.tool_calls`。
- 使用标准 `ToolMessage` 返回 Observation。
- DeepSeek 消息格式适配。

项目继续负责：

- 整批 Action 预处理和跨轮去重。
- 等待资源锁的任务不进入线程池。
- 同资源冲突调用保持顺序。
- 稳定 `ToolResult` 错误码和模型恢复路由。
- 幂等工具自动重试和 Pydantic 返回值校验。
- 副作用等级、长结果摘要、分页和截断。

## 为什么没有直接使用 `create_agent`

预构建 `create_agent` 适合标准模型—工具循环，但默认 ToolNode 不能直接表达当前项目的资源调度要求：

```text
等待资源锁的任务不能占用 Worker
同资源冲突调用保持模型原始顺序
整批 Action 在执行前统一去重
ToolResult 错误策略决定恢复或终止
```

如果完全换成默认 ToolNode，需要通过 Middleware 或自定义 ToolNode 把这些能力重新补回。当前直接使用 `StateGraph` 自定义 `tools` 节点更清晰。

## 保留与删除总结

| 能力 | 结果 | 原因 |
|---|---|---|
| DeepSeek HTTP 请求 | 删除 | `ChatDeepSeek` 已适配 |
| 原始消息字典 | 删除 | 改用 LangChain Message |
| 手写 Agent 循环 | 删除 | 改用 LangGraph 节点和边 |
| 手动消息追加 | 删除 | 改用 `add_messages` reducer |
| Tool Schema 业务定义 | 保留 | 属于项目工具契约 |
| `ToolExecutor` | 保留 | 包含项目安全策略 |
| `ToolBatchExecutor` | 保留 | 默认调度不满足资源要求 |
| `ToolResult` | 保留 | 稳定错误和恢复语义 |
| Action 去重 | 保留 | 框架默认不提供当前语义 |
| Observation 管理 | 保留 | 长结果策略由业务决定 |
| 副作用策略 | 保留 | 授权规则不能交给模型 |

## 验证结果

迁移后运行：

```powershell
uv run python -m unittest discover -s tests -v
```

50 项测试通过，覆盖 LangChain 工具绑定、LangGraph 路由、批量工具调用、错误恢复、Action 去重、自动重试、资源锁、副作用策略、MCP 适配和 Observation 管理。

此外使用真实 DeepSeek 接口完成了 `123 + 456` 冒烟测试，实际链路为：

```text
ChatDeepSeek
→ AIMessage.tool_calls
→ LangGraph tools 节点
→ calculator
→ ToolMessage
→ ChatDeepSeek 最终答案 579
```

## FastMCP 远程工具接入

项目已经基于 FastMCP 实现 `MCPClientManager`，启动时连接 DeepWiki 公共 MCP Server：

```text
FastMCP Client.list_tools()
  ↓
转换为项目 Tool 定义
  ↓
与本地工具组成 Agent 专属 registry
  ↓
ChatDeepSeek.bind_tools()
  ↓
模型调用后路由到 FastMCP Client.call_tool()
```

实现要点：

- FastMCP Client 在独立 asyncio 事件循环线程中保持 Session，不为每次工具调用重建连接。
- 远程工具增加 `deepwiki__` 命名空间，避免与本地工具重名。
- MCP `input_schema` 转换为现有 Function Tool Schema。
- MCP annotations 映射为副作用等级和幂等属性；DeepWiki 配置为已知只读 Server。
- `call_tool()` 结果转换为现有 Tool 业务值，继续复用 `ToolExecutor`、`ToolResult` 和 Observation 管理。
- MCP 连接失败时保留本地工具并继续启动 Agent。

已通过真实链路验证：

```text
DeepSeek
→ deepwiki__read_wiki_structure
→ FastMCP Client.call_tool
→ DeepWiki MCP Server
→ ToolResult / ToolMessage
→ DeepSeek 最终答案
```

后续可以继续学习 LangGraph Checkpointer、上下文裁剪、长期记忆和人工审批恢复。

## 目录结构演进

随着工具执行、资源调度和 MCP 集成逐步增加，项目已不再把所有模块平铺在根目录。当前按职责划分为：

```text
agent/        LangGraph 编排、配置和结果摘要
tooling/      工具定义、契约、执行、策略、调度和资源锁
integrations/ FastMCP 等外部协议适配
main.py       应用入口和依赖组装
```

本次只调整模块边界和导入路径，没有保留旧模块兼容副本，也没有改变 Agent Loop、工具注册或执行策略。
