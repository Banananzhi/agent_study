# Minimal Agent Loop

## 工具

工具统一定义和注册在 `tools.py`：

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

同一轮的多个原生 `tool_calls` 由 `ToolBatchExecutor` 使用线程池动态调度，默认最大并行数为 4，可通过 `Agent(max_parallel_tools=...)` 调整。调度器只将已经一次性获得全部资源 Lease 的调用提交 Worker，等待锁的调用保留在调度队列中，不占用线程；后续无冲突调用可以先执行，但同资源冲突调用始终保持模型生成时的顺序。`read_file` 申请文件 READ 资源，`create_file` 和 `write_file` 申请 WRITE 资源，因此同文件读读可以并行、读写和写写互斥、不同文件可以并行。最终 ToolResult 与 tool message 始终按原始 `tool_calls` 顺序返回。

工具还通过 `side_effect_level` 独立声明副作用等级：`none`、`local_write`、`external_write` 或 `destructive`。`ToolExecutionPolicy` 在工具进入资源调度和线程池前进行程序化判断；当前默认允许无副作用和本地写入工具，外部写入及破坏性工具返回 `approval_required`，不会实际执行。该字段只负责权限和风险控制，并行关系仍由具体资源声明决定。

使用搜索前，需要在 `.env` 中设置博查 API Key：

```dotenv
BOCHA_API_KEY=你的博查API-Key
BOCHA_BASE_URL=https://api.bochaai.com/v1/web-search
```

一个不依赖 LangChain、也不依赖第三方 Python 包的最小 Agent Loop。模型可以决定直接回答，或调用内置的安全计算器，再根据工具结果生成最终答案。

## 运行

先编辑项目根目录的 `.env`，将占位符替换成自己的真实 DeepSeek API Key：

```dotenv
DEEPSEEK_API_KEY=sk-你的真实API-Key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

然后运行：

```powershell
python main.py
```

启动后输入一次问题，Agent 输出最终答案后程序结束。这里的“单轮”指用户只进行一次提问；Agent 内部仍可进行多次 `think → act → observe`。

运行期间会输出中文日志，包括步骤、模型名、模型是否决定调用工具、Action、工具执行、Observation 和最终答案；不会伪造模型的内部 Thought，API Key 也不会写入日志。

`.env` 已被 Git 忽略。如果你使用的第三方服务确实提供 `deepseek-flash`，请在 `.env` 中同时修改模型名和服务地址。

`agent.py` 包含 `Agent` 类及所有实现细节：`think` 返回 DeepSeek 原生 assistant 消息，`act` 解析并执行 `tool_calls`，`run` 负责完整循环。工具结果会通过包含 `tool_call_id` 的 `tool` 消息回传模型；模型不再返回 `tool_calls` 时，其 `content` 就是最终答案。`main.py` 只负责创建 Agent 和调用 `run`。

Agent 会使用“工具名称 + 按 Schema 补齐默认值后的标准参数”生成 Action 指纹。如果上一个 Action 已经成功，模型又立即生成完全相同的调用，Agent 会拦截重复执行并通过 `repeated_action` Observation 要求模型使用已有结果或调整调用。

单条 Observation 默认限制为 8000 个字符，可通过 `Agent(max_observation_chars=...)` 调整。超限时会保留合法 JSON 结构，优先截断成功结果的 `value` 或失败结果的 `error.message`，并在 `truncation` 中返回原始长度和实际保留长度。

默认中文系统提示词定义在 `agent.py` 的 `SYSTEM` 常量中，也可以通过 `Agent(system="...")` 覆盖。
