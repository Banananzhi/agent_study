# Minimal Agent Loop

## 工具

工具统一定义和注册在 `tools.py`：

- `calculator`：基础算术计算
- `web_search`：通过博查搜索实时网页信息
- `read_webpage`：读取公开网页正文
- `get_current_time`：获取指定时区的当前时间

每个工具都先定义标准 Function Tool Schema（名称、描述、参数类型、必填项、范围和额外参数策略），再通过 `register_tool()` 与实际 Python 函数绑定。Schema 会通过 DeepSeek API 的 `tools` 字段发送给模型，ToolExecutor 也会在执行前使用同一份 Schema 校验参数。

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

默认中文系统提示词定义在 `agent.py` 的 `SYSTEM` 常量中，也可以通过 `Agent(system="...")` 覆盖。
