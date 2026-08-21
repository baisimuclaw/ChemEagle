# ChemEAGLE 多后端重构开发 Prompt

你正在维护仓库 `/home/baisimu/Project/ChemEagle`。当前应位于分支
`feat/codex-vllm-backends`。请在这个分支上完成开发，不要直接修改或推送
`main`，不要丢弃用户已有改动，也不要在未经请求时 push、开 PR 或改写 Git
历史。

## 目标

把 ChemEAGLE 从“业务代码直接创建 `AzureOpenAI` 客户端”的结构重构为可切换
的 LLM 后端，同时保留原有化学信息抽取流程和输出格式：

1. Azure OpenAI 后端：保持当前云端使用方式和结果兼容性。
2. Codex App Server 后端：使用用户的 ChatGPT/Codex 订阅登录，不需要 OpenAI
   Platform API key，也不依赖 OpenClaw。
3. 本地 OpenAI-compatible 后端：连接 vLLM/Qwen，保留并整理现有
   `ChemEagle_OS` 能力，适合超算批处理。

最终用户应能通过统一入口和配置选择 `azure`、`codex` 或 `local`，不应再维护
三套大段重复的 agent 实现。

## 开发前必须完成的检查

1. 阅读根目录的 `README.md`、`LICENSE.txt`、`requirements.txt`，以及以下文件：
   `main.py`、`get_observer.py`、`get_molecular_agent.py`、
   `get_reaction_agent.py`、`get_R_group_sub_agent.py`、`get_text_agent.py`、
   `chemietoolkit/helper.py`。
2. 用 `rg` 枚举所有 `AzureOpenAI`、`OpenAI(`、
   `client.chat.completions.create`、`API_KEY`、`response_format`、`tools=` 和
   硬编码模型名调用，建立迁移清单；不能只修改 `main.py`。
3. 运行 `git status --short --branch`，确认分支正确并识别已有改动。
4. 不下载大型模型权重，不在普通测试中初始化真实的 ChemIEToolkit/RxnIM 模型。

## 关键设计约束

### 1. 延迟校验配置

删除模块 import 阶段的此类逻辑：

```python
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
```

没有 Azure key 时，以下操作必须仍然成功：

```bash
python -c "import main"
```

只有真正选择并调用 Azure 后端时才校验 Azure 配置，并给出明确、可操作、不得
泄露凭据的错误信息。

### 2. 建立统一 provider 层

新增一个职责清晰的小型包，例如：

```text
chemeagle_llm/
├── __init__.py
├── config.py
├── types.py
├── base.py
├── factory.py
├── azure.py
├── local_openai.py
├── codex_app_server.py
└── codex_auth.py
```

具体文件名可根据仓库情况调整，但必须具备：

- 统一的请求/响应类型，不让业务 agent 依赖 OpenAI SDK 的
  `ChatCompletion` 对象形状。
- 支持 system/user/assistant/tool 消息、文本、Base64 图片、JSON 结构化输出、
  function tools、超时、重试和取消。
- 统一返回文本、结构化 JSON、工具调用、用量/诊断信息；业务代码不得通过
  `response.choices[0]...` 读取 provider 专有对象。
- provider factory 和显式配置优先级：函数参数 > 环境变量 > 合理默认值。
- 配置对象应可注入，测试中不能依赖真实环境变量。

建议提供高层接口而非机械模仿 OpenAI SDK：

```python
class LLMBackend(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...

    def run_tool_loop(
        self,
        request: LLMRequest,
        tools: list[ToolDefinition],
        executor: ToolExecutor,
    ) -> LLMResponse: ...
```

如果实际代码证明 async 更合适，可以内部使用 async，但应保留易用、稳定的同步
入口以兼容当前 ChemEAGLE 调用方式。

### 3. Azure 后端

- 封装当前 `AzureOpenAI` 行为，不改变现有 Azure 环境变量的兼容性：
  `API_KEY`、`AZURE_ENDPOINT`、`API_VERSION`。
- 同时接受更明确的别名，例如 `AZURE_OPENAI_API_KEY` 和
  `AZURE_OPENAI_ENDPOINT`，并记录优先级。
- 不要顺便更换当前 Azure 模型或改写化学 prompt；硬编码模型应迁移到配置，
  但默认值保持当前行为。
- Azure provider contract 应由 mock 测试覆盖，测试不得产生真实网络请求。

### 4. 本地 vLLM/OpenAI-compatible 后端

- 整理当前 `ChemEagle_OS`、`*_OS` 实现，尽量与云端 agent 共享编排代码，而不是
  继续复制整套函数。
- 保持现有环境变量兼容：`VLLM_BASE_URL`、`VLLM_API_KEY`，必要时继续接受已有的
  `OLLAMA_*` 回退。
- 默认地址可以保持 `http://localhost:8000/v1`，但必须允许函数参数或配置覆盖。
- 兼容不同服务对 `response_format`、tool calling、`temperature` 的支持差异；用
  capability flags 和本地 JSON 校验/重试处理，不要到处散落 provider 判断。
- 不把本地模型名写死为开发者机器上的绝对路径。

### 5. 原生 Codex App Server 后端

使用官方 `codex app-server`，不要把 ChatGPT/Codex 订阅伪装成
`OPENAI_API_KEY`，不要读取、复制、解析或提交 `~/.codex/auth.json`，不要调用
未公开的 ChatGPT 私有 HTTP endpoint。认证、token 持久化和刷新全部交给 Codex
App Server。

官方协议参考（实现时必须重新核对当前版本）：

- <https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md>
- <https://github.com/openai/codex/tree/main/codex-rs/app-server-protocol/schema>

首个稳定实现使用本机 stdio transport；不要以实验性 WebSocket transport 作为
默认。客户端至少需要实现：

1. 安全启动和关闭 `codex app-server --stdio` 子进程。
2. stdout 逐行解析 JSONL，stderr 独立采集；绝不能把 stderr 当协议消息。
3. 单连接 `initialize` 请求与 `initialized` notification 握手。
4. 请求 ID 关联、server notification 分发、server-to-client request 应答。
5. `account/read` 检查认证状态。
6. `account/login/start` 的 `chatgpt` 浏览器登录和 `chatgptDeviceCode` 设备码登录。
7. 等待 `account/login/completed`，并提供 `status`、`login`、`logout` 命令。
8. `model/list`：默认优先使用账户/App Server 的可用模型或用户显式选择；不得把
   某个可能无权限的最新模型作为唯一硬编码值。
9. `thread/start`、`turn/start` 和事件流，直到 `turn/completed`。
10. 从 `agentMessage` 事件组装最终文本；处理失败、取消、进程退出和协议错误。
11. 文本和图片输入映射：Base64 图片使用 App Server 支持的 data URL；不得假设
    它接受远程图片 URL。
12. 用 `turn/start.outputSchema` 实现结构化 JSON 输出，并在本地做
    `json.loads`/schema 校验和有限次数纠错重试。
13. 查询/暴露 `account/rateLimits/read`，在达到订阅限额时给用户明确错误，不得
    静默切换到需要付费的 Platform API key。

Codex App Server 的 `dynamicTools`/`item/tool/call` 当前是实验性接口。ChemEAGLE
现有多个子 agent 依赖 OpenAI function calling，因此必须作出有测试的显式设计：

- 在 `initialize` 中按需启用 experimental API。
- 将现有 OpenAI function schema 规范化后映射到 `dynamicTools`。
- 收到 `item/tool/call` 时，只允许调用当前请求显式注册的白名单函数；校验工具名
  和 JSON 参数，不允许模型构造任意 Python 调用、shell 命令或路径。
- 将现有 Python 工具结果转换为 App Server 要求的 content items 并回复原请求。
- 捕获工具异常并返回结构化失败，不能让 reader loop 崩溃或死锁。
- 把协议版本兼容性集中在 Codex provider 内，并检测 `codex --version`；错误信息应
  告诉用户期望与实际版本。
- 将动态工具协议隔离为可替换模块；如果当前 Codex 版本不支持，应明确报
  `UnsupportedCapabilityError`，不能悄悄产出不完整化学结果。

如果可以在保持结果语义的前提下，把部分“先让模型选择工具、Python 执行、再把
结果交回模型”的流程重构成 provider 无关的显式编排，可以这样做；但不得擅自
改变 agent 选择逻辑、化学模型调用顺序或最终 schema。任何语义变化必须通过
回归测试说明。

### 6. Codex 权限与工作目录

这个用途是模型推理和受控的 ChemEAGLE 函数调用，不需要 Codex 修改仓库或执行
任意 shell。Codex thread 应使用最小权限：

- 使用固定、受控的工作目录；不要把用户 home 或 `/` 作为可写工作区。
- 默认拒绝 Codex 的 shell、文件编辑及其他非 ChemEAGLE 工具请求。
- 只响应本次调用注册的 dynamic tool。
- 不在日志中打印 OAuth URL 查询参数、access token、refresh token、API key 或
  完整认证对象。

### 7. 公共 API 与向后兼容

保留以下入口：

```python
from main import ChemEagle, ChemEagle_OS
```

推荐把统一入口扩展为：

```python
ChemEagle(
    image_path,
    provider="codex",       # azure | codex | local
    model=None,
    use_plan_observer=False,
    use_action_observer=False,
)
```

其中：

- 旧的无 `provider` 调用应保持兼容；具体默认值必须在 README 中明确。
- `ChemEagle_OS` 可作为调用 `provider="local"` 的兼容 wrapper，并给出非破坏性
  deprecation 说明；本次不要求删除它。
- observer 和所有子 agent 必须接收同一 provider/config，不能在内部偷偷重新创建
  Azure 客户端。
- 返回值继续是普通 Python `dict`，最终 reaction/text schema 不应因 provider 改变。

## 可靠性要求

- 为协议请求设置超时；子进程异常退出时终止所有 pending request。
- 对 App Server overload `-32001` 和明确的瞬时错误使用带 jitter 的指数退避；
  不要对认证失败、参数错误或永久性错误盲目重试。
- JSON 结构化输出必须经过校验。重试耗尽后保留原始响应的安全摘要并抛出领域错误，
  不要返回看似成功的空字典。
- 不要用宽泛 `except Exception: return default` 隐藏 provider 故障。observer 的可选
  降级行为可以保留，但必须记录可诊断原因。
- 日志使用标准 `logging`，避免当前散落的 `print` 扩散；不要在本任务中无关地
  重写全部历史日志。
- 子进程、reader thread/task、临时文件在成功、失败、取消路径上都必须释放。

## 测试要求

新增轻量测试目录，不加载真实模型、不需要 GPU、不需要 Azure key，也默认不访问
网络。至少覆盖：

1. 无任何 API 环境变量时可以 import 主要模块。
2. provider factory 的选择、配置优先级和错误信息。
3. Azure 请求映射和结构化响应归一化（mock OpenAI SDK）。
4. local OpenAI-compatible 请求映射、capability fallback 和 JSON 校验（mock）。
5. 假 App Server 子进程或内存 transport：
   - initialize/initialized；
   - 已登录和未登录；
   - browser/device-code login 事件；
   - model/list；
   - text + Base64 image turn；
   - outputSchema；
   - agent message 组装；
   - dynamic tool 成功、未知工具、参数错误、工具抛错；
   - overload 重试、超时、取消、EOF/崩溃；
   - rate-limit 错误。
6. 一个不加载化学权重的端到端 orchestration contract 测试，确认三个 provider 的
   规范化假响应产生相同最终 dictionary schema。
7. 可选 live smoke test 必须显式 opt-in，例如
   `CHEMEAGLE_RUN_LIVE_CODEX=1`；默认测试套件必须 skip，不能消耗订阅额度。

测试过程中如果发现仓库当前没有统一测试框架，优先使用 `pytest` 并把开发依赖与
运行命令写清楚。不要为了测试而强制安装整套 GPU 依赖；必要时把 provider 层做成
可单独测试的纯 Python 模块。

## 文档和用户体验

更新 README，并新增详细文档（例如 `docs/llm_backends.md`），至少包含：

- 三个后端的用途、依赖、认证、环境变量和最小示例。
- Codex CLI/App Server 的安装检查、浏览器登录和超算 device-code 登录。
- 明确说明 Codex/ChatGPT 订阅不等于 OpenAI Platform API key，Codex 模式消耗
  订阅额度并受限额约束。
- 3080 本地测试、超算 vLLM 批处理和 Codex 订阅交互使用的推荐部署方式。
- 常见错误：未登录、Codex binary 不存在、版本不兼容、模型不可用、订阅限额、
  vLLM 无法 tool-call、图片过大、JSON 校验失败。
- 从旧环境变量和 `ChemEagle_OS` 迁移的方法。

提供不会泄露真实凭据的 `.env.example`；不要提交 `.env`、token、auth 文件、模型
权重、输出数据集或本机绝对路径。

## 非目标

- 不重写化学视觉模型、OCR、SMILES 修复算法或论文 prompt。
- 不下载/提交 ChemEAGLE、Qwen 或其他大模型权重。
- 不通过逆向 ChatGPT 私有 endpoint 实现“订阅 API”。
- 不复制 OpenClaw 整个代码库，也不以 OpenClaw 作为运行依赖。
- 不删除 Azure 或现有本地后端。
- 不声称不同 LLM 后端具有相同科学准确率；只保证接口和 schema 兼容。
- 不进行与该目标无关的大规模格式化或重命名。

## 实施顺序

严格按以下顺序推进，每一阶段保持代码可测试：

1. 建立测试基线，修复 import-time Azure key 依赖。
2. 定义 provider-neutral 类型、异常、配置和 factory。
3. 封装 Azure，保持现有行为并通过 contract tests。
4. 封装/去重本地 vLLM 路线，并通过 contract tests。
5. 实现 Codex stdio JSONL/RPC 核心、认证和模型发现。
6. 实现图片、结构化输出和受控 dynamic tool bridge。
7. 逐个迁移 main、observer 和五个 agent 文件，不遗漏嵌套客户端创建点。
8. 增加单元、协议、集成和可选 live smoke tests。
9. 更新 README、迁移文档和示例。
10. 运行完整的轻量验证，检查 diff、凭据泄露、硬编码路径和分支状态。

## 完成标准

只有同时满足以下条件才能报告完成：

- 所有 Azure/OpenAI 客户端创建集中在 provider 层；业务 agent 中没有新的直接
  SDK client 创建。
- 无 Azure key 时 import 成功，Codex/local 模式不读取或要求 Azure 配置。
- Codex 登录由 App Server 管理；项目代码和日志中没有 OAuth/API secrets。
- Azure、Codex、local 三个后端都实现同一 contract。
- 现有公共入口和最终 dictionary schema 保持兼容。
- 默认测试无需 GPU、模型权重、真实账号或网络即可通过。
- README 给出三种后端可复制执行的示例和限制说明。
- `git diff --check` 通过；`git status` 只包含本任务有意修改的文件。

## 最终汇报格式

开发完成后，先给结果，再简要列出：

1. 架构和关键文件变化。
2. Codex 登录/调用方式。
3. 本地 vLLM 和 Azure 的兼容情况。
4. 实际运行的测试命令和结果。
5. 尚未验证的 live/GPU 部分及原因。
6. 已知限制，尤其是 Codex dynamic tools 的实验性和订阅限额。

不要只给计划或示例代码；应实际修改仓库、运行可行的验证，并在遇到真实阻塞时给
出证据和最小化的下一步。
