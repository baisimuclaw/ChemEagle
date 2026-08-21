# ChemEAGLE LLM backends

ChemEAGLE separates chemical vision/OCR tools from the language-model backend. The
public result remains a normal Python `dict`; changing the backend changes the model
and billing/compute route, not the intended reaction schema. It does **not** guarantee
identical scientific accuracy between models.

## What must be installed

All modes need Python 3.10, the packages in `requirements.txt`, Tesseract, and the
ChemEAGLE vision checkpoints linked from the main README. Those checkpoints are
separate from the LLM: Azure and Codex host the language model remotely, while local
mode also requires a vision-language model served by vLLM or Ollama.

```bash
conda create -n chemeagle python=3.10
conda activate chemeagle
python -m pip install -r requirements.txt
```

Do not put model weights, `.env`, Codex authentication files, or extracted datasets in
Git. Copy `.env.example` to a private `.env` only if your environment loader uses it;
ChemEAGLE itself reads process environment variables and does not automatically load
that file.

## One public API

```python
from main import ChemEagle

result = ChemEagle(
    "./examples/1.png",
    provider="codex",             # azure | codex | local
    model=None,                    # account/server default where supported
    use_plan_observer=False,
    use_action_observer=False,
)
print(result)
```

Calling `ChemEagle(path)` without `provider` remains backward-compatible and selects
Azure. `ChemEagle_OS(...)` remains an alias for `provider="local"`.

For a batch, reuse one backend so Codex reuses its app-server process and local/Azure
reuse their HTTP client:

```python
from chemeagle_llm import create_backend
from main import ChemEagle

backend = create_backend("codex")
try:
    results = [ChemEagle(path, backend=backend) for path in image_paths]
finally:
    backend.close()
```

## Codex subscription backend

This route launches the official `codex app-server --stdio` and delegates OAuth token
storage and refresh to Codex. ChemEAGLE never reads `~/.codex/auth.json`, never turns a
subscription into `OPENAI_API_KEY`, and removes API-key variables from the managed
child process. A ChatGPT/Codex subscription is not an OpenAI Platform API key: requests
consume subscription capacity and stop with an explicit error when its limit is
reached.

Install the official CLI, then verify that the app-server command and required version
are available:

```bash
npm install -g @openai/codex
codex --version
codex app-server --help
```

See the official [Codex CLI documentation](https://developers.openai.com/codex/cli/)
and [App Server protocol README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md).
This implementation currently requires `codex-cli >= 0.146.0` by default because its
controlled function bridge uses the experimental dynamic-tools protocol.

Desktop/browser login:

```bash
python -m chemeagle_llm.codex_auth login
python -m chemeagle_llm.codex_auth status
python -m chemeagle_llm.codex_auth models
```

Headless supercomputer login:

```bash
python -m chemeagle_llm.codex_auth login --device-code
```

Open the displayed verification URL on another device and enter the one-time code.
Use `python -m chemeagle_llm.codex_auth logout` to sign out. After login:

```python
result = ChemEagle("./examples/1.png", provider="codex")
```

If `model=None`, App Server/account defaults decide the model. Inspect available models
with the `models` command and set `CODEX_MODEL` or the function's `model=` only when you
need an explicit choice.

Codex threads use a temporary controlled directory, read-only sandbox, and `never`
approval policy. Shell execution and file edits are denied. Only functions explicitly
registered for the current ChemEAGLE request can run. The `dynamicTools` interface is
experimental; a CLI version without it fails clearly instead of returning partial
chemistry.

## Azure OpenAI backend

Azure remains the no-argument default. Both explicit names and the original aliases
are accepted; explicit function arguments take precedence over environment variables.

```bash
export CHEMEAGLE_LLM_PROVIDER=azure
export AZURE_OPENAI_API_KEY='...'
export AZURE_OPENAI_ENDPOINT='https://YOUR-RESOURCE.openai.azure.com/'
export AZURE_OPENAI_API_VERSION='2024-10-21'
export CHEMEAGLE_LLM_MODEL='YOUR-AZURE-DEPLOYMENT-NAME'
```

```python
result = ChemEagle(
    "./examples/1.png",
    provider="azure",
    model="YOUR-AZURE-DEPLOYMENT-NAME",
)
```

Legacy `API_KEY`, `AZURE_ENDPOINT`, and `API_VERSION` still work. Credentials are
validated only when Azure is actually called, so importing ChemEAGLE or selecting
Codex/local does not require them.

## Local vLLM/Qwen backend

Serve a vision-capable instruction model with automatic function calling enabled. The
served model identifier—not a developer-specific filesystem path—is passed to
ChemEAGLE.

```bash
vllm serve /models/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name Qwen/Qwen3-VL-32B-Instruct-AWQ \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 64000 \
  --limit-mm-per-prompt video=0

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-AWQ
```

```python
result = ChemEagle("./examples/1.png", provider="local")
# Equivalent compatibility entry:
# result = ChemEagle_OS("./examples/1.png")
```

Capability flags centralize server differences:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_SUPPORTS_TOOLS` | `true` | Server supports function tools. |
| `VLLM_SUPPORTS_RESPONSE_FORMAT` | `true` | Server accepts JSON response format. |
| `VLLM_SUPPORTS_RESPONSE_FORMAT_WITH_TOOLS` | `false` | Server accepts tools and response format together. |
| `VLLM_SUPPORTS_JSON_SCHEMA` | `false` | Server accepts strict JSON Schema format. |

When a capability is off, the provider omits the unsupported request field and still
validates JSON locally. Invalid JSON is corrected a finite number of times and then
raises `InvalidResponseError`; it is not converted into a successful empty object.
`OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, and `OLLAMA_MODEL` remain fallback aliases.

## Hardware deployment recommendations

- **RTX 3080 (10/12 GB):** use Codex/Azure for the MLLM while the 3080 runs the
  ChemEAGLE vision checkpoints, or test a quantized 4B/8B vision-language model with a
  deliberately smaller context. A 32B model does not fit normally on one 3080. Running
  both the chemical vision stack and vLLM on the same card can exhaust VRAM.
- **One 48 GB class group GPU (often an NVIDIA L40S):** an AWQ/FP8 32B deployment may
  fit, but confirm the actual card and free memory with `nvidia-smi`; leave headroom for
  the vision encoder, KV cache, images, and ChemEAGLE models.
- **Multi-GPU supercomputer:** run vLLM as a long-lived scheduler job and point many
  ChemEAGLE workers at its `/v1` endpoint. Use vLLM tensor parallelism appropriate for
  the allocated cards. Keep job concurrency below both vLLM capacity and the memory
  needs of the chemical extraction models.
- **Interactive/low-volume work:** Codex subscription mode is the simplest route and
  needs no local LLM GPU. **Large reproducible batches:** local vLLM avoids subscription
  limits and is the recommended route once validated against your chemistry set.

These are capacity guidelines, not guarantees; model quantization, context length,
image resolution, vLLM version, and concurrent requests materially change memory use.

## Output

The return value is a Python dictionary suitable for JSON serialization. Its exact
contents depend on the detected diagram type, but the stable outer form is represented
by:

```json
{
  "reactions": [
    {
      "reaction_id": "0_1",
      "reactants": [{"smiles": "..."}],
      "conditions": [],
      "products": [{"smiles": "..."}]
    }
  ],
  "text_extraction": [{"annotated_text": "..."}]
}
```

R-group/table paths may add labels, `additional_info`, reaction templates, or the
original recognized molecule list during intermediate processing; the final data
structure agent normalizes the result.

## Troubleshooting

- **Codex binary missing/version incompatible:** run `codex --version` and
  `codex app-server --help`; update the official CLI or deliberately change
  `CHEMEAGLE_CODEX_MIN_VERSION` only after checking protocol compatibility.
- **Not logged in/API-key account rejected:** run the `codex_auth login` command. The
  backend requires a `chatgpt` account type and will not silently use an API-key route.
- **Model unavailable:** run `codex_auth models`, check the Azure deployment name, or
  query local `GET /v1/models` and pass exactly the served identifier.
- **Subscription limit reached:** wait for the displayed reset window or explicitly
  choose Azure/local. There is no automatic paid fallback.
- **vLLM will not tool-call:** start it with automatic tool choice and a parser supported
  by that model/vLLM release; verify `VLLM_SUPPORTS_TOOLS=true` only after a tool-call
  smoke test.
- **Image too large/GPU out of memory:** reduce image resolution/context/concurrency or
  use a larger/multi-GPU allocation. Preserve enough resolution for chemical labels.
- **JSON validation failed:** inspect the safe error preview, verify the selected model
  is instruction/tool capable, and reduce prompt/context pressure. Retries are finite.
- **Azure config missing:** set key, endpoint, version, and the Azure deployment model;
  the exception names the missing variables without printing their values.

## Migration checklist

1. Existing Azure users can keep `API_KEY`, `AZURE_ENDPOINT`, `API_VERSION`, and
   `ChemEagle(path)` unchanged.
2. Existing local users can keep `ChemEagle_OS`; remove absolute local model paths and
   set `VLLM_MODEL`/`VLLM_BASE_URL`, or use `ChemEagle(provider="local")`.
3. Subscription users install/login through Codex and use
   `ChemEagle(provider="codex")`; no OpenAI or Azure API key is required.

Run provider-only tests without a GPU or network:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The optional live subscription smoke test is disabled by default:

```bash
CHEMEAGLE_RUN_LIVE_CODEX=1 python -m pytest -q tests/test_live_codex.py
```
