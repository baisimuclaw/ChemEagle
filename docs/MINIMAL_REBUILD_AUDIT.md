# Minimal rebuild and upstream audit

Audit date: 2026-08-22

## Recovery points

- Initial fork / local upstream baseline:
  `5a02479e52a4692ddee155e9ab618355b0d75a75`.
- Baseline is identical to `origin/main`, `upstream/main`, and
  `upstream-official/main` at audit time.
- Pre-rebuild backup branch:
  `backup/pre-minimal-rebuild-20260822` at
  `9af20626b1bd67e2bdad64b3656b553340ce425e`.
- Verified recovery bundle:
  `/home/baisimu/Project/ChemEagle-upstream-audit-20260822/pre_minimal_rebuild_backup/ChemEagle-pre-minimal-rebuild.bundle`.
- Full pre-rebuild patch and status records are in the same backup directory.
- File-by-file comparison source:
  `/home/baisimu/Project/ChemEagle-upstream-audit-20260822/upstream`.

## Allowed change set

Every difference must belong to one of these categories:

1. use a local ChatGPT/Codex subscription through the official Codex App
   Server protocol, without extracting credentials or silently using an API
   key;
2. execute only the upstream neural vision calls on a local, SSH, or
   Slurm-over-SSH GPU worker;
3. add bounded robustness for inactivity, transport errors, transient Codex
   failures, and empty neural detections, with no more than three attempts;
4. add configuration, documentation, and non-external tests for the preceding
   changes.

No unrelated feature work is accepted by this audit.

## Modified upstream files

- `main.py`: replaces direct SDK construction with provider selection and a
  separately selected vision backend. The upstream planner, ordered-agent
  execution, original image, final prompt, fallback SMILES postprocessors, and
  public result dictionary are retained.
- `get_reaction_agent.py`: uses the provider abstraction and remote vision
  proxy; retries an empty RxnIM result three times; retains the first complete
  raw prediction for the correction stage instead of rerunning RxnIM; an
  exhausted empty result remains empty.
- `get_molecular_agent.py`: uses the provider and vision abstractions; retries
  empty molecule detections three times; gives the LLM the same compact tool
  projection while retaining the complete raw graph for upstream
  postprocessing.
- `get_R_group_sub_agent.py`: ports the same upstream two-completion tool
  workflows to the provider abstraction. The product-variant agent still sends
  its full bounding-box annotation in the second completion, table agents keep
  the upstream single-tool fallback, and malformed table output fails rather
  than being converted into fabricated empty chemistry.
- `get_text_agent.py`: keeps upstream OCR sentence splitting and full-paragraph
  ChemNER scope, but runs ChemRxnExtractor and ChemNER through the selected
  vision backend. Tesseract discovery is lazy and portable instead of using
  machine-specific absolute paths.
- `get_observer.py`: sends the unchanged observer prompts through the selected
  LLM backend and retains the upstream safe observer fallbacks.
- `chemietoolkit/interface.py`: accepts an explicit staged model directory and
  offline mode for a remote worker. With no new options, model repositories,
  weights, devices, and lazy loading remain upstream-compatible.
- `rxnim/interface.py`: resolves the nested MolNexTR checkpoint from the staged
  worker model directory when configured; otherwise uses the exact upstream
  Hugging Face download.
- `molnextr/chemistry.py`: routes the optional symbol-to-SMILES LLM fallback
  through the active provider while preserving legacy standalone Azure
  behavior, embedded-JSON tolerance, and RDKit validation. External chemical
  resolvers are disabled only in the explicitly offline remote process. Final
  Graph2SMILES correction is executed locally with the complete graph.

## Added LLM files

- `chemeagle_llm/__init__.py`: public provider-neutral exports.
- `chemeagle_llm/types.py`: normalized requests, responses, tool calls, and
  supplemental full-image evidence.
- `chemeagle_llm/errors.py`: provider-neutral typed failures.
- `chemeagle_llm/config.py`: environment resolution and hard one-to-three
  attempt bounds.
- `chemeagle_llm/context.py`: request-scoped backend propagation for nested
  agents.
- `chemeagle_llm/base.py`: common JSON validation and upstream-compatible
  two-completion tool loop.
- `chemeagle_llm/openai_compatible.py`: Azure/local OpenAI-compatible request
  mapping and bounded retry handling.
- `chemeagle_llm/azure.py`: lazy upstream-compatible Azure backend.
- `chemeagle_llm/local_openai.py`: optional local vLLM/Ollama backend retained
  for upstream OS compatibility.
- `chemeagle_llm/factory.py`: lazy provider construction.
- `chemeagle_llm/codex_dynamic_tools.py`: dynamic-tool schema/result mapping.
- `chemeagle_llm/codex_app_server.py`: subscription-only App Server JSONL
  client, native image/history mapping, per-thread tool allowlists, inactivity
  tracking, bounded retries, and successful-tool reuse.
- `chemeagle_llm/codex_auth.py`: safe status/login/logout/model diagnostics
  delegated to Codex.

## Added vision files

- `chemeagle_vision/__init__.py`: public vision exports.
- `chemeagle_vision/base.py`: minimal backend protocol.
- `chemeagle_vision/errors.py`: typed configuration, process, protocol, and
  timeout failures.
- `chemeagle_vision/config.py`: local/SSH/Slurm configuration with L40S, eight
  CPUs, 24G memory, and one-to-three attempt defaults.
- `chemeagle_vision/codec.py`: lossless PNG, NumPy, tuple, and scalar JSON wire
  codec.
- `chemeagle_vision/context.py`: request-scoped backend and atomic successful
  prediction cache.
- `chemeagle_vision/factory.py`: lazy local or remote backend construction.
- `chemeagle_vision/proxies.py`: compatibility surface used by existing agents;
  neural calls go to the selected backend and pure upstream Graph2SMILES stays
  local.
- `chemeagle_vision/request_cache.py`: non-mutating compact LLM projection plus
  complete raw molecular graph retention.
- `chemeagle_vision/runtime.py`: fixed neural-method allowlist and one lazy
  upstream toolkit instance; ChemRxnExtractor retains the upstream
  sentence-by-sentence fallback.
- `chemeagle_vision/worker.py`: offline stdio JSONL worker with diagnostics kept
  off protocol stdout.
- `chemeagle_vision/remote.py`: persistent SSH / nested-submit-host / Slurm
  transport, keepalives, local stderr log, timeouts, and bounded worker restart.

## Added configuration, documentation, and tests

- `.env.example`: all LLM, Codex, SSH, Slurm, model, timeout, retry, log, and
  24G memory settings.
- `docs/codex_remote_gpu.md`: local Codex plus WSL or CUHK Slurm setup.
- `tests/__init__.py`: test package marker.
- `tests/test_llm_config.py`: provider parity, JSON/tool contracts, and bounds.
- `tests/test_codex_app_server.py`: protocol, authentication, timeout, transient
  failure, nested-tool isolation, history, safety, and cache tests.
- `tests/test_vision_backends.py`: codec, worker, SSH/Slurm command, lifecycle,
  local Graph2SMILES, and concurrent cache tests.
- `tests/test_orchestration_contract.py`: static and mocked checks against the
  upstream prompts, payloads, temperature policy, raw-result retention, empty
  handling, and output schema.

## Global comparison result

- No upstream file is missing.
- Exactly nine upstream files are modified; every other common file is
  byte-identical to the local upstream copy.
- Every file under `prompt/` is byte-identical to upstream, including
  `prompt_plan.txt` and `prompt_final_simple_version.txt`.
- There is no added image cropping, resizing, downsampling, JPEG conversion,
  payload truncation, or prompt summarization.
- Full-resolution images use lossless PNG on the GPU wire and the original
  inline image for Codex.
- PDF extraction, checkpoint `strict` policy, neural architecture, pretrained
  initialization, and final fallback chemistry functions are unchanged.
- Direct OS helper defaults retain the upstream model paths.
- The official schema generated by the installed `codex-cli 0.146.0` confirms
  the fields used here: `baseInstructions`, `dynamicTools`, `ephemeral`,
  `thread/inject_items`, `turn/start.outputSchema`, `item/tool/call`, and
  `turn/completed`.

Codex cannot be perfectly identical to the upstream Azure deployments: the
subscription exposes its own selected/default model, and App Server does not
offer the upstream Chat Completions `temperature` parameter. These are explicit
provider differences, not silent changes to prompts or chemistry data.

## Verification boundary

Verification is intentionally static/mocked only. It includes compile checks,
diff whitespace checks, protocol-schema inspection, file inventory comparison,
and the local unit suite. It does not submit a paper, call Codex inference,
connect to SSH, allocate a Slurm GPU, or run a neural model.
