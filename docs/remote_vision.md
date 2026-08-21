# Remote ChemEAGLE vision workers

ChemEAGLE can keep language-model orchestration on an internet-connected PC while
running only CUDA vision models on another machine. The remote worker does not contact
OpenAI and needs no Codex login, API key, VPN, or inbound listening port.

```text
workstation: ChemEAGLE + Codex + network post-processing
                         |
                         | SSH JSONL (images and JSON-safe values)
                         v
remote:      optional Slurm srun -> offline PyTorch worker -> GPU
```

`CHEMEAGLE_LLM_PROVIDER` and `CHEMEAGLE_VISION_PROVIDER` are independent. Vision
providers are `local`, `ssh`, and `slurm-ssh`.

## Security and network boundary

The worker protocol allowlists reaction recognition, molecule/coreference recognition,
MolNexTR graph inference, and ChemNER. It cannot execute a model-provided shell command.
Images and NumPy arrays are encoded in the JSONL stream; workstation paths are never
trusted as paths on the remote machine. Diagnostics go to stderr so they cannot corrupt
protocol responses.

Remote mode defaults to `CHEMEAGLE_VISION_OFFLINE=true` and sets
`CHEMEAGLE_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. Web chemical
lookups remain in workstation-side orchestration. Do not copy `~/.codex`, API keys, or
`.env` secrets to the worker.

## Prepare one shared remote environment

The remote Python environment needs ChemEAGLE vision dependencies, CUDA PyTorch, and
the project source. On Slurm, place the environment, source, and weights in storage
visible from login and compute nodes. Install them once, not once per GPU.

On a machine that can reach package indexes, create the Python 3.10 environment and
verify CUDA before copying weights:

```bash
conda create -n chemeagle-worker python=3.10 -y
conda activate chemeagle-worker
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last value must be `True` when run inside the WSL GPU host or a Slurm GPU
allocation. The login node does not need to expose CUDA. If the remote machine cannot
reach PyPI or Hugging Face, build a wheelhouse or a `conda-pack` archive on a compatible
internet-connected Linux machine, then copy and unpack it in shared storage. No VPN or
OpenAI access is required on the worker.

On an internet-connected Linux machine with the dependencies installed, prefetch the
worker assets:

```bash
python -m chemeagle_vision.prefetch --output /tmp/chemeagle-models
```

This creates explicit checkpoints, the ChemRxnExtractor snapshot, BioBERT files,
EasyOCR files, and `manifest.json`. Copy that directory and the repository to the
remote machine, for example:

```bash
rsync -aP /tmp/chemeagle-models/ cuhk:/SHARED/PATH/chemeagle-models/
rsync -aP ./ cuhk:/SHARED/PATH/ChemEagle/
```

Use a shared path supplied by the administrator. An offline worker fails clearly when
an asset is absent instead of attempting a download.

Before using ChemEAGLE, verify the copied interpreter directly:

```bash
/REMOTE/PATH/env/bin/python -c "import torch; print(torch.__version__)"
```

On Slurm, run that command under the same `srun` resource flags shown below and also
check `torch.cuda.is_available()`.

## Direct SSH GPU (WSL/3060 Ti)

Set these variables on the workstation:

```bash
export CHEMEAGLE_VISION_PROVIDER=ssh
export CHEMEAGLE_VISION_SSH_HOST=wangwucai_wsl
export CHEMEAGLE_VISION_SSH_CONFIG="$HOME/.ssh/config"
export CHEMEAGLE_VISION_REMOTE_DIR=/REMOTE/PATH/ChemEagle
export CHEMEAGLE_VISION_REMOTE_PYTHON=/REMOTE/PATH/env/bin/python
export CHEMEAGLE_VISION_MODEL_DIR=/REMOTE/PATH/chemeagle-models
export CHEMEAGLE_VISION_DEVICE=cuda
export CHEMEAGLE_VISION_CHEMRXN_DEVICE=cpu
export CHEMEAGLE_VISION_OFFLINE=true
```

The workstation starts one persistent SSH process. The worker loads models lazily and
reuses them across nested ChemEAGLE tool calls.

## Per-request vision inference lifecycle

Reaction-template and molecular-recognition agents retain the complete first vision
prediction for the lifetime of one image request. The LLM receives a deep-copied,
compact projection with large graph fields such as `coords`, `edges`, `atoms`, and
`molfile` removed. After the LLM corrects atom symbols or expands R-groups, ChemEAGLE
applies those corrections to the retained prediction and runs Graph2SMILES.

This applies to all four molecular-recognition entry points: the base, `correctR`,
`correctmultiR`, and local-vLLM (`OS`) variants. They do not run MolDetector/Image2Graph
a second time merely to recover fields removed from the LLM payload. Reaction-template
entry points use the same lifecycle for RxnIM predictions.

The retained prediction is request-scoped, not process-global. Concurrent images cannot
reuse or overwrite one another's graph data. If an LLM backend returns without invoking
the vision tool, ChemEAGLE performs one fallback vision call so downstream graph
reconstruction still has a complete prediction.

Request-scoped vision selection is propagated into Codex dynamic-tool threads. This is
important: without that propagation, a nested agent could fall back to loading vision
models on the workstation instead of using the configured SSH worker.

## CUHK Slurm L40S

This matches the identified group reservation; adjust paths and resource sizes:

```bash
export CHEMEAGLE_VISION_PROVIDER=slurm-ssh
export CHEMEAGLE_VISION_SSH_HOST=cuhk
export CHEMEAGLE_VISION_SSH_CONFIG="$HOME/.ssh/config"
export CHEMEAGLE_VISION_REMOTE_DIR=/SHARED/PATH/ChemEagle
export CHEMEAGLE_VISION_REMOTE_PYTHON=/SHARED/PATH/env/bin/python
export CHEMEAGLE_VISION_MODEL_DIR=/SHARED/PATH/chemeagle-models
export CHEMEAGLE_VISION_DEVICE=cuda
export CHEMEAGLE_VISION_CHEMRXN_DEVICE=cpu
export CHEMEAGLE_VISION_OFFLINE=true

export CHEMEAGLE_VISION_SLURM_ACCOUNT=xlzhang
export CHEMEAGLE_VISION_SLURM_QOS=xlzhang
export CHEMEAGLE_VISION_SLURM_RESERVATION=xlzhang_gpu
export CHEMEAGLE_VISION_SLURM_PARTITION=chpc
export CHEMEAGLE_VISION_SLURM_SUBMIT_HOST=sandbox
export CHEMEAGLE_VISION_SLURM_GPU_TYPE=L40S
export CHEMEAGLE_VISION_SLURM_GPUS=1
export CHEMEAGLE_VISION_SLURM_CPUS=8
export CHEMEAGLE_VISION_SLURM_MEMORY=64G
export CHEMEAGLE_VISION_SLURM_TIME=08:00:00
```

With `CHEMEAGLE_VISION_SLURM_SUBMIT_HOST=sandbox`, the generated command is equivalent
to `ssh cuhk 'ssh sandbox "srun ... python -m chemeagle_vision.worker --stdio"'`.
This keeps the long-running `srun` client off the login node while preserving worker
stdin/stdout through both SSH connections. Leave the variable unset on clusters that
permit `srun` directly from the login host. Closing the backend ends the worker and
releases the GPU allocation.

## Health check before a full run

This checks SSH, Slurm, the remote Python, CUDA visibility, and PyTorch. It does not
load model weights or call OpenAI:

```bash
python - <<'PY'
from chemeagle_vision import create_vision_backend

backend = create_vision_backend()
try:
    print(backend.health())
finally:
    backend.close()
PY
```

A CUDA worker should report `status: ok`, `cuda_available: true`, and the expected GPU.
Model files are validated lazily on the first inference call.

## Application use and batching

Environment configuration is sufficient:

```python
from main import ChemEagle

result = ChemEagle("./examples/1.png", provider="codex")
```

Or select vision explicitly:

```python
result = ChemEagle(
    "./examples/1.png",
    provider="codex",
    vision_provider="slurm-ssh",
)
```

For batches, reuse both backends so the Slurm allocation and loaded models remain alive:

```python
from chemeagle_llm import create_backend
from chemeagle_vision import create_vision_backend
from main import ChemEagle

llm = create_backend("codex")
vision = create_vision_backend("slurm-ssh")
try:
    results = [
        ChemEagle(path, backend=llm, vision_backend=vision)
        for path in image_paths
    ]
finally:
    vision.close()
    llm.close()
```

One worker serializes requests, appropriate for one 8 GB 3060 Ti. For L40S batch
parallelism, start several workers with separate Slurm GPU allocations.

Codex uses a four-minute response window and retries the same request at most
three times. Time spent inside a running vision tool is excluded from that
window; the response timer restarts after the tool result reaches Codex. Enable
stage tracing while validating a deployment:

```bash
export CHEMEAGLE_LLM_TIMEOUT=240
export CHEMEAGLE_LLM_MAX_RETRIES=3
export CHEMEAGLE_TRACE=1
```

Vision inference has a separate bounded fallback. RxnIM and molecule/coreference
inference retry twice when a call succeeds but returns no detections (three total
attempts). GPU, SSH, and Slurm exceptions still fail immediately. If all RxnIM
attempts are empty, the image-aware LLM path continues with the empty result; an
exhausted molecule/coreference call receives an empty schema instead of failing
with an index error. Reaction-template tools retain the RxnIM prediction in their
result; the common compact projection removes graph-heavy fields before it is
sent to the LLM.

`CHEMEAGLE_TRACE` reports only turn/tool identifiers, elapsed time, status, and
tool-payload character counts. It does not print credentials or tool results.

ChemRxnExtractor defaults to CPU, matching the upstream text agent and avoiding
contention with the PyTorch vision models already resident on the GPU. Set
`CHEMEAGLE_VISION_CHEMRXN_DEVICE=cuda` only after validating that combination on
the target environment. Scheme-only OCR is filtered before ChemNER and
ChemRxnExtractor by default; set `CHEMEAGLE_TEXT_PROSE_FILTER=0` on the workstation
to restore the upstream all-text behaviour.

## Common failures

- `CUDA is unavailable`: remote PyTorch is CPU-only or Slurm did not allocate a GPU.
- startup timeout: the Slurm job is pending; inspect `squeue -u "$USER"`.
- offline asset missing: prefetch and copy the complete model directory, then verify
  that `CHEMEAGLE_VISION_MODEL_DIR` is a remote path.
- invalid JSONL: custom model code wrote to stdout instead of stderr.
- SSH exits: verify the alias with `ssh -F ~/.ssh/config HOST true`.
- submit-host SSH exits: first run `ssh cuhk 'ssh sandbox true'`; the internal host
  may deliberately accept only a key installed on the login node.
