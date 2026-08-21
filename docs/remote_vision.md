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
export CHEMEAGLE_VISION_OFFLINE=true
```

The workstation starts one persistent SSH process. The worker loads models lazily and
reuses them across nested ChemEAGLE tool calls.

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
export CHEMEAGLE_VISION_OFFLINE=true

export CHEMEAGLE_VISION_SLURM_ACCOUNT=xlzhang
export CHEMEAGLE_VISION_SLURM_QOS=xlzhang
export CHEMEAGLE_VISION_SLURM_RESERVATION=xlzhang_gpu
export CHEMEAGLE_VISION_SLURM_PARTITION=chpc
export CHEMEAGLE_VISION_SLURM_GPU_TYPE=L40S
export CHEMEAGLE_VISION_SLURM_GPUS=1
export CHEMEAGLE_VISION_SLURM_CPUS=8
export CHEMEAGLE_VISION_SLURM_MEMORY=64G
export CHEMEAGLE_VISION_SLURM_TIME=08:00:00
```

The generated command is equivalent to `ssh cuhk 'srun ... python -m
chemeagle_vision.worker --stdio'`. SSH transports stdin/stdout; `srun` still performs
the mandatory allocation. Closing the backend ends the worker and releases the GPU.

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

## Common failures

- `CUDA is unavailable`: remote PyTorch is CPU-only or Slurm did not allocate a GPU.
- startup timeout: the Slurm job is pending; inspect `squeue -u "$USER"`.
- offline asset missing: prefetch and copy the complete model directory, then verify
  that `CHEMEAGLE_VISION_MODEL_DIR` is a remote path.
- invalid JSONL: custom model code wrote to stdout instead of stderr.
- SSH exits: verify the alias with `ssh -F ~/.ssh/config HOST true`.
