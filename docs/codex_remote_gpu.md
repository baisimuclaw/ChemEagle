# Codex orchestration with a remote GPU

This fork keeps ChemEAGLE orchestration and the Codex subscription on the local
workstation, while only the original chemical-vision calls run on a remote GPU.
The remote worker does not need Codex login, an OpenAI API key, a VPN, or an
inbound listening port.

```text
local ChemEAGLE + codex app-server + upstream Graph2SMILES postprocessing
              |
              | persistent SSH JSONL (lossless images and model results)
              v
remote Python environment [optional srun] -> CUDA neural vision models
```

The language-model and vision providers are selected independently:

```python
from main import ChemEagle

result = ChemEagle(
    "./examples/1.png",
    provider="codex",
    vision_provider="slurm-ssh",
)
```

## Local Codex setup

Install the official Codex CLI and sign in once on the workstation:

```bash
npm install -g @openai/codex
python -m chemeagle_llm.codex_auth login
python -m chemeagle_llm.codex_auth status
```

ChemEAGLE launches `codex app-server --stdio`. It delegates token storage and
refresh to Codex and does not read the credential file or convert a subscription
into an API key. The managed App Server process is stripped of API-key variables.

## Remote worker setup

The repository, Python environment, and model files must be visible from the GPU
machine. For Slurm, place all three on storage shared by the submit and compute
nodes. Install the upstream `requirements.txt` in that environment and use its
Python executable in `CHEMEAGLE_VISION_REMOTE_PYTHON`.

When `CHEMEAGLE_VISION_OFFLINE=true`, put these existing upstream model assets in
`CHEMEAGLE_VISION_MODEL_DIR`:

- `molnextr.pth`
- `rxn.ckpt`
- `moldet.ckpt`
- `corefdet.ckpt`
- `ner.ckpt`
- `chemrxnextractor-training-modules/`

Hugging Face and EasyOCR caches may also live below that directory. Missing
offline assets raise an error; ChemEAGLE does not silently switch models.

### WSL / direct SSH

```bash
export CHEMEAGLE_LLM_PROVIDER=codex
export CHEMEAGLE_VISION_PROVIDER=ssh
export CHEMEAGLE_VISION_SSH_HOST=wangwucai_wsl
export CHEMEAGLE_VISION_REMOTE_DIR=<REMOTE_CHEMEAGLE_PATH>
export CHEMEAGLE_VISION_REMOTE_PYTHON=<REMOTE_ENV_PATH>/bin/python
export CHEMEAGLE_VISION_MODEL_DIR=<REMOTE_MODEL_PATH>
export CHEMEAGLE_VISION_DEVICE=cuda
export CHEMEAGLE_VISION_OFFLINE=true
```

### CUHK Slurm / L40S

Use the actual Lustre paths and scheduler fields assigned to the account:

```bash
export CHEMEAGLE_LLM_PROVIDER=codex
export CHEMEAGLE_VISION_PROVIDER=slurm-ssh
export CHEMEAGLE_VISION_SSH_HOST=cuhk
export CHEMEAGLE_VISION_SLURM_SUBMIT_HOST=sandbox
export CHEMEAGLE_VISION_REMOTE_DIR=<LUSTRE_PATH>/ChemEagle
export CHEMEAGLE_VISION_REMOTE_PYTHON=<LUSTRE_PATH>/env/bin/python
export CHEMEAGLE_VISION_MODEL_DIR=<LUSTRE_PATH>/chemeagle-models
export CHEMEAGLE_VISION_DEVICE=cuda
export CHEMEAGLE_VISION_OFFLINE=true
export CHEMEAGLE_VISION_SLURM_GPU_TYPE=L40S
export CHEMEAGLE_VISION_SLURM_GPUS=1
export CHEMEAGLE_VISION_SLURM_CPUS=8
export CHEMEAGLE_VISION_SLURM_MEMORY=24G
export CHEMEAGLE_VISION_SLURM_TIME=08:00:00
```

With the submit host set, ChemEAGLE automatically creates the equivalent of:

```text
ssh cuhk -> ssh sandbox -> srun ... python -m chemeagle_vision.worker --stdio
```

The local process owns this persistent worker. Closing the backend ends the SSH
process and releases the Slurm allocation. Set `CHEMEAGLE_VISION_REMOTE_LOG` to a
local path if remote stderr should be retained.

## Bounded robustness behavior

- Ordinary Codex turns use a 240-second inactivity window; the final synthesis
  keeps the upstream-compatible 600-second window.
- Codex progress resets the inactivity window. Time inside a running vision tool
  is governed by the separate tool timeout.
- LLM attempts, remote-worker transport attempts, empty RxnIM predictions, and
  empty molecule detections are each bounded at three total attempts.
- A successful vision result is cached only within the current ChemEAGLE request,
  so retrying Codex does not rerun the GPU model.
- Retries preserve the original full-resolution image and the complete successful
  tool result. There is no automatic cropping, resizing, summarizing, or field
  deletion beyond the behavior already present upstream.
- If reaction/molecule recognition is still empty after three attempts, it remains
  empty; no LLM-only chemical structure is substituted.

These changes address transient network, scheduler, worker, and empty-result
failures. They do not change prompts, checkpoint loading strictness, neural model
architecture, PDF extraction, or the final output schema.
