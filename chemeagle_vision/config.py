"""Configuration for local, SSH, and Slurm-over-SSH vision backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


def _first(env: Mapping[str, str], *names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class VisionConfig:
    provider: str = "local"
    device: str = "auto"
    chemrxn_device: str = "cpu"
    timeout: float = 600.0
    startup_timeout: float = 1800.0
    offline: bool = False
    model_dir: Optional[str] = None

    ssh_host: Optional[str] = None
    ssh_config: Optional[str] = None
    ssh_binary: str = "ssh"
    remote_dir: Optional[str] = None
    remote_python: str = "python3"

    slurm_account: Optional[str] = None
    slurm_qos: Optional[str] = None
    slurm_reservation: Optional[str] = None
    slurm_partition: Optional[str] = None
    slurm_submit_host: Optional[str] = None
    slurm_gpu_type: str = "L40S"
    slurm_gpus: int = 1
    slurm_cpus: int = 8
    slurm_memory: str = "64G"
    slurm_time: str = "08:00:00"

    @classmethod
    def from_env(
        cls,
        provider: Optional[str] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        **overrides: object,
    ) -> "VisionConfig":
        values = os.environ if env is None else env
        selected = (
            provider or values.get("CHEMEAGLE_VISION_PROVIDER") or "local"
        ).strip().lower().replace("_", "-")
        aliases = {"remote": "ssh", "slurm": "slurm-ssh", "cluster": "slurm-ssh"}
        selected = aliases.get(selected, selected)
        if selected not in {"local", "ssh", "slurm-ssh"}:
            raise ValueError(
                "CHEMEAGLE_VISION_PROVIDER must be one of: local, ssh, slurm-ssh"
            )

        remote = selected != "local"
        defaults = dict(
            provider=selected,
            device=values.get("CHEMEAGLE_VISION_DEVICE", "cuda" if remote else "auto"),
            chemrxn_device=values.get(
                "CHEMEAGLE_VISION_CHEMRXN_DEVICE", "cpu"
            ).strip().lower(),
            timeout=float(values.get("CHEMEAGLE_VISION_TIMEOUT", "600")),
            startup_timeout=float(
                values.get("CHEMEAGLE_VISION_STARTUP_TIMEOUT", "1800")
            ),
            offline=_env_bool(values, "CHEMEAGLE_VISION_OFFLINE", remote),
            model_dir=values.get("CHEMEAGLE_VISION_MODEL_DIR"),
            ssh_host=values.get("CHEMEAGLE_VISION_SSH_HOST"),
            ssh_config=_first(
                values,
                "CHEMEAGLE_VISION_SSH_CONFIG",
                default=os.path.expanduser("~/.ssh/config"),
            ),
            ssh_binary=values.get("CHEMEAGLE_VISION_SSH_BIN", "ssh"),
            remote_dir=values.get("CHEMEAGLE_VISION_REMOTE_DIR"),
            remote_python=values.get("CHEMEAGLE_VISION_REMOTE_PYTHON", "python3"),
            slurm_account=values.get("CHEMEAGLE_VISION_SLURM_ACCOUNT"),
            slurm_qos=values.get("CHEMEAGLE_VISION_SLURM_QOS"),
            slurm_reservation=values.get("CHEMEAGLE_VISION_SLURM_RESERVATION"),
            slurm_partition=values.get("CHEMEAGLE_VISION_SLURM_PARTITION"),
            slurm_submit_host=values.get(
                "CHEMEAGLE_VISION_SLURM_SUBMIT_HOST"
            ),
            slurm_gpu_type=values.get("CHEMEAGLE_VISION_SLURM_GPU_TYPE", "L40S"),
            slurm_gpus=int(values.get("CHEMEAGLE_VISION_SLURM_GPUS", "1")),
            slurm_cpus=int(values.get("CHEMEAGLE_VISION_SLURM_CPUS", "8")),
            slurm_memory=values.get("CHEMEAGLE_VISION_SLURM_MEMORY", "64G"),
            slurm_time=values.get("CHEMEAGLE_VISION_SLURM_TIME", "08:00:00"),
        )
        for key, value in overrides.items():
            if value is not None:
                if key not in defaults:
                    raise TypeError(f"Unknown vision configuration field: {key}")
                defaults[key] = value
        return cls(**defaults)

    def validate_remote(self) -> None:
        if self.chemrxn_device not in {"cpu", "cuda", "auto"}:
            raise ValueError(
                "CHEMEAGLE_VISION_CHEMRXN_DEVICE must be one of: cpu, cuda, auto"
            )
        if self.provider == "local":
            return
        if not self.ssh_host:
            raise ValueError("Remote vision requires CHEMEAGLE_VISION_SSH_HOST")
        if not self.remote_dir:
            raise ValueError("Remote vision requires CHEMEAGLE_VISION_REMOTE_DIR")
        if self.slurm_gpus < 1 or self.slurm_cpus < 1:
            raise ValueError("Slurm GPU and CPU counts must be positive")
