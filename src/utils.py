"""Utilitarios compartilhados: carga de config, seeds, device, IO."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml", overrides: dict[str, Any] | None = None) -> dict:
    """Carrega o YAML e aplica overrides com chave pontuada (ex.: {'train.epochs': 30})."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_data_dir(cfg: dict, cli_data: str | None) -> str:
    """Ordem de prioridade: --data > config > download via kagglehub."""
    data_dir = cli_data or cfg["data"].get("data_dir")
    if data_dir:
        return data_dir
    import kagglehub  # import tardio: so precisa no Colab

    print(f"[data] baixando '{cfg['data']['dataset_slug']}' via kagglehub...")
    return kagglehub.dataset_download(cfg["data"]["dataset_slug"])
