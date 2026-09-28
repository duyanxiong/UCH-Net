"""Portable state-dict loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_public_weights(model, path, strict=True):
    """Load a portable file produced by tools/export_public_weights.py."""
    payload = _torch_load(Path(path))
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict) or not all(torch.is_tensor(value) for value in state_dict.values()):
        raise TypeError("Expected a tensor state_dict or a {'state_dict': ...} portable checkpoint")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)
    if isinstance(payload, dict) and payload.get("names") is not None:
        model.names = payload["names"]
    return payload

