"""Export an original-project checkpoint to a portable state-dict.

Run this script in the original repository/environment so torch can import the
classes referenced by the legacy checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Legacy checkpoints reference modules from the original repository. Ensure that
# repository wins over any unrelated Ultralytics installation on sys.path.
ORIGINAL_REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ORIGINAL_REPOSITORY))

import torch


def load_legacy_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="Export portable Mid-Prior weights")
    parser.add_argument("input", help="Original best.pt or last.pt")
    parser.add_argument("output", help="Output .pth path")
    args = parser.parse_args()

    print(f"Loading legacy checkpoint: {Path(args.input).resolve()}", flush=True)
    checkpoint = load_legacy_checkpoint(args.input)
    if not isinstance(checkpoint, dict):
        model = checkpoint
    else:
        model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None or not hasattr(model, "state_dict"):
        raise TypeError("Checkpoint does not contain a PyTorch model under 'ema' or 'model'")

    print("Preparing FP32 state-dict...", flush=True)
    model = model.float()
    head = model.model[-1]
    yaml_config = getattr(model, "yaml", {}) or {}
    payload = {
        "format": "mid-prior-state-dict-v1",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "nc": int(getattr(head, "nc")),
        "names": getattr(model, "names", None),
        "scale": yaml_config.get("scale"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing portable checkpoint: {output.resolve()}", flush=True)
    torch.save(payload, output)
    print(f"Exported {len(payload['state_dict'])} tensors to {output.resolve()}")


if __name__ == "__main__":
    main()

