from pathlib import Path

import torch
import yaml

from mid_prior_yolo import MidPriorDetectionModel


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model.yaml"
PACKAGED_CONFIG = Path(__file__).resolve().parents[1] / "src" / "mid_prior_yolo" / "configs" / "model.yaml"


def test_packaged_config_matches_checkout():
    assert yaml.safe_load(CONFIG.read_text(encoding="utf-8")) == yaml.safe_load(
        PACKAGED_CONFIG.read_text(encoding="utf-8")
    )


def test_n_scale_training_forward():
    model = MidPriorDetectionModel(CONFIG, nc=2, scale="n", verbose=False).train()
    outputs = model(torch.zeros(1, 6, 128, 128))
    assert [tuple(x.shape[-2:]) for x in outputs] == [(16, 16), (8, 8), (4, 4)]
    assert all(x.shape[1] == 66 for x in outputs)


def test_three_channel_profile_fallback():
    model = MidPriorDetectionModel(CONFIG, nc=2, scale="n", verbose=False).eval()
    output, features = model(torch.zeros(1, 3, 128, 128))
    assert output.shape[1] == 6
    assert len(features) == 3

