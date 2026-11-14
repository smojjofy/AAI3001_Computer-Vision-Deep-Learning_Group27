import torch

from modules.inference.checkpoint import load_checkpoint, save_checkpoint
from modules.inference.depth import TinyDepthNet


def test_checkpoint_round_trip(tmp_path) -> None:
    model = TinyDepthNet(base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "depth.pt"

    save_checkpoint(
        path,
        task="depth",
        model=model,
        model_config={"in_channels": 3, "base_channels": 8},
        epoch=2,
        global_step=12,
        optimizer=optimizer,
        metrics={"validation_loss": 0.25},
    )
    payload = load_checkpoint(path, expected_task="depth")

    assert payload["epoch"] == 2
    assert payload["global_step"] == 12
    assert payload["model_config"]["base_channels"] == 8
    assert payload["metrics"]["validation_loss"] == 0.25
