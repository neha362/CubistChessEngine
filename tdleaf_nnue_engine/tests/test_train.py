from pathlib import Path

from tdleaf_nnue_engine.train import train_model


def test_training_pipeline_writes_checkpoint(tmp_path: Path):
    out = tmp_path / "nnue_model.pt"
    ckpt = train_model(
        output_checkpoint=str(out),
        games=1,
        max_plies=10,
        depth=1,
        epochs=1,
        batch_size=8,
        seed=1,
    )
    assert ckpt.exists()
