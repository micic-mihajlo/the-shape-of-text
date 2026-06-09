import json

from scripts.prepare_mlx_lora_data import convert_dataset


def test_prepare_mlx_lora_data_writes_train_and_valid(tmp_path):
    train_file = tmp_path / "train_source.jsonl"
    valid_file = tmp_path / "valid_source.jsonl"
    train_file.write_text(
        json.dumps({"prompt": "Rewrite A", "completion": "Finished A"}) + "\n",
        encoding="utf-8",
    )
    valid_file.write_text(
        json.dumps({"prompt": "Rewrite B", "completion": "Finished B"}) + "\n",
        encoding="utf-8",
    )

    counts = convert_dataset(
        train_file=train_file,
        valid_file=valid_file,
        output_dir=tmp_path / "mlx_data",
    )

    assert counts == {"train": 1, "valid": 1, "test": 0}
    train = json.loads((tmp_path / "mlx_data" / "train.jsonl").read_text())
    valid = json.loads((tmp_path / "mlx_data" / "valid.jsonl").read_text())
    assert train == {"prompt": "Rewrite A", "completion": "Finished A"}
    assert valid == {"prompt": "Rewrite B", "completion": "Finished B"}
