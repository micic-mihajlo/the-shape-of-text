#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from shape_of_text.style_metrics import aggregate_style_metrics


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completions(records: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, str):
            values.append(value)
    return values


def metric_distance(candidate: dict[str, float], target: dict[str, float]) -> float:
    return float(sum(abs(candidate[key] - target[key]) for key in target))


def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = str(record.get("id") or index)
        keyed[key] = record
    return keyed


def pairwise_summary(
    adapter_records: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
    *,
    completion_field: str,
) -> dict[str, float]:
    adapter_by_id = records_by_id(adapter_records)
    base_by_id = records_by_id(base_records)
    shared_ids = sorted(set(adapter_by_id) & set(base_by_id))
    if not shared_ids:
        return {
            "shared_count": 0.0,
            "same_output_rate": 0.0,
            "adapter_empty_rate": 0.0,
            "base_empty_rate": 0.0,
            "avg_char_delta_adapter_minus_base": 0.0,
        }

    same_count = 0
    adapter_empty_count = 0
    base_empty_count = 0
    char_deltas: list[int] = []
    for item_id in shared_ids:
        adapter_text = str(adapter_by_id[item_id].get(completion_field) or "").strip()
        base_text = str(base_by_id[item_id].get(completion_field) or "").strip()
        same_count += int(adapter_text == base_text)
        adapter_empty_count += int(not adapter_text)
        base_empty_count += int(not base_text)
        char_deltas.append(len(adapter_text) - len(base_text))

    shared_count = len(shared_ids)
    return {
        "shared_count": float(shared_count),
        "same_output_rate": same_count / shared_count,
        "adapter_empty_rate": adapter_empty_count / shared_count,
        "base_empty_rate": base_empty_count / shared_count,
        "avg_char_delta_adapter_minus_base": sum(char_deltas) / shared_count,
    }


def comparison_report(
    *,
    adapter_file: Path,
    base_file: Path,
    target_file: Path,
    adapter_field: str = "completion",
    base_field: str = "completion",
    target_field: str = "completion",
) -> dict[str, Any]:
    adapter_records = read_records(adapter_file)
    base_records = read_records(base_file)
    target_records = read_records(target_file)

    adapter_metrics = aggregate_style_metrics(completions(adapter_records, adapter_field))
    base_metrics = aggregate_style_metrics(completions(base_records, base_field))
    target_metrics = aggregate_style_metrics(completions(target_records, target_field))

    adapter_distance = metric_distance(adapter_metrics, target_metrics)
    base_distance = metric_distance(base_metrics, target_metrics)
    return {
        "adapter_metrics": adapter_metrics,
        "base_metrics": base_metrics,
        "target_metrics": target_metrics,
        "adapter_target_distance": adapter_distance,
        "base_target_distance": base_distance,
        "style_distance_improvement": base_distance - adapter_distance,
        "pairwise": pairwise_summary(
            adapter_records,
            base_records,
            completion_field=adapter_field,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and adapter social outputs")
    parser.add_argument("--adapter-file", type=Path, required=True)
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--adapter-field", default="completion")
    parser.add_argument("--base-field", default="completion")
    parser.add_argument("--target-field", default="completion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = comparison_report(
        adapter_file=args.adapter_file,
        base_file=args.base_file,
        target_file=args.target_file,
        adapter_field=args.adapter_field,
        base_field=args.base_field,
        target_field=args.target_field,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
