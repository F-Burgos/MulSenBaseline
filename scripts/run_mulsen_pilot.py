from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mulsen_data import assert_protocol_index, build_sample_index


def load_config(path: Path) -> dict:
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)


def command_index(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    categories = config["data"].get("categories", "all")
    records = build_sample_index(config["paths"]["dataset_root"], categories=categories)
    assert_protocol_index(records)
    summary = {
        "num_samples": len(records),
        "num_train": sum(record.split == "train" for record in records),
        "num_test": sum(record.split == "test" for record in records),
        "categories": sorted({record.category for record in records}),
    }
    output = Path(config["paths"]["artifacts_root"]) / "metrics" / "sample_index_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MulSen-AD pilot runner")
    parser.add_argument("--config", type=Path, default=Path("configs/mulsen_pilot.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Validate official split and modality pairing")
    args = parser.parse_args()

    if args.command == "index":
        command_index(args)


if __name__ == "__main__":
    main()
