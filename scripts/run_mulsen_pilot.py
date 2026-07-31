from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

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


def parse_categories(value: str) -> Union[str, list[str]]:
    if value == "all":
        return "all"
    return [item.strip() for item in value.split(",") if item.strip()]


def command_extract_embeddings(args: argparse.Namespace) -> None:
    from src.extract_embeddings import extract_embeddings_from_official_model

    config = load_config(args.config)
    output = extract_embeddings_from_official_model(
        config,
        categories=parse_categories(args.categories),
        output_path=args.output,
        max_samples_per_split=args.max_samples_per_split,
        device=args.device,
    )
    print(json.dumps({"embedding_cache": str(output)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MulSen-AD pilot runner")
    parser.add_argument("--config", type=Path, default=Path("configs/mulsen_pilot.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Validate official split and modality pairing")
    extract_parser = subparsers.add_parser("extract-embeddings", help="Cache official encoder embeddings")
    extract_parser.add_argument("--categories", default="all", help="Comma-separated categories or 'all'")
    extract_parser.add_argument("--output", type=Path, default=None)
    extract_parser.add_argument("--max-samples-per-split", type=int, default=None)
    extract_parser.add_argument("--device", default=None, help="Device override such as 'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.command == "index":
        command_index(args)
    elif args.command == "extract-embeddings":
        command_extract_embeddings(args)


if __name__ == "__main__":
    main()
