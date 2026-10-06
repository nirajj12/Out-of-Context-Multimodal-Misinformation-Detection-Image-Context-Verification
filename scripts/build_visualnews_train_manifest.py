"""Freeze only the ID-keyed VisualNews TRAIN metadata inspected in notebook 02."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

if __package__:
    from .check_missing_assets import validate_image_path
else:
    from check_missing_assets import validate_image_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRAIN_COUNT = 385_003
REQUIRED_FIELDS = ["id", "image_path", "caption", "source"]
OPTIONAL_FIELDS = ["title", "article_path", "full_article_path", "timestamp", "topic"]


def build_train_table(metadata_path, expected_count=EXPECTED_TRAIN_COUNT):
    with Path(metadata_path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("TRAIN metadata must be a nonempty ID-keyed dictionary")
    if len(raw) != expected_count:
        raise ValueError(f"TRAIN count changed: {len(raw):,}; expected {expected_count:,}")
    for key, record in raw.items():
        if not isinstance(record, dict):
            raise ValueError(f"Malformed record: {key!r}")
        metadata_id = record.get("id")
        if type(metadata_id) is not int or metadata_id < 0 or str(metadata_id) != key:
            raise ValueError(f"Original ID/key mismatch: {key!r}")
        for field in ["image_path", "caption", "source"]:
            value = record.get(field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"Unusable {field} for ID {metadata_id}")
        image_path = validate_image_path(record["image_path"])
        if image_path.parts[2] != record["source"]:
            raise ValueError(f"Source/path mismatch for ID {metadata_id}")
        if "split" in record and record["split"] != "train":
            raise ValueError(f"Non-TRAIN row: {metadata_id}")
    fields = REQUIRED_FIELDS.copy()
    for field in OPTIONAL_FIELDS:
        if any(field in record for record in raw.values()):
            fields.append(field)
    table = pd.DataFrame(raw.values())[fields].copy()
    table["split"] = "train"  # Provenance of the TRAIN file, not an event label.
    table = table.sort_values("id", kind="stable").reset_index(drop=True)
    if not table.id.is_unique or not table.image_path.is_unique:
        raise ValueError("TRAIN IDs and image paths must be unique")
    return table


def save_manifest(table, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(output_path), table)
        return "existing identical manifest preserved"
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        table.to_parquet(temporary_path, engine="pyarrow", compression="zstd", compression_level=9, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(temporary_path), table)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "created and reloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/raw/visualnews_metadata/train.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/manifests/visualnews_train_evidence.parquet")
    args = parser.parse_args()
    table = build_train_table(args.metadata)
    status = save_manifest(table, args.output)
    summary = {
        "validation_passed": True, "train_count": len(table), "columns": table.columns.tolist(),
        "manifest_path": str(args.output), "size_bytes": args.output.stat().st_size,
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "unique_ids": int(table.id.nunique()), "unique_image_paths": int(table.image_path.nunique()),
        "source_counts": table.source.value_counts().to_dict(),
        "null_title_count": int(table.title.isna().sum()) if "title" in table else None,
        "all_train": bool(table.split.eq("train").all()), "status": status,
        "ids_paths_captions_preserved": True, "construction_labels_added": False,
        "timestamp_semantics": "Article-associated timestamp; not asserted event/capture date.",
    }
    report_path = PROJECT_ROOT / "outputs/large_corpus/train_manifest_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
