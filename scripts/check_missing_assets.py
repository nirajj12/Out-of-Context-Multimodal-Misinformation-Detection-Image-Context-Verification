"""Inspect selected staged images without changing them."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHERS = {"bbc", "guardian", "usa_today", "washington_post"}


def safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Path must be a nonempty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"Invalid path characters: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe or noncanonical relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in parts[0]:
        raise ValueError(f"Absolute/drive path is not allowed: {value!r}")
    return path


def validate_image_path(value):
    path = safe_relative_path(value)
    parts = path.parts
    if len(parts) != 6 or parts[:2] != ("visual_news", "origin"):
        raise ValueError(f"Unexpected VisualNews image path: {value!r}")
    if parts[2] not in PUBLISHERS or parts[3] != "images":
        raise ValueError(f"Unexpected publisher/image directory: {value!r}")
    if path.suffix != ".jpg":
        raise ValueError(f"Expected .jpg extension: {value!r}")
    return path


def default_image_directory():
    config_path = PROJECT_ROOT / "configs" / "paths.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    pilot_root = Path(config["pilot_root"])
    if not pilot_root.is_absolute():
        pilot_root = PROJECT_ROOT / pilot_root
    return pilot_root / "images"


def load_asset_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Asset manifest does not exist: {path}")
    assets = pd.read_parquet(path, engine="pyarrow")
    required = {"image_path", "is_query_image", "is_evidence_image", "asset_role", "source"}
    missing_columns = required - set(assets.columns)
    if missing_columns:
        raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")
    if assets.empty or not assets["image_path"].is_unique:
        raise ValueError("Asset paths must be nonempty and unique")
    for column in ("is_query_image", "is_evidence_image"):
        if assets[column].isna().any() or not pd.api.types.is_bool_dtype(assets[column]):
            raise ValueError(f"{column} must contain non-null booleans")
    if "normalized_archive_path" not in assets.columns:
        assets["normalized_archive_path"] = assets["image_path"]

    for asset in assets.itertuples(index=False):
        image_path = validate_image_path(asset.image_path)
        safe_relative_path(asset.normalized_archive_path)
        if image_path.parts[2] != asset.source:
            raise ValueError(f"Publisher disagrees with path: {asset.image_path}")
        if asset.is_query_image and asset.is_evidence_image:
            expected_role = "both"
        elif asset.is_query_image:
            expected_role = "query_only"
        elif asset.is_evidence_image:
            expected_role = "evidence_only"
        else:
            raise ValueError(f"Asset has neither role: {asset.image_path}")
        if asset.asset_role != expected_role:
            raise ValueError(f"Inconsistent asset role: {asset.image_path}")
    if not assets["normalized_archive_path"].is_unique:
        raise ValueError("Archive paths must be unique")
    return assets


def staged_path(image_directory, image_path):
    root = Path(image_directory).resolve()
    relative_path = validate_image_path(image_path)
    destination = root.joinpath(*relative_path.parts)
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"Staged path escapes the image directory: {image_path}")
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in staged path: {current}")
    return destination


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path):
    result = {
        "file_found": False, "readable": False, "size_bytes": None,
        "format": None, "width": None, "height": None, "sha256": None,
        "status": "missing", "error": "",
    }
    if not path.exists():
        return result
    if not path.is_file():
        result.update(status="not_regular_file", error="Expected a regular file")
        return result
    result["file_found"] = True
    try:
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
        if result["size_bytes"] == 0:
            result.update(status="zero_byte", error="File is empty")
            return result
        with Image.open(path) as image:
            result["format"] = image.format
            result["width"], result["height"] = image.size
            image.verify()
        # Decoding also catches truncated files that pass a header-only check.
        with Image.open(path) as image:
            image.load()
        result["readable"] = True
        result["status"] = "ok" if result["format"] == "JPEG" else "unexpected_format"
    except (
        OSError, ValueError, SyntaxError, EOFError,
        UnidentifiedImageError, Image.DecompressionBombError,
    ) as error:
        result.update(status="unreadable", error=str(error))
    return result


def check_assets(assets, image_directory):
    results = []
    for asset in assets.itertuples(index=False):
        try:
            path = staged_path(image_directory, asset.image_path)
        except ValueError as error:
            result = {
                "file_found": False, "readable": False, "size_bytes": None,
                "format": None, "width": None, "height": None, "sha256": None,
                "status": "unsafe_staged_path", "error": str(error),
            }
        else:
            result = inspect_image(path)
        result["image_path"] = asset.image_path
        results.append(result)
    return pd.DataFrame(results)


def summarize_assets(results):
    hash_counts = results["sha256"].dropna().value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1]
    corrupt_mask = results["status"].isin(["zero_byte", "unreadable"])
    summary = {
        "expected_images": len(results),
        "found_images": int(results["file_found"].sum()),
        "missing_local_images": int((results["status"] == "missing").sum()),
        "available_pct": 100 * results["file_found"].mean(),
        "readable_images": int(results["readable"].sum()),
        "corrupt_or_unreadable_images": int(corrupt_mask.sum()),
        "zero_byte_images": int((results["status"] == "zero_byte").sum()),
        "unexpected_formats": int((results["status"] == "unexpected_format").sum()),
        "unsafe_staged_paths": int((results["status"] == "unsafe_staged_path").sum()),
        "not_regular_files": int((results["status"] == "not_regular_file").sum()),
        "hashed_files": int(results["sha256"].notna().sum()),
        "unique_hashes": len(hash_counts),
        "duplicate_hash_groups": len(duplicate_hashes),
        "files_in_duplicate_groups": int(duplicate_hashes.sum()),
    }
    return summary, duplicate_hashes


def print_summary(results):
    summary, duplicate_hashes = summarize_assets(results)
    print(pd.Series(summary, name="value", dtype=object).to_frame().to_string())
    readable = results[results["readable"]]
    if readable.empty:
        print("No readable staged images; dimensions and format counts are unavailable.")
    else:
        print("Image formats:")
        print(readable["format"].value_counts().to_string())
        print("Dimensions of readable images:")
        print(readable[["width", "height"]].describe().to_string())
    if summary["hashed_files"] == 0:
        print("No staged files were hashed; exact duplicate checks remain pending.")
    elif duplicate_hashes.empty:
        print("No exact duplicates among the files successfully hashed.")
    else:
        print("Exact SHA-256 duplicate groups (first 10):")
        for digest, count in duplicate_hashes.head(10).items():
            paths = results.loc[results["sha256"] == digest, "image_path"]
            print(f"{digest}: {count} files; examples: {paths.head(2).tolist()}")
    problems = results[results["status"] != "ok"]
    if not problems.empty:
        print("Local staging issues (first 10; not a dataset-availability claim):")
        print(problems[["image_path", "status", "error"]].head(10).to_string(index=False))
    print("SHA-256 detects exact file duplicates, not visual near-duplicates.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_manifest = PROJECT_ROOT / "data/manifests/pilot_image_assets.parquet"
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--output", type=Path,
        help="Staged image directory; defaults to configured pilot_root/images",
    )
    arguments = parser.parse_args(argv)
    image_directory = arguments.output
    if image_directory is None:
        image_directory = default_image_directory()
    try:
        assets = load_asset_manifest(arguments.manifest)
        results = check_assets(assets, image_directory)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Asset check failed: {error}\n")
    print(f"Staged image directory: {image_directory.resolve()}")
    print_summary(results)
    return 0 if bool((results["status"] == "ok").all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
