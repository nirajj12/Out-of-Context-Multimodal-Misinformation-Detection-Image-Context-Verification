"""Resumable tar ranges, bounded image batches, validated checkpoints, separate text jobs.

Notebook 07 supplies the frozen encoder and controls the smoke/full run boundary.
Only this module's checkpoint directories are writable. Pilot artifacts are read-only.
Use a single writer per checkpoint directory.
"""

import csv
import hashlib
import json
import tempfile
import warnings
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

if __package__:
    from .check_missing_assets import safe_relative_path, validate_image_path
    from .extract_selected_images import canonical_member_name, open_archive
    from .visualnews_archive_resume import (ArchiveResumeError, HTTPRangeSource, ArchiveCursor,
                                          member_boundary, writer_lock, sync_handle, sync_directory)
else:
    from check_missing_assets import safe_relative_path, validate_image_path
    from extract_selected_images import canonical_member_name, open_archive
    from visualnews_archive_resume import (ArchiveResumeError, HTTPRangeSource, ArchiveCursor,
                                         member_boundary, writer_lock, sync_handle, sync_directory)


ARCHIVE_URL = "https://www.cs.rice.edu/~vo9/visualnews/origin.tar"
FAILURE_COLUMNS = ["metadata_id", "image_path", "error_type", "error_message", "timestamp_utc"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            sync_handle(handle)
        assert json.loads(temporary.read_text()) == value
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def validate_train_manifest(table, expected_count=385_003):
    required = {"id", "image_path", "caption", "source", "split"}
    if not required.issubset(table.columns) or len(table) != expected_count:
        raise ValueError("TRAIN manifest schema or count mismatch")
    forbidden = {"falsified", "source_dataset", "newsclip_split", "is_query_provenance", "provenance_role"}
    if forbidden.intersection(table.columns):
        raise ValueError("Construction/query-provenance labels are not evidence fields")
    if not table.id.is_unique or not table.image_path.is_unique or not table.split.eq("train").all():
        raise ValueError("Non-TRAIN or duplicate targets")
    for row in table.itertuples(index=False):
        if not isinstance(row.id, (int, np.integer)) or row.id < 0:
            raise ValueError("Invalid original metadata ID")
        if not isinstance(row.caption, str) or not row.caption.strip():
            raise ValueError(f"Unusable caption: {row.id}")
        path = validate_image_path(row.image_path)
        if path.parts[2] != row.source:
            raise ValueError(f"Source/path mismatch: {row.id}")


def encoder_signature(pilot_config):
    keys = ["model_name", "pretrained", "activation", "force_quick_gelu", "embedding_dimension",
            "saved_dtype", "frozen", "image_preprocessing", "image_loading", "tokenizer_source",
            "tokenizer_context_length", "normalization_method", "openclip_version"]
    return {key: pilot_config[key] for key in keys}


def load_compatible_encoder(pilot_config, require_t4=True):
    import torch
    import open_clip
    from importlib.metadata import version

    expected = {"model_name": "ViT-B-32", "pretrained": "openai", "activation": "QuickGELU",
                "force_quick_gelu": True, "embedding_dimension": 512, "saved_dtype": "float32", "frozen": True,
                "image_loading": "Pillow default frame, RGB conversion in memory",
                "tokenizer_source": "open_clip.get_tokenizer(model_name)",
                "normalization_method": "float32 L2 normalization per row before CPU numpy conversion"}
    for key, value in expected.items():
        if pilot_config.get(key) != value:
            raise ValueError(f"STOP: frozen pilot config mismatch: {key}")
    if version("open_clip_torch") != pilot_config["openclip_version"]:
        raise ValueError("STOP: OpenCLIP version differs from frozen pilot; do not silently change it")
    if not torch.cuda.is_available():
        raise RuntimeError("STOP: CUDA unavailable; this validation requires the Colab GPU")
    if require_t4 and "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("STOP: smoke-test runtime is not a T4")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", precision="fp32", device="cuda", force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    model.requires_grad_(False)
    activations = {type(module).__name__ for module in model.modules()
                   if type(module).__name__ in ("GELU", "QuickGELU")}
    if activations != {"QuickGELU"} or model.visual.output_dim != 512:
        raise ValueError("STOP: activation/dimension mismatch")
    if repr(preprocess) != pilot_config["image_preprocessing"]:
        raise ValueError("STOP: evaluation preprocessing differs from frozen pilot")
    if model.context_length != pilot_config["tokenizer_context_length"]:
        raise ValueError("STOP: tokenizer context length differs from frozen pilot")
    if tokenizer([""]).shape[1] != model.context_length:
        raise ValueError("STOP: tokenizer/model mismatch")
    return model, preprocess, tokenizer


def normalized_cpu_embeddings(features):
    import torch
    features = features.float()
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if not torch.isfinite(features).all().item() or not (norms > 0).all().item():
        raise RuntimeError("STOP: invalid model features")
    return (features / norms).cpu().numpy().astype(np.float32, copy=False)


def make_batch_encoder(model, modality, tokenizer=None):
    import torch
    if modality not in ("image", "text"):
        raise ValueError("Unknown modality")

    def encode(items):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("STOP: encoder must remain frozen and in evaluation mode")
        with torch.inference_mode():
            if modality == "image":
                batch = torch.stack(items).to("cuda")
                features = model.encode_image(batch)
            else:
                batch = tokenizer(items).to("cuda")
                features = model.encode_text(batch)
            result = normalized_cpu_embeddings(features)
        del batch, features
        return result
    return encode


def validate_chunk(matrix, mapping, targets):
    if matrix.dtype != np.float32 or matrix.shape != (len(mapping), 512) or not len(mapping):
        raise ValueError("Chunk must contain N aligned float32 512-D rows")
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite embeddings")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError("Embeddings must be L2-normalized")
    if mapping.embedding_row_within_chunk.tolist() != list(range(len(mapping))):
        raise ValueError("Within-chunk rows are not contiguous")
    if not mapping.metadata_id.is_unique or not mapping.image_path.is_unique:
        raise ValueError("Duplicate ID/path inside chunk")
    lookup = targets if targets.index.name == "id" else targets.set_index("id")
    expected = lookup.reindex(mapping.metadata_id)
    if expected.image_path.isna().any():
        raise ValueError("Checkpoint contains non-target IDs")
    for field in ["image_path", "source", "split"]:
        if mapping[field].tolist() != expected[field].tolist():
            raise ValueError(f"Checkpoint metadata misalignment: {field}")
    return {"rows": len(mapping), "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "min_norm": float(norms.min()), "max_norm": float(norms.max()), "finite": True}


def read_validated_chunk(array_path, map_path, marker_path, targets, signature):
    """Read one completed chunk without changing its files or progress state."""
    array_path, map_path, marker_path = Path(array_path), Path(map_path), Path(marker_path)
    if not all(path.is_file() for path in [array_path, map_path, marker_path]):
        raise ValueError("STOP: chunk array, mapping and completion marker are required")
    marker = json.loads(marker_path.read_text())
    if marker["signature"] != signature:
        raise ValueError("STOP: chunk belongs to a different manifest/model/modality")
    if marker["embedding_file"] != array_path.name or marker["mapping_file"] != map_path.name:
        raise ValueError("STOP: chunk filenames differ from completion marker")
    if marker["number"] != int(array_path.stem.rsplit("_", 1)[1]):
        raise ValueError("STOP: chunk number differs from completion marker")
    if file_sha256(array_path) != marker["embedding_sha256"] or file_sha256(map_path) != marker["mapping_sha256"]:
        raise ValueError("STOP: completed chunk checksum mismatch")
    matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
    mapping = pd.read_parquet(map_path)
    validation = validate_chunk(matrix, mapping, targets)
    if validation != marker["validation"]:
        raise ValueError("STOP: chunk validation differs from completion marker")
    return matrix, mapping


def validate_merged_embeddings(path, expected_count, block_size=5000):
    """Check a memory-mapped final array one small block at a time."""
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (expected_count, 512) or matrix.dtype != np.float32 or expected_count < 1:
        raise ValueError("STOP: final array must have N rows, 512 dimensions and float32 dtype")
    minimum_norm, maximum_norm = float("inf"), 0.0
    for start in range(0, expected_count, block_size):
        block = matrix[start:start + block_size]
        if not np.isfinite(block).all():
            raise ValueError("STOP: nonfinite final embeddings")
        norms = np.linalg.norm(block, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
            raise ValueError("STOP: final embeddings are not normalized")
        minimum_norm = min(minimum_norm, float(norms.min()))
        maximum_norm = max(maximum_norm, float(norms.max()))
    return {"rows": expected_count, "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "finite": True, "min_norm": minimum_norm, "max_norm": maximum_norm}


class EmbeddingCheckpoints:
    """Commit array + map + checksum marker; reconstruct completion from validated maps."""

    def __init__(self, chunk_dir, state_path, targets, manifest_hash, model_config, modality,
                 checkpoint_size=1000, failure_log_format="jsonl", gpu_batch_size=64):
        if modality not in ("image", "text") or checkpoint_size < 1:
            raise ValueError("Invalid modality/checkpoint size")
        validate_train_manifest(targets, expected_count=len(targets))
        self.chunk_dir = Path(chunk_dir)
        self.state_path = Path(state_path)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.with_name(self.state_path.name + ".lock").exists():
            raise ArchiveResumeError("STOP: writer lock exists; confirm old writer stopped before removing only its lock")
        self._expected_state_hash = file_sha256(self.state_path) if self.state_path.exists() else None
        self.targets = targets
        self.target_lookup = targets.set_index("id")[["image_path", "source", "split"]]
        self.modality = modality
        self.checkpoint_size = checkpoint_size
        self.gpu_batch_size = gpu_batch_size
        if failure_log_format not in ("jsonl", "csv"):
            raise ValueError("Failure log must be jsonl or csv")
        self.failures_path = self.state_path.with_name(f"{modality}_embedding_failures.{failure_log_format}")
        ordered = targets[["id", "image_path"]].sort_values("id").to_csv(index=False)
        self.signature = {"manifest_sha256": manifest_hash, "model_config": model_config,
                          "modality": modality, "target_count": len(targets),
                          "target_selection_sha256": hashlib.sha256(ordered.encode()).hexdigest()}
        self.state = {"signature": self.signature, "started_at_utc": utc_now(),
                      "archive_passes_opened": 0, "scanned_archive_member_count": 0}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("signature") != self.signature:
                raise ValueError("STOP: checkpoint model/manifest/target scope mismatch")
            if self.state.get("failures_log") != self.failures_path.name:
                raise ValueError("STOP: keep the existing failure log format when resuming")
        elif list(self.chunk_dir.glob(f"{modality}_*chunk_*")):
            raise ValueError("STOP: chunks exist without a scope-bearing progress file")
        self.completed_paths = set()
        self.completed_ids = set()
        self.chunks = []
        self.pending_rows = []
        self.pending_vectors = []
        self.pending_paths = set()
        with writer_lock(self.state_path):
            self.save_progress("validating_checkpoints")
            self.reload_completed()
            self.save_progress("ready")

    def chunk_paths(self, number):
        suffix = f"chunk_{number:05d}"
        return (self.chunk_dir / f"{self.modality}_embeddings_{suffix}.npy",
                self.chunk_dir / f"{self.modality}_embedding_map_{suffix}.parquet",
                self.chunk_dir / f"{self.modality}_{suffix}.complete.json")

    def chunk_marker(self, number, matrix, mapping):
        array_path, map_path, _ = self.chunk_paths(number)
        validation = validate_chunk(matrix, mapping, self.target_lookup)
        return {"signature": self.signature, "number": number, "validation": validation,
                "embedding_file": array_path.name, "mapping_file": map_path.name,
                "embedding_sha256": file_sha256(array_path), "mapping_sha256": file_sha256(map_path)}

    def reload_completed(self):
        numbers = set()
        for path in self.chunk_dir.glob(f"{self.modality}_*chunk_*"):
            if path.suffix == ".tmp":
                continue
            tail = path.name.split("chunk_", 1)[1].split(".", 1)[0]
            if tail.isdecimal():
                numbers.add(int(tail))
        if sorted(numbers) != list(range(len(numbers))):
            raise ValueError("STOP: checkpoint numbering has a gap")
        for number in sorted(numbers):
            array_path, map_path, marker_path = self.chunk_paths(number)
            if not array_path.exists() or not map_path.exists():
                raise ValueError(f"STOP: incomplete checkpoint pair: {number}; preserve and repair it")
            matrix = np.load(array_path, allow_pickle=False)
            mapping = pd.read_parquet(map_path)
            marker = self.chunk_marker(number, matrix, mapping)
            if marker_path.exists():
                if json.loads(marker_path.read_text()) != marker:
                    raise ValueError(f"STOP: completed chunk checksum/scope mismatch: {number}")
            else:
                # Disconnect after publishing both files but before publishing the commit marker.
                write_json_atomic(marker_path, marker)
            paths = set(mapping.image_path)
            ids = set(mapping.metadata_id)
            if self.completed_paths.intersection(paths) or self.completed_ids.intersection(ids):
                raise ValueError("STOP: duplicates across completed chunks")
            self.completed_paths.update(paths)
            self.completed_ids.update(ids)
            self.chunks.append(marker)

    def check_state_unchanged(self):
        actual = file_sha256(self.state_path) if self.state_path.exists() else None
        if actual != self._expected_state_hash:
            raise ArchiveResumeError("STOP: progress changed by another writer; reload validated checkpoints")

    def save_progress(self, status, **details):
        self.check_state_unchanged()
        self.state.update(details)
        config = self.signature["model_config"]
        self.state.update(status=status, updated_at_utc=utc_now(),
                          completed_target_count=len(self.completed_paths),
                          completed_chunks=[chunk["embedding_file"] for chunk in self.chunks],
                          failures_log=self.failures_path.name,
                          manifest_sha256=self.signature["manifest_sha256"],
                          target_count=self.signature["target_count"],
                          completed_count=len(self.completed_paths), chunk_count=len(self.chunks),
                          failure_count=self.state.get("failure_attempt_count", 0),
                          archive_members_scanned=self.state["scanned_archive_member_count"],
                          model=config.get("model_name"), pretrained=config.get("pretrained"),
                          batch_size=self.gpu_batch_size, checkpoint_size=self.checkpoint_size)
        write_json_atomic(self.state_path, self.state)
        self._expected_state_hash = file_sha256(self.state_path)

    def add(self, rows, matrix):
        self.check_state_unchanged()
        mapping = pd.DataFrame(rows).rename(columns={"id": "metadata_id"})
        mapping.insert(0, "embedding_row_within_chunk", range(len(mapping)))
        validate_chunk(matrix, mapping, self.target_lookup)
        for row, vector in zip(rows, matrix):
            path = row["image_path"]
            if path in self.completed_paths or path in self.pending_paths:
                raise ValueError("STOP: refusing to recompute/append a completed target")
            self.pending_rows.append(row)
            self.pending_vectors.append(vector.copy())
            self.pending_paths.add(path)
            if len(self.pending_rows) == self.checkpoint_size:
                self.flush()

    def flush(self):
        if not self.pending_rows:
            return
        self.check_state_unchanged()
        number = len(self.chunks)
        array_path, map_path, marker_path = self.chunk_paths(number)
        if any(path.exists() for path in [array_path, map_path, marker_path]):
            raise FileExistsError("STOP: never overwrite a published checkpoint")
        matrix = np.stack(self.pending_vectors).astype(np.float32, copy=False)
        mapping = pd.DataFrame(self.pending_rows).rename(columns={"id": "metadata_id"})
        mapping.insert(0, "embedding_row_within_chunk", range(len(mapping)))
        validate_chunk(matrix, mapping, self.target_lookup)
        temporary_paths = []
        try:
            for final_path in [array_path, map_path]:
                with tempfile.NamedTemporaryFile(dir=self.chunk_dir, suffix=".tmp", delete=False) as handle:
                    temporary_paths.append(Path(handle.name))
            with temporary_paths[0].open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                sync_handle(handle)
            mapping.to_parquet(temporary_paths[1], index=False)
            with temporary_paths[1].open("rb") as handle:
                sync_handle(handle)
            reloaded = np.load(temporary_paths[0], allow_pickle=False)
            reloaded_map = pd.read_parquet(temporary_paths[1])
            validate_chunk(reloaded, reloaded_map, self.target_lookup)
            if not np.array_equal(reloaded, matrix):
                raise ValueError("Array save/reload mismatch")
            pd.testing.assert_frame_equal(mapping, reloaded_map)
            for temporary, final in zip(temporary_paths, [array_path, map_path]):
                if final.exists():
                    raise FileExistsError("Checkpoint appeared during save; single writer required")
                temporary.replace(final)
                sync_directory(final.parent)
            marker = self.chunk_marker(number, np.load(array_path, allow_pickle=False), pd.read_parquet(map_path))
            write_json_atomic(marker_path, marker)
            self.chunks.append(marker)
            self.completed_paths.update(mapping.image_path)
            self.completed_ids.update(mapping.metadata_id)
            self.pending_rows.clear()
            self.pending_vectors.clear()
            self.pending_paths.clear()
            self.save_progress("checkpoint_saved")
            print(f"{self.modality} checkpoint {number:05d}: {len(mapping):,} rows; completed {len(self.completed_paths):,}", flush=True)
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)

    def record_failure(self, row, error):
        record = {"metadata_id": int(row["id"]), "image_path": row["image_path"],
                  "error_type": type(error).__name__, "error_message": str(error)[:1000],
                  "timestamp_utc": utc_now()}
        needs_header = not self.failures_path.exists() or self.failures_path.stat().st_size == 0
        with self.failures_path.open("a", encoding="utf-8", newline="") as handle:
            if self.failures_path.suffix == ".csv":
                writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "error"])
                if needs_header:
                    writer.writeheader()
                writer.writerow({"id": record["metadata_id"], "image_path": record["image_path"],
                                 "error": record["error_type"] + ": " + record["error_message"]})
            else:
                handle.write(json.dumps(record) + "\n")
            sync_handle(handle)
        self.state["failure_attempt_count"] = self.state.get("failure_attempt_count", 0) + 1
        self.save_progress("image_failure_recorded")


def mapping_record(row):
    return {field: row[field] for field in ["id", "image_path", "source", "split"]}


def decode_member_tensor(archive, member, preprocess, max_image_bytes=32 * 1024 * 1024):
    if not member.isfile() or member.size <= 0 or member.size > max_image_bytes:
        raise ValueError("Target must be a bounded, nonempty regular file")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("Target member cannot be read")
    with source:
        image_bytes = source.read(max_image_bytes + 1)
    if len(image_bytes) != member.size:
        raise ValueError("Incomplete/oversized target bytes")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(image_bytes)) as image:
            with image.convert("RGB") as rgb:
                tensor = preprocess(rgb)
    del image_bytes
    return tensor


def stream_image_embeddings(store, preprocess, encode_batch, gpu_batch_size=64,
                            archive_url=ARCHIVE_URL, archive_path=None, progress_every=10000,
                            max_consecutive_failures=5, failure_fraction_limit=0.10,
                            failure_rate_min_attempts=20, no_match_member_limit=100000):
    if gpu_batch_size < 1 or progress_every < 1:
        raise ValueError("Batch/progress sizes must be positive")
    if store.modality != "image":
        raise ValueError("Image streaming requires image checkpoints")
    remaining = store.targets.loc[~store.targets.image_path.isin(store.completed_paths)]
    result = {"target_count": len(store.targets), "already_completed": len(store.completed_paths),
              "found_this_run": 0, "encoded_this_run": 0, "failed_this_run": 0,
              "scanned_members_this_run": 0, "archive_passes_this_run": 0,
              "unsafe_members_skipped": 0, "missing_targets": 0, "gpu_batch_size": gpu_batch_size}
    if remaining.empty:
        store.save_progress("complete", last_run=result)
        return result  # Resume proves zero network scans and zero GPU calls when done.
    requested = {}
    for row in remaining[["id", "image_path", "source", "split"]].to_dict("records"):
        member_name = row["image_path"].removeprefix("visual_news/")
        safe_relative_path(member_name)
        requested[member_name] = row
    tensors, batch_rows, matched = [], [], set()
    consecutive_failures = 0
    recognized_archive_layout = False

    def flush_batch():
        if tensors:
            embeddings = encode_batch(tensors)  # GPU/model errors are systemic, never image failures.
            store.add(batch_rows, embeddings)
            result["encoded_this_run"] += len(batch_rows)
            tensors.clear()
            batch_rows.clear()

    store.state["archive_passes_opened"] += 1
    result["archive_passes_this_run"] = 1
    store.save_progress("streaming", last_run=result)
    try:
        remote = None if archive_path is not None else archive_url
        with open_archive(archive_path=archive_path, archive_url=remote) as (archive, _):
            for member in archive:
                # Python 3.12 caches TarInfo records even in streaming mode. No random access is used.
                archive.members.clear()
                result["scanned_members_this_run"] += 1
                store.state["scanned_archive_member_count"] += 1
                try:
                    name = canonical_member_name(member.name)
                except ValueError:
                    result["unsafe_members_skipped"] += 1
                    continue
                if member.isfile() and not recognized_archive_layout:
                    try:
                        validate_image_path("visual_news/" + name)
                        recognized_archive_layout = True
                    except ValueError:
                        pass
                if result["scanned_members_this_run"] >= no_match_member_limit and not recognized_archive_layout:
                    raise RuntimeError("STOP: archive layout unrecognized; inspect prefixes before continuing")
                if result["scanned_members_this_run"] % progress_every == 0:
                    print(f"Scanned {result['scanned_members_this_run']:,}; matched {len(matched):,}/{len(requested):,}", flush=True)
                    store.save_progress("streaming", last_run=result)
                if name not in requested:
                    continue
                if name in matched:
                    raise ValueError(f"STOP: duplicate selected archive member: {name!r}")
                matched.add(name)
                result["found_this_run"] += int(member.isfile())
                row = requested[name]
                try:
                    tensor = decode_member_tensor(archive, member, preprocess)
                except (OSError, ValueError, SyntaxError, EOFError, UnidentifiedImageError,
                        Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                    store.record_failure(row, error)
                    result["failed_this_run"] += 1
                    consecutive_failures += 1
                    attempts = len(matched)
                    high_fraction = (attempts >= failure_rate_min_attempts
                                     and result["failed_this_run"] / attempts > failure_fraction_limit)
                    if consecutive_failures >= max_consecutive_failures or high_fraction:
                        raise RuntimeError("STOP: image failure safety limit reached; inspect the persistent log") from error
                else:
                    consecutive_failures = 0
                    tensors.append(tensor)
                    batch_rows.append(mapping_record(row))
                    del tensor
                    if len(tensors) == gpu_batch_size:
                        flush_batch()
                if len(matched) == len(requested):
                    break
        flush_batch()
        store.flush()
        missing = set(requested) - matched
        result["missing_targets"] = len(missing)
        for name in sorted(missing):
            store.record_failure(requested[name], FileNotFoundError("Target not found in scanned archive"))
        store.save_progress("complete" if len(store.completed_paths) == len(store.targets) else "incomplete", last_run=result)
        if missing:
            raise RuntimeError(f"STOP: {len(missing):,} target paths unmatched; verify archive layout")
        return result
    except BaseException:
        # Persist already encoded vectors, but never hide a storage-validation failure.
        store.flush()
        store.save_progress("interrupted", last_run=result)
        raise
    finally:
        tensors.clear()
        batch_rows.clear()


def encode_text_embeddings(store, encode_batch, gpu_batch_size=64):
    with writer_lock(store.state_path):
        store.check_state_unchanged()
        return _encode_text_embeddings(store, encode_batch, gpu_batch_size)


def _encode_text_embeddings(store, encode_batch, gpu_batch_size=64):
    if store.modality != "text" or gpu_batch_size < 1:
        raise ValueError("Text encoding requires text checkpoints and a positive batch size")
    remaining = store.targets.loc[~store.targets.image_path.isin(store.completed_paths)]
    result = {"target_count": len(store.targets), "already_completed": len(store.completed_paths),
              "encoded_this_run": 0, "gpu_batch_size": gpu_batch_size, "archive_passes_this_run": 0}
    try:
        for start in range(0, len(remaining), gpu_batch_size):
            batch = remaining.iloc[start:start + gpu_batch_size]
            matrix = encode_batch(batch.caption.tolist())
            rows = [mapping_record(row) for row in batch.to_dict("records")]
            store.add(rows, matrix)
            result["encoded_this_run"] += len(rows)
        store.flush()
        store.save_progress("complete", last_run=result)
        return result
    except BaseException:
        store.flush()
        store.save_progress("interrupted", last_run=result)
        raise


def stream_resumable_image_embeddings(store, preprocess, encode_batch, gpu_batch_size=64,
                                      archive_url=ARCHIVE_URL, source=None,
                                      max_new_images_per_session=None, progress_every=10000,
                                      max_consecutive_failures=5, failure_fraction_limit=0.10,
                                      failure_rate_min_attempts=20, no_match_member_limit=100000):
    """Resume at a logical tar boundary after durable vectors/failures, never HTTP tell.

    The session limit is checked after a GPU batch; overshoot is at most batch_size-1.
    Decode failures before the cursor are retried with bounded conditional ranges.
    """
    if store.modality != "image" or gpu_batch_size < 1 or progress_every < 1:
        raise ValueError("Invalid image store/batch/progress configuration")
    if max_new_images_per_session is not None and (type(max_new_images_per_session) is not int or max_new_images_per_session < 1):
        raise ValueError("Session limit must be a positive integer or None")
    if store.pending_rows or store.pending_vectors:
        raise ArchiveResumeError("STOP: reload checkpoint store after an interrupted call; discard unsaved buffers")
    started = time.perf_counter()
    initial_done = len(store.completed_ids)
    result = {"target_count": len(store.targets), "already_completed": initial_done,
              "encoded_this_run": 0, "failed_this_run": 0, "found_this_run": 0,
              "scanned_members_this_run": 0, "archive_passes_this_run": 0,
              "unsafe_members_skipped": 0, "resume_start_offset": None,
              "committed_archive_offset": None, "archive_size": None,
              "encoder_seconds": 0.0, "checkpoint_seconds": 0.0}
    with writer_lock(store.state_path):
        store.check_state_unchanged()
        if initial_done == len(store.targets):
            previous = store.state.get("last_run", {})
            for key in ["committed_archive_offset", "archive_size", "archive_scan_percent"]:
                result[key] = previous.get(key)
            result.update(status="complete", completed_targets=initial_done, remaining_targets=0,
                          unresolved_failures=0, checkpoint_count=len(store.chunks),
                          target_completion_percent=100.0, elapsed_seconds=time.perf_counter() - started)
            store.save_progress("complete", last_run=result)
            return result  # No source construction, HTTP probe or GPU call.
        cursor = ArchiveCursor(store)
        expected = cursor.state["source"] if cursor.state else None
        if expected is None and cursor.failures:
            identities = [entry["source"] for entry in cursor.failures.values()]
            if any(item != identities[0] for item in identities):
                raise ArchiveResumeError("STOP: conflicting source identities in failure journal")
            expected = identities[0]
        source = source or HTTPRangeSource(archive_url)
        identity = source.probe(expected)
        if cursor.state is None:
            cursor.commit(0, {}, identity)
        base = cursor.state["next_offset"]
        boundary, pax = base, cursor.state["pax_headers"].copy()
        eof = cursor.state["archive_eof"]
        result.update(resume_start_offset=base, archive_size=identity["size"])
        requested = {row["image_path"].removeprefix("visual_news/"): row
                     for row in store.targets[["id", "image_path", "source", "split"]].to_dict("records")}
        tensors, rows, matched = [], [], set()
        consecutive = 0
        attempts = 0

        def report(status, print_status=False):
            unresolved = set(cursor.failures) - store.completed_paths
            result.update(status=status, completed_targets=len(store.completed_ids),
                          remaining_targets=len(store.targets) - len(store.completed_ids),
                          unresolved_failures=len(unresolved),
                          failed_attempts_total=store.state.get("failure_attempt_count", 0),
                          checkpoint_count=len(store.chunks),
                          committed_archive_offset=cursor.state["next_offset"],
                          archive_scan_percent=100 * cursor.state["next_offset"] / identity["size"],
                          target_completion_percent=100 * len(store.completed_ids) / len(store.targets),
                          elapsed_seconds=time.perf_counter() - started)
            store.save_progress(status, last_run=result)
            if print_status:
                print(json.dumps(result, indent=2), flush=True)

        def commit(eof_state=False):
            if tensors or rows or store.pending_vectors:
                raise ArchiveResumeError("STOP: candidate cursor still has uncommitted target work")
            old_chunk_count = len(cursor.state["checkpoints"])
            before = time.perf_counter()
            cursor.commit(boundary, pax, identity, eof_state)
            result["checkpoint_seconds"] += time.perf_counter() - before
            report("streaming")
            if len(store.chunks) > old_chunk_count:
                print(f"Images {result['completed_targets']:,}/{result['target_count']:,}; "
                      f"remaining {result['remaining_targets']:,}; new {result['encoded_this_run']:,}; "
                      f"failed attempts {result['failed_this_run']:,}; unresolved {result['unresolved_failures']:,}; "
                      f"chunks {len(store.chunks):,}; committed byte {boundary:,}; "
                      f"scan {result['archive_scan_percent']:.2f}%; targets {result['target_completion_percent']:.2f}%; "
                      f"elapsed {result['elapsed_seconds']:.1f}s; encode {result['encoder_seconds']:.1f}s; "
                      f"checkpoint/cursor {result['checkpoint_seconds']:.1f}s", flush=True)

        def encode_pending():
            if not tensors:
                return
            before = time.perf_counter()
            embeddings = encode_batch(tensors)  # GPU errors propagate; never logged as decode errors.
            result["encoder_seconds"] += time.perf_counter() - before
            before = time.perf_counter()
            store.add(rows, embeddings)
            result["checkpoint_seconds"] += time.perf_counter() - before
            result["encoded_this_run"] += len(rows)
            tensors.clear()
            rows.clear()

        def flush_vectors():
            before = time.perf_counter()
            store.flush()
            result["checkpoint_seconds"] += time.perf_counter() - before

        def limited():
            return max_new_images_per_session is not None and result["encoded_this_run"] >= max_new_images_per_session

        def decode(payload, row, member_offset, data_offset, size, regular):
            nonlocal consecutive, attempts
            attempts += 1
            result["found_this_run"] += 1
            try:
                if not regular or not 0 < size <= 32 * 1024 * 1024:
                    raise ValueError("Target must be a bounded, nonempty regular file")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(payload)) as image:
                        with image.convert("RGB") as rgb:
                            tensor = preprocess(rgb)
            except (OSError, ValueError, SyntaxError, EOFError, UnidentifiedImageError,
                    Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                cursor.record_failure(row, error, member_offset, data_offset, size, regular, identity)
                result["failed_this_run"] += 1
                consecutive += 1
                if consecutive >= max_consecutive_failures or (attempts >= failure_rate_min_attempts and result["failed_this_run"] / attempts > failure_fraction_limit):
                    raise RuntimeError("STOP: image decode failure safety limit; inspect CSV and range failure journal") from error
            else:
                consecutive = 0
                tensors.append(tensor)
                rows.append(mapping_record(row))
                # Short checkpoint-ending batches produce exactly 1000-row new chunks.
                if len(tensors) >= gpu_batch_size or len(store.pending_vectors) + len(tensors) >= store.checkpoint_size:
                    encode_pending()

        try:
            # Retry only durable failures behind the saved boundary. Later records replay in the stream.
            for record in list(cursor.failures.values()):
                if record["image_path"] in store.completed_paths or record["data_offset"] + record["size"] > base:
                    continue
                row = requested[record["image_path"].removeprefix("visual_news/")]
                payload = source.read_image_bytes(record["data_offset"], record["size"], identity) if record["regular"] and 0 < record["size"] <= 32 * 1024 * 1024 else b""
                decode(payload, row, record["member_offset"], record["data_offset"], record["size"], record["regular"])
                if not tensors and not store.pending_vectors:
                    commit(eof)
                if limited():
                    flush_vectors()
                    commit(eof)
                    report("stopped_intentionally", True)
                    return result
            encode_pending()
            flush_vectors()
            commit(eof)  # Retry vectors never move the forward boundary.
            if limited() and len(store.completed_ids) != len(store.targets):
                report("stopped_intentionally", True)
                return result
            if len(store.completed_ids) != len(store.targets) and not eof:
                recognized = base > 0
                store.state["archive_passes_opened"] += 1
                result["archive_passes_this_run"] += 1
                with source.open_tar(base, pax, identity) as archive:
                    for member in archive:
                        archive.members.clear()
                        boundary, pax = member_boundary(archive, member, base, identity["size"])
                        result["scanned_members_this_run"] += 1
                        store.state["scanned_archive_member_count"] += 1
                        try:
                            name = canonical_member_name(member.name)
                        except ValueError:
                            name = None
                            result["unsafe_members_skipped"] += 1
                        if name is not None and member.isfile():
                            try:
                                validate_image_path("visual_news/" + name)
                                recognized = True
                            except ValueError:
                                pass
                        if not recognized and result["scanned_members_this_run"] >= no_match_member_limit:
                            raise ArchiveResumeError("STOP: archive path layout unrecognized")
                        row = requested.get(name)
                        if row is not None:
                            if name in matched:
                                raise ArchiveResumeError("STOP: duplicate target member in archive")
                            matched.add(name)
                            if row["image_path"] not in store.completed_paths:
                                payload = b""
                                if member.isfile() and 0 < member.size <= 32 * 1024 * 1024:
                                    with archive.extractfile(member) as handle:
                                        payload = handle.read(member.size)
                                    if len(payload) != member.size:
                                        raise ArchiveResumeError("STOP: truncated image transport payload")
                                decode(payload, row, base + member.offset, base + member.offset_data, member.size, member.isfile())
                        if not tensors and not store.pending_vectors:
                            newly_durable = (len(store.chunks) > len(cursor.state["checkpoints"]) or
                                             cursor.journal_size > cursor.state["failure_journal_bytes"])
                            if newly_durable or result["scanned_members_this_run"] % progress_every == 0:
                                commit()
                        if limited():
                            flush_vectors()
                            commit()
                            report("stopped_intentionally", True)
                            return result
                        if result["scanned_members_this_run"] % progress_every == 0:
                            report("streaming", True)
                        if len(store.completed_ids) == len(store.targets):
                            break
                    else:
                        eof = True  # Strict parser verified two complete zero blocks.
                encode_pending()
                flush_vectors()
                commit(eof)
            complete = len(store.completed_ids) == len(store.targets)
            unknown_missing = set(store.targets.image_path) - store.completed_paths - set(cursor.failures)
            report("complete" if complete else "incomplete", True)
            if eof and unknown_missing:
                raise ArchiveResumeError(f"STOP: {len(unknown_missing)} targets absent at verified archive EOF; preserve state and check manifest/source; no reduced-corpus merge")
            return result
        except BaseException:
            # Do not publish a candidate cursor that includes a still-pending GPU batch.
            # Already validated chunks survive; at most uncommitted work is replayed.
            if store._expected_state_hash == (file_sha256(store.state_path) if store.state_path.exists() else None):
                report("interrupted")
            raise
        finally:
            tensors.clear()
            rows.clear()
