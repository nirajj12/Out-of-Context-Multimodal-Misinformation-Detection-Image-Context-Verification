# VisualNews archive and embedding recovery

Notebook `07_visualnews_large_evidence_colab.ipynb` generates frozen embeddings for the existing 385,003-row TRAIN manifest. It does not train a model. The model, QuickGELU, fp32 inference/output, preprocessing, tokenizer, L2 normalization and recorded OpenCLIP version remain those of the pilot. No pilot artifacts are modified.

## Two recovery records

Validated `.npy` arrays, aligned Parquet maps and checksum `.complete.json` markers establish completed IDs. The new `image_archive_cursor.json` separately records an **absolute logical tar boundary**, source identity, global PAX state, checkpoint references and a checksum of the durable failure-journal prefix. Cursor contents and individual journal records also carry integrity checksums. A progress JSON never proves completion.

The boundary is the next member position calculated from the parsed member's data offset, declared payload size and 512-byte padding, plus the resumed stream's base offset. GNU long-name/link and local PAX headers are consumed with their following member. Global PAX state is saved for subsequent members. The transport's physical position/read-ahead is never saved as the cursor.

The cursor advances only after all earlier selected members have either validated committed embeddings or durable failure records, with an empty GPU batch and empty vector buffer. Publication order is array/map, validation, marker, then cursor. A checkpoint published before a cursor interruption is reused during harmless replay. Completed files are never overwritten. Existing 5,000-row chunks remain valid; new chunks target 1,000 rows, with short final/session/retry chunks accepted. GPU batches have a maximum of 64; a 40-image batch closes a 1,000-row checkpoint after fifteen 64-image batches.

Each session revalidates a 512-byte probe and source identity. The forward request starts at the saved offset with `Range` and conditional validators. It requires HTTP 206, matching start/end/total, consistent Content-Length when supplied, and an unencoded body. Strong ETags use If-Range and If-Match; a sufficiently old Last-Modified/Date combination is the alternative. A changed URL, resolved URL, size, available ETag/Last-Modified, HTTP 200 fallback or malformed response stops before accepting its body. Truncated bodies and malformed tar are systemic errors, never image failures.

Isolated decode failures preserve the original `image_embedding_failures.csv` schema (`id,image_path,error`) and also append `image_archive_failures.jsonl`, including source identity, scope, original ID/path, absolute member/data offsets and size. Failures behind the cursor are retried with bounded conditional payload requests. Failed IDs remain unfinished; past failed attempts stay in the audit logs even after eventual success. Five consecutive failures or more than 10% failures after at least 20 attempts stop the stage. GPU/model/storage errors propagate without becoming decode failures.

## Colab setup and execution

Before Colab reloads code, these runtime files must reach GitHub main through your normal review/push process:

- `notebooks/07_visualnews_large_evidence_colab.ipynb`
- `scripts/visualnews_stream_embeddings.py`
- `scripts/visualnews_archive_resume.py` (new, required import)

Include these validation/documentation files in the same reviewed change:

- `tests/test_visualnews_stream_embeddings.py`
- `tests/test_visualnews_archive_resume.py` (new)
- `docs/visualnews_archive_resume.md` (this guide)

The existing manifest and `scripts/extract_selected_images.py` / `scripts/check_missing_assets.py` must remain in the checkout. There is no new manifest, runtime ZIP, FAISS build or image extraction tree. Notebook repository setup accepts a clean matching `main` checkout and fast-forwards; it stops on local changes, another branch/origin or divergence without resetting them. All five saved production flags are **False**.

Cell numbers below are **one-based notebook cell positions**, including Markdown; use the section names if Colab displays execution counts instead.

1. Run GPU setup and Drive mount, then **Repository**, **Paths**, **Packages**, **Helpers**, **Manifest**, **Model** (cells 2–26). Keep the repository under `/content`, production chunks in their existing Drive directories and `smoke_test/` separate. The manifest stays exactly 385,003 rows. Helper imports are refreshed after the repository pull.
2. The settings cell is **cell 15**, in Paths. Keep `GPU_BATCH_SIZE = 64` and `CHECKPOINT_SIZE = 1000`. Set only `RUN_IMAGE_EMBEDDING = True` for the image stage. For an inexpensive manual recovery validation, set `MAX_NEW_IMAGES_PER_SESSION = 1000` (the limit counts **newly encoded images**, excluding previously completed IDs).
3. Run **Image Progress** (cells 28–29) and the code cells under **Stream Images** (cells 32 and 36). Cell 32 defines the unchanged frozen GPU encoder; cell 36 calls the range-resume helper. The helper validates existing chunks, opens the source only when unfinished targets remain, saves a safe cursor and reports remaining targets, new encodings, failed attempts/unresolved failures, chunks, starting/committed offsets, source bytes, elapsed and measured encoder/checkpoint time. Archive scan percentage and target completion percentage are separate; no speculative ETA is printed.
4. A planned limit ends successfully as `stopped_intentionally`. It finishes the current batch (possible overshoot at most 63 new images), publishes a short checkpoint if necessary and commits the cursor. Real archive EOF with unresolved decode failures reports incomplete and the notebook stops before merging. Missing targets at verified EOF produce an actionable error.

## Manual interruption and restart validation

This procedure remains **manual**. Local synthetic tests do not establish a real Colab/Drive restart result. The previous real 100-image/100-caption smoke test does not need to be repeated.

1. Run the bounded 1,000-new-image session above. Save its `resume_start_offset`, `committed_archive_offset`, completed count and checkpoint filenames/checksums. Confirm the cursor offset is nonzero and the status is resumable.
2. Restart/disconnect the Colab runtime. Mount the same Drive and rerun setup, Image Progress and Stream Images with the same manifest/model and limit. Verify `resume_start_offset` equals the previous committed offset, completed IDs are reused, and the old checkpoint checksums do not change. A small byte-zero **512-byte probe** is expected; the forward archive request must begin at the saved nonzero offset.
3. To exercise an unplanned interruption, start another bounded session and interrupt while a batch or checkpoint is in flight. Record the most recently committed cursor before restarting. Reload setup/progress. The cursor may trail newly published checkpoints; those checkpoint IDs must be skipped during replay. Unsaved GPU/CPU work can repeat. Do not manually advance the cursor.
4. A killed runtime can leave `image_embedding_progress.json.lock` (or text equivalent). Confirm that the old runtime/process has stopped, then remove **only that stale lock**. Do not delete progress, cursor, journals, arrays, maps or markers. A normal planned stop or Python interruption releases its lock automatically.
5. Check the session report, CSV and range journal. Unresolved failures may require manual diagnosis; retrying does not authorize dropping them or changing the frozen encoder/source. Correcting/replacing the archive is a source change that requires reconciliation, not silently accepting a new ETag.

## Transition to the full run and later stages

After manual recovery validation succeeds, keep the same production directories/checkpoints/cursor and set `MAX_NEW_IMAGES_PER_SESSION = None`. Keep only `RUN_IMAGE_EMBEDDING = True`. Run Image Progress and Stream Images again; validated work continues from the saved position. Do not delete the cursor to start the full run. A preexisting legacy run without a cursor requires an initial scan from zero to establish one, skipping completed IDs; subsequent sessions use byte-range resume.

When image completion is exactly 385,003 with zero unresolved image IDs, disable the image flag and enable only `RUN_TEXT_EMBEDDING`. Run **Text Embeddings** (cells 38–40). Text has its own lock/progress/chunks and resumes captions by original IDs, without any archive HTTP requests. Then disable text encoding and manually enable/run **Merge Image Chunks** (cells 42–45), **Merge Text Chunks** (cell 47), and **Final Validation** (cells 49–51) as needed. Merge/final validation still require exactly all original IDs, ascending metadata-ID order for both arrays/common map, unique aligned paths, 512 columns, fp32, finite unit vectors and completion checksums. No reduced corpus is permitted by these changes.

## Limits and repair behavior

- Supported source format is ordinary **uncompressed 512-byte tar**, with GNU long-name/link and bounded local/global PAX extensions. Compressed/sparse archives, unsupported member types/sizes, global PAX size overrides, excessive extension state and malformed/dangling headers stop clearly. Target images must be bounded regular payloads, at most 32 MiB. Links are never followed or extracted. The parser verifies two zero end blocks rather than treating a malformed header as EOF.
- The source must continue providing valid ranges and usable identity validators. A URL/identity change needs inspection while preserving the existing checkpoints; do not bypass it by editing the cursor. No full download verifies source identity. The small Rice server probes observed 206/correct ranges and a 97,916,641,280-byte archive; those observations are not hard-coded trust.
- Unsaved batches/vectors and archive bytes after the last committed boundary can repeat. A checkpoint-before-cursor crash can replay a larger stretch while skipping those IDs. Existing legacy chunks are reused but cannot reconstruct an old byte position by themselves.
- Checkpoint pairs missing either file stop and preserve files. A valid complete pair whose marker publication was interrupted is validated and its marker recovered on reload. Malformed cursor/journal or committed checksum mismatch stops for repair. Restore trusted artifacts or inspect an uncommitted partial journal append before editing anything; never erase a committed failure prefix to bypass validation.
- Exclusive lock-file creation, state conflict detection, atomic replace and fsync provide practical single-writer protection on local filesystems. Drive FUSE/backend does not guarantee distributed locking, immediate remote persistence or atomic multi-file transactions; fsync may be unsupported. Reload checksum validation detects missing/corrupted artifacts. Use one connected writer per stage/directory and avoid concurrent sync writers. A hard-disconnect stale lock requires the manual check above.
- At most a checkpoint of fp32 vectors, one GPU batch and one bounded image payload are held for processing, plus manifest/index metadata. No image directory or full archive is persisted. Archive `TarInfo` history is cleared as members are scanned.
- Synthetic HTTP/tar/fake-encoder tests cover parser/checkpoint recovery locally. Live Colab T4 encoding/restart, Drive remote durability and the full production run remain unverified until the manual sessions are observed.

## Local verification observed on 7 October 2026

Command: `/opt/anaconda3/envs/major-project/bin/python -m pytest -q` using Python 3.11.16: **66 passed in 13.47 seconds**. Tests use synthetic tar archives, mocked HTTP and fake unit-vector encoders; no GPU/network download is part of the automated suite. Coverage includes multiple nonzero resumes, GNU long names/links, global/local PAX and size overrides, malformed/compressed/sparse tar, bad 200/range/identity/encoding headers, truncated bodies, uncommitted batches, interrupted checkpoint pairs/markers, checkpoint-before-cursor replay, bounded retries, planned stops, an actual 5,000-row synthetic legacy chunk, 1,000-row new chunks with maximum-64 batches, fully completed zero-HTTP/GPU runs, corruption/conflicting writers, independent text and deterministic image/text/common-map merges.

Notebook format validation, Python source compilation and `git diff --check` passed. All 52 protected local manifest/pilot/annotation/retrieval/earlier-output files matched their pre-edit SHA-256 hashes. The encoder contract and existing chunk/final validation functions were unchanged by AST comparison. The existing repository setup, model, merge/final-validation code and the user's trailing notebook cells were preserved; the text execution cell only adds its writer lock.

Two small read-only Rice probes returned 206 for bytes 0–511 and 1024–1535, with the requested Content-Range and total 97,916,641,280 bytes, matching ETag `"16cc496000-5dcdf02f54740"`, matching Last-Modified `Sun, 17 Apr 2022 19:54:29 GMT` and no Content-Encoding. Only 1,024 payload bytes were read. No production embedding job, repeated live smoke test, real Colab restart, commit or push was performed.
