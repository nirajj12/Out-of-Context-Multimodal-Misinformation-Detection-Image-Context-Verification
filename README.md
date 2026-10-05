# Out-of-Context Multimodal Misinformation Detection / Image-Context Verification

## Problem

The input is an image and a caption/claim. The goal is to determine whether the
image belongs to the claimed real-world context, or whether an authentic image
has been reused with misleading event, place, person, or time information.

## Current Stage

**Dataset Preparation and Verification.** No model training has started yet.
This repository currently contains only the development setup and placeholders;
no dataset download, processing, retrieval, or verification logic is implemented.

## Dataset Plan

- **NewsCLIPpings:** controlled OOC image-caption benchmark/query dataset.
- **VisualNews-derived metadata / later VisualNews subset:** fixed evidence/reference corpus.
- **Manual annotation overlay later:** `SAME_EVENT`, `RELATED_DIFFERENT_EVENT`,
  `IRRELEVANT`, and `AMBIGUOUS_CANNOT_DETERMINE`.
- **VERITE:** locked external evaluation later.

## Immediate Objectives

1. Download NewsCLIPpings metadata.
2. Inspect train/val/test splits.
3. Inspect VisualNews-derived metadata.
4. Verify `id` and `image_id` mapping.
5. Manually inspect 20–50 mappings.
6. Freeze a small pilot evidence manifest.
7. Stage selected images.
8. Only then move to CLIP and FAISS.

## Project Structure

| Directory | Responsibility |
| --- | --- |
| `configs/` | Paths and phase-specific configuration placeholders. |
| `data/` | Raw metadata, processed records, manifests, reports, pilot assets, and annotations. |
| `notebooks/` | Metadata inspection, mapping validation, pilot analysis, and later retrieval inspection. |
| `scripts/` | Standalone placeholders for future dataset preparation tasks. |
| `src/` | Reusable placeholders for data, preprocessing, retrieval, reranking, context, verification, and utilities. |
| `embeddings/` | Future frozen embeddings; generated contents are ignored. |
| `indexes/` | Future FAISS indexes; generated contents are ignored. |
| `models/` | Future trained models/checkpoints; generated contents are ignored. |
| `outputs/` | Figures, tables, and ignored experiment logs. |
| `tests/` | Directory existence check for the initial project structure. |
| `colab/` | Reserved notebooks for later GPU encoding and optional reranker experiments. |

## Environment Setup

Run these commands manually from the project root:

```bash
conda env create -f environment.yml
conda activate major-project

python -m ipykernel install --user \
    --name major-project \
    --display-name "Python (major-project)"
```

The environment uses Python 3.11 and core development dependencies only.
`requirements.txt` provides lightweight pip compatibility. Heavy model,
retrieval, NLP, and application dependencies will be added in their relevant phases.

After setup, check the directory structure with:

```bash
pytest tests/test_project_structure.py
```

## VS Code

Select the `major-project` Conda interpreter and the **Python (major-project)**
notebook kernel inside VS Code.

## Git

Starter commands to run manually when ready (`git init` can be skipped if already initialized):

```bash
git init
git status
git add .
git commit -m "Initialize project structure"
```

Review staged files before committing. No remote is configured automatically.

## GitHub

Create a new empty GitHub repository manually. Do not initialize it with a README,
`.gitignore`, or license if these already exist locally. After your initial local
commit, connect and push manually:

```bash
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Data Safety

- Do not commit raw datasets or VisualNews images.
- Do not commit embeddings, indexes, or models.
- Keep raw downloaded data unchanged.
- Put transformed data under `data/processed/`.
- Put reproducible corpus definitions under `data/manifests/`.
- Put dataset audit results under `data/reports/`.

Empty storage folders contain `.gitkeep` files so their structure remains trackable.
Small manifests, reports, annotations, configs, code, and notebooks remain trackable.

## Next Steps

1. NewsCLIPpings metadata download
2. NewsCLIPpings inspection/audit
3. VisualNews-derived metadata download
4. metadata inspection
5. ID mapping
6. manual mapping validation
7. fixed pilot manifest
8. selected asset staging
9. frozen CLIP embeddings
10. FAISS
11. retrieval inspection
12. annotation pilot
