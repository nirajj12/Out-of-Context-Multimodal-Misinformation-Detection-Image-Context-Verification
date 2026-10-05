"""Verify that the initial project directories exist."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = (
    "configs",
    "data",
    "data/raw",
    "data/raw/newsclippings",
    "data/raw/visualnews_metadata",
    "data/processed",
    "data/manifests",
    "data/reports",
    "data/pilot",
    "data/pilot/images",
    "data/pilot/captions",
    "data/pilot/articles",
    "data/annotations",
    "notebooks",
    "scripts",
    "src",
    "src/data",
    "src/preprocessing",
    "src/retrieval",
    "src/reranking",
    "src/context",
    "src/verification",
    "src/utils",
    "embeddings",
    "indexes",
    "models",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/logs",
    "tests",
    "colab",
)


def test_project_structure():
    missing = [path for path in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / path).is_dir()]
    assert not missing, f"Missing project directories: {missing}"
