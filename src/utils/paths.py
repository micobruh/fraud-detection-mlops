from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path
