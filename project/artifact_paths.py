"""Portable, non-personal path labels for saved research metadata."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def relative_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return "external_input/" + path.name
