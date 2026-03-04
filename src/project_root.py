from pathlib import Path

def find_project_root(current_path: Path, markers=("pyproject.toml", ".git", "requirements.txt", "README.md")):
    for parent in current_path.resolve().parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found")

# Usage
PROJECT_ROOT = find_project_root(Path(__file__))