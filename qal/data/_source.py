"""Data provenance shared by runtime resolution and setuptools (stdlib only)."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

DEFAULT_REPOSITORY = "QUEL-Imaging/quel-qal"
DATA_DIR = Path(__file__).resolve().parent


def _git(root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def checkout_source(data_dir=None):
    """Recognize only the checkout containing this package, including worktrees."""
    data_dir = Path(data_dir or DATA_DIR).resolve()
    root = data_dir.parent.parent
    if not (root / ".git").exists():
        return None
    top = _git(root, "rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != root:
        return None
    if not _git(root, "ls-files", "--error-unmatch", "qal/data/_registry.py"):
        return None
    ref = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    ref = ref or _git(root, "rev-parse", "HEAD")
    if not ref:
        return None
    origin = _git(root, "remote", "get-url", "origin")
    match = re.fullmatch(r"(?:https?://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?", origin)
    repository = match.group(1) if match else DEFAULT_REPOSITORY
    return Source(ref, repository, data_dir)


@dataclass(frozen=True)
class Source:
    ref: str
    repository: str
    checkout: Path | None = None

    def url(self, filename):
        # media.githubusercontent serves hydrated LFS objects.
        return (f"https://media.githubusercontent.com/media/{self.repository}/"
                f"{quote(self.ref, safe='')}/qal/data/{quote(filename, safe='/')}")

    def raw_url(self, filename):
        return (f"https://raw.githubusercontent.com/{self.repository}/"
                f"{quote(self.ref, safe='')}/qal/data/{quote(filename, safe='/')}")


def recorded_source(data_dir=None):
    path = Path(data_dir or DATA_DIR) / "_data_source.json"
    if path.is_file():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return Source(metadata["ref"], metadata["repository"])
    return None


def resolve_source(data_dir=None):
    checkout = checkout_source(data_dir)
    source = checkout or recorded_source(data_dir) or Source("main", DEFAULT_REPOSITORY)
    ref = os.environ.get("QAL_DATA_REF") or source.ref
    repository = os.environ.get("QAL_DATA_REPOSITORY") or source.repository
    # An explicit different source must never use this checkout's data.
    local = source.checkout if (ref, repository) == (source.ref, source.repository) else None
    return Source(ref, repository, local)
