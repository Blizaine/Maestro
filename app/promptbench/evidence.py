"""Fingerprints and artifacts for the code actually used in an experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_files(app_root):
    root = Path(app_root)
    files = {root / "launch.py", root / "wgp.py"}
    for directory, pattern in (("services", "*.py"), ("services/llm_guides/enhance", "*.md"),
                               ("promptbench", "*.py"), ("promptbench", "*.md"),
                               ("models/minimax_h3", "*.py"),
                               ("defaults", "*minimax_h3*.json"), ("model_definitions", "*minimax_h3*.json")):
        files.update((root / directory).rglob(pattern))
    return sorted(p for p in files if p.is_file())


def source_fingerprint(app_root):
    root = Path(app_root)
    return {p.relative_to(root).as_posix(): file_digest(p) for p in source_files(root)}


def snapshot_source(app_root, output):
    root, output = Path(app_root).resolve(), Path(output)
    hashes = source_fingerprint(root)
    for relative in hashes:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    repo = root.parent
    command = ["git", "-c", f"safe.directory={repo.as_posix()}"]
    def git(*arguments):
        try:
            result = subprocess.run(command + list(arguments), cwd=repo, capture_output=True, timeout=30)
            return result.stdout.decode("utf-8", "replace") if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    head = git("rev-parse", "HEAD")
    # Exact copied sources include uncommitted and untracked code. Avoid archiving
    # unrelated private configuration or media from a whole-worktree diff.
    patch = git("diff", "HEAD", "--", *["app/" + p for p in hashes])
    if patch is not None:
        (output / "source.patch").write_text(patch, encoding="utf-8")
    return {"git_head": head.strip() if head else None, "source_sha256": hashes,
            "source_digest": digest(hashes), "scope": "enhancement source snapshot including untracked files"}
