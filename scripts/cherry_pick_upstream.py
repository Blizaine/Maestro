#!/usr/bin/env python3
"""
Cherry-pick upstream Wan2GP commits with path rewriting.

Upstream (deepbeepmeep/Wan2GP) puts `wgp.py`, `models/`, `shared/`,
`preprocessing/`, `postprocessing/`, `defaults/`, `profiles/`, and
`plugins/` at the repo root. Maestro nests all of these under `app/`.

A naive `git cherry-pick <upstream-hash>` fails because the paths don't
match. This script generates a patch via `git format-patch`, rewrites
the paths in the patch metadata lines only (never touching hunk
content), and applies the rewritten patch with `git am --3way`.

Usage:
    python scripts/cherry_pick_upstream.py <hash> [<hash> ...]
    python scripts/cherry_pick_upstream.py --dry-run <hash>

Requires:
    - Remote `upstream-wgp` configured and fetched.
      git remote add upstream-wgp https://github.com/deepbeepmeep/Wan2GP.git
      git fetch upstream-wgp
    - Python 3.8+

Behavior on conflict:
    `git am` pauses. Resolve files, `git add` them, then run
    `git am --continue`. To abort, run `git am --abort`.

Adding new path mappings:
    Edit PATH_PREFIXES (directories) or FILE_PATHS (single files) below.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Path-rewrite configuration

# Upstream top-level directories that map to app/<same>.
PATH_PREFIXES = (
    "models/",
    "shared/",
    "preprocessing/",
    "postprocessing/",
    "defaults/",
    "profiles/",
    "plugins/",
)

# Upstream root-level files that map to app/<same>.
FILE_PATHS = (
    "wgp.py",
)

REMOTE = "upstream-wgp"


def rewrite_path(p: str) -> str:
    """Return the Maestro-side path for an upstream path.

    Leaves untouched any path that doesn't match a known prefix/file —
    e.g. README.md, .gitignore, scripts/* stay at the repo root.
    """
    for prefix in PATH_PREFIXES:
        if p.startswith(prefix):
            return "app/" + p
    if p in FILE_PATHS:
        return "app/" + p
    return p


# ---------------------------------------------------------------------------
# Patch rewriting — only touches metadata lines, never hunk content.
#
# Diff metadata line formats handled:
#   diff --git a/PATH b/PATH
#   --- a/PATH
#   +++ b/PATH
#   rename from PATH
#   rename to PATH
#   copy from PATH
#   copy to PATH
#   Binary files a/PATH and b/PATH differ
#
# Special cases that stay untouched:
#   --- /dev/null            (new files)
#   +++ /dev/null            (deleted files)
#   Lines inside hunks (+, -, context, @@)
#   Commit message lines

_DIFF_LINE = re.compile(r"^(diff --git )a/(\S+) b/(\S+)$")
_TRIPLE_MINUS = re.compile(r"^(--- a/)(\S+)$")
_TRIPLE_PLUS = re.compile(r"^(\+\+\+ b/)(\S+)$")
_RENAME = re.compile(r"^(rename (?:from|to) )(\S+)$")
_COPY = re.compile(r"^(copy (?:from|to) )(\S+)$")
_BINARY = re.compile(r"^(Binary files )a/(\S+) and b/(\S+)( differ)$")


def rewrite_patch(patch_text: str) -> str:
    """Rewrite path-bearing patch metadata lines. Leaves hunks untouched."""
    out_lines = []
    for line in patch_text.split("\n"):
        if m := _DIFF_LINE.match(line):
            prefix, a, b = m.group(1), m.group(2), m.group(3)
            out_lines.append(f"{prefix}a/{rewrite_path(a)} b/{rewrite_path(b)}")
        elif m := _TRIPLE_MINUS.match(line):
            out_lines.append(m.group(1) + rewrite_path(m.group(2)))
        elif m := _TRIPLE_PLUS.match(line):
            out_lines.append(m.group(1) + rewrite_path(m.group(2)))
        elif m := _RENAME.match(line):
            out_lines.append(m.group(1) + rewrite_path(m.group(2)))
        elif m := _COPY.match(line):
            out_lines.append(m.group(1) + rewrite_path(m.group(2)))
        elif m := _BINARY.match(line):
            prefix, a, b, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
            out_lines.append(
                f"{prefix}a/{rewrite_path(a)} and b/{rewrite_path(b)}{suffix}"
            )
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Git operations

def get_patch(commit: str) -> str:
    """Return git format-patch output for a single commit."""
    result = subprocess.run(
        ["git", "format-patch", "-1", "--stdout", commit],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def verify_remote() -> None:
    """Exit if upstream-wgp remote isn't configured."""
    remotes = subprocess.run(
        ["git", "remote"], check=True, capture_output=True, text=True
    ).stdout.split()
    if REMOTE not in remotes:
        print(f"!! Remote `{REMOTE}` not configured.", file=sys.stderr)
        print(
            f"   Add it with:\n"
            f"       git remote add {REMOTE} "
            f"https://github.com/deepbeepmeep/Wan2GP.git\n"
            f"       git fetch {REMOTE}",
            file=sys.stderr,
        )
        sys.exit(1)


def apply_patch(patch_text: str) -> None:
    """Apply a rewritten patch via git am.

    On conflict, git am pauses and the script exits with the failing
    return code. The user is expected to resolve, `git add`, and run
    `git am --continue` (or `git am --abort`).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as f:
        f.write(patch_text)
        patch_path = f.name

    try:
        result = subprocess.run(
            ["git", "am", "--3way", patch_path],
            check=False,
        )
        if result.returncode != 0:
            print("", file=sys.stderr)
            print("!! git am failed — likely a merge conflict.", file=sys.stderr)
            print("   Resolve the conflicting files, `git add` them,", file=sys.stderr)
            print("   then run:", file=sys.stderr)
            print("       git am --continue", file=sys.stderr)
            print("   Or abort with:", file=sys.stderr)
            print("       git am --abort", file=sys.stderr)
            sys.exit(result.returncode)
    finally:
        Path(patch_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "commits",
        nargs="+",
        help="Upstream commit hash(es) to cherry-pick, in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rewritten patch to stdout instead of applying.",
    )
    args = parser.parse_args()

    verify_remote()

    for commit in args.commits:
        print(f"==> Cherry-picking {commit}", file=sys.stderr)
        try:
            patch = get_patch(commit)
        except subprocess.CalledProcessError:
            print(
                f"!! Commit {commit} not found in {REMOTE}. "
                f"Did you `git fetch {REMOTE}`?",
                file=sys.stderr,
            )
            sys.exit(1)

        rewritten = rewrite_patch(patch)

        if args.dry_run:
            sys.stdout.write(rewritten)
            sys.stdout.write("\n")
            continue

        apply_patch(rewritten)
        print(f"    Applied {commit}", file=sys.stderr)

    if not args.dry_run:
        n = len(args.commits)
        print(f"\nApplied {n} commit(s).", file=sys.stderr)
        print(f"Review with: git log -n {n} --oneline", file=sys.stderr)


if __name__ == "__main__":
    main()
