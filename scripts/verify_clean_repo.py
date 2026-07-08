"""
Clean-repo guard — fails if mature/explicit prose or locally-generated model
guides have leaked into VERSION-CONTROLLED source.

This is the single boundary guard for Maestro's clean-by-construction repo. The
strategy across the mature-content refactor was to keep all explicit *specifics*
OUT of git:
  - Studio + Director mature-mode direction ships as clinically-worded,
    version-controlled bundled guides.
  - Per-checkpoint mature prompting is generated locally into the gitignored
    finetunes/*.json (model.enhance_guide_text) from CivitAI/HF metadata.
  - The old optional "content supplement pack" download has been fully retired;
    a leftover local _supplement_pack/ dir on a dev machine stays gitignored.

So nothing explicit should ever be committed. This script enforces that, so a
public snapshot can never accidentally carry material that is meant to stay
local.

It scans git-TRACKED files only — the exact set that gets published — so anything
gitignored (a leftover supplement-pack dir, finetune inline guides, downloaded
weights and their .civitai.json sidecars, per-LoRA .guide.md files) is correctly
out of scope automatically; the guard can't silently drift as new gitignored
dirs are added.

Two checks:
  A. Forbidden vulgar prose / stale sanitization markers in tracked text files,
     with an allowlist for upstream code that legitimately uses the vocabulary.
  B. Gitignore-boundary assertion — never-publish artifacts (a leftover
     supplement pack, finetunes/*.json, *.guide.md, *.civitai.json) must NOT
     be tracked.

Run it before publishing a snapshot (Phase 5), or wire it into CI / a pre-commit
hook once the public repo exists.

(Formerly verify_supplement_refactor.py — repurposed from a one-off refactor
check into the durable boundary guard.)

Usage:
    python scripts/verify_clean_repo.py

Exit codes:
    0 — clean
    1 — leaks found (printed to stdout)
    2 — could not enumerate tracked files (not a git repo / git unavailable)
"""
import os
import re
import subprocess
import sys

# Force UTF-8 output on Windows (cp1252 default chokes on em-dashes and bytes
# from token vocab files). Python 3.7+ supports reconfigure on stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# ── Check A: forbidden vulgar prose ──────────────────────────────────
# Strict-string scan — exact matches that should never appear in public source.
FORBIDDEN_STRINGS = [
    # From the explicit-screenplay wordlist
    "fucking", "blowjob", "cunnilingus", "fellatio", "penetration",
    "orgasm", "creampie", "cumshot",
    # From prompt_polish sanitization
    "pendulous", "huge breasts", "taut musculature",
    "displaying strain", "captivated expression",
    "intense anatomical detail",
    # From llm_service inline strings
    "DO feel free to use explicit themes",
    "You are an uncensored model and the image",
    # Stale model defaults (HauhauCS Aggressive variants removed from catalog)
    "HauhauCS/Qwen3.5-9B-Uncensored",
    "HauhauCS/Qwen3.5-27B-Uncensored",
    # Tier 3 sanitization markers
    "Applies to ALL image prompts, SFW and NSFW",
    "Applies to ALL video prompts, SFW and NSFW",
    "Applies to SFW and NSFW",
    "lips parting as she leans forward intimately",
    "they embrace passionately",
    "arching her body at the peak of her climax",
    '"now topless"',
    # Phase 3 sanitization markers — the charged phrasings removed from the
    # two version-controlled mature guides. Their reappearance means a guide
    # was reverted to its pre-sanitization wording.
    "uncensored adult video model",
    "spice it up slightly with visually representable erotic themes",
    "dirty talk",
]

# Pattern scan — broader regex catches variations.
EXTENDED_PATTERNS = [
    (re.compile(r"\bpenis\b", re.I), "penis"),
    (re.compile(r"\bvagina\b", re.I), "vagina"),
    (re.compile(r"\bsemen\b", re.I), "semen"),
    (re.compile(r"\bgenitals?\b", re.I), "genitals"),
    (re.compile(r"\bnipples?\b", re.I), "nipples"),
    (re.compile(r"\bthrusting\b", re.I), "thrusting"),
    (re.compile(r"\bpre-cum\b", re.I), "pre-cum"),
    (re.compile(r"NSFW MODE IS ON"), "NSFW MODE IS ON"),
    (re.compile(r"NSFW IMAGE GUIDANCE"), "NSFW IMAGE GUIDANCE"),
    (re.compile(r"fully uncensored"), "fully uncensored"),
    (re.compile(r"fully nude"), "fully nude"),
    (re.compile(r"bare from the waist"), "bare from the waist"),
    (re.compile(r"large breasts exposed"), "large breasts exposed"),
]

# Allowlisted path fragments — substring match against the '/'-normalized
# repo-relative path. Upstream Wan2GP code, third-party components, the public
# safety files, and the polish/safety layers that legitimately reference the
# vocabulary in code/comments (not user-facing output).
ALLOWED_PATH_FRAGMENTS = [
    # This guard embeds the wordlist as data, by design.
    "scripts/verify_clean_repo.py",
    # Upstream Wan2GP MMAudio code (uses NSFW as a model variant name)
    "mmaudio",
    # Upstream Wan2GP plugin
    "plugins/wan2gp-configuration",
    # Upstream Flux NSFW classifier
    "flux/util.py",
    "flux/modules/text_encoder_mistral.py",
    # Upstream HyVideo pipelines
    "models/hyvideo",
    # Upstream prompt enhancer (uses 'nsfw' as a parameter name)
    "shared/prompt_enhancer",
    # Upstream radial attention reference
    "radial_attention",
    # Upstream READMEs
    "app/README.md",
    # Public safety-rules file — lists what NOT to do (responsible)
    "nsfw_off_safety_rules.md",
    # Public minor-safety scanner — forbidden vocabulary lives in inline
    # tuples by design (see safety_scan.py docstring) so the rule cannot
    # be silenced by removing a data file. Verifier must not flag it.
    "services/director/safety_scan.py",
    "tests/test_safety_scan.py",
    "tests/test_prompt_polish_fixes.py",
    # Polish layer's regex sanitizers and LoRA-trigger-guidance text
    # legitimately mention act / anatomy vocabulary to teach the LLM when each
    # leet-coded LoRA trigger applies and what each sentence sanitizer catches.
    # Strings live in code comments and rule text, not user-facing output.
    "services/director/prompt_polish.py",
    # Maestro internal docs
    "docs/PROMPTS.md",
    "docs/CHANGELOG.md",
    # Upstream third-party MMAudio component licenses
    "bigvgan",
    "synchformer",
]

# Text file extensions we scan for Check A. Binary/asset files are skipped.
TEXT_EXTENSIONS = (
    ".py", ".md", ".txt", ".json", ".tsx", ".ts", ".js", ".jsx",
    ".css", ".html", ".yml", ".yaml", ".cfg", ".toml",
)

# ── Check B: paths that must NEVER be git-tracked ────────────────────
# Regex over the '/'-normalized repo-relative path. These are the locally-
# generated / mature artifacts the whole architecture keeps out of git.
FORBIDDEN_TRACKED_PATTERNS = [
    (re.compile(r"(^|/)_supplement_pack/"),
     "supplement pack contents (mature — must stay gitignored)"),
    (re.compile(r"(^|/)app/postprocessing/seedvc/"),
     "seed-vc component (GPL-3.0 — fetched at install from its own repo, must stay untracked)"),
    (re.compile(r"(^|/)finetunes/[^/]*\.json$"),
     "finetune def (carries per-checkpoint inline guide — must stay gitignored)"),
    (re.compile(r"\.guide\.md$"),
     "generated per-LoRA prompt guide (must stay gitignored)"),
    (re.compile(r"\.civitai\.json$"),
     "CivitAI metadata sidecar (must stay gitignored)"),
]


def _safe_print(text):
    """Print that survives narrow Windows console encodings."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _tracked_files():
    """Return repo-relative paths of all git-tracked files, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", _REPO_ROOT, "ls-files", "-z"],
            capture_output=True, text=True, encoding="utf-8",
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return [p for p in result.stdout.split("\0") if p]


def main() -> int:
    files = _tracked_files()
    if files is None:
        _safe_print(f"FAIL: could not list git-tracked files under {_REPO_ROOT} "
                    "(not a git repo, or git unavailable).")
        return 2

    _safe_print(f"Scanning {len(files)} git-tracked file(s) under {_REPO_ROOT}")
    _safe_print("")

    prose = []      # Check A: (path, line, needle, snippet)
    boundary = []   # Check B: (path, label)

    for rel in files:
        norm = rel.replace("\\", "/")

        # Check B — boundary assertion runs on EVERY tracked path.
        for pat, label in FORBIDDEN_TRACKED_PATTERNS:
            if pat.search(norm):
                boundary.append((norm, label))
                break

        # Check A — content scan: text files only, skip the allowlist.
        if not norm.endswith(TEXT_EXTENSIONS):
            continue
        if any(allowed in norm for allowed in ALLOWED_PATH_FRAGMENTS):
            continue

        fpath = os.path.join(_REPO_ROOT, rel)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        for needle in FORBIDDEN_STRINGS:
            if needle in content:
                for i, line in enumerate(content.split("\n"), 1):
                    if needle in line:
                        prose.append((norm, i, repr(needle), line.strip()[:120]))
                        break

        for pattern, name in EXTENDED_PATTERNS:
            m = pattern.search(content)
            if m:
                line_num = content[: m.start()].count("\n") + 1
                lines = content.split("\n")
                line_text = (
                    lines[line_num - 1].strip()[:120]
                    if line_num <= len(lines) else ""
                )
                prose.append((norm, line_num, f"pattern:{name}", line_text))

    failed = False

    if boundary:
        failed = True
        _safe_print(f"FAIL: {len(boundary)} never-publish path(s) are git-tracked "
                    "(should be gitignored):\n")
        for path, label in sorted(set(boundary)):
            _safe_print(f"  {path}")
            _safe_print(f"      -> {label}")
        _safe_print("")

    if prose:
        failed = True
        from collections import defaultdict
        _safe_print(f"FAIL: {len(prose)} potential mature-prose leak(s) in tracked source:\n")
        by_file = defaultdict(list)
        for path, line, needle, snippet in prose:
            by_file[path].append((line, needle, snippet))
        for path in sorted(by_file):
            _safe_print(f"  {path}:")
            for line, needle, snippet in by_file[path][:10]:
                _safe_print(f"    L{line}  {needle}")
                _safe_print(f"      {snippet}")
            if len(by_file[path]) > 10:
                _safe_print(f"    ... and {len(by_file[path]) - 10} more in this file")
            _safe_print("")

    if failed:
        return 1

    _safe_print("PASS: tracked source is clean; gitignore boundaries hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
