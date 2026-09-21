"""Versioned token dialects and adapter semantics, independent of GPU imports."""
import hashlib
import re

TOKENIZER_REVISION = "f2278a2e005dc4ecc421c53a0929f62b3aeb2280"
V9_REVISION = "430084f7c8eeeb5fc6947ea31b3f2f25f7602def"
PAIRS = {
    "v4": {"revision": TOKENIZER_REVISION, "head": "tokenizer", "nar": "nar"},
    "v9": {"revision": V9_REVISION, "head": "tokenizer_v9", "nar": "nar_v9"},
}


def tokenizer_pair(project=None):
    name = (project or {}).get("tokenizer_pair", "v4")
    if not isinstance(name, str) or name not in PAIRS:
        raise ValueError("Choose the v4 or v9 matched music tokenizer and decoder pair")
    result = dict(PAIRS[name])
    adapted = (project or {}).get('adapted_pair')
    if adapted:
        for key in ('head_sha256', 'nar_sha256'):
            if not isinstance(adapted, dict) or not re.fullmatch('[a-f0-9]{64}', str(adapted.get(key, ''))):
                raise ValueError('The adapted music pair needs valid head and decoder fingerprints')
        result['revision'] = 'local-' + hashlib.sha256(
            (adapted['head_sha256'] + adapted['nar_sha256']).encode()).hexdigest()
    return result


def pair_asset(project, branch, *, cancelled=lambda: False, report=lambda message: None):
    """Resolve only managed assets or an immutable project's verified local pair."""
    from models.TTS.yue2.music_assets import ensure_asset
    pair = tokenizer_pair(project)
    if not project.get('adapted_pair'):
        return ensure_asset(pair[branch], cancelled=cancelled, report=report)
    from .music_training import project_directory
    from .music_styles import file_digest
    path = project_directory(project['id']) / 'adapted_pair' / (branch + '.safetensors')
    if not path.is_file() or file_digest(path) != project['adapted_pair'][branch + '_sha256']:
        raise ValueError('This project is missing its matched adapted tokenizer/decoder; prepare a new experiment')
    return path


def adapter_contract(manifest):
    """Old bundles always use a fixed decoder companion with full I/O weights."""
    mode = manifest.get("adapter_mode", "separate")
    if not isinstance(mode, str) or mode not in {"separate", "joint"}:
        raise ValueError("Unsupported music adapter mode")
    if mode == "joint" and manifest.get("version", 1) < 2:
        raise ValueError("Joint music adapters require a version 2 bundle")
    return mode
