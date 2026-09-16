"""Optional real-audio assets, downloaded only for covers or music training."""
from pathlib import Path
import json
import shutil
import threading

from services.managed_assets import managed_asset_matches, write_managed_asset_receipt
from services.music_styles import file_digest

ROOT = Path("ckpts/yue2_music_training")
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
TOKENIZER_REVISION = "f2278a2e005dc4ecc421c53a0929f62b3aeb2280"
REGULARIZER_REVISION = "5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35"
VAE_REVISION = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"

ASSETS = {
    "lyric_aligner": {"url": "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
                      "file": "mms_fa.pt", "size": 1262047414, "sha256": "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680"},
    "vocal_separator": {"url": "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt",
                        "file": "hdemucs_high_trained.pt", "size": 334697255, "sha256": "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"},
    "nar_training": {"repo_id": "DeepBeepMeep/TTS", "revision": "864a479cbf3e810e1b2c1993b438510750e383b2",
                     "remote_path": "YuE2_Acoustic_bf16.safetensors", "file": "nar_training_bf16.safetensors",
                     "size": 2929494672, "sha256": "975253f29d0f9d3229fa96e92234ab8c82221c6e511f0c8b68ad3a3f3fee151e"},
    "vae_training": {"repo_id": "m-a-p/YuE2-Vae", "revision": VAE_REVISION,
                     "remote_path": "model.safetensors", "file": "vae_training.safetensors",
                     "size": 530512720, "sha256": "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346"},
    "text_tokenizer": {"repo_id": "DeepBeepMeep/TTS", "revision": "864a479cbf3e810e1b2c1993b438510750e383b2",
                       "remote_path": "YuE2_AR/qwen.tiktoken", "file": "qwen.tiktoken", "size": 2561218},
    "ar_training": {"repo_id": "DeepBeepMeep/TTS", "revision": "864a479cbf3e810e1b2c1993b438510750e383b2",
                    "remote_path": "YuE2_AR/YuE2_AR_bf16.safetensors", "file": "ar_training_bf16.safetensors",
                    "size": 4331951160, "sha256": "59ce30757ae3960405dc6116c3a919541d61e5e55b28c4927dbb879870209366"},
    "mert": {"repo_id": "m-a-p/MERT-v2-FullSong", "revision": MERT_REVISION,
             "remote_path": "model.safetensors", "file": "mert.safetensors",
             "size": 2529812848, "sha256": "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d"},
    "mert_config": {"repo_id": "m-a-p/MERT-v2-FullSong", "revision": MERT_REVISION,
                    "remote_path": "config.json", "file": "mert_config.json"},
    "mert_processor": {"repo_id": "m-a-p/MERT-v2-FullSong", "revision": MERT_REVISION,
                       "remote_path": "preprocessor_config.json", "file": "mert_processor.json"},
    "tokenizer": {"repo_id": "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4", "revision": TOKENIZER_REVISION,
                  "remote_path": "tokenizer_head_joint_v4.pt", "file": "tokenizer_head_joint_v4.pt",
                  "size": 171305291, "sha256": "d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e"},
    "nar": {"repo_id": "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4", "revision": TOKENIZER_REVISION,
            "remote_path": "nar_lora_joint_v4.pt", "file": "nar_lora_joint_v4.pt",
            "size": 140626123, "sha256": "df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe"},
    "regularizer": {"repo_id": "Mothersuperior/yue2-minted-corpus", "repo_type": "dataset", "revision": REGULARIZER_REVISION,
                    "remote_path": "regularizer/minted_regularizer_pack.pt", "file": "minted_regularizer_pack.pt",
                    "size": 101878861, "sha256": "bdd9b9780de46bb0752c3e3bc101869759493443c6e35b1b2d4a2c5033eadc4e"},
}

_acoustic_manifest = json.loads(Path(__file__).with_name('acoustic_regularizer.json').read_text())
for _entry in _acoustic_manifest['files']:
    ASSETS['acoustic:' + _entry['path']] = {
        'repo_id': 'Mothersuperior/yue2-minted-corpus', 'repo_type': 'dataset',
        'revision': _acoustic_manifest['revision'], 'remote_path': _entry['path'],
        'file': 'acoustic_regularizer/' + _entry['path'], 'size': _entry['size'],
        'sha256': _entry.get('sha256'),
    }
_asset_locks = {name: threading.Lock() for name in ASSETS}


def ensure_asset(name, *, cancelled=lambda: False, report=lambda message: None):
    with _asset_locks[name]:
        return _download_asset(name, cancelled=cancelled, report=report)


def _download_asset(name, *, cancelled, report):
    from huggingface_hub import hf_hub_download
    spec = ASSETS[name]
    path = ROOT / spec["file"]
    if managed_asset_matches(str(path), spec):
        return path
    if cancelled():
        raise InterruptedError("Music preparation cancelled")
    report(f"Downloading optional music asset: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if spec.get('url'):
        import requests
        with requests.get(spec['url'], stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for block in response.iter_content(1024 * 1024):
                    if cancelled():
                        raise InterruptedError('Music preparation cancelled')
                    output.write(block)
        source = temporary
    else:
        source = hf_hub_download(spec["repo_id"], spec["remote_path"], revision=spec["revision"],
                                 repo_type=spec.get("repo_type", "model"))
    if cancelled():
        raise InterruptedError("Music preparation cancelled")
    digest = file_digest(Path(source))
    if spec.get("sha256") and digest != spec["sha256"]:
        raise ValueError(f"The {name} download failed its checksum check")
    if spec.get('size') and Path(source).stat().st_size != spec['size']:
        raise ValueError(f'The {name} download has an incorrect size')
    if Path(source) != temporary:
        shutil.copyfile(source, temporary)
    temporary.replace(path)
    write_managed_asset_receipt(str(path), spec, actual_sha256=digest)
    return path
