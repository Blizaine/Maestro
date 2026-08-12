#!/usr/bin/env python3
"""Compare Maestro's observed H3 Turbo revision with Hugging Face main."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "models"
    / "minimax_h3"
    / "turbo_presets.json"
)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("repo_id"):
        raise ValueError(f"Invalid H3 Turbo manifest: {path}")
    return manifest


def fetch_upstream_state(repo_id: str, *, opener=urlopen) -> dict:
    encoded_repo = quote(repo_id, safe="/")
    model_url = (
        f"https://huggingface.co/api/models/{encoded_repo}"
        "?expand[]=sha&expand[]=lastModified&expand[]=siblings"
    )
    request = Request(
        model_url,
        headers={"User-Agent": "Maestro-H3-Turbo-Monitor/1.0"},
    )
    with opener(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not payload.get("sha"):
        raise RuntimeError("Hugging Face returned no repository revision")

    latest_revision = str(payload["sha"])
    tree_url = (
        f"https://huggingface.co/api/models/{encoded_repo}/tree/"
        f"{quote(latest_revision, safe='')}?recursive=true&expand=true"
    )
    request = Request(
        tree_url,
        headers={"User-Agent": "Maestro-H3-Turbo-Monitor/1.0"},
    )
    with opener(request, timeout=30) as response:
        tree = json.load(response)
    if not isinstance(tree, list):
        raise RuntimeError("Hugging Face returned no repository file tree")

    safetensor_metadata = []
    for item in tree:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("path") or "")
        if not filename.endswith(".safetensors"):
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        safetensor_metadata.append(
            {
                "filename": filename,
                "size": int(item["size"]) if item.get("size") is not None else None,
                # This is the downloaded file's content SHA-256. xetHash and
                # the response ETag identify storage objects and are not
                # interchangeable with this value.
                "sha256": str(lfs.get("oid") or ""),
                "xet_hash": str(item.get("xetHash") or ""),
            }
        )
    safetensor_metadata.sort(key=lambda item: item["filename"])
    return {
        "repo_id": repo_id,
        "latest_revision": latest_revision,
        "last_modified": str(payload.get("lastModified") or ""),
        "safetensors": [item["filename"] for item in safetensor_metadata],
        "safetensor_metadata": safetensor_metadata,
    }


def compare_manifest_to_upstream(manifest: dict, upstream: dict) -> dict:
    watch = manifest.get("upstream_watch") or {}
    observed = str(watch.get("observed_main_revision") or "")
    latest = str(upstream.get("latest_revision") or "")
    return {
        **upstream,
        "observed_revision": observed,
        "changed": bool(latest and latest != observed),
        "model_card_url": str(
            watch.get("model_card_url")
            or f"https://huggingface.co/{manifest['repo_id']}"
        ),
    }


def write_github_outputs(path: Path, result: dict) -> None:
    values = {
        "changed": str(bool(result["changed"])).lower(),
        "repo_id": result["repo_id"],
        "latest_revision": result["latest_revision"],
        "observed_revision": result["observed_revision"],
        "last_modified": result["last_modified"],
        "model_card_url": result["model_card_url"],
        "safetensors_json": json.dumps(result["safetensors"], separators=(",", ":")),
        "safetensor_metadata_json": json.dumps(
            result.get("safetensor_metadata", []),
            separators=(",", ":"),
        ),
    }
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            clean_value = str(value).replace("\r", "").replace("\n", " ")
            handle.write(f"{key}={clean_value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    upstream = fetch_upstream_state(str(manifest["repo_id"]))
    result = compare_manifest_to_upstream(manifest, upstream)
    print(json.dumps(result, indent=2, sort_keys=True))

    output_path = args.github_output
    if output_path is None and os.environ.get("GITHUB_OUTPUT"):
        output_path = Path(os.environ["GITHUB_OUTPUT"])
    if output_path is not None:
        write_github_outputs(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
