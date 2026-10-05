"""Explicit LM Studio model management; inference/idle cleanup never calls this.

Native API contracts: https://lmstudio.ai/docs/developer/rest/list and /unload.
"""

from urllib.parse import urlsplit, urlunsplit

import requests


class ModelManagementError(RuntimeError):
    pass


class ModelSelectionError(ValueError):
    pass


class ModelBusyError(RuntimeError):
    pass


def server_url(value: str) -> str:
    """Accept server roots or inference bases, retaining reverse-proxy prefixes."""
    if not isinstance(value, str):
        raise ModelSelectionError("Configure the LM Studio server URL first.")
    try:
        parsed = urlsplit(value.strip())
        parsed.port  # Validate malformed and out-of-range ports before requests.
    except ValueError:
        raise ModelSelectionError("Configure a valid LM Studio server URL.") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ModelSelectionError("Use an HTTP(S) server URL without embedded credentials, query or fragment.")
    path = parsed.path.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _request(method: str, root: str, path: str, api_key: str, **kwargs) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.request(
            method, root + path, headers=headers, timeout=(5, 30),
            allow_redirects=False, **kwargs,
        )
    except requests.RequestException:
        raise ModelManagementError("Could not reach the configured LM Studio server; check its URL and connection.") from None
    if response.status_code in {404, 405}:
        raise ModelManagementError("This server does not expose LM Studio's native model-management API. Use LM Studio 0.4 or newer with its server enabled.")
    if response.status_code in {401, 403}:
        raise ModelManagementError("LM Studio rejected authentication. Check the remote provider's API key.")
    if not 200 <= response.status_code < 300:
        raise ModelManagementError(f"LM Studio model management failed (HTTP {response.status_code}).")
    try:
        payload = response.json()
    except ValueError:
        raise ModelManagementError("LM Studio returned invalid model-management JSON.") from None
    if not isinstance(payload, dict):
        raise ModelManagementError("LM Studio returned an invalid model-management response.")
    return payload


def loaded_models(remote_url: str, api_key: str = "") -> dict:
    root = server_url(remote_url)
    payload = _request("GET", root, "/api/v1/models", api_key)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ModelManagementError("This server did not return LM Studio's native model list.")
    instances = []
    model_keys = []
    seen = set()
    for model in models:
        if not isinstance(model, dict) or model.get("type") != "llm":
            continue
        key = model.get("key")
        loaded = model.get("loaded_instances")
        if not isinstance(key, str) or not key or not isinstance(loaded, list):
            raise ModelManagementError("LM Studio returned invalid loaded-model metadata.")
        model_keys.append(key)
        for instance in loaded:
            identifier = instance.get("id") if isinstance(instance, dict) else None
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ModelManagementError("LM Studio returned an invalid or ambiguous model instance.")
            seen.add(identifier)
            instances.append({
                "instance_id": identifier, "model_key": key,
                "display_name": str(model.get("display_name") or key),
            })
    return {"server_url": root, "instances": instances, "model_keys": model_keys}


def unload_model(remote_url: str, api_key: str = "", *,
                 instance_id: str | None = None, model_id: str = "") -> dict:
    """Unload one explicit instance or an unambiguous configured model only."""
    if instance_id is not None and (not isinstance(instance_id, str) or not instance_id.strip()):
        raise ModelSelectionError("Choose a loaded LM Studio model instance.")
    if instance_id is None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ModelSelectionError("Choose a loaded instance or configure a remote model first.")
    if instance_id is not None:
        instance_id = instance_id.strip()
    else:
        model_id = model_id.strip()
    before = loaded_models(remote_url, api_key)
    if instance_id is not None:
        matches = [item for item in before["instances"] if item["instance_id"] == instance_id]
    else:
        matches = [item for item in before["instances"]
                   if model_id in {item["instance_id"], item["model_key"]}]
    if len(matches) > 1:
        raise ModelSelectionError("This model has multiple loaded instances. Choose the exact instance to unload.")
    if not matches:
        if instance_id is None and model_id not in before["model_keys"]:
            raise ModelSelectionError("The configured model was not found in LM Studio. Choose a loaded instance.")
        return {"status": "not_loaded", "instance_id": instance_id,
                "model_key": model_id if instance_id is None else None,
                "server_url": before["server_url"]}
    selected = matches[0]
    payload = _request("POST", before["server_url"], "/api/v1/models/unload", api_key,
                       json={"instance_id": selected["instance_id"]})
    if payload.get("instance_id") != selected["instance_id"]:
        raise ModelManagementError("LM Studio did not confirm the selected instance. Refresh loaded models to check its state.")
    try:
        after = loaded_models(remote_url, api_key)
    except ModelManagementError:
        raise ModelManagementError("The unload request was accepted, but its result could not be verified. Refresh loaded models to check its state.") from None
    if any(item["instance_id"] == selected["instance_id"] for item in after["instances"]):
        raise ModelManagementError("LM Studio still reports the selected instance as loaded. Its memory release is not confirmed.")
    return {"status": "unloaded", **selected, "server_url": before["server_url"]}
