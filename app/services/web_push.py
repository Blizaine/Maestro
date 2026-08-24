"""Local Web Push subscriptions for Maestro completion notifications.

The browser owns the push endpoint and encryption keys; Maestro stores those
details only on the machine running Maestro.  There is no Maestro cloud relay.
The browser vendor's standards-based Web Push service is contacted directly by
``pywebpush`` when a top-level Studio or Director item reaches a terminal state.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


_DEFAULT_PREFERENCES = {
    "notifyCompleted": True,
    "notifyFailed": True,
    "notifyQueue": True,
    "onlyWhenHidden": True,
}
_CATEGORY_PREFERENCE = {
    "completion": "notifyCompleted",
    "failure": "notifyFailed",
    "queue": "notifyQueue",
    "test": None,
}


class WebPushUnavailable(RuntimeError):
    """Raised when optional Web Push delivery is not available."""


@dataclass(frozen=True)
class PushDeliveryResult:
    attempted: int
    delivered: int
    removed: int
    errors: tuple[str, ...] = ()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def _new_vapid_pair() -> tuple[str, str]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_numbers = private_key.public_key().public_numbers()
    public_raw = (
        b"\x04"
        + public_numbers.x.to_bytes(32, "big")
        + public_numbers.y.to_bytes(32, "big")
    )
    return private_pem, _b64url(public_raw)


def _clean_preferences(raw: Any) -> dict[str, bool]:
    source = raw if isinstance(raw, dict) else {}
    return {
        key: bool(source.get(key, default))
        for key, default in _DEFAULT_PREFERENCES.items()
    }


def _clean_origin(raw: Any) -> str:
    value = str(raw or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "/"
    return f"{parsed.scheme}://{parsed.netloc}"


def _validate_subscription(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("subscription must be an object")
    endpoint = str(raw.get("endpoint") or "").strip()
    keys = raw.get("keys")
    if not endpoint.startswith("https://"):
        raise ValueError("subscription endpoint must use HTTPS")
    if not isinstance(keys, dict):
        raise ValueError("subscription keys are missing")
    p256dh = str(keys.get("p256dh") or "").strip()
    auth = str(keys.get("auth") or "").strip()
    if not p256dh or not auth:
        raise ValueError("subscription p256dh/auth keys are missing")
    return {
        "endpoint": endpoint,
        "expirationTime": raw.get("expirationTime"),
        "keys": {"p256dh": p256dh, "auth": auth},
    }


class WebPushService:
    """Thread-safe, machine-local subscription store and sender."""

    def __init__(self, settings_dir: str | os.PathLike[str]):
        self._path = Path(settings_dir) / "web_push.json"
        self._lock = threading.RLock()
        self._state = self._load_state()

    @staticmethod
    def dependency_available() -> bool:
        try:
            import pywebpush  # noqa: F401
        except Exception:
            return False
        return True

    def _load_state(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                candidate = json.load(handle)
            if isinstance(candidate, dict):
                loaded = candidate
        except (OSError, ValueError, TypeError):
            loaded = {}

        private_key = loaded.get("vapid_private_key")
        public_key = loaded.get("vapid_public_key")
        if not isinstance(private_key, str) or not isinstance(public_key, str):
            private_key, public_key = _new_vapid_pair()

        subscriptions = loaded.get("subscriptions")
        if not isinstance(subscriptions, list):
            subscriptions = []
        state = {
            "version": 1,
            "vapid_private_key": private_key,
            "vapid_public_key": public_key,
            "subscriptions": [
                item for item in subscriptions
                if isinstance(item, dict) and item.get("endpoint")
            ],
        }
        try:
            _atomic_write_json(self._path, state)
        except OSError as exc:
            print(f"[Web Push] Could not persist local notification keys: {exc}")
        return state

    @property
    def public_key(self) -> str:
        return str(self._state["vapid_public_key"])

    def status(self) -> dict[str, Any]:
        with self._lock:
            count = len(self._state["subscriptions"])
        available = self.dependency_available()
        return {
            "supported": available,
            "public_key": self.public_key,
            "subscription_count": count,
            "reason": None if available else (
                "The Web Push runtime is not installed. Run Maestro Update once."
            ),
        }

    def subscribe(
        self,
        subscription: Any,
        *,
        preferences: Any = None,
        origin: Any = None,
        label: Any = None,
    ) -> dict[str, Any]:
        clean = _validate_subscription(subscription)
        record = {
            **clean,
            "preferences": _clean_preferences(preferences),
            "origin": _clean_origin(origin),
            "label": str(label or "This device")[:160],
        }
        with self._lock:
            items = self._state["subscriptions"]
            index = next(
                (i for i, item in enumerate(items)
                 if item.get("endpoint") == clean["endpoint"]),
                None,
            )
            if index is None:
                items.append(record)
            else:
                items[index] = record
            _atomic_write_json(self._path, self._state)
            count = len(items)
        return {"subscribed": True, "subscription_count": count}

    def unsubscribe(self, endpoint: Any) -> dict[str, Any]:
        endpoint_text = str(endpoint or "").strip()
        with self._lock:
            before = len(self._state["subscriptions"])
            self._state["subscriptions"] = [
                item for item in self._state["subscriptions"]
                if item.get("endpoint") != endpoint_text
            ]
            removed = before - len(self._state["subscriptions"])
            if removed:
                _atomic_write_json(self._path, self._state)
            count = len(self._state["subscriptions"])
        return {"unsubscribed": bool(removed), "subscription_count": count}

    def _eligible(
        self, category: str, endpoint: str | None = None
    ) -> list[dict[str, Any]]:
        preference_key = _CATEGORY_PREFERENCE.get(category)
        with self._lock:
            items = [dict(item) for item in self._state["subscriptions"]]
        eligible = []
        for item in items:
            if endpoint and item.get("endpoint") != endpoint:
                continue
            prefs = _clean_preferences(item.get("preferences"))
            if preference_key and not prefs[preference_key]:
                continue
            item["preferences"] = prefs
            eligible.append(item)
        return eligible

    def _send(
        self,
        *,
        category: str,
        title: str,
        body: str,
        tag: str,
        endpoint: str | None = None,
    ) -> PushDeliveryResult:
        if not self.dependency_available():
            raise WebPushUnavailable(
                "Web Push dependencies are missing; run Maestro Update once."
            )
        from pywebpush import WebPushException, webpush

        subscriptions = self._eligible(category, endpoint)
        delivered = 0
        stale: list[str] = []
        errors: list[str] = []
        private_key = str(self._state["vapid_private_key"])
        for item in subscriptions:
            payload = {
                "category": category,
                "title": title,
                "body": body,
                "tag": tag,
                "url": item.get("origin") or "/",
                "onlyWhenHidden": bool(
                    item.get("preferences", {}).get("onlyWhenHidden", True)
                ),
            }
            subscription_info = {
                "endpoint": item["endpoint"],
                "expirationTime": item.get("expirationTime"),
                "keys": item["keys"],
            }
            try:
                webpush(
                    subscription_info=subscription_info,
                    data=json.dumps(payload, ensure_ascii=False),
                    vapid_private_key=private_key,
                    vapid_claims={"sub": "mailto:notifications@maestro.local"},
                    ttl=86400,
                )
                delivered += 1
            except WebPushException as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code in {404, 410}:
                    stale.append(item["endpoint"])
                else:
                    errors.append(str(exc))
            except Exception as exc:
                errors.append(str(exc))

        if stale:
            stale_set = set(stale)
            with self._lock:
                self._state["subscriptions"] = [
                    item for item in self._state["subscriptions"]
                    if item.get("endpoint") not in stale_set
                ]
                _atomic_write_json(self._path, self._state)

        return PushDeliveryResult(
            attempted=len(subscriptions),
            delivered=delivered,
            removed=len(stale),
            errors=tuple(errors),
        )

    def send_test(self, endpoint: Any = None) -> PushDeliveryResult:
        endpoint_text = str(endpoint or "").strip() or None
        return self._send(
            category="test",
            title="Maestro background notifications are ready",
            body="This device can receive alerts even after Maestro is closed.",
            tag="maestro-web-push-test",
            endpoint=endpoint_text,
        )

    def dispatch(
        self,
        *,
        category: str,
        title: str,
        body: str,
        tag: str,
    ) -> None:
        """Send without delaying job publication or releasing the GPU lock."""

        if category not in _CATEGORY_PREFERENCE:
            return

        def worker() -> None:
            try:
                result = self._send(
                    category=category,
                    title=title,
                    body=body,
                    tag=tag,
                )
                if result.errors:
                    print(
                        "[Web Push] Delivery warning: "
                        + "; ".join(result.errors[:2])
                    )
            except WebPushUnavailable:
                return
            except Exception as exc:
                print(f"[Web Push] Delivery failed: {exc}")

        threading.Thread(
            target=worker,
            daemon=True,
            name="maestro_web_push",
        ).start()


def dispatch_queue_complete(
    service: WebPushService | None,
    *,
    title: str,
    body: str,
    tag: str,
) -> None:
    if service is not None:
        service.dispatch(
            category="queue", title=title, body=body, tag=tag
        )
