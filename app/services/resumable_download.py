"""Bounded retries and validated HTTP Range resumption for large downloads."""
from __future__ import annotations

import re
import time

import requests


class ResumableDownload:
    """One logical byte stream, reopening the original URL after interruptions.

    Only identity-encoded responses can be resumed. If-Range and a matching
    strong ETag (or Last-Modified) protect against combining different files.
    The caller must close this object even when its consumer stops early.
    """

    def __init__(self, url, headers, *, on_retry=None, get=None, sleep=time.sleep, max_retries=4):
        self.url = url
        self.request_headers = {**headers, "Accept-Encoding": "identity"}
        self.on_retry = on_retry
        self.get = get or requests.get
        self.sleep = sleep
        self.max_retries = max_retries
        self.retries = 0
        self.response = None
        self.offset = 0
        self._open()
        self.headers = self.response.headers
        raw_total = self.headers.get("Content-Length", "")
        self.total = int(raw_total) if str(raw_total).isdigit() else 0
        self.validator_key = "ETag"
        self.validator = self.headers.get("ETag", "")
        if not self.validator or self.validator.startswith("W/"):
            self.validator_key = "Last-Modified"
            self.validator = self.headers.get("Last-Modified", "")

    def _retry(self):
        if self.retries >= self.max_retries:
            raise RuntimeError(f"CivitAI download remained interrupted after {self.max_retries} automatic retries. Please retry later.") from None
        self.retries += 1
        if self.on_retry:
            self.on_retry(self.retries, self.offset)
        self.sleep(min(2 ** self.retries, 16))

    def _open(self):
        while True:
            headers = dict(self.request_headers)
            if self.offset:
                headers.update({"Range": f"bytes={self.offset}-", "If-Range": self.validator})
            try:
                response = self.get(self.url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                self._retry()
                continue
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                response.close()
                self._retry()
                continue
            if response.status_code >= 400:
                status = response.status_code
                response.close()
                raise RuntimeError(f"CivitAI download server returned HTTP {status}. Check access or retry later.")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                response.close()
                raise RuntimeError("CivitAI returned a compressed download despite an identity request; byte-range resumption is unavailable.")
            if not self.offset and response.status_code != 200:
                response.close()
                raise RuntimeError("CivitAI returned an unexpected partial response for a new download.")
            self.response = response
            return

    def raise_for_status(self):
        # _open already checks status, without exposing signed URLs or tokens.
        return None

    def iter_content(self, chunk_size=1024 * 1024):
        while True:
            try:
                for chunk in self.response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    self.offset += len(chunk)
                    if self.total and self.offset > self.total:
                        raise RuntimeError("Download server sent more bytes than the declared file size.")
                    yield chunk
                if self.total and self.offset < self.total:
                    raise requests.exceptions.ChunkedEncodingError("Incomplete response")
                return
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
                self.close()
                if self.offset and (not self.validator or not self.total):
                    raise RuntimeError("CivitAI interrupted the download and supplied no file identity/size for safe resumption. Please retry.") from None
                self._retry()
                self._open()
                if not self.offset:
                    # No bytes escaped: allow the fresh response to establish identity.
                    self.headers = self.response.headers
                    raw_total = self.headers.get("Content-Length", "")
                    self.total = int(raw_total) if str(raw_total).isdigit() else 0
                    self.validator = self.headers.get(self.validator_key, "")
                    continue
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", self.response.headers.get("Content-Range", ""))
                if (self.response.status_code != 206 or not match
                    or int(match[1]) != self.offset or int(match[2]) != self.total - 1
                    or int(match[3]) != self.total
                    or self.response.headers.get(self.validator_key) != self.validator
                    or self.response.headers.get("Content-Encoding", "identity") != "identity"):
                    self.close()
                    raise RuntimeError("CivitAI could not safely resume this file (changed file or unsupported byte range). Please retry.")

    def close(self):
        if self.response is not None:
            self.response.close()
