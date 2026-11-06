#!/usr/bin/env python3
"""One bounded retry of the author-linked FFNet SPD checkpoint.

The process refuses to issue a request before the frozen backoff boundary.  It
uses only the already-running local proxy, performs at most the initial Drive
request and one explicit confirmation request, and refuses all redirects.  A
text response is preserved as evidence and is never installed as a checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import requests


ROOT = Path("/path/to/project")
FILE_ID = "1k-WIcGvGY4YF02ITMTTIDdPblA2cjaji"
SOURCE_URL = "https://drive.google.com/uc?export=download&id=" + FILE_ID
PROXY = "http://127.0.0.1:7897"
NOT_BEFORE = dt.datetime.fromisoformat("2026-10-31T23:46:30+08:00")
TARGET = ROOT / "artifacts/checkpoints/ffnet_spd_52164dfe/ffnet-v2x-spd.pth"
DEFAULT_OUTPUT = ROOT / ("artifacts/research_autopilot/"
                         "status_audit_20261031_ffnet_checkpoint_retry_after_freeze_v1/download_v1")
ALLOWED_CONFIRM_HOST = "drive.usercontent.google.com"


class DownloadForm(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.inputs: dict[str, str] = {}
        self.active = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "form" and values.get("id") == "download-form":
            self.action = values.get("action")
            self.active = True
        elif tag == "input" and self.active and values.get("name"):
            self.inputs[str(values["name"])] = values.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.active = False


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def response_record(response: requests.Response, ordinal: int) -> dict[str, Any]:
    return {
        "request_ordinal": ordinal,
        "http_status": response.status_code,
        "resolved_url": response.url,
        "host": urlparse(response.url).hostname,
        "content_type": response.headers.get("Content-Type"),
        "content_length": response.headers.get("Content-Length"),
        "content_disposition": response.headers.get("Content-Disposition"),
        "location": response.headers.get("Location"),
    }


def save_text_evidence(output: Path, name: str, body: bytes) -> dict[str, Any]:
    path = output / name
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.relative_to(ROOT)), "bytes": len(body), "sha256": sha256(path)}


def is_textual(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    return "text/" in content_type or "html" in content_type or "json" in content_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate local invariants without opening a network connection")
    args = parser.parse_args()
    output = args.output.resolve()
    state_path = output / "download_status_v1.json"
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        raise FileExistsError(f"one-shot output already exists: {state_path}")

    started_wall = now()
    started_mono = time.monotonic()
    state: dict[str, Any] = {
        "schema": "ffnet_author_checkpoint_after_freeze_download_v1",
        "status": "preflight",
        "started_at": started_wall.isoformat(timespec="seconds"),
        "not_before": NOT_BEFORE.isoformat(),
        "public_file_id": FILE_ID,
        "source_url": SOURCE_URL,
        "source_repository": "https://github.com/haibao-yu/FFNet-VIC3D",
        "source_commit": "52164dfe00764c9a9925539e99689cf25b88eace",
        "proxy": PROXY,
        "target": str(TARGET.relative_to(ROOT)),
        "request_limit": 2,
        "requests_made": 0,
        "redirect_policy": "disabled_fail_closed",
        "responses": [],
        "boundary": ("Single author-linked same-file retry after the frozen backoff; no quota bypass, "
                     "mirror, alternate identity, external contact, or global proxy mutation."),
    }
    exit_code = 1
    response: requests.Response | None = None
    partial = TARGET.with_suffix(TARGET.suffix + ".partial.after_freeze_v1")
    try:
        if TARGET.exists() or partial.exists():
            raise RuntimeError("checkpoint or one-shot partial already exists; refusing overwrite")
        if args.preflight_only:
            state["status"] = "passed_local_preflight_no_network"
            state["checked_at"] = now().isoformat(timespec="seconds")
            exit_code = 0
            return
        current = now()
        if current < NOT_BEFORE:
            raise RuntimeError(f"frozen retry boundary not reached: {current.isoformat()} < {NOT_BEFORE.isoformat()}")

        session = requests.Session()
        session.trust_env = False
        session.proxies.update({"http": PROXY, "https": PROXY})
        response = session.get(SOURCE_URL, stream=True, timeout=(15, 90), allow_redirects=False)
        state["requests_made"] = 1
        state["responses"].append(response_record(response, 1))
        if response.is_redirect or response.is_permanent_redirect:
            raise RuntimeError("initial request returned a redirect; bounded policy refuses a third-party hop")
        response.raise_for_status()

        if is_textual(response):
            body = response.content
            evidence = save_text_evidence(output, "initial_response_v1.html", body)
            state["responses"][-1]["body_evidence"] = evidence
            parser_html = DownloadForm()
            parser_html.feed(body.decode(response.encoding or "utf-8", errors="replace"))
            response.close()
            response = None
            action = parser_html.action or ""
            if urlparse(action).scheme != "https" or urlparse(action).hostname != ALLOWED_CONFIRM_HOST:
                raise RuntimeError("initial text response is not the expected author-file confirmation form")
            if parser_html.inputs.get("id") != FILE_ID:
                raise RuntimeError("confirmation form file identity differs from frozen Drive ID")
            response = session.get(action, params=parser_html.inputs, stream=True,
                                   timeout=(15, 120), allow_redirects=False)
            state["requests_made"] = 2
            state["responses"].append(response_record(response, 2))
            if response.is_redirect or response.is_permanent_redirect:
                raise RuntimeError("confirmation request redirected; bounded retry refuses another request")
            if urlparse(response.url).hostname != ALLOWED_CONFIRM_HOST:
                raise RuntimeError("confirmation response escaped the allowed Google user-content host")
            response.raise_for_status()

        if is_textual(response):
            body = response.content
            evidence = save_text_evidence(output, "confirmation_error_response_v1.html", body)
            state["responses"][-1]["body_evidence"] = evidence
            raise RuntimeError("expected checkpoint but received a textual quota/error response")

        expected_text = response.headers.get("Content-Length")
        expected = int(expected_text) if expected_text and expected_text.isdigit() else None
        digest = hashlib.sha256()
        downloaded = 0
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("xb") as stream:
            for block in response.iter_content(1 << 20):
                if not block:
                    continue
                stream.write(block)
                digest.update(block)
                downloaded += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if expected is not None and downloaded != expected:
            raise RuntimeError(f"byte count differs from HTTP Content-Length: {downloaded} != {expected}")
        if downloaded < 100_000:
            raise RuntimeError(f"checkpoint is unexpectedly small: {downloaded} bytes")
        os.replace(partial, TARGET)
        state.update({
            "status": "downloaded_hash_recorded_pending_format_gate",
            "completed_at": now().isoformat(timespec="seconds"),
            "bytes": downloaded,
            "sha256": digest.hexdigest(),
        })
        exit_code = 0
    except BaseException as exc:
        state.update({
            "status": "failed_preserved_owner_not_released",
            "failed_at": now().isoformat(timespec="seconds"),
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "partial": ({"path": str(partial.relative_to(ROOT)), "bytes": partial.stat().st_size,
                         "sha256": sha256(partial)} if partial.exists() else None),
        })
    finally:
        if response is not None:
            response.close()
        state["elapsed_s"] = time.monotonic() - started_mono
        atomic_json(state_path, state)
        print(json.dumps({"status": state["status"], "requests_made": state["requests_made"],
                          "output": str(state_path)}, sort_keys=True), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
