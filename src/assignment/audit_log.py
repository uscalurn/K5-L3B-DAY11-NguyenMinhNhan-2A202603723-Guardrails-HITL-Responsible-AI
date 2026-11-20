"""
Assignment 11 — Sanitized request audit log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

from core.config import DEMO_SECRETS
from guardrails.output_guardrails import content_filter


def redact_sensitive(text: str) -> str:
    """Apply CP2 redaction plus credentials/DB rules before retaining any text.

    Environment credentials are compared in memory only, never exported.
    Unicode normalization prevents invisible separators bypassing the rules.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    secrets = list(DEMO_SECRETS)
    secrets.extend(value for name, value in os.environ.items()
                   if re.search(r"(?:key|password|secret|token)", name, re.I) and value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        text = re.sub(re.escape(secret), "[REDACTED]", text, flags=re.I)
    patterns = (
        r"\b(?:password|passwd|pwd|mật\s*khẩu|api[_\s-]*key|access[_\s-]*token)"
        r"[\s\"']*(?::|=|\bis\b)\s*(?:\"[^\"]*\"|'[^']*')",
        r"\b(?:password|passwd|pwd|mật\s*khẩu|api[_\s-]*key|access[_\s-]*token|authorization)"
        r"[\s\"']*(?::|=|\bis\b)[\s\"']*(?:bearer\s+)?[^\s,;\"'<>]+",
        r"\b(?:password|passwd|pwd|api[_\s-]*key|mật\s*khẩu)\s+[^\s,;\"'<>]+",
        r"\b(?:sk|pk|api|key|token)-[A-Za-z0-9_-]{6,}\b",
        r"\bBearer\s+[A-Za-z0-9._~+/-]+=*",
        r"\b(?:db|database)[_.-][A-Za-z0-9_.-]+(?::\d+)?\b",
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s<>]+",
        r"\b(?:db[_\s-]*host|database[_\s-]*host)[\s\"']*(?::|=|\bis\b)[\s\"']*[^\s,;\"'<>]+",
        r"(?<!\w)(?:\+?\d[\s().-]*){10,15}(?!\w)",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    )
    for pattern in patterns:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.I)
    text = content_filter(text)["redacted"]
    # Match punctuation/spacing obfuscation of the protected lab values.
    compact = re.sub(r"[^a-z0-9]", "", text.casefold())
    if any(re.sub(r"[^a-z0-9]", "", s.casefold()) in compact for s in DEMO_SECRETS if s):
        return "[REDACTED]"
    return text


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}
        self._pending: dict[str, list[str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request. Return its ID so concurrent requests can be paired."""
        # Correlation IDs are opaque metadata. Content/PII redaction can mistake
        # UUID digit runs for national IDs and collapse distinct request IDs.
        request_id = uuid4().hex if request_id is None else request_id
        if request_id in self._open:
            raise ValueError("request_id already has an unfinished input")
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": redact_sensitive(user_id),
            "input": redact_sensitive(text),
            "timestamp": utc_now_iso(),
            "started": time.monotonic(),
        }
        self._pending.setdefault(user_id, []).append(request_id)
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        error: str | None = None,
        status_code: int | None = None,
    ):
        """Finish a matching request; omit raw input/output from retained state."""
        pending = self._pending.get(user_id, [])
        if request_id is None:
            request_id = pending[0] if pending else None
        if request_id not in pending or request_id not in self._open:
            raise ValueError("record_output requires a matching record_input")
        entry = self._open.pop(request_id)
        pending.remove(request_id)
        if not pending:
            self._pending.pop(user_id, None)
        latency = max(0.0, time.monotonic() - entry.pop("started"))
        entry.update(response=redact_sensitive(text), blocked=bool(blocked),
                     layer=redact_sensitive(layer) if layer is not None else None,
                     latency=latency, latency_ms=round(latency * 1000, 3))
        entry["status"] = "error" if error is not None else "blocked" if blocked else "completed"
        if error is not None:
            entry["error"] = redact_sensitive(error)
        if status_code is not None:
            entry["status_code"] = status_code
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
