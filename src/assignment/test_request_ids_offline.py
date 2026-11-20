"""Offline regression: real CP3 pipeline and exports, with only the SDK mocked.

Run from the repository root:
    .venv/Scripts/python.exe -m pytest src/assignment/test_request_ids_offline.py -q

All generated artifacts live in pytest's temporary directory. Submission JSON,
model selection and guardrail rules are never replaced. No API key is needed.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil
import socket
import sys
from types import CodeType, SimpleNamespace
from uuid import UUID

import httpx
from openai import RateLimitError
import pytest

from assignment import audit_log, monitoring, pipeline
from core.config import BLUE_MODEL, DEMO_SECRETS
from core.openai_runtime import OpenAIRunner


ROOT = Path(__file__).resolve().parents[2]
HEX_ID = re.compile(r"[0-9a-f]{32}")
TRICKY_IDS = [
    UUID("abcdefab-cdef-4abc-8abc-def123456789"),
    UUID("abcdefab-cdef-4abc-8abc-def987654321"),
    UUID("abcdefab-cdef-4abc-8abc-123456789012"),
]
PRIVATE_VALUES = [
    "offline-login-73xy",
    "sk-offline-regression-1234567890",
    "customer@example.org",
    "0901234567",
    "123456789012",
    "db.regression.internal",
    "offline-environment-credential-73xy",
    *DEMO_SECRETS,
]
PRIVATE_TEXT = (
    "Bank savings: password=offline-login-73xy; "
    "api_key=sk-offline-regression-1234567890; "
    + "; ".join(PRIVATE_VALUES[2:])
)


def _submission_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "outputs").glob("*") if path.is_file()
    }


def _rows(result):
    return (result["safe_queries"] + result["attack_queries"]
            + result["rate_limit"]["requests"] + result["edge_cases"])


def test_runtime_provenance_matches_current_source():
    """Check the loaded bytecode, not just inspect.getsource's disk contents."""
    assert pipeline.AuditLogPlugin is audit_log.AuditLogPlugin
    for module, names in (
        (audit_log, ["AuditLogPlugin.record_input", "AuditLogPlugin.record_output",
                     "AuditLogPlugin.export_json"]),
        (pipeline, ["run_assignment_suite"]),
    ):
        expected_path = ROOT / "src/assignment" / Path(module.__file__).name
        assert Path(module.__file__).resolve() == expected_path
        compiled = compile(expected_path.read_text(encoding="utf-8"),
                           str(expected_path), "exec", dont_inherit=True)
        for name in names:
            expected, loaded = compiled, module
            for part in name.split("."):
                expected = next(item for item in expected.co_consts
                                if isinstance(item, CodeType) and item.co_name == part)
                loaded = getattr(loaded, part)
            assert loaded.__code__ == expected, f"Stale bytecode: {name}"


@pytest.fixture
def offline(monkeypatch, tmp_path):
    before = _submission_hashes()
    runner = asyncio.Runner()
    # Windows creates a local wakeup socket pair when initializing an event
    # loop. Initialize it before blocking every subsequent connection attempt.
    runner.get_loop()

    def deny_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in this offline test")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setenv("AUDIT_OFFLINE_API_KEY", PRIVATE_VALUES[6])
    (tmp_path / "schemas").mkdir()
    shutil.copyfile(ROOT / "schemas/results.schema.json",
                    tmp_path / "schemas/results.schema.json")
    # Exercise the actual production writers, redirected to a temporary root.
    monkeypatch.setattr(pipeline, "__file__",
                        str(tmp_path / "src/assignment/pipeline.py"))
    monkeypatch.setattr(audit_log, "default_audit_log_path",
                        lambda: str(tmp_path / "outputs/audit_log.json"))
    monkeypatch.setattr(monitoring, "default_metrics_path",
                        lambda: str(tmp_path / "outputs/metrics.json"))
    try:
        yield tmp_path, runner.run
    finally:
        runner.close()
        assert _submission_hashes() == before, "Submission artifacts were modified"


@pytest.mark.parametrize("provider_error,batches", [(False, 34), (True, 1)])
def test_real_pipeline_round_trip(monkeypatch, offline, provider_error, batches):
    """1,085 UUIDs across success/error paths, through the real suite writers."""
    offline, run = offline
    rng = random.Random(11)
    generated, admitted, finished = [], [], []
    sdk_calls = []

    def next_uuid():
        index = len(generated)
        value = (TRICKY_IDS[index] if index < len(TRICKY_IDS)
                 else UUID(int=rng.getrandbits(128), version=4))
        generated.append(value.hex)
        return value

    class TracingAudit(audit_log.AuditLogPlugin):
        def record_input(self, **kwargs):
            value = super().record_input(**kwargs)
            assert value == generated[-1]
            assert self._open[value]["request_id"] == value
            admitted.append(value)
            return value

        def record_output(self, **kwargs):
            value = kwargs["request_id"]
            assert value in admitted
            entry = super().record_output(**kwargs)
            assert entry["request_id"] == value
            finished.append(value)
            return entry

    def sdk_create(**kwargs):
        assert kwargs["model"] == BLUE_MODEL
        sdk_calls.append(True)
        if provider_error:
            response = httpx.Response(
                429, request=httpx.Request("POST", "https://offline.invalid"))
            raise RateLimitError("Offline simulated provider error",
                                 response=response, body=None)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=PRIVATE_TEXT))])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=sdk_create)))
    monkeypatch.setattr(OpenAIRunner, "_client", lambda self: client)
    monkeypatch.setattr(audit_log, "uuid4", next_uuid)
    original_read_text = Path.read_text

    def no_artifact_reads(path, *args, **kwargs):
        assert path.name not in {"results.json", "audit_log.json", "metrics.json"}, (
            "The suite must not reuse existing artifacts")
        return original_read_text(path, *args, **kwargs)

    for batch in range(batches):
        start = len(generated)
        audit = TracingAudit()
        plugins = pipeline.build_production_plugins(use_llm_judge=False)
        suite = {"plugins": plugins, "audit": audit,
                 "monitor": monitoring.MonitoringAlert()}
        with monkeypatch.context() as guarded:
            guarded.setattr(Path, "read_text", no_artifact_reads)
            if provider_error:
                with pytest.raises(RuntimeError, match="CP3 suite failed"):
                    run(pipeline.run_assignment_suite(suite))
            else:
                in_memory = run(pipeline.run_assignment_suite(suite))

        result = json.loads((offline / "outputs/results.json").read_text("utf-8"))
        logs = json.loads((offline / "outputs/audit_log.json").read_text("utf-8"))
        rows = _rows(result)
        expected = generated[start:]
        assert len(expected) == len(rows) == len(logs) == 31
        assert [row["request_id"] for row in rows] == expected
        assert admitted[start:] == expected
        assert set(finished[start:]) == set(expected)
        assert all(HEX_ID.fullmatch(value) for value in expected)
        by_id = {entry["request_id"]: entry for entry in logs}
        assert len(by_id) == len(expected)
        assert set(by_id) == set(expected)
        assert logs == audit.logs
        assert not audit._open and not audit._pending
        for row in rows:
            entry = by_id[row["request_id"]]
            for key in ("input", "blocked", "layer", "user_id", "status"):
                assert row[key] == entry[key]
            assert row["response_preview"] == entry["response"][:500]
            assert row.get("error") == entry.get("error")
            assert row.get("status_code") == entry.get("status_code")
            assert all(value not in entry["response"] for value in PRIVATE_VALUES)
        if provider_error:
            assert result["errors"] == 16
            assert result["checks"]["model_requests_succeeded"] is False
        else:
            assert in_memory == result
            assert all(result["checks"].values())
            assert plugins[2].redacted_count == 16
        assert len(sdk_calls) == (batch + 1) * 16
    assert len(generated) == len(set(generated)) == batches * 31


def test_sensitive_content_and_distinct_ids(monkeypatch, offline):
    """Keep metadata intact while identical digits in content remain filtered."""
    offline, _ = offline
    first, second = [value.hex for value in TRICKY_IDS[:2]]
    assert audit_log.redact_sensitive(first) == audit_log.redact_sensitive(second)
    audit = audit_log.AuditLogPlugin()
    for value in (first, second):
        assert audit.record_input(user_id="one-user", text=PRIVATE_TEXT,
                                  request_id=value) == value
        assert all(secret not in audit._open[value]["input"]
                   for secret in PRIVATE_VALUES)
    with pytest.raises(ValueError):
        audit.record_input(user_id="one-user", text="duplicate", request_id=first)
    with pytest.raises(ValueError):
        audit.record_output(user_id="other-user", text="wrong", request_id=first)
    # Changing credentials must not affect correlation of already-open IDs.
    monkeypatch.setenv("AUDIT_ROTATED_API_KEY", first)
    for value in (second, first):
        entry = audit.record_output(user_id="one-user", request_id=value,
                                    text=PRIVATE_TEXT, error="password=offline-error")
        assert entry["request_id"] == value
        assert all(secret not in entry["response"] for secret in PRIVATE_VALUES)
        assert "offline-error" not in entry["error"]
    audit.export_json()
    logs = json.loads((offline / "outputs/audit_log.json").read_text("utf-8"))
    assert [entry["request_id"] for entry in logs] == [second, first]
    assert all(HEX_ID.fullmatch(entry["request_id"]) for entry in logs)


def test_running_module_keeps_old_code_after_disk_edit(monkeypatch, offline):
    """Reproduce import-before-fix, source-edit, export-after-fix without live CP3."""
    offline, _ = offline
    fixed_source = Path(audit_log.__file__).read_text(encoding="utf-8")
    old_line = "request_id = redact_sensitive(request_id or uuid4().hex)"
    new_line = "request_id = uuid4().hex if request_id is None else request_id"
    assert fixed_source.count(new_line) == 1
    legacy_source = fixed_source.replace(new_line, old_line, 1)
    module_path = offline / "cached_audit_probe.py"
    module_path.write_text(legacy_source, encoding="utf-8")

    def load(name):
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    name = "_lab11_cached_audit_probe"
    loaded = load(name)
    first, second = [value.hex for value in TRICKY_IDS[:2]]
    monkeypatch.setattr(loaded, "uuid4", lambda: TRICKY_IDS[0])
    audit = loaded.AuditLogPlugin()
    before_edit = audit.record_input(user_id="before-edit", text="Bank savings")
    assert before_edit != first and "[REDACTED]" in before_edit

    module_path.write_text(fixed_source, encoding="utf-8")
    assert importlib.import_module(name) is loaded  # Python's sys.modules cache.
    monkeypatch.setattr(loaded, "uuid4", lambda: TRICKY_IDS[1])
    after_edit = loaded.AuditLogPlugin().record_input(
        user_id="after-edit", text="Bank savings")
    assert after_edit != second and "[REDACTED]" in after_edit
    audit.record_output(user_id="before-edit", request_id=before_edit,
                        text="Bank savings reply")
    path = offline / "exported_after_fix.json"
    audit.export_json(str(path))
    assert json.loads(path.read_text("utf-8"))[0]["request_id"] == before_edit

    fresh = load(name + "_fresh")
    monkeypatch.setattr(fresh, "uuid4", lambda: TRICKY_IDS[0])
    assert fresh.AuditLogPlugin().record_input(
        user_id="fresh-process-equivalent", text="Bank savings") == first
