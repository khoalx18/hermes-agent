"""Regression tests for the spec/plan gate on kanban auto-decompose.

Source of truth: .hermes/plans/2026-08-03_174325-spec-gate-policy.md
(API SURFACE, GATE FLOW, FAIL-CLOSED, OVERRIDE PATH, AC-1..AC-7).

This suite targets ONLY the documented contract — nothing else:

    TaskGate(level, reasons, override)
    classify_task(title, body) -> TaskGate
    is_gated(gate) -> bool
    validate_gate_graph(gate, children, *, architect_assignee="architect")
        -> tuple[bool, str]
    decompose_task(...) integration (gate flow inside the decomposer)
    _SYSTEM_PROMPT rule block (AC-7)

Expected lifecycle: the implementation task lands the gate API in
hermes_cli/kanban_decompose.py AFTER this file. Until then every test
here is RED by design (AttributeError on the missing API); the review
task runs this suite against the implemented code and it must be green.

Test doubles mirror tests/hermes_cli/test_kanban_decompose.py (mocked
auxiliary LLM, real DB via kanban_home fixture).
"""

from __future__ import annotations

import json as jsonlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    # decompose_task routes through call_llm — mock it at the source module.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim for call-site compatibility (see anchor test file).
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist (same shape as the anchor test)."""
    from types import SimpleNamespace

    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def _fanout_link_count(conn, root_id: str) -> int:
    """Count children of a decomposed root.

    decompose_triage_task links the ROOT under every child (task_links
    rows with child_id = root), so this count equals the number of
    children created — 0 means the decomposition never wrote the DB.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM task_links WHERE child_id = ?", (root_id,)
    ).fetchone()
    return int(row[0])


def _run_decompose(tid: str, llm_payload: str, *, config: dict | None = None, author: str = "me"):
    """Run decomp.decompose_task with the standard test environment.

    Profiles and _load_config are patched so kanban.spec_gate is fully
    controlled by the caller (default: no explicit key -> gate enabled).
    """
    cfg = {"kanban": {}} if config is None else config
    patches = _patch_list_profiles(["orchestrator", "architect", "coding", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=cfg,
        ):
            return decomp.decompose_task(tid, author=author)
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# classify_task — AC-1 (deterministic per spec PHÂN LOẠI TASK)
# ---------------------------------------------------------------------------

class TestClassifyTask:

    def test_dangerous_path_zone_keywords(self):
        # Body rỗng fail-closed thành uncertain (spec), nên keyword tests
        # phải có body không rỗng để thực sự test pattern matching.
        body = "cần thực hiện gấp"
        cases = [
            ("sửa file trong data/ cho macro", body),
            ("chạy migrations/ upgrade", body),
            ("journal.db bị lỗi, sửa giúp", body),
            ("trade_plans.db cần sửa", body),
            ("đụng engine/ code", body),
            ("đổi schema bảng tasks", body),
            ("sửa pipeline macro data", body),   # documented example
            ("update CONTRACTS.md", body),
        ]
        for title, b in cases:
            gate = decomp.classify_task(title, b)
            assert gate.level == "dangerous", (title, gate.reasons)

    def test_dangerous_irreversible_actions(self):
        body = "cần thực hiện gấp"
        cases = [
            ("deploy gateway", body),            # documented example
            ("publish release", body),
            ("release new version", body),
            ("transfer funds between accounts", body),
            ("process payment", body),
            ("send email blast", body),
            ("migrate dữ liệu", body),
            ("drop table trade_plans", body),    # documented example
            ("truncate bảng logs", body),
            ("delete records cũ", body),
            ("remove file config", body),
            ("rm -rf data/", body),
            ("wipe cache", body),
            ("format ổ đĩa", body),
            ("đặt lệnh thật", body),             # documented example
        ]
        for title, b in cases:
            gate = decomp.classify_task(title, b)
            assert gate.level == "dangerous", (title, gate.reasons)

    def test_case_insensitive_matching(self):
        gate = decomp.classify_task("DEPLOY GATEWAY", "PUBLISH")
        assert gate.level == "dangerous"

    def test_complex_keywords(self):
        body = "cần thực hiện gấp"
        cases = [
            ("design new architecture", body),
            ("refactor auth module", body),
            ("new module cho reporting", body),
            ("cross-module change", body),
            ("handoff sang team khác", body),
            (">3 files cần sửa", body),
            ("implement per spec/plan", body),   # spec/plan context -> complex
            ("viết acceptance criteria", body),
        ]
        for title, b in cases:
            gate = decomp.classify_task(title, b)
            assert gate.level == "complex", (title, gate.reasons)

    def test_uncertain_keywords_fail_closed(self):
        body = "cần thực hiện gấp"
        cases = [
            ("touches database", body),
            ("external service bị lỗi", body),
            ("production incident", body),
            ("scheduler chạy sai giờ", body),
            ("cron task chết", body),
            ("integration test fail", body),
            ("api version bump", body),
        ]
        for title, b in cases:
            gate = decomp.classify_task(title, b)
            assert gate.level == "uncertain", (title, gate.reasons)

    def test_empty_body_is_uncertain(self):
        for body in ("", "   ", "\n\n"):
            gate = decomp.classify_task("some task", body)
            assert gate.level == "uncertain", repr(body)

    def test_safe_task(self):
        gate = decomp.classify_task("fix typo trong README", "đổi câu chữ trong docs")
        assert gate.level == "safe"
        gate2 = decomp.classify_task("thêm test cho hàm X đã có", "bổ sung unit test")
        assert gate2.level == "safe"

    def test_override_marker_in_title(self):
        gate = decomp.classify_task("deploy gateway [spec-gate:skip]", "có chủ đích")
        assert gate.level == "safe"
        assert gate.override is True
        assert "override marker" in gate.reasons

    def test_override_marker_in_body(self):
        gate = decomp.classify_task("deploy gateway", "cho phép lần này [spec-gate:skip]")
        assert gate.level == "safe"
        assert gate.override is True

    def test_override_beats_dangerous_keywords(self):
        gate = decomp.classify_task("drop table trade_plans [spec-gate:skip]", "")
        assert gate.level == "safe"
        assert gate.override is True


# ---------------------------------------------------------------------------
# is_gated — AC-1 (GATED <=> level != "safe")
# ---------------------------------------------------------------------------

class TestIsGated:

    def test_safe_not_gated(self):
        gate = decomp.TaskGate(level="safe", reasons=[])
        assert decomp.is_gated(gate) is False

    @pytest.mark.parametrize("level", ["dangerous", "complex", "uncertain"])
    def test_gated_levels(self, level):
        gate = decomp.TaskGate(level=level, reasons=[])
        assert decomp.is_gated(gate) is True

    def test_override_marker_safe_not_gated(self):
        gate = decomp.TaskGate(level="safe", reasons=["override marker"], override=True)
        assert decomp.is_gated(gate) is False


# ---------------------------------------------------------------------------
# validate_gate_graph — AC-2/AC-3 structural rules (exact strings from spec)
# ---------------------------------------------------------------------------

def _architect_child(title: str = "Spec/plan cho X", parents: list | None = None):
    return {
        "title": title,
        "body": "GOAL APPROACH ACCEPTANCE OUT-OF-SCOPE",
        "assignee": "architect",
        "parents": parents or [],
    }


def _coding_child(title: str = "Implement X", parents: list | None = None):
    return {
        "title": title,
        "body": "Theo spec/plan cua architect",
        "assignee": "coding",
        "parents": parents or [],
    }


class TestValidateGateGraph:

    def test_missing_architect_child(self):
        gate = decomp.TaskGate(level="dangerous", reasons=["deploy"])
        children = [_coding_child(parents=[])]
        ok, reason = decomp.validate_gate_graph(gate, children)
        assert ok is False
        assert reason == "missing architect spec/plan child"

    def test_architect_child_with_parents_rejected(self):
        gate = decomp.TaskGate(level="complex", reasons=["design"])
        children = [
            {"title": "Spec/plan", "body": "b", "assignee": "architect", "parents": [1]},
            {"title": "review", "body": "b", "assignee": "reviewer", "parents": [0]},
        ]
        ok, reason = decomp.validate_gate_graph(gate, children)
        assert ok is False
        assert reason == "architect child must have no parents"

    def test_coding_child_without_architect_parent_rejected(self):
        gate = decomp.TaskGate(level="dangerous", reasons=["deploy"])
        children = [
            _architect_child(),
            _coding_child(parents=[1]),  # wrong index — not the architect (0)
        ]
        ok, reason = decomp.validate_gate_graph(gate, children)
        assert ok is False
        # implementation định dạng title bằng repr (có quotes) — assert theo
        # semantic của spec: prefix "child ", có title, suffix chuẩn.
        assert reason.startswith("child ")
        assert reason.endswith("lacks architect spec/plan parent")
        assert "Implement X" in reason

    def test_non_architect_child_without_architect_parent_rejected(self):
        # any non-architect child (review/research/...) must reference the architect
        gate = decomp.TaskGate(level="uncertain", reasons=["api"])
        children = [
            _architect_child(),
            {"title": "Review", "body": "b", "assignee": "reviewer", "parents": []},
        ]
        ok, reason = decomp.validate_gate_graph(gate, children)
        assert ok is False
        assert reason.startswith("child ")
        assert reason.endswith("lacks architect spec/plan parent")
        assert "Review" in reason

    def test_valid_graph_accepted(self):
        gate = decomp.TaskGate(level="dangerous", reasons=["deploy"])
        children = [
            _architect_child(),
            _coding_child(parents=[0]),
            {"title": "Review", "body": "b", "assignee": "reviewer", "parents": [0]},
        ]
        ok, reason = decomp.validate_gate_graph(gate, children)
        assert ok is True
        assert reason == ""

    def test_custom_architect_assignee(self):
        gate = decomp.TaskGate(level="dangerous", reasons=["deploy"])
        children = [
            {"title": "Spec/plan", "body": "b", "assignee": "planner", "parents": []},
            {"title": "Implement", "body": "b", "assignee": "coding", "parents": [0]},
        ]
        ok, reason = decomp.validate_gate_graph(gate, children, architect_assignee="planner")
        assert ok is True
        assert reason == ""
        # default architect_assignee="architect" must NOT accept a "planner" graph
        ok2, _ = decomp.validate_gate_graph(gate, children)
        assert ok2 is False


# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT rule block — AC-7 (exact text from spec GATE FLOW)
# ---------------------------------------------------------------------------

class TestSystemPromptGateRules:

    def test_spec_gate_rule_block_present(self):
        prompt = decomp._SYSTEM_PROMPT
        assert "SPEC/PLAN GATE" in prompt
        assert "GATE ASSESSMENT" in prompt
        assert "Never emit a coding child without the architect parent" in prompt
        assert '"parents": [0]' in prompt          # short example present
        assert "let the architect judge" in prompt  # no silent fan-out escape


# ---------------------------------------------------------------------------
# decompose_task gate flow — AC-2, AC-3, AC-4, AC-5, AC-6 + fail-closed
# ---------------------------------------------------------------------------

class TestDecomposeGateIntegration:

    def test_gated_fanout_false_forces_architect_assignee(self, kanban_home):
        """AC-4: GATED + fanout=false -> assignee ép thành architect, không coding child."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": False,
            "rationale": "single unit",
            "title": "Deploy gateway cẩn thận",
            "body": "spec chi tiết",
            "assignee": "coding",   # LLM sai — gate phải ép architect
        })

        outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok, outcome.reason
        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
            children = _fanout_link_count(conn, tid)
        assert task.assignee == "architect"
        assert children == 0

    def test_gated_fanout_true_missing_architect_rejected(self, kanban_home):
        """AC-2: GATED + fanout=true + thiếu architect child -> ok=False, task ở lại triage."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "dangerous",
            "tasks": [
                {"title": "Deploy", "body": "b", "assignee": "coding", "parents": []},
            ],
        })

        outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok is False
        assert "spec-gate" in outcome.reason
        with kb.connect() as conn:
            root = kb.get_task(conn, tid)
            assert root.status == "triage"
            assert _fanout_link_count(conn, tid) == 0

    def test_gated_fanout_true_coding_without_architect_parent_rejected(self, kanban_home):
        """AC-2: coding child không parent architect -> từ chối toàn bộ, không ghi một phần."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "dangerous",
            "tasks": [
                {"title": "Spec/plan deploy", "body": "GOAL...", "assignee": "architect", "parents": []},
                {"title": "Implement deploy", "body": "Theo spec", "assignee": "coding", "parents": []},  # thiếu [0]
            ],
        })

        outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok is False
        assert "spec-gate" in outcome.reason
        with kb.connect() as conn:
            root = kb.get_task(conn, tid)
            assert root.status == "triage"
            assert _fanout_link_count(conn, tid) == 0

    def test_gated_fanout_true_valid_graph_creates_dependency(self, kanban_home):
        """AC-3: graph hợp lệ -> ok=True; coding child chờ architect parent done (status=todo)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "spec first",
            "tasks": [
                {"title": "Spec/plan cho deploy", "body": "GOAL APPROACH ACCEPTANCE OUT-OF-SCOPE",
                 "assignee": "architect", "parents": []},
                {"title": "Implement deploy", "body": "Theo spec/plan cua architect",
                 "assignee": "coding", "parents": [0]},
            ],
        })

        outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok, outcome.reason
        assert outcome.child_ids and len(outcome.child_ids) == 2
        with kb.connect() as conn:
            root = kb.get_task(conn, tid)
            c0 = kb.get_task(conn, outcome.child_ids[0])
            c1 = kb.get_task(conn, outcome.child_ids[1])
            arch_to_coding = conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
                (outcome.child_ids[0], outcome.child_ids[1]),
            ).fetchone()[0]
            linked_children = _fanout_link_count(conn, tid)
        assert root.status == "todo"
        assert c0.assignee == "architect"
        assert c0.status == "ready"    # architect chạy ngay
        assert c1.assignee == "coding"
        assert c1.status == "todo"     # block tới khi architect done ("đã duyệt")
        assert arch_to_coding == 1     # dependency architect -> coding tồn tại trong DB
        assert linked_children == 2    # root được link dưới cả 2 child

    def test_safe_task_fanout_unchanged(self, kanban_home):
        """AC-5: safe task giữ nguyên fanout hiện tại, không bắt buộc architect."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="fix typo trong README", body="đổi câu chữ", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "test split",
            "tasks": [
                {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
                {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
            ],
        })

        outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok, outcome.reason
        assert len(outcome.child_ids) == 2
        with kb.connect() as conn:
            root = kb.get_task(conn, tid)
            c0 = kb.get_task(conn, outcome.child_ids[0])
            c1 = kb.get_task(conn, outcome.child_ids[1])
        assert root.status == "todo"
        assert c0.assignee == "researcher"
        assert c0.status == "ready"
        assert c1.assignee == "engineer"
        assert c1.status == "todo"

    def test_override_marker_bypasses_gate(self, kanban_home):
        """AC-6: [spec-gate:skip] -> classification safe -> fanout như cũ, không ép architect."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway [spec-gate:skip]", body="có chủ đích", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "override",
            "tasks": [
                {"title": "Deploy", "body": "b", "assignee": "coding", "parents": []},
            ],
        })

        outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok, outcome.reason
        with kb.connect() as conn:
            c0 = kb.get_task(conn, outcome.child_ids[0])
        assert c0.assignee == "coding"   # không bị ép thành architect

    def test_override_marker_logged(self, kanban_home, caplog):
        """AC-6 audit: override marker phải xuất hiện trong log decompose."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway [spec-gate:skip]", body="có chủ đích", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": False,
            "rationale": "single",
            "title": "Deploy",
            "body": "b",
            "assignee": "coding",
        })

        with caplog.at_level(logging.INFO, logger="hermes_cli.kanban_decompose"):
            outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok, outcome.reason
        assert any(
            "spec-gate" in r.message and "override" in r.message.lower()
            for r in caplog.records
        ), "missing spec-gate override audit log"

    def test_config_spec_gate_false_disables_gate(self, kanban_home, caplog):
        """AC-6: kanban.spec_gate=false -> bỏ qua gate + warning log (kill switch)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "bypass",
            "tasks": [
                {"title": "Deploy", "body": "b", "assignee": "coding", "parents": []},
            ],
        })

        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_decompose"):
            outcome = _run_decompose(tid, llm_payload, config={"kanban": {"spec_gate": False}})

        assert outcome.ok, outcome.reason
        with kb.connect() as conn:
            assert _fanout_link_count(conn, tid) == 1
        assert any(
            "spec-gate" in r.message and "DISABLED" in r.message
            for r in caplog.records
        ), "missing spec-gate DISABLED warning log"

    def test_classify_exception_fails_closed(self, kanban_home):
        """FAIL-CLOSED: lỗi classify -> coi như uncertain (GATED), không crash, không ghi child."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": True,
            "rationale": "normal fanout",
            "tasks": [
                {"title": "Deploy", "body": "b", "assignee": "coding", "parents": []},
            ],
        })

        with patch("hermes_cli.kanban_decompose.classify_task", side_effect=RuntimeError("boom")):
            outcome = _run_decompose(tid, llm_payload)

        assert outcome.ok is False
        assert "spec-gate" in outcome.reason
        with kb.connect() as conn:
            root = kb.get_task(conn, tid)
            assert root.status == "triage"
            assert _fanout_link_count(conn, tid) == 0

    def test_gate_assessment_injected_into_user_message(self, kanban_home):
        """GATE FLOW: GATE ASSESSMENT được inject vào user message sau body, trước roster."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="deploy gateway", body="deploy lên prod", triage=True)

        llm_payload = jsonlib.dumps({
            "fanout": False,
            "rationale": "single",
            "title": "Deploy cẩn thận",
            "body": "spec",
            "assignee": "architect",
        })

        captured: dict = {}

        def fake_call_llm(**kwargs):
            captured["messages"] = kwargs.get("messages", [])
            return _fake_aux_response(llm_payload)

        patches = _patch_list_profiles(["orchestrator", "architect"])
        for p in patches:
            p.start()
        try:
            with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm), patch(
                "hermes_cli.kanban_decompose._load_config",
                return_value={"kanban": {}},
            ):
                outcome = decomp.decompose_task(tid, author="me")
        finally:
            for p in patches:
                p.stop()

        assert outcome.ok, outcome.reason
        user_msg = captured["messages"][1]["content"]
        assert "GATE ASSESSMENT" in user_msg
        assert "dangerous" in user_msg
        # position: sau body, trước roster (spec GATE FLOW)
        assert user_msg.index("GATE ASSESSMENT") > user_msg.index("Body:")
        assert user_msg.index("GATE ASSESSMENT") < user_msg.index("Available profiles")
