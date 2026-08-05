"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the
auto-decompose path in the gateway dispatcher loop. Reads the user's
profile roster (with descriptions) and asks the auxiliary LLM to
return a task graph in JSON. Then atomically creates the children,
links them under the root, and flips the root ``triage -> todo``.

The root task stays alive and becomes the parent of every leaf child,
so when the whole graph completes the root wakes back up — its
assignee (the orchestrator profile) gets a chance to judge completion
and add more tasks if the work isn't done yet.

Design notes
------------

* Mirrors the shape of ``hermes_cli/kanban_specify.py``: lazy aux
  client import inside the function, lenient response parse, never
  raises on expected failure modes.

* The system prompt sees the *configured* profile roster — names plus
  descriptions plus the default fallback. Profiles without a
  description are still listed (with a note) so the decomposer can
  match on name as a fallback, but the user has an obvious incentive
  to describe them.

* ``fanout=false`` collapses to the same effect as ``kanban specify``:
  we tighten the body and flip ``triage -> todo`` as a single task,
  no children created. This makes ``decompose`` a strict superset of
  ``specify`` from the user's perspective.

* If the LLM picks an assignee that doesn't exist as a profile, we
  rewrite it to the configured ``default_assignee`` (or the default
  profile if unset). A child task NEVER ends up with ``assignee=None``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

SPEC/PLAN GATE:
  The GATE ASSESSMENT section in the user message shows the task's classification.
  - safe: decompose normally (rules below).
  - dangerous / complex / uncertain (GATED): create EXACTLY ONE architect child
    whose deliverable is a spec + plan (body must contain: Goal, Approach,
    Acceptance criteria, Out of scope). That architect child has NO parents.
    EVERY other child (coding, review, research) MUST list the architect child's
    index in its "parents". Never emit a coding child without the architect parent.
    Example:
      {"title": "Spec/plan cho X", "body": "GOAL... ACCEPTANCE... NON-GOALS... PLAN...", "assignee": "architect", "parents": []},
      {"title": "Implement X", "body": "Theo spec/plan cua architect", "assignee": "coding", "parents": [0]}
  - If GATED but you believe the task is actually safe, do NOT silently fan out:
    create the architect task anyway (let the architect judge).

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}
{gate_assessment}
Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None


@dataclass
class TaskGate:
    """Deterministic classification of a task for the spec/plan gate.

    ``level`` is one of ``safe`` | ``dangerous`` | ``complex`` | ``uncertain``.
    ``reasons`` names the patterns that matched. ``override`` is True only
    for the explicit ``[spec-gate:skip]`` marker (auditable bypass).
    """

    level: str
    reasons: list[str]
    override: bool = False


def is_gated(gate: TaskGate) -> bool:
    """True when a task must go through the spec/plan gate (level != safe)."""
    return gate.level != "safe"


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Dangerous: destructive zones/actions — first match wins (fail-closed).
# Minimum mandatory set from the spec; keep this list maintained when new
# dangerous zones/actions appear.
_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_rx(r"data/"), "path data/"),
    (_rx(r"migrations/"), "path migrations/"),
    (_rx(r"journal\.db"), "journal.db"),
    (_rx(r"trade_plans\.db"), "trade_plans.db"),
    (_rx(r"\w+\.db\b"), "db file"),
    (_rx(r"engine/"), "path engine/"),
    (_rx(r"\bschema"), "schema"),
    (_rx(r"\bpipeline"), "pipeline"),
    (_rx(r"contracts\.md"), "CONTRACTS.md"),
    (_rx(r"\bdeploy"), "deploy"),
    (_rx(r"\bpublish"), "publish"),
    (_rx(r"\brelease"), "release"),
    (_rx(r"\btransfer"), "transfer"),
    (_rx(r"\bpayment"), "payment"),
    (_rx(r"\bsend\b"), "send"),
    (_rx(r"\bmigrate"), "migrate"),
    (_rx(r"\bdrop\b"), "drop"),
    (_rx(r"\btruncate\b"), "truncate"),
    (_rx(r"\bdelete\b"), "delete"),
    (_rx(r"\bremove\b"), "remove"),
    (_rx(r"\brm\b"), "rm"),
    (_rx(r"\bwipe\b"), "wipe"),
    (_rx(r"\bformat\b"), "format"),
    (_rx(r"\bx[oó]a\b"), "xóa"),
    (_rx(r"đặt lệnh"), "đặt lệnh"),
    (_rx(r"sửa lệnh"), "sửa lệnh"),
    (_rx(r"đóng lệnh"), "đóng lệnh"),
    (_rx(r"\bdat lenh\b"), "đặt lệnh"),
    (_rx(r"\bsua lenh\b"), "sửa lệnh"),
    (_rx(r"\bdong lenh\b"), "đóng lệnh"),
    (_rx(r"live trad\w*"), "live trading"),
]

# Complex: needs design / touches many modules.
_COMPLEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_rx(r"\bdesign"), "design"),
    (_rx(r"\barchitecture"), "architecture"),
    (_rx(r"\brefactor"), "refactor"),
    (_rx(r"\bnew module"), "new module"),
    (_rx(r"cross[\s-]module"), "cross-module"),
    (_rx(r">\s*3\s+files\b"), ">3 files"),
    (_rx(r"\bhandoff\b"), "handoff"),
    (_rx(r"\bspec\b"), "spec"),
    (_rx(r"\bplan\b"), "plan"),
    (_rx(r"\bacceptance\b"), "acceptance"),
]

# Uncertain: ambiguous zones — fail closed.
_UNCERTAIN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_rx(r"\bdatabase\b"), "database"),
    (_rx(r"\bexternal service"), "external service"),
    (_rx(r"\bproduction\b"), "production"),
    (_rx(r"\bscheduler\b"), "scheduler"),
    (_rx(r"\bcron\b"), "cron"),
    (_rx(r"\bintegration"), "integration"),
    (_rx(r"\bapi\b"), "api"),
]

_OVERRIDE_MARKER = "[spec-gate:skip]"


def classify_task(title: str, body: str) -> TaskGate:
    """Classify a task deterministically (no LLM).

    Matching is case-insensitive over title + body. Priority:
    override marker > dangerous > complex > uncertain > empty-body > safe.
    An empty body never rescues a title that matches a gate pattern;
    a body-less task with no pattern match is classified uncertain
    (fail-closed per spec: \"Body rỗng/thiếu body -> uncertain — không
    đánh giá được thì gate\"). The override marker stays the only
    auditable bypass.
    """
    title = (title or "").strip()
    body = (body or "").strip()
    text = f"{title}\n{body}".lower()

    if _OVERRIDE_MARKER in text:
        return TaskGate("safe", ["override marker"], override=True)
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            return TaskGate("dangerous", [reason])
    for pattern, reason in _COMPLEX_PATTERNS:
        if pattern.search(text):
            return TaskGate("complex", [reason])
    for pattern, reason in _UNCERTAIN_PATTERNS:
        if pattern.search(text):
            return TaskGate("uncertain", [reason])
    if not body:
        return TaskGate("uncertain", ["empty body (no spec)"])
    return TaskGate("safe", ["no gate patterns matched"])


def validate_gate_graph(
    gate: TaskGate,
    children: list[dict],
    *,
    architect_assignee: str = "architect",
) -> tuple[bool, str]:
    """Structural check for a gated decomposition graph (see spec).

    Only meaningful when ``is_gated(gate)``: the graph must contain
    exactly one architect child with no parents, and every other child
    must list that architect child's index in its ``parents``. Any
    violation rejects the whole decomposition (fail-closed, no partial
    DB writes). Non-gated input always validates.
    """
    if not is_gated(gate):
        return True, ""
    arch_idx = None
    for i, child in enumerate(children):
        if (child.get("assignee") or "") == architect_assignee:
            if arch_idx is not None:
                return False, "multiple architect spec/plan children"
            arch_idx = i
    if arch_idx is None:
        return False, "missing architect spec/plan child"
    if children[arch_idx].get("parents"):
        return False, "architect child must have no parents"
    for i, child in enumerate(children):
        if i == arch_idx:
            continue
        parents = child.get("parents") or []
        if not isinstance(parents, list) or arch_idx not in parents:
            title = child.get("title") or str(i)
            return False, f"child {title} lacks architect spec/plan parent"
    return True, ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_json_blob(raw: str) -> Optional[dict]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(val, dict):
        return None
    return val


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return (
        os.environ.get("HERMES_PROFILE")
        or os.environ.get("USER")
        or "decomposer"
    )


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_orchestrator_profile(cfg: dict) -> str:
    """Resolve which profile owns the root/orchestration task after fan-out.

    Falls back to the active default profile when ``kanban.orchestrator_profile``
    is unset, so a task is never stranded for lack of an orchestrator.
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("orchestrator_profile") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    # Fall back to the active default profile.
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _resolve_default_assignee(cfg: dict) -> str:
    """Resolve which profile catches child tasks the orchestrator can't route."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get("default_assignee") or "").strip()
    if explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """Return (roster_for_prompt, valid_assignee_names).

    Each roster entry is ``{name, description, has_description}``. The
    valid-set is used after the LLM responds to rewrite invalid
    assignees to the default fallback.
    """
    roster: list[dict] = []
    valid: set[str] = set()
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return roster, valid
    for p in all_profiles:
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
        valid.add(p.name)
    return roster, valid


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    lines = []
    for entry in roster:
        tag = "" if entry["has_description"] else " ⚠ undescribed"
        lines.append(f"  - {entry['name']}{tag}: {entry['description']}")
    return "\n".join(lines)


def _normalize_assignee_choice(
    assignee: object,
    *,
    default_assignee: str,
    valid_names: set[str],
) -> str:
    """Return a valid assignee, falling back to ``default_assignee``.

    Fan-out children and the single-task fallback should share the same
    routing guarantee: promoted work must not be left unassigned.
    """
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    if chosen not in valid_names:
        return default_assignee
    return chosen


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks.

    Returns an outcome describing what happened. Never raises for
    expected failure modes (task not in triage, no aux client
    configured, API error, malformed response, decomposer returned
    fanout=true with empty task list) — those surface via ``ok=False``.
    """
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return DecomposeOutcome(
            task_id, False, f"task is not in triage (status={task.status!r})"
        )

    cfg = _load_config()
    orchestrator = _resolve_orchestrator_profile(cfg)
    default_assignee = _resolve_default_assignee(cfg)
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    auto_promote = bool(kanban_cfg.get("auto_promote_children", True))
    spec_gate_enabled = bool(kanban_cfg.get("spec_gate", True))
    roster, valid_names = _build_roster()

    try:
        from agent.auxiliary_client import call_llm  # type: ignore
    except Exception as exc:
        logger.debug("decompose: auxiliary client import failed: %s", exc)
        return DecomposeOutcome(task_id, False, "auxiliary client unavailable")

    # SPEC/PLAN GATE — deterministic classification before the LLM call.
    # GATED (dangerous/complex/uncertain) tasks may only fan out into a
    # graph rooted at an architect spec/plan child; validation happens
    # structurally below so an LLM drift can never silently bypass it.
    gate: Optional[TaskGate] = None
    gate_assessment = ""
    if not spec_gate_enabled:
        logger.warning(
            "spec-gate DISABLED by config — dangerous tasks may fan out directly",
        )
    else:
        try:
            gate = classify_task(task.title or "", task.body or "")
        except Exception as exc:
            logger.warning(
                "spec-gate: classification failed for %s (%s) — failing closed",
                task_id, exc,
            )
            gate = TaskGate("uncertain", ["classification error"])
        if gate.override:
            logger.info("spec-gate: override marker on task %s", task_id)
        logger.info(
            "spec-gate: task %s classified level=%s reasons=%s",
            task_id, gate.level, gate.reasons,
        )
        gate_assessment = f"GATE ASSESSMENT: level={gate.level} reasons={gate.reasons}"

    user_msg = _USER_TEMPLATE.format(
        task_id=task.id,
        title=_truncate(task.title or "", 400),
        body=_truncate(task.body or "(no body)", 4000),
        gate_assessment=gate_assessment,
        roster=_format_roster(roster),
        default_assignee=default_assignee,
    )

    try:
        # Route through call_llm so auxiliary.kanban_decomposer.* config
        # (provider/model/base_url, extra_body, reasoning_effort, retries)
        # all apply — the previous direct client.chat.completions.create()
        # path dropped auxiliary.<task>.extra_body entirely (#35566).
        resp = call_llm(
            task="kanban_decomposer",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4000,
            timeout=timeout or 180,
        )
    except Exception as exc:
        logger.info(
            "decompose: API call failed for %s (%s)", task_id, exc,
        )
        return DecomposeOutcome(task_id, False, f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _extract_json_blob(raw)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    fanout = bool(parsed.get("fanout"))
    audit_author = author or _profile_author()

    if not fanout:
        # Fall back to single-task spec promotion (same effect as specify).
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = new_title.strip() if isinstance(new_title, str) and new_title.strip() else None
        body_val = new_body if isinstance(new_body, str) and new_body.strip() else None
        assignee_val = None
        if gate is not None and is_gated(gate):
            # GATED single-task promotion: force the architect profile so a
            # spec/plan gets produced; never hand a gated task straight to
            # coding or any other implementer.
            assignee_val = "architect"
            logger.info(
                "spec-gate: task %s level=%s fanout=false — forcing assignee=%s",
                task_id, gate.level, assignee_val,
            )
        elif not task.assignee:
            assignee_val = _normalize_assignee_choice(
                parsed.get("assignee"),
                default_assignee=default_assignee,
                valid_names=valid_names,
            )
        if title_val is None and body_val is None:
            return DecomposeOutcome(
                task_id, False, "decomposer returned fanout=false with no title/body",
            )
        with kb.connect_closing() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title=title_val,
                body=body_val,
                assignee=assignee_val,
                author=audit_author,
            )
        if not ok:
            return DecomposeOutcome(
                task_id, False, "task moved out of triage before promotion",
            )
        return DecomposeOutcome(
            task_id, True, "single task (no fanout)",
            fanout=False, new_title=title_val,
        )

    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(
            task_id, False, "decomposer returned fanout=true with empty tasks list",
        )

    # Rewrite invalid assignees to the default fallback. Never leave a
    # task with assignee=None — the user explicitly does not want that.
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}] is not an object",
            )
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}].title is missing or empty",
            )
        body = entry.get("body")
        if not isinstance(body, str):
            body = ""
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee,
            default_assignee=default_assignee,
            valid_names=valid_names,
        )
        if (
            isinstance(assignee, str)
            and assignee.strip()
            and assignee.strip() not in valid_names
        ):
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # Clean parent indices: drop non-int and out-of-range.
        clean_parents = [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx]
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
        })

    # SPEC/PLAN GATE — structural validation before any DB write. If the
    # LLM drifted from the gated graph rules (missing architect child, or
    # a child without the architect parent), reject the whole decomposition
    # and leave the task in triage. Fail-closed: never write a partial
    # graph that lets coding children run before a spec/plan is approved.
    if gate is not None and is_gated(gate):
        ok, reason = validate_gate_graph(gate, children)
        if not ok:
            logger.info(
                "spec-gate: task %s graph rejected: %s", task_id, reason,
            )
            return DecomposeOutcome(task_id, False, f"spec-gate: {reason}")

    try:
        with kb.connect_closing() as conn:
            child_ids = kb.decompose_triage_task(
                conn,
                task_id,
                root_assignee=orchestrator,
                children=children,
                author=audit_author,
                auto_promote=auto_promote,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")

    if child_ids is None:
        return DecomposeOutcome(
            task_id, False, "task moved out of triage before decomposition",
        )

    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children",
        fanout=True, child_ids=child_ids,
    )


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kb.connect_closing() as conn:
        rows = kb.list_tasks(
            conn,
            status="triage",
            tenant=tenant,
            limit=1000,
        )
    return [row.id for row in rows]
