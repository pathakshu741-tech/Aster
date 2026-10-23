"""
Shoko Missions: long-running goals with a saved plan, real progress tracking,
blockers, decisions, an activity timeline and a "Continue Mission" briefing.

This module is self-contained. The main app only needs to:
  * call open_missions(...) from a rail button,
  * add get_context_block(...) to the system prompt,
  * optionally expose the three mission_* tools (tool_log).

Storage: ~/.ava_assistant/missions.json (atomic write + .bak copy of the last good file).
"""
import json
import os
import re
import shutil
import threading
import tkinter as tk
import uuid
from datetime import datetime, date
from tkinter import messagebox

import ollama

MODEL = "llama3.2:3b"
BASE_DIR = os.path.join(os.path.expanduser("~"), ".ava_assistant")
MISSIONS_FILE = os.path.join(BASE_DIR, "missions.json")

MAX_MILESTONES = 6
MAX_TASKS = 7
MAX_ACTIVITY = 300
CONTEXT_MAX_CHARS = 2200

TASK_STATUSES = ("todo", "doing", "paused", "done")
MISSION_STATUSES = ("draft", "active", "paused", "completed", "archived")

# Same palette as dashboard.py so the window matches the rest of the app.
P = {
    "bg": "#0a0c1a", "rail": "#0d1024", "rail_active": "#1c1f45",
    "card": "#12152b", "edge": "#1c2040", "row": "#171a35",
    "text": "#eceeff", "muted": "#8a8fb8", "accent": "#7c6cff",
    "accent2": "#a99cff", "green": "#4ade80", "green_bg": "#14281f",
    "red": "#ff8f8f", "amber": "#e0b050",
}
FONT = "Segoe UI"

_lock = threading.RLock()
_load_state = {"error": None, "readonly": False}
_ui = {"window": None, "refresh": None, "busy": False}


# =====================================================================
# Small helpers
# =====================================================================
def _now():
    return datetime.now().isoformat(timespec="seconds")


def _new_id():
    return uuid.uuid4().hex[:8]


def _s(value, limit=2000):
    return str(value).strip()[:limit] if value is not None else ""


def _str_list(value, max_items, limit):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        text = _s(v, limit)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _ago(iso):
    try:
        secs = (datetime.now() - datetime.fromisoformat(iso)).total_seconds()
    except Exception:
        return "no activity yet"
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


def _deadline_text(m):
    if not m.get("deadline"):
        return ""
    try:
        left = (date.fromisoformat(m["deadline"]) - date.today()).days
    except ValueError:
        return ""
    if left < 0:
        return f"overdue by {-left} d"
    if left == 0:
        return "due today"
    return f"{left} d left"


def friendly_error(e):
    msg = str(e)
    low = msg.lower()
    if any(k in low for k in ("connect", "refused", "10061", "winerror", "timed out", "unreachable")):
        return "Couldn't reach Ollama. Make sure it is running, then try again."
    if "not found" in low or "404" in low:
        return f"The model '{MODEL}' isn't installed. Run:  ollama pull {MODEL}"
    return msg or "Something went wrong."


# =====================================================================
# Normalisation (tolerates missing / malformed fields)
# =====================================================================
def _norm_task(t):
    if not isinstance(t, dict):
        return None
    title = _s(t.get("title"), 200)
    if not title:
        return None
    deps = t.get("depends_on")
    return {
        "id": _s(t.get("id"), 16) or _new_id(),
        "title": title,
        "notes": _s(t.get("notes")),
        "effort": _s(t.get("effort"), 40),
        "status": t.get("status") if t.get("status") in TASK_STATUSES else "todo",
        "depends_on": [d for d in deps if isinstance(d, str)] if isinstance(deps, list) else [],
        "created_at": _s(t.get("created_at")) or _now(),
        "completed_at": _s(t.get("completed_at")),
    }


def _norm_milestone(ms):
    if not isinstance(ms, dict):
        return None
    title = _s(ms.get("title"), 200)
    if not title:
        return None
    tasks = [x for x in (_norm_task(t) for t in (ms.get("tasks") or [])) if x]
    return {"id": _s(ms.get("id"), 16) or _new_id(), "title": title,
            "description": _s(ms.get("description")), "tasks": tasks}


def _norm_mission(m):
    if not isinstance(m, dict):
        return None
    status = m.get("status") if m.get("status") in MISSION_STATUSES else "active"
    now = _now()
    out = {
        "id": _s(m.get("id"), 16) or _new_id(),
        "title": _s(m.get("title"), 120) or "Untitled mission",
        "description": _s(m.get("description")),
        "goal": _s(m.get("goal"), 4000),
        "interpretation": _s(m.get("interpretation")),
        "success_criteria": _str_list(m.get("success_criteria"), 12, 300),
        "deadline": _s(m.get("deadline"), 10),
        "experience": _s(m.get("experience"), 300),
        "constraints": _s(m.get("constraints"), 500),
        "status": status,
        "first_action": _s(m.get("first_action"), 300),
        "chat_id": _s(m.get("chat_id"), 32),
        "milestones": [x for x in (_norm_milestone(ms) for ms in (m.get("milestones") or [])) if x],
        "risks": [], "blockers": [], "notes": [], "activity": [],
        "pending_revision": m.get("pending_revision") if isinstance(m.get("pending_revision"), dict) else None,
        "created_at": _s(m.get("created_at")) or now,
        "updated_at": _s(m.get("updated_at")) or now,
    }
    for r in (m.get("risks") or []):
        if isinstance(r, dict) and _s(r.get("text")):
            out["risks"].append({"text": _s(r.get("text"), 300), "handling": _s(r.get("handling"), 300)})
    for b in (m.get("blockers") or []):
        if isinstance(b, dict) and _s(b.get("text")):
            out["blockers"].append({
                "id": _s(b.get("id"), 16) or _new_id(), "text": _s(b.get("text"), 500),
                "attempted": _s(b.get("attempted"), 500), "task_id": _s(b.get("task_id"), 16),
                "status": "resolved" if b.get("status") == "resolved" else "open",
                "resolution": _s(b.get("resolution"), 500),
                "created_at": _s(b.get("created_at")) or now, "resolved_at": _s(b.get("resolved_at")),
            })
    for n in (m.get("notes") or []):
        if isinstance(n, dict) and _s(n.get("text")):
            out["notes"].append({
                "id": _s(n.get("id"), 16) or _new_id(), "text": _s(n.get("text"), 1500),
                "kind": n.get("kind") if n.get("kind") in ("note", "decision", "progress") else "note",
                "created_at": _s(n.get("created_at")) or now,
            })
    for a in (m.get("activity") or []):
        if isinstance(a, dict) and _s(a.get("text")):
            out["activity"].append({"id": _s(a.get("id"), 16) or _new_id(), "ts": _s(a.get("ts")) or now,
                                    "type": _s(a.get("type"), 40) or "event", "text": _s(a.get("text"), 400)})
    return out


# =====================================================================
# Persistence (atomic write, .bak, corruption never deletes data)
# =====================================================================
def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path, payload):
    os.makedirs(BASE_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _extract_list(raw):
    if isinstance(raw, dict):
        raw = raw.get("missions", [])
    return raw if isinstance(raw, list) else []


def load_all():
    """Returns all missions. A damaged file is renamed (never deleted) and the .bak is tried."""
    with _lock:
        _load_state["error"] = None
        _load_state["readonly"] = False
        if not os.path.exists(MISSIONS_FILE):
            return []
        try:
            raw = _read_json(MISSIONS_FILE)
        except ValueError as e:                       # file exists but isn't valid JSON
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            kept = f"{MISSIONS_FILE}.corrupt-{stamp}"
            try:
                os.replace(MISSIONS_FILE, kept)
            except OSError:
                _load_state["readonly"] = True
                _load_state["error"] = f"missions.json is damaged ({e}) and couldn't be moved aside."
                return []
            try:
                raw = _read_json(MISSIONS_FILE + ".bak")
                _atomic_write(MISSIONS_FILE, {"version": 1, "missions": _extract_list(raw)})
                _load_state["error"] = (f"missions.json was damaged, so the last backup was restored. "
                                        f"The damaged file was kept as {os.path.basename(kept)}.")
            except Exception:
                _load_state["error"] = (f"missions.json was damaged and no backup could be read. "
                                        f"Your data was kept as {os.path.basename(kept)}.")
                return []
        except OSError as e:                          # locked / unreadable: don't touch anything
            _load_state["readonly"] = True
            _load_state["error"] = f"Couldn't read missions.json ({e}). Nothing will be overwritten."
            return []
        missions = [x for x in (_norm_mission(m) for m in _extract_list(raw)) if x]
        return missions


def save_all(missions):
    with _lock:
        if _load_state["readonly"]:
            raise RuntimeError("The missions file couldn't be read, so saving is disabled to protect it.")
        if os.path.exists(MISSIONS_FILE):
            try:
                shutil.copy2(MISSIONS_FILE, MISSIONS_FILE + ".bak")
            except OSError:
                pass
        _atomic_write(MISSIONS_FILE, {"version": 1, "missions": missions})


def get_mission(mid):
    for m in load_all():
        if m["id"] == mid:
            return m
    return None


def update_mission(mid, fn):
    """Loads fresh data, applies fn(mission) in memory, then saves. If fn raises, nothing is saved."""
    with _lock:
        missions = load_all()
        for m in missions:
            if m["id"] == mid:
                fn(m)
                m["updated_at"] = _now()
                save_all(missions)
                return m
    raise KeyError("That mission no longer exists.")


def _log(m, kind, text):
    m["activity"].append({"id": _new_id(), "ts": _now(), "type": kind, "text": _s(text, 400)})
    if len(m["activity"]) > MAX_ACTIVITY:
        m["activity"] = m["activity"][-MAX_ACTIVITY:]


# =====================================================================
# Queries and progress (single, transparent calculation)
# =====================================================================
def all_tasks(m):
    return [(ms, t) for ms in m["milestones"] for t in ms["tasks"]]


def find_task(m, tid):
    for ms in m["milestones"]:
        for i, t in enumerate(ms["tasks"]):
            if t["id"] == tid:
                return ms, t, i
    return None, None, -1


def find_milestone(m, msid):
    for i, ms in enumerate(m["milestones"]):
        if ms["id"] == msid:
            return ms, i
    return None, -1


def progress(m):
    """Progress = completed tasks / all tasks. Every task counts equally; 'doing' counts as 0.
    Returns (done, total, percent). Milestones show their own done/total separately."""
    tasks = all_tasks(m)
    done = sum(1 for _, t in tasks if t["status"] == "done")
    total = len(tasks)
    return done, total, (round(100 * done / total) if total else 0)


def milestone_state(ms):
    total = len(ms["tasks"])
    done = sum(1 for t in ms["tasks"] if t["status"] == "done")
    if total and done == total:
        return "done"
    if any(t["status"] in ("doing", "done") for t in ms["tasks"]):
        return "in progress"
    return "not started"


def current_milestone(m):
    for ms in m["milestones"]:
        if milestone_state(ms) != "done":
            return ms
    return None


def unmet_deps(m, t):
    tasks = {x["id"]: x for _, x in all_tasks(m)}
    return [d for d in t["depends_on"] if d in tasks and tasks[d]["status"] != "done"]


def open_blockers(m):
    return [b for b in m["blockers"] if b["status"] == "open"]


def effective_status(m):
    if m["status"] == "active" and open_blockers(m):
        return "blocked"
    return m["status"]


def next_task(m):
    """Returns (milestone, task, waiting). Order: task in progress, then first ready task,
    then first unfinished task whose prerequisites aren't done (waiting=True)."""
    pending = [(ms, t) for ms, t in all_tasks(m) if t["status"] not in ("done", "paused")]
    for ms, t in pending:
        if t["status"] == "doing":
            return ms, t, False
    for ms, t in pending:
        if not unmet_deps(m, t):
            return ms, t, False
    if pending:
        return pending[0][0], pending[0][1], True
    return None, None, False


def label_map(m):
    return {f"{i}.{j}": t["id"] for i, ms in enumerate(m["milestones"], 1)
            for j, t in enumerate(ms["tasks"], 1)}


def label_of(m, tid):
    for k, v in label_map(m).items():
        if v == tid:
            return k
    return "?"


def would_cycle(m, tid, deps):
    tasks = {x["id"]: x for _, x in all_tasks(m)}
    seen, stack = set(), list(deps)
    while stack:
        x = stack.pop()
        if x == tid:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(tasks.get(x, {}).get("depends_on", []))
    return False


# =====================================================================
# Mission mutations (each one logs a timeline entry)
# =====================================================================
def create_mission_from_plan(goal, plan, deadline="", experience="", constraints=""):
    m = _norm_mission({
        "title": plan.get("title") or goal[:60], "goal": goal, "description": plan.get("interpretation", ""),
        "interpretation": plan.get("interpretation", ""), "success_criteria": plan.get("success_criteria", []),
        "deadline": deadline, "experience": experience, "constraints": constraints,
        "status": "draft", "first_action": plan.get("first_action", ""),
        "milestones": plan.get("milestones", []), "risks": plan.get("risks", []),
    })
    _log(m, "mission_created", "Mission created and plan generated (waiting for approval).")
    with _lock:
        missions = load_all()
        missions.append(m)
        save_all(missions)
    return m["id"]


def approve_plan(mid):
    def fn(m):
        if m["status"] != "draft":
            raise ValueError("This plan is already approved.")
        m["status"] = "active"
        _log(m, "plan_approved", "Plan approved. Mission is now active.")
    return update_mission(mid, fn)


def set_mission_status(mid, status, text):
    def fn(m):
        m["status"] = status
        _log(m, "status_" + status, text)
    return update_mission(mid, fn)


def complete_mission(mid):
    def fn(m):
        done, total, _ = progress(m)
        if total == 0 or done < total:
            raise ValueError(f"{total - done} task(s) still aren't done, so the mission can't be completed yet.")
        m["status"] = "completed"
        _log(m, "mission_completed", "Mission marked complete by you.")
    return update_mission(mid, fn)


def delete_mission(mid):
    with _lock:
        missions = [m for m in load_all() if m["id"] != mid]
        save_all(missions)


def edit_mission(mid, title, goal, criteria, deadline, constraints):
    def fn(m):
        m["title"] = _s(title, 120) or m["title"]
        m["goal"] = _s(goal, 4000) or m["goal"]
        m["success_criteria"] = _str_list(criteria.splitlines(), 12, 300)
        m["deadline"] = deadline
        m["constraints"] = _s(constraints, 500)
        _log(m, "mission_edited", "Mission details edited.")
    return update_mission(mid, fn)


def link_chat(mid, chat_id):
    return update_mission(mid, lambda m: m.update(chat_id=chat_id))


def add_milestone(mid, title, description=""):
    def fn(m):
        m["milestones"].append({"id": _new_id(), "title": _s(title, 200), "description": _s(description), "tasks": []})
        _log(m, "milestone_added", f"Milestone added: {_s(title, 80)}")
    return update_mission(mid, fn)


def edit_milestone(mid, msid, title, description):
    def fn(m):
        ms, _ = find_milestone(m, msid)
        if ms is None:
            raise KeyError("Milestone not found.")
        ms["title"], ms["description"] = _s(title, 200) or ms["title"], _s(description)
        _log(m, "milestone_edited", f"Milestone edited: {ms['title'][:80]}")
    return update_mission(mid, fn)


def delete_milestone(mid, msid):
    def fn(m):
        ms, i = find_milestone(m, msid)
        if ms is None:
            raise KeyError("Milestone not found.")
        gone = {t["id"] for t in ms["tasks"]}
        m["milestones"].pop(i)
        for _, t in all_tasks(m):
            t["depends_on"] = [d for d in t["depends_on"] if d not in gone]
        _log(m, "milestone_deleted", f"Milestone deleted: {ms['title'][:80]}")
    return update_mission(mid, fn)


def move_milestone(mid, msid, delta):
    def fn(m):
        ms, i = find_milestone(m, msid)
        j = i + delta
        if ms is None or not 0 <= j < len(m["milestones"]):
            return
        m["milestones"][i], m["milestones"][j] = m["milestones"][j], m["milestones"][i]
    return update_mission(mid, fn)


def add_task(mid, msid, title, notes="", effort=""):
    def fn(m):
        ms, _ = find_milestone(m, msid)
        if ms is None:
            raise KeyError("Milestone not found.")
        ms["tasks"].append(_norm_task({"title": title, "notes": notes, "effort": effort}))
        _log(m, "task_added", f"Task added: {_s(title, 80)}")
    return update_mission(mid, fn)


def edit_task(mid, tid, title, notes, effort, depends_on):
    def fn(m):
        _, t, _ = find_task(m, tid)
        if t is None:
            raise KeyError("Task not found.")
        if tid in depends_on or would_cycle(m, tid, depends_on):
            raise ValueError("Those prerequisites would create a loop.")
        t["title"] = _s(title, 200) or t["title"]
        t["notes"], t["effort"], t["depends_on"] = _s(notes), _s(effort, 40), list(depends_on)
        _log(m, "task_edited", f"Task edited: {t['title'][:80]}")
    return update_mission(mid, fn)


def delete_task(mid, tid):
    def fn(m):
        ms, t, i = find_task(m, tid)
        if t is None:
            raise KeyError("Task not found.")
        ms["tasks"].pop(i)
        for _, x in all_tasks(m):
            x["depends_on"] = [d for d in x["depends_on"] if d != tid]
        for b in m["blockers"]:
            if b["task_id"] == tid:
                b["task_id"] = ""
        _log(m, "task_deleted", f"Task deleted: {t['title'][:80]}")
    return update_mission(mid, fn)


def move_task(mid, tid, delta):
    def fn(m):
        ms, t, i = find_task(m, tid)
        j = i + delta
        if t is None or not 0 <= j < len(ms["tasks"]):
            return
        ms["tasks"][i], ms["tasks"][j] = ms["tasks"][j], ms["tasks"][i]
    return update_mission(mid, fn)


_TASK_VERBS = {"doing": "Started", "done": "Completed", "paused": "Paused", "todo": "Reopened"}


def set_task_status(mid, tid, status):
    def fn(m):
        _, t, _ = find_task(m, tid)
        if t is None:
            raise KeyError("Task not found.")
        old = t["status"]
        t["status"] = status
        if status == "done":
            t["completed_at"] = _now()
        elif old == "done":
            t["completed_at"] = ""
        _log(m, "task_" + status, f"{_TASK_VERBS[status]}: {t['title'][:100]}")
        if status != "done" and m["status"] == "completed":
            m["status"] = "active"
            _log(m, "status_active", "Mission reopened because a task was reopened.")
    return update_mission(mid, fn)


def add_note(mid, text, kind="note"):
    def fn(m):
        m["notes"].append({"id": _new_id(), "text": _s(text, 1500), "kind": kind, "created_at": _now()})
        label = {"decision": "Decision recorded", "progress": "Progress update", "note": "Note added"}[kind]
        _log(m, kind, f"{label}: {_s(text, 120)}")
    return update_mission(mid, fn)


def add_blocker(mid, text, attempted="", task_id=""):
    def fn(m):
        m["blockers"].append({"id": _new_id(), "text": _s(text, 500), "attempted": _s(attempted, 500),
                              "task_id": task_id, "status": "open", "resolution": "",
                              "created_at": _now(), "resolved_at": ""})
        _log(m, "blocker_added", f"Blocker added: {_s(text, 120)}")
    return update_mission(mid, fn)


def resolve_blocker(mid, bid, resolution):
    def fn(m):
        for b in m["blockers"]:
            if b["id"] == bid:
                b["status"], b["resolution"], b["resolved_at"] = "resolved", _s(resolution, 500), _now()
                _log(m, "blocker_resolved", f"Blocker resolved: {b['text'][:100]}")
                return
        raise KeyError("Blocker not found.")
    return update_mission(mid, fn)


# =====================================================================
# AI: planning, revisions, briefing
# =====================================================================
def _parse_json(text):
    text = (text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except ValueError:
        found = re.search(r"\{.*\}", text, re.DOTALL)
        if found:
            return json.loads(found.group(0))
        raise


def _chat_json(system, user, max_tokens=1800):
    for _ in range(2):
        resp = ollama.chat(
            model=MODEL, format="json", keep_alive="30m",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0.3, "num_ctx": 4096, "num_predict": max_tokens},
        )
        try:
            return _parse_json(resp["message"]["content"])
        except ValueError:
            continue
    raise ValueError("The model didn't return a usable answer. Please try again.")


PLAN_SYSTEM = (
    "You are the planning engine of a personal AI assistant. Turn the user's goal into a realistic, "
    "ordered plan. Reply with ONLY a JSON object:\n"
    '{"title": "short title", "interpretation": "1-2 sentences", '
    '"success_criteria": ["measurable criterion"], '
    '"milestones": [{"title": "...", "tasks": [{"title": "specific action", "effort": "e.g. 2 hours", '
    '"depends_on": ["exact title of an EARLIER task"]}]}], '
    '"risks": [{"obstacle": "...", "handling": "..."}], "first_action": "one concrete first step"}\n'
    "Rules: 3-5 milestones, 3-6 tasks each, tasks must be concrete actions, depends_on only when truly needed "
    "and only naming earlier tasks exactly, 2-4 risks, 2-5 success criteria. Never mark anything as done. "
    'If the goal is too vague to plan, reply ONLY {"clarifying_question": "one short question"}.'
)


def sanitize_plan(raw):
    if not isinstance(raw, dict):
        raise ValueError("The model's answer wasn't a plan.")
    plan = {
        "title": _s(raw.get("title"), 80), "interpretation": _s(raw.get("interpretation"), 400),
        "success_criteria": _str_list(raw.get("success_criteria"), 6, 250),
        "first_action": _s(raw.get("first_action"), 300),
        "clarifying_question": _s(raw.get("clarifying_question"), 300),
        "milestones": [], "risks": [],
    }
    by_title = {}
    for ms in (raw.get("milestones") or [])[:MAX_MILESTONES]:
        if not isinstance(ms, dict) or not _s(ms.get("title")):
            continue
        tasks = []
        for t in (ms.get("tasks") or [])[:MAX_TASKS]:
            if isinstance(t, str):
                t = {"title": t}
            if not isinstance(t, dict) or not _s(t.get("title")):
                continue
            deps = t.get("depends_on") or []
            deps = [deps] if isinstance(deps, str) else deps
            dep_ids = [by_title[_s(d).lower()] for d in deps if isinstance(d, str) and _s(d).lower() in by_title]
            task = _norm_task({"title": t.get("title"), "effort": t.get("effort"), "depends_on": dep_ids})
            by_title[task["title"].lower()] = task["id"]
            tasks.append(task)
        plan["milestones"].append({"id": _new_id(), "title": _s(ms.get("title"), 200),
                                   "description": _s(ms.get("description")), "tasks": tasks})
    for r in (raw.get("risks") or [])[:5]:
        if isinstance(r, str):
            r = {"obstacle": r}
        if isinstance(r, dict) and _s(r.get("obstacle")):
            plan["risks"].append({"text": _s(r.get("obstacle"), 300), "handling": _s(r.get("handling"), 300)})
    return plan


def generate_plan(goal, deadline="", experience="", constraints="", desired=""):
    user = (f"Today: {date.today().isoformat()}\nGoal: {goal}\nDeadline: {deadline or 'none given'}\n"
            f"My level/experience: {experience or 'not given'}\nTime/constraints: {constraints or 'not given'}\n"
            f"Desired outcome: {desired or 'not given'}")
    plan = sanitize_plan(_chat_json(PLAN_SYSTEM, user))
    if not plan["milestones"] and not plan["clarifying_question"]:
        raise ValueError("Shoko couldn't produce a usable plan. Try again, or add more detail to the goal.")
    if desired and not plan["success_criteria"]:
        plan["success_criteria"] = [desired]
    return plan


REVISE_SYSTEM = (
    "You revise an existing mission plan. Propose the SMALLEST set of changes that fixes the problem. "
    "Never touch tasks whose status is done. Reply with ONLY a JSON object:\n"
    '{"reason": "why the plan should change", '
    '"add_tasks": [{"milestone": "milestone id or title", "title": "...", "effort": "..."}], '
    '"modify_tasks": [{"id": "task id", "new_title": "...", "notes": "..."}], '
    '"remove_tasks": [{"id": "task id"}], '
    '"new_milestones": [{"title": "...", "tasks": [{"title": "...", "effort": "..."}]}], '
    '"risks": [{"obstacle": "...", "handling": "..."}]}\n'
    "Use empty lists for anything that doesn't change. Use the exact ids given."
)


def _tasks_for_prompt(m):
    lines = []
    for ms in m["milestones"]:
        lines.append(f"Milestone [{ms['id']}] {ms['title']}")
        for t in ms["tasks"]:
            lines.append(f"  Task [{t['id']}] ({t['status']}) {t['title']}")
    return "\n".join(lines) or "(no milestones yet)"


def _resolve_task(m, ref):
    ref = _s(ref, 200).lower()
    for _, t in all_tasks(m):
        if t["id"].lower() == ref or t["title"].lower() == ref:
            return t
    return None


def _resolve_milestone(m, ref):
    ref = _s(ref, 200).lower()
    for ms in m["milestones"]:
        if ms["id"].lower() == ref or ms["title"].lower() == ref:
            return ms
    return current_milestone(m) or (m["milestones"][-1] if m["milestones"] else None)


def sanitize_revision(m, raw):
    if not isinstance(raw, dict):
        raise ValueError("The model's answer wasn't a revision.")
    rev = {"id": _new_id(), "created_at": _now(), "reason": _s(raw.get("reason"), 400) or "Plan adjustment",
           "add_tasks": [], "modify_tasks": [], "remove_tasks": [], "new_milestones": [], "risks": [],
           "kept_done": progress(m)[0]}
    removed = set()
    for r in (raw.get("remove_tasks") or [])[:8]:
        t = _resolve_task(m, r.get("id") if isinstance(r, dict) else r)
        if t and t["status"] != "done" and t["id"] not in removed:
            removed.add(t["id"])
            rev["remove_tasks"].append({"task_id": t["id"], "title": t["title"], "status": t["status"]})
    for a in (raw.get("modify_tasks") or [])[:8]:
        if not isinstance(a, dict):
            continue
        t = _resolve_task(m, a.get("id"))
        new_title = _s(a.get("new_title"), 200)
        if t and t["status"] != "done" and t["id"] not in removed and new_title and new_title != t["title"]:
            rev["modify_tasks"].append({"task_id": t["id"], "old_title": t["title"], "new_title": new_title,
                                        "notes": _s(a.get("notes"), 500)})
    for a in (raw.get("add_tasks") or [])[:8]:
        if not isinstance(a, dict) or not _s(a.get("title")):
            continue
        ms = _resolve_milestone(m, a.get("milestone"))
        if ms:
            rev["add_tasks"].append({"milestone_id": ms["id"], "milestone_title": ms["title"],
                                     "title": _s(a.get("title"), 200), "effort": _s(a.get("effort"), 40)})
    for ms in (raw.get("new_milestones") or [])[:3]:
        if isinstance(ms, dict) and _s(ms.get("title")):
            tasks = []
            for t in (ms.get("tasks") or [])[:MAX_TASKS]:
                t = {"title": t} if isinstance(t, str) else t
                if isinstance(t, dict) and _s(t.get("title")):
                    tasks.append({"title": _s(t.get("title"), 200), "effort": _s(t.get("effort"), 40)})
            rev["new_milestones"].append({"title": _s(ms.get("title"), 200), "tasks": tasks})
    for r in (raw.get("risks") or [])[:4]:
        r = {"obstacle": r} if isinstance(r, str) else r
        if isinstance(r, dict) and _s(r.get("obstacle")):
            rev["risks"].append({"text": _s(r.get("obstacle"), 300), "handling": _s(r.get("handling"), 300)})
    if not (rev["add_tasks"] or rev["modify_tasks"] or rev["remove_tasks"] or rev["new_milestones"]):
        raise ValueError("Shoko didn't find anything that needs to change.")
    return rev


def propose_revision(mid, reason):
    m = get_mission(mid)
    if m is None:
        raise KeyError("That mission no longer exists.")
    blockers = "; ".join(b["text"] for b in open_blockers(m)) or "none"
    user = (f"Mission goal: {m['goal']}\nPlan status: {m['status']}\nDeadline: {m['deadline'] or 'none'}\n"
            f"Constraints: {m['constraints'] or 'none'}\nOpen blockers: {blockers}\n"
            f"Current plan:\n{_tasks_for_prompt(m)}\n\nWhy the plan should change: {reason}")
    rev = sanitize_revision(m, _chat_json(REVISE_SYSTEM, user))

    def fn(mm):
        mm["pending_revision"] = rev
        _log(mm, "revision_proposed", f"Plan revision proposed: {rev['reason'][:120]}")
    update_mission(mid, fn)
    return rev


def reject_revision(mid):
    def fn(m):
        m["pending_revision"] = None
        _log(m, "revision_rejected", "Proposed plan revision rejected. Plan unchanged.")
    return update_mission(mid, fn)


def apply_revision(mid):
    def fn(m):
        rev = m.get("pending_revision")
        if not rev:
            raise ValueError("There is no proposed revision to apply.")
        counts = {"added": 0, "changed": 0, "removed": 0}
        for r in rev["remove_tasks"]:
            ms, t, i = find_task(m, r["task_id"])
            if t and t["status"] != "done":
                ms["tasks"].pop(i)
                for _, x in all_tasks(m):
                    x["depends_on"] = [d for d in x["depends_on"] if d != t["id"]]
                counts["removed"] += 1
        for r in rev["modify_tasks"]:
            _, t, _ = find_task(m, r["task_id"])
            if t and t["status"] != "done":
                t["title"] = r["new_title"]
                if r.get("notes"):
                    t["notes"] = r["notes"]
                counts["changed"] += 1
        for a in rev["add_tasks"]:
            ms, _ = find_milestone(m, a["milestone_id"])
            ms = ms or current_milestone(m) or (m["milestones"][-1] if m["milestones"] else None)
            if ms:
                ms["tasks"].append(_norm_task({"title": a["title"], "effort": a["effort"]}))
                counts["added"] += 1
        for nm in rev["new_milestones"]:
            m["milestones"].append({"id": _new_id(), "title": nm["title"], "description": "",
                                    "tasks": [x for x in (_norm_task(t) for t in nm["tasks"]) if x]})
            counts["added"] += len(nm["tasks"])
        known = {r["text"].lower() for r in m["risks"]}
        m["risks"].extend(r for r in rev["risks"] if r["text"].lower() not in known)
        m["pending_revision"] = None
        if m["status"] == "completed" and progress(m)[0] < progress(m)[1]:
            m["status"] = "active"
        _log(m, "plan_revised", f"Plan revised (+{counts['added']} added, ~{counts['changed']} changed, "
                                f"-{counts['removed']} removed): {rev['reason'][:100]}")
    return update_mission(mid, fn)


# ---------------------------------------------------------------- briefing
def _recent_events(m, n=2):
    skip = {"revision_proposed", "revision_rejected"}
    events = [a for a in m["activity"] if a["type"] not in skip][-n:]
    return [f"{a['text']} ({_ago(a['ts'])})" for a in reversed(events)]


def build_briefing(m, chat_summary=""):
    """Facts are assembled from saved data (never invented). The model only phrases
    the 'Reason' and 'My input' lines, and a plain fallback is used if it fails."""
    done, total, pct = progress(m)
    ms, t, waiting = next_task(m)
    blockers = open_blockers(m)

    events = _recent_events(m)
    left_off = "; ".join(events) if len(m["activity"]) > 1 else "The plan is set up, but no work has been recorded yet."
    if chat_summary:
        left_off += f"\n   Earlier conversation: {chat_summary[:300]}"

    if t is None:
        objective = "All planned tasks are done." if total else "There are no tasks yet."
        action = ("Review the success criteria and mark the mission complete when they are met."
                  if total else "Add milestones and tasks, or propose a plan revision.")
    else:
        objective = f"Milestone \"{ms['title']}\" > task \"{t['title']}\" ({t['status']})"
        action = t["title"] + (f"  (about {t['effort']})" if t["effort"] else "")
        if waiting:
            names = [x["title"] for _, x in all_tasks(m) if x["id"] in unmet_deps(m, t)]
            action += f"\n   Note: it waits on: {', '.join(names)}"

    if t is None:
        reason = "Nothing is left to do in the saved plan."
    elif t["status"] == "doing":
        reason = "You already started it, so finishing it keeps momentum."
    elif waiting:
        reason = "Every remaining task is waiting on something else, so this is the closest to ready."
    else:
        reason = "It's the first unfinished task in your plan and nothing it depends on is still open."
    need = ("Tell me what you've tried on the blocker(s) above or how you want to handle them."
            if blockers else "Tell me when you start or finish this, or anything that gets in the way.")

    try:
        facts = (f"Goal: {m['goal'][:200]}\nProgress: {done}/{total} tasks done\n"
                 f"Next task: {t['title'] if t else 'none'} (status {t['status'] if t else 'n/a'})\n"
                 f"Open blockers: {'; '.join(b['text'] for b in blockers) or 'none'}\n"
                 f"Deadline: {m['deadline'] or 'none'} {_deadline_text(m)}")
        raw = _chat_json(
            "You write two short sentences for a mission briefing. Use ONLY the facts given and never say "
            "anything is done unless the facts say so. Reply ONLY with JSON: "
            '{"reason": "why the next task is the right next step", '
            '"need_from_user": "one short question or request for the user"}', facts, 200)
        r, n = _s(raw.get("reason"), 260), _s(raw.get("need_from_user"), 220)
        if r:
            reason = r
        if n:
            need = n
    except Exception as e:
        print("Briefing AI skipped:", e)

    block_text = ("\n".join(f"   - {b['text']}" + (f" (tried: {b['attempted']})" if b["attempted"] else "")
                            for b in blockers) if blockers else "None recorded.")
    deadline = f"   |   Deadline: {m['deadline']} ({_deadline_text(m)})" if m["deadline"] else ""
    return (f"Mission: {m['title']}   [{pct}%  -  {done}/{total} tasks done]{deadline}\n\n"
            f"Where we left off: {left_off}\n\nCurrent objective: {objective}\n\n"
            f"Next action: {action}\n\nReason: {reason}\n\nBlockers:\n{block_text}\n\nMy input: {need}")


# =====================================================================
# Chat integration (called from the main app)
# =====================================================================
def mission_for_chat(chat_id):
    if not chat_id:
        return None
    for m in load_all():
        if m["chat_id"] == chat_id and m["status"] != "archived":
            return m
    return None


def _words(text):
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


def _compact_context(m, user_text=""):
    done, total, pct = progress(m)
    ms, t, waiting = next_task(m)
    lines = [f"Mission: {m['title']} (status: {effective_status(m)}, {pct}% = {done}/{total} tasks done)",
             f"Goal: {m['goal'][:300]}"]
    if m["deadline"]:
        lines.append(f"Deadline: {m['deadline']} ({_deadline_text(m)})")
    if m["constraints"] or m["experience"]:
        lines.append(f"User level/constraints: {m['experience']} {m['constraints']}".strip())
    if m["success_criteria"]:
        lines.append("Success criteria: " + "; ".join(m["success_criteria"][:5]))
    cm = current_milestone(m)
    if cm:
        lines.append(f"Current milestone: {cm['title']}")
    if t:
        lines.append(f"Next task: {t['title']} ({t['status']}{', waiting on prerequisites' if waiting else ''})")
    doing = [x["title"] for _, x in all_tasks(m) if x["status"] == "doing"][:3]
    if doing:
        lines.append("In progress: " + "; ".join(doing))
    recent_done = [x for _, x in all_tasks(m) if x["status"] == "done"]
    recent_done.sort(key=lambda x: x["completed_at"], reverse=True)
    if recent_done:
        lines.append("Recently completed: " + "; ".join(x["title"] for x in recent_done[:3]))
    for b in open_blockers(m)[:3]:
        lines.append(f"OPEN BLOCKER: {b['text']}" + (f" (tried: {b['attempted']})" if b["attempted"] else ""))
    qwords = _words(user_text)
    notes = list(reversed(m["notes"]))
    picked = [n for n in notes if n["kind"] == "decision"][:3]
    picked += [n for n in notes if n not in picked and qwords & _words(n["text"])][:3]
    picked += [n for n in notes if n["kind"] == "progress" and n not in picked][:2]
    for n in picked:
        lines.append(f"{n['kind'].title()}: {n['text'][:200]}")
    return "\n".join(lines)[:CONTEXT_MAX_CHARS]


def get_context_block(chat_id, user_text=""):
    """Text appended to the system prompt. Only the mission linked to THIS chat is included,
    so other missions can never leak in. Never raises."""
    try:
        m = mission_for_chat(chat_id)
        if m is None:
            return ""
        return ("\n\nACTIVE MISSION (this chat is dedicated to this one mission; ignore any other mission)\n"
                + _compact_context(m, user_text) +
                "\nMission rules: rely on this saved state. Never claim a task is finished; the user marks "
                "tasks done in the Missions screen. When the user reports progress, a problem or a decision, "
                "call mission_log_progress, mission_add_blocker or mission_record_decision. For plan changes, "
                "tell the user to use 'Propose plan revision' in Missions. Suggest one concrete next step "
                "when it helps.")
    except Exception as e:
        print("Mission context error:", e)
        return ""


def tool_log(kind, text, chat_id):
    try:
        m = mission_for_chat(chat_id)
        if m is None:
            return "Error: this chat isn't linked to a mission."
        text = _s(text, 500)
        if not text:
            return "Error: no text given."
        if kind == "blocker":
            add_blocker(m["id"], text)
            return "Blocker recorded on the mission."
        add_note(m["id"], text, "decision" if kind == "decision" else "progress")
        return "Decision recorded." if kind == "decision" else "Progress update recorded (no task was marked done)."
    except Exception as e:
        return f"Error: {e}"


def refresh_open_window():
    cb = _ui.get("refresh")
    if cb:
        try:
            cb()
        except tk.TclError:
            pass


# =====================================================================
# UI helpers
# =====================================================================
def _btn(parent, text, cmd, primary=False, danger=False, small=False):
    fg = "white" if primary else (P["red"] if danger else P["text"])
    return tk.Button(parent, text=text, command=cmd, bg=P["accent"] if primary else P["row"], fg=fg, bd=0,
                     relief="flat", cursor="hand2", font=(FONT, 8 if small else 9, "bold"),
                     activebackground=P["accent2"] if primary else P["edge"], activeforeground=fg,
                     padx=6 if small else 12, pady=2 if small else 5)


def _link(parent, text, cmd, fg=None, bg=None, size=8):
    lbl = tk.Label(parent, text=text, fg=fg or P["muted"], bg=bg or P["card"], font=(FONT, size, "bold"),
                   cursor="hand2", padx=4)
    lbl.bind("<Button-1>", lambda e: cmd())
    return lbl


def _walk(widget):
    out = []
    for c in widget.winfo_children():
        out.append(c)
        out.extend(_walk(c))
    return out


def _scroll_area(parent):
    outer = tk.Frame(parent, bg=P["bg"])
    canvas = tk.Canvas(outer, bg=P["bg"], highlightthickness=0)
    sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=P["bg"])
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    def wheel(e):
        canvas.yview_scroll(int(-e.delta / 120), "units")
        return "break"

    def bind_wheel():
        # bound per-widget on purpose: bind_all would replace the chat window's own scrolling
        for w in [canvas, inner] + _walk(inner):
            w.bind("<MouseWheel>", wheel)
    return outer, inner, canvas, bind_wheel


def _progress_bar(parent, pct, width=260, color=None):
    c = tk.Canvas(parent, width=width, height=8, bg=parent["bg"], highlightthickness=0)
    c.create_rectangle(0, 0, width, 8, fill=P["edge"], outline="")
    if pct > 0:
        c.create_rectangle(0, 0, max(4, width * pct / 100), 8, fill=color or P["accent"], outline="")
    return c


STATUS_STYLE = {
    "draft": ("NEEDS APPROVAL", P["amber"]), "active": ("ACTIVE", P["green"]), "blocked": ("BLOCKED", P["red"]),
    "paused": ("PAUSED", P["muted"]), "completed": ("COMPLETED", P["accent2"]), "archived": ("ARCHIVED", P["muted"]),
}


def _status_pill(parent, status, bg):
    text, color = STATUS_STYLE.get(status, (status.upper(), P["muted"]))
    return tk.Label(parent, text=text, fg=color, bg=bg, font=(FONT, 7, "bold"), padx=6)


def _form_dialog(parent, title, fields, validate=None, ok_text="Save", intro=""):
    """fields: [{'key','label','value','lines'}]; lines>1 gives a multi-line box. Returns dict or None."""
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.configure(bg=P["bg"])
    dlg.transient(parent)
    dlg.resizable(False, False)
    result = {"value": None}
    widgets = {}
    if intro:
        tk.Label(dlg, text=intro, bg=P["bg"], fg=P["muted"], font=(FONT, 9), wraplength=480,
                 justify="left").pack(anchor="w", padx=18, pady=(14, 0))
    for f in fields:
        tk.Label(dlg, text=f["label"], bg=P["bg"], fg=P["muted"], font=(FONT, 8)).pack(anchor="w", padx=18, pady=(10, 0))
        lines = f.get("lines", 1)
        if lines > 1:
            w = tk.Text(dlg, height=lines, width=62, wrap="word", bg=P["card"], fg=P["text"],
                        insertbackground=P["text"], bd=0, relief="flat", font=(FONT, 10), padx=6, pady=4)
            w.insert("1.0", f.get("value", ""))
            w.pack(padx=18, pady=(2, 0), fill="x")
        else:
            w = tk.Entry(dlg, width=62, bg=P["card"], fg=P["text"], insertbackground=P["text"], bd=0,
                         relief="flat", font=(FONT, 10))
            w.insert(0, f.get("value", ""))
            w.pack(padx=18, pady=(2, 0), fill="x", ipady=5)
            w.bind("<Return>", lambda e: ok())
        widgets[f["key"]] = (w, lines)
    err = tk.Label(dlg, text="", bg=P["bg"], fg=P["red"], font=(FONT, 8), wraplength=480, justify="left")
    err.pack(anchor="w", padx=18, pady=(8, 0))

    def ok():
        vals = {k: (w.get("1.0", "end-1c") if n > 1 else w.get()).strip() for k, (w, n) in widgets.items()}
        if validate:
            problem = validate(vals)
            if problem:
                err.config(text=problem)
                return
        result["value"] = vals
        dlg.destroy()

    row = tk.Frame(dlg, bg=P["bg"])
    row.pack(fill="x", padx=18, pady=(6, 16))
    _btn(row, ok_text, ok, primary=True).pack(side="right")
    _btn(row, "Cancel", dlg.destroy).pack(side="right", padx=(0, 8))
    dlg.bind("<Escape>", lambda e: dlg.destroy())
    dlg.update_idletasks()
    try:
        x = parent.winfo_rootx() + max(20, (parent.winfo_width() - dlg.winfo_width()) // 2)
        y = parent.winfo_rooty() + 60
        dlg.geometry(f"+{x}+{y}")
        dlg.grab_set()
    except tk.TclError:
        pass
    first = widgets[fields[0]["key"]][0]
    first.focus_set()
    parent.wait_window(dlg)
    return result["value"]


def _valid_deadline(text):
    if not text:
        return ""
    try:
        date.fromisoformat(text)
        return ""
    except ValueError:
        return "Deadline must look like 2026-12-31 (or leave it empty)."


# =====================================================================
# The Missions window
# =====================================================================
def open_missions(parent, theme, on_continue, get_chat_summary=None, assistant_name="Shoko"):
    existing = _ui.get("window")
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.deiconify()
                existing.lift()
                existing.focus_force()
                return
        except tk.TclError:
            pass

    win = tk.Toplevel(parent)
    win.title(f"{assistant_name} - Missions")
    win.geometry("1000x720")
    win.minsize(760, 520)
    win.configure(bg=P["bg"])
    _ui["window"] = win
    S = {"view": "dash", "mid": None, "canvas": None}

    # ---------------- top bar ----------------
    top = tk.Frame(win, bg=P["rail"])
    top.pack(fill="x")
    tk.Label(top, text="🎯  Missions", bg=P["rail"], fg=P["text"], font=(FONT, 13, "bold")).pack(side="left", padx=18, pady=12)
    status_lbl = tk.Label(top, text="", bg=P["rail"], fg=P["accent2"], font=("Consolas", 9))
    status_lbl.pack(side="left", padx=10)
    _btn(top, "＋  New mission", lambda: do_create(), primary=True).pack(side="right", padx=18)
    body = tk.Frame(win, bg=P["bg"])
    body.pack(fill="both", expand=True)

    # ---------------- async helper ----------------
    def run_async(label, fn, on_done):
        if _ui["busy"]:
            status_lbl.config(text="Shoko is still working on the previous request...")
            return
        _ui["busy"] = True
        box = {"done": False, "tick": 0}

        def worker():
            try:
                box["value"] = fn()
            except Exception as e:
                box["error"] = e
            finally:
                box["done"] = True
                if _ui.get("window") is None:
                    _ui["busy"] = False

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            try:
                if not win.winfo_exists():
                    return
            except tk.TclError:
                return
            if not box["done"]:
                box["tick"] += 1
                status_lbl.config(text=f"{label}{'.' * (box['tick'] % 4)}")
                win.after(200, poll)
                return
            _ui["busy"] = False
            status_lbl.config(text="")
            on_done(box.get("value"), box.get("error"))
        poll()

    def safe(fn, *args, **kw):
        """Runs a storage action; shows a friendly error instead of crashing the UI."""
        try:
            return fn(*args, **kw)
        except Exception as e:
            messagebox.showerror("Missions", str(e), parent=win)
            return None

    def moved(fn, *args):
        safe(fn, *args)
        render()

    # ---------------- navigation ----------------
    def show_dash():
        S["view"], S["mid"] = "dash", None
        render()

    def show_detail(mid):
        S["view"], S["mid"] = "detail", mid
        render()

    def render():
        prev = None
        if S["view"] == "detail" and S["canvas"] is not None:
            try:
                prev = S["canvas"].yview()[0]
            except tk.TclError:
                prev = None
        S["canvas"] = None
        for w in body.winfo_children():
            w.destroy()
        try:
            if S["view"] == "detail":
                m = get_mission(S["mid"])
                if m is None:
                    S["view"] = "dash"
                    draw_dash()
                else:
                    draw_detail(m)
            else:
                draw_dash()
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(body, text=f"Couldn't draw this screen: {e}", bg=P["bg"], fg=P["red"],
                     font=(FONT, 10), wraplength=700).pack(pady=30)
        if prev is not None and S["canvas"] is not None:
            win.update_idletasks()
            S["canvas"].yview_moveto(prev)

    _ui["refresh"] = render

    # ---------------- dashboard ----------------
    def draw_dash():
        missions = load_all()
        missions.sort(key=lambda m: m["updated_at"], reverse=True)
        if _load_state["error"]:
            tk.Label(body, text="⚠ " + _load_state["error"], bg="#3a2f1a", fg=P["amber"], font=(FONT, 9),
                     wraplength=900, justify="left", padx=12, pady=6).pack(fill="x", padx=20, pady=(12, 0))
        outer, inner, canvas, bind_wheel = _scroll_area(body)
        outer.pack(fill="both", expand=True)
        if not missions:
            tk.Label(inner, text="🎯", bg=P["bg"], fg=P["accent2"], font=(FONT, 34)).pack(pady=(90, 4))
            tk.Label(inner, text="No missions yet.\nGive Shoko a big goal and it will plan it, track it, and\n"
                                 "remember where you stopped.", bg=P["bg"], fg=P["muted"],
                     font=(FONT, 10), justify="center").pack()
            _btn(inner, "＋  Start your first mission", lambda: do_create(), primary=True).pack(pady=16)
            bind_wheel()
            return
        groups = [("ACTIVE", ("active", "draft")), ("PAUSED", ("paused",)),
                  ("COMPLETED", ("completed",)), ("ARCHIVED", ("archived",))]
        for name, statuses in groups:
            items = [m for m in missions if m["status"] in statuses]
            if not items:
                continue
            tk.Label(inner, text=f"// {name}  ({len(items)})", bg=P["bg"], fg=P["accent"],
                     font=("Consolas", 9, "bold")).pack(anchor="w", padx=22, pady=(16, 4))
            for m in items:
                draw_card(inner, m)
        bind_wheel()

    def draw_card(parent_frame, m):
        mid = m["id"]
        eff = effective_status(m)
        card = tk.Frame(parent_frame, bg=P["card"], highlightthickness=1, highlightbackground=P["edge"])
        card.pack(fill="x", padx=20, pady=5)
        head = tk.Frame(card, bg=P["card"])
        head.pack(fill="x", padx=14, pady=(10, 2))
        title = tk.Label(head, text=m["title"], bg=P["card"], fg=P["text"], font=(FONT, 11, "bold"),
                         cursor="hand2", anchor="w")
        title.pack(side="left")
        title.bind("<Button-1>", lambda e: show_detail(mid))
        _status_pill(head, eff, P["card"]).pack(side="left", padx=8)
        done, total, pct = progress(m)
        meta = tk.Frame(card, bg=P["card"])
        meta.pack(fill="x", padx=14, pady=2)
        _progress_bar(meta, pct, 220, P["green"] if pct == 100 else P["accent"]).pack(side="left")
        tk.Label(meta, text=f"{pct}%  ·  {done}/{total} tasks", bg=P["card"], fg=P["muted"],
                 font=(FONT, 8)).pack(side="left", padx=8)
        ms, t, _ = next_task(m)
        cm = current_milestone(m)
        info = []
        if cm:
            info.append(f"Milestone: {cm['title'][:45]}")
        if t:
            info.append(f"Next: {t['title'][:55]}")
        if info:
            tk.Label(card, text="   ·   ".join(info), bg=P["card"], fg=P["accent2"], font=(FONT, 9),
                     anchor="w").pack(fill="x", padx=14)
        last = m["activity"][-1]["ts"] if m["activity"] else m["created_at"]
        foot = f"Last activity: {_ago(last)}"
        if m["deadline"]:
            foot += f"   ·   Deadline {m['deadline']} ({_deadline_text(m)})"
        tk.Label(card, text=foot, bg=P["card"], fg=P["muted"], font=(FONT, 8), anchor="w").pack(fill="x", padx=14, pady=(2, 4))
        acts = tk.Frame(card, bg=P["card"])
        acts.pack(fill="x", padx=14, pady=(2, 10))
        _btn(acts, "Open", lambda: show_detail(mid), small=True).pack(side="left", padx=(0, 6))
        if m["status"] in ("active", "paused"):
            _btn(acts, "▶ Continue", lambda: do_continue(mid), primary=True, small=True).pack(side="left", padx=(0, 6))
        if m["status"] == "active":
            _btn(acts, "Pause", lambda: do_pause(mid), small=True).pack(side="left", padx=(0, 6))
        elif m["status"] == "paused":
            _btn(acts, "Resume", lambda: do_resume(mid), small=True).pack(side="left", padx=(0, 6))
        if m["status"] == "archived":
            _btn(acts, "Restore", lambda: do_restore(mid), small=True).pack(side="left", padx=(0, 6))
        elif m["status"] != "draft":
            _btn(acts, "Archive", lambda: do_archive(mid), small=True).pack(side="left", padx=(0, 6))
        _btn(acts, "Delete", lambda: do_delete(mid, m["title"]), danger=True, small=True).pack(side="right")

    # ---------------- detail screen ----------------
    def draw_detail(m):
        mid = m["id"]
        eff = effective_status(m)
        draft = m["status"] == "draft"
        labels = label_map(m)
        head = tk.Frame(body, bg=P["bg"])
        head.pack(fill="x", padx=20, pady=(12, 0))
        _link(head, "‹ All missions", show_dash, fg=P["accent2"], bg=P["bg"], size=9).pack(anchor="w")
        trow = tk.Frame(head, bg=P["bg"])
        trow.pack(fill="x", pady=(4, 0))
        tk.Label(trow, text=m["title"], bg=P["bg"], fg=P["text"], font=(FONT, 16, "bold"),
                 wraplength=700, justify="left", anchor="w").pack(side="left")
        _status_pill(trow, eff, P["bg"]).pack(side="left", padx=10)
        tk.Label(head, text="Goal: " + m["goal"], bg=P["bg"], fg=P["muted"], font=(FONT, 9), wraplength=900,
                 justify="left", anchor="w").pack(fill="x", pady=(2, 6))
        done, total, pct = progress(m)
        prow = tk.Frame(head, bg=P["bg"])
        prow.pack(fill="x")
        _progress_bar(prow, pct, 320, P["green"] if pct == 100 else P["accent"]).pack(side="left")
        ptxt = f"{pct}%  ·  {done} of {total} tasks done" if total else "No tasks yet"
        if m["deadline"]:
            ptxt += f"  ·  deadline {m['deadline']} ({_deadline_text(m)})"
        tk.Label(prow, text=ptxt, bg=P["bg"], fg=P["muted"], font=(FONT, 8)).pack(side="left", padx=10)

        btns = tk.Frame(head, bg=P["bg"])
        btns.pack(fill="x", pady=(10, 6))
        if draft:
            _btn(btns, "✔  Approve plan", lambda: do_approve(mid), primary=True).pack(side="left", padx=(0, 8))
        elif m["status"] in ("active", "paused"):
            _btn(btns, "▶  Continue Mission", lambda: do_continue(mid), primary=True).pack(side="left", padx=(0, 8))
        if m["status"] == "active":
            _btn(btns, "Pause", lambda: do_pause(mid)).pack(side="left", padx=(0, 8))
        elif m["status"] == "paused":
            _btn(btns, "Resume", lambda: do_resume(mid)).pack(side="left", padx=(0, 8))
        if m["status"] != "archived":
            _btn(btns, "Propose plan revision", lambda: do_revision(mid)).pack(side="left", padx=(0, 8))
        if m["status"] in ("active", "paused"):
            _btn(btns, "Mark complete", lambda: do_complete(mid)).pack(side="left", padx=(0, 8))
        _btn(btns, "Edit details", lambda: do_edit_mission(mid)).pack(side="left", padx=(0, 8))
        if m["status"] == "archived":
            _btn(btns, "Restore", lambda: do_restore(mid)).pack(side="left", padx=(0, 8))
        elif not draft:
            _btn(btns, "Archive", lambda: do_archive(mid)).pack(side="left", padx=(0, 8))
        _btn(btns, "Delete", lambda: do_delete(mid, m["title"]), danger=True).pack(side="right")

        if draft:
            tk.Label(head, text="⚠ Review this plan. Edit anything you like, then press Approve. "
                                "Nothing is tracked as progress until you approve it.",
                     bg="#3a2f1a", fg=P["amber"], font=(FONT, 9), wraplength=900, justify="left",
                     padx=10, pady=6).pack(fill="x", pady=(0, 4))
        if m.get("pending_revision"):
            rb = tk.Frame(head, bg="#2a2370")
            rb.pack(fill="x", pady=(0, 4))
            tk.Label(rb, text="Shoko proposed a plan revision. Nothing changes until you approve it.",
                     bg="#2a2370", fg=P["text"], font=(FONT, 9), padx=10, pady=6).pack(side="left")
            _btn(rb, "Review", lambda: review_revision(mid), primary=True, small=True).pack(side="right", padx=8, pady=4)

        outer, inner, canvas, bind_wheel = _scroll_area(body)
        outer.pack(fill="both", expand=True, pady=(4, 0))
        S["canvas"] = canvas
        pad = {"padx": 20}

        def section(title, action_text=None, action=None):
            r = tk.Frame(inner, bg=P["bg"])
            r.pack(fill="x", pady=(16, 4), **pad)
            tk.Label(r, text=f"// {title}", bg=P["bg"], fg=P["accent"], font=("Consolas", 9, "bold")).pack(side="left")
            if action:
                _link(r, action_text, action, fg=P["accent2"], bg=P["bg"]).pack(side="right")

        # success criteria
        section("SUCCESS CRITERIA")
        if m["success_criteria"]:
            for c in m["success_criteria"]:
                tk.Label(inner, text="•  " + c, bg=P["bg"], fg=P["text"], font=(FONT, 9), wraplength=880,
                         justify="left", anchor="w").pack(fill="x", **pad)
        else:
            tk.Label(inner, text="None set. Use 'Edit details' to add some.", bg=P["bg"], fg=P["muted"],
                     font=(FONT, 9)).pack(anchor="w", **pad)

        # milestones and tasks
        section("PLAN", "＋ Add milestone", lambda: do_add_milestone(mid))
        if not m["milestones"]:
            tk.Label(inner, text="No milestones yet.", bg=P["bg"], fg=P["muted"], font=(FONT, 9)).pack(anchor="w", **pad)
        cm = current_milestone(m)
        for i, ms in enumerate(m["milestones"], 1):
            state = milestone_state(ms)
            mdone = sum(1 for t in ms["tasks"] if t["status"] == "done")
            box = tk.Frame(inner, bg=P["card"], highlightthickness=1,
                           highlightbackground=P["accent"] if cm is ms else P["edge"])
            box.pack(fill="x", pady=4, **pad)
            mh = tk.Frame(box, bg=P["card"])
            mh.pack(fill="x", padx=10, pady=(8, 2))
            tk.Label(mh, text=f"{i}.  {ms['title']}", bg=P["card"], fg=P["text"], font=(FONT, 10, "bold")).pack(side="left")
            color = P["green"] if state == "done" else (P["accent2"] if state == "in progress" else P["muted"])
            tk.Label(mh, text=f"{state}  ·  {mdone}/{len(ms['tasks'])}", bg=P["card"], fg=color,
                     font=(FONT, 8, "bold")).pack(side="left", padx=10)
            for txt, cmd, fg in (("✕", lambda s=ms: do_delete_milestone(mid, s), P["red"]),
                                 ("↓", lambda s=ms: moved(move_milestone, mid, s["id"], 1), None),
                                 ("↑", lambda s=ms: moved(move_milestone, mid, s["id"], -1), None),
                                 ("✎", lambda s=ms: do_edit_milestone(mid, s), None),
                                 ("＋ Task", lambda s=ms: do_add_task(mid, s), P["accent2"])):
                _link(mh, txt, cmd, fg=fg).pack(side="right")
            for j, t in enumerate(ms["tasks"], 1):
                draw_task(box, m, t, f"{i}.{j}", draft, labels)
            tk.Frame(box, bg=P["card"], height=6).pack()

        # blockers
        section("BLOCKERS", "＋ Add blocker", lambda: do_add_blocker(mid))
        if not m["blockers"]:
            tk.Label(inner, text="No blockers recorded.", bg=P["bg"], fg=P["muted"], font=(FONT, 9)).pack(anchor="w", **pad)
        for b in sorted(m["blockers"], key=lambda x: (x["status"] != "open", x["created_at"])):
            r = tk.Frame(inner, bg=P["row"])
            r.pack(fill="x", pady=2, **pad)
            is_open = b["status"] == "open"
            tk.Label(r, text="⚠" if is_open else "✔", fg=P["red"] if is_open else P["green"], bg=P["row"],
                     font=(FONT, 10)).pack(side="left", padx=(10, 6), pady=6)
            col = tk.Frame(r, bg=P["row"])
            col.pack(side="left", fill="x", expand=True, pady=4)
            tk.Label(col, text=b["text"], bg=P["row"], fg=P["text"], font=(FONT, 9, "bold"), wraplength=640,
                     justify="left", anchor="w").pack(fill="x")
            sub = []
            if b["attempted"]:
                sub.append("Tried: " + b["attempted"])
            if b["task_id"]:
                sub.append("Task " + label_of(m, b["task_id"]))
            if b["resolution"]:
                sub.append("Resolved by: " + b["resolution"])
            if sub:
                tk.Label(col, text="   ·   ".join(sub), bg=P["row"], fg=P["muted"], font=(FONT, 8),
                         wraplength=640, justify="left", anchor="w").pack(fill="x")
            if is_open:
                _link(r, "Resolve", lambda x=b: do_resolve_blocker(mid, x), fg=P["green"], bg=P["row"]).pack(side="right", padx=8)

        # notes and decisions
        section("NOTES, DECISIONS & PROGRESS")
        nrow = tk.Frame(inner, bg=P["bg"])
        nrow.pack(fill="x", **pad)
        _btn(nrow, "＋ Progress update", lambda: do_note(mid, "progress"), small=True).pack(side="left", padx=(0, 6))
        _btn(nrow, "＋ Decision", lambda: do_note(mid, "decision"), small=True).pack(side="left", padx=(0, 6))
        _btn(nrow, "＋ Note", lambda: do_note(mid, "note"), small=True).pack(side="left")
        for n in list(reversed(m["notes"]))[:8]:
            r = tk.Frame(inner, bg=P["row"])
            r.pack(fill="x", pady=2, **pad)
            tk.Label(r, text=n["kind"].upper(), bg=P["row"], fg=P["accent2"], font=("Consolas", 7, "bold"),
                     width=9).pack(side="left", padx=(8, 0), pady=6)
            tk.Label(r, text=n["text"], bg=P["row"], fg=P["text"], font=(FONT, 9), wraplength=700,
                     justify="left", anchor="w").pack(side="left", fill="x", expand=True)
            tk.Label(r, text=_ago(n["created_at"]), bg=P["row"], fg=P["muted"], font=(FONT, 7)).pack(side="right", padx=8)

        # risks
        if m["risks"]:
            section("LIKELY OBSTACLES")
            for r in m["risks"]:
                text = r["text"] + (f"  ->  {r['handling']}" if r["handling"] else "")
                tk.Label(inner, text="•  " + text, bg=P["bg"], fg=P["muted"], font=(FONT, 9), wraplength=880,
                         justify="left", anchor="w").pack(fill="x", **pad)

        # activity
        section("RECENT ACTIVITY")
        for a in list(reversed(m["activity"]))[:8]:
            tk.Label(inner, text=f"{_ago(a['ts']):>12}   {a['text']}", bg=P["bg"], fg=P["muted"],
                     font=("Consolas", 8), anchor="w", justify="left", wraplength=900).pack(fill="x", **pad)
        tk.Frame(inner, bg=P["bg"], height=30).pack()
        bind_wheel()

    def draw_task(box, m, t, label, draft, labels):
        mid = m["id"]
        row = tk.Frame(box, bg=P["row"])
        row.pack(fill="x", padx=10, pady=2)
        icon, color = {"todo": ("☐", P["accent2"]), "doing": ("▶", P["accent"]),
                       "paused": ("⏸", P["muted"]), "done": ("☑", P["green"])}[t["status"]]
        tk.Label(row, text=icon, fg=color, bg=P["row"], font=("Segoe UI Symbol", 11)).pack(side="left", padx=(8, 4), pady=4)
        col = tk.Frame(row, bg=P["row"])
        col.pack(side="left", fill="x", expand=True, pady=2)
        done = t["status"] == "done"
        tk.Label(col, text=f"{label}  {t['title']}", bg=P["row"], fg=P["muted"] if done else P["text"],
                 font=(FONT, 9, "overstrike" if done else "normal"), wraplength=520, justify="left",
                 anchor="w").pack(fill="x")
        sub = []
        waits = unmet_deps(m, t)
        if waits and not done:
            sub.append("⛓ waits for " + ", ".join(label_of(m, w) for w in waits))
        if any(b["status"] == "open" and b["task_id"] == t["id"] for b in m["blockers"]):
            sub.append("⚠ has an open blocker")
        if t["effort"]:
            sub.append(t["effort"])
        if t["notes"]:
            sub.append(t["notes"][:80])
        if done and t["completed_at"]:
            sub.append("done " + _ago(t["completed_at"]))
        if sub:
            tk.Label(col, text="   ·   ".join(sub), bg=P["row"], fg=P["red"] if "⚠" in sub[0] else P["muted"],
                     font=(FONT, 8), anchor="w", justify="left", wraplength=520).pack(fill="x")
        for txt, cmd, fg in (("✕", lambda: do_delete_task(mid, t), P["red"]),
                             ("↓", lambda: moved(move_task, mid, t["id"], 1), None),
                             ("↑", lambda: moved(move_task, mid, t["id"], -1), None),
                             ("✎", lambda: do_edit_task(mid, t), None)):
            _link(row, txt, cmd, fg=fg, bg=P["row"]).pack(side="right")
        if not draft:
            if done:
                _link(row, "Reopen", lambda: set_status_task(mid, t, "todo"), fg=P["accent2"], bg=P["row"]).pack(side="right")
            else:
                _link(row, "Done", lambda: set_status_task(mid, t, "done"), fg=P["green"], bg=P["row"]).pack(side="right")
                if t["status"] == "doing":
                    _link(row, "Pause", lambda: set_status_task(mid, t, "paused"), bg=P["row"]).pack(side="right")
                else:
                    _link(row, "Start", lambda: set_status_task(mid, t, "doing"), fg=P["accent"], bg=P["row"]).pack(side="right")

    # ---------------- actions ----------------
    def set_status_task(mid, t, status):
        if status == "doing":
            m = get_mission(mid)
            waits = unmet_deps(m, t) if m else []
            if waits and not messagebox.askyesno(
                    "Prerequisites not done",
                    "This task waits for: " + ", ".join(label_of(m, w) for w in waits) + "\n\nStart it anyway?", parent=win):
                return
        safe(set_task_status, mid, t["id"], status)
        render()

    def do_create(prefill=None):
        prefill = prefill or {}

        def check(v):
            if len(v["goal"].split()) < 3:
                return "Describe the goal in at least a few words."
            return _valid_deadline(v["deadline"])
        v = _form_dialog(win, "New mission", [
            {"key": "goal", "label": "What do you want to achieve?", "lines": 4, "value": prefill.get("goal", "")},
            {"key": "deadline", "label": "Deadline (optional, YYYY-MM-DD)", "value": prefill.get("deadline", "")},
            {"key": "experience", "label": "Your current level (optional)", "value": prefill.get("experience", "")},
            {"key": "constraints", "label": "Time or other constraints (optional)", "value": prefill.get("constraints", "")},
            {"key": "desired", "label": "What would success look like? (optional)", "value": prefill.get("desired", "")},
        ], validate=check, ok_text="Create plan",
            intro="Only the goal is required. Shoko will draft a plan that you review before it starts.")
        if not v:
            return

        def work():
            plan = generate_plan(v["goal"], v["deadline"], v["experience"], v["constraints"], v["desired"])
            if plan["clarifying_question"] and not plan["milestones"]:
                return ("clarify", plan["clarifying_question"])
            mid = create_mission_from_plan(v["goal"], plan, v["deadline"], v["experience"], v["constraints"])
            return ("ok", mid)

        def done(res, err):
            if err:
                messagebox.showerror("Couldn't create the mission", friendly_error(err), parent=win)
                do_create(v)
            elif res[0] == "clarify":
                messagebox.showinfo("One quick question", res[1], parent=win)
                do_create(v)
            else:
                show_detail(res[1])
        run_async("Shoko is planning your mission", work, done)

    def do_approve(mid):
        safe(approve_plan, mid)
        render()

    def do_pause(mid):
        safe(set_mission_status, mid, "paused", "Mission paused.")
        render()

    def do_resume(mid):
        safe(set_mission_status, mid, "active", "Mission resumed.")
        render()

    def do_archive(mid):
        safe(set_mission_status, mid, "archived", "Mission archived.")
        render()

    def do_restore(mid):
        safe(set_mission_status, mid, "paused", "Mission restored from archive (paused).")
        render()

    def do_complete(mid):
        m = get_mission(mid)
        if m is None:
            return
        done, total, _ = progress(m)
        if total == 0 or done < total:
            messagebox.showinfo("Not finished yet", f"{total - done} task(s) are still open. Finish them, "
                                "or remove tasks you no longer need, before completing the mission.", parent=win)
            return
        crit = "\n".join("- " + c for c in m["success_criteria"]) or "(no success criteria set)"
        if messagebox.askyesno("Complete mission", f"All tasks are done. Have you met your success criteria?\n\n{crit}", parent=win):
            safe(complete_mission, mid)
            render()

    def do_delete(mid, title):
        if messagebox.askyesno("Delete mission", f"Permanently delete \"{title}\"?\n\nThis can't be undone. "
                               "The linked chat (if any) is kept.", icon="warning", parent=win):
            safe(delete_mission, mid)
            show_dash()

    def do_edit_mission(mid):
        m = get_mission(mid)
        if m is None:
            return
        v = _form_dialog(win, "Edit mission", [
            {"key": "title", "label": "Title", "value": m["title"]},
            {"key": "goal", "label": "Goal", "lines": 3, "value": m["goal"]},
            {"key": "criteria", "label": "Success criteria (one per line)", "lines": 5,
             "value": "\n".join(m["success_criteria"])},
            {"key": "deadline", "label": "Deadline (YYYY-MM-DD, or empty)", "value": m["deadline"]},
            {"key": "constraints", "label": "Constraints", "value": m["constraints"]},
        ], validate=lambda v: "Title can't be empty." if not v["title"] else _valid_deadline(v["deadline"]))
        if v:
            safe(edit_mission, mid, v["title"], v["goal"], v["criteria"], v["deadline"], v["constraints"])
            render()

    def do_add_milestone(mid):
        v = _form_dialog(win, "Add milestone", [{"key": "title", "label": "Milestone title"}],
                         validate=lambda v: "" if v["title"] else "Give it a title.", ok_text="Add")
        if v:
            safe(add_milestone, mid, v["title"])
            render()

    def do_edit_milestone(mid, ms):
        v = _form_dialog(win, "Edit milestone", [
            {"key": "title", "label": "Title", "value": ms["title"]},
            {"key": "desc", "label": "Description", "lines": 3, "value": ms["description"]}],
            validate=lambda v: "" if v["title"] else "Title can't be empty.")
        if v:
            safe(edit_milestone, mid, ms["id"], v["title"], v["desc"])
            render()

    def do_delete_milestone(mid, ms):
        n = len(ms["tasks"])
        if messagebox.askyesno("Delete milestone", f"Delete \"{ms['title']}\" and its {n} task(s)?\n\n"
                               "Completed tasks in it will be removed from progress too.", icon="warning", parent=win):
            safe(delete_milestone, mid, ms["id"])
            render()

    def do_add_task(mid, ms):
        v = _form_dialog(win, "Add task", [
            {"key": "title", "label": "Task"},
            {"key": "effort", "label": "Estimated effort (optional)"},
            {"key": "notes", "label": "Notes (optional)", "lines": 2}],
            validate=lambda v: "" if v["title"] else "Give the task a title.", ok_text="Add")
        if v:
            safe(add_task, mid, ms["id"], v["title"], v["notes"], v["effort"])
            render()

    def do_edit_task(mid, t):
        m = get_mission(mid)
        if m is None:
            return
        labels = label_map(m)
        own = label_of(m, t["id"])
        initial = ", ".join(label_of(m, d) for d in t["depends_on"])

        def check(v):
            if not v["title"]:
                return "Title can't be empty."
            for part in [p.strip() for p in v["waits"].split(",") if p.strip()]:
                if part not in labels:
                    return f"'{part}' isn't a task number. Use numbers like 1.1, 2.3."
                if part == own:
                    return "A task can't wait for itself."
            return ""
        v = _form_dialog(win, f"Edit task {own}", [
            {"key": "title", "label": "Task", "value": t["title"]},
            {"key": "effort", "label": "Estimated effort", "value": t["effort"]},
            {"key": "notes", "label": "Notes", "lines": 3, "value": t["notes"]},
            {"key": "waits", "label": "Waits for tasks (numbers like 1.1, 1.2; optional)", "value": initial}], validate=check)
        if v:
            deps = [labels[p.strip()] for p in v["waits"].split(",") if p.strip()]
            safe(edit_task, mid, t["id"], v["title"], v["notes"], v["effort"], deps)
            render()

    def do_delete_task(mid, t):
        if messagebox.askyesno("Delete task", f"Delete \"{t['title']}\"?", parent=win):
            safe(delete_task, mid, t["id"])
            render()

    def do_add_blocker(mid):
        m = get_mission(mid)
        if m is None:
            return
        labels = label_map(m)

        def check(v):
            if not v["text"]:
                return "Describe the blocker."
            if v["task"] and v["task"] not in labels:
                return "Task must be a number like 1.2 (or empty)."
            return ""
        v = _form_dialog(win, "Add blocker", [
            {"key": "text", "label": "What is in the way?", "lines": 3},
            {"key": "attempted", "label": "What have you tried? (optional)"},
            {"key": "task", "label": "Related task number, e.g. 1.2 (optional)"}], validate=check, ok_text="Add blocker")
        if not v:
            return
        safe(add_blocker, mid, v["text"], v["attempted"], labels.get(v["task"], ""))
        render()
        if messagebox.askyesno("Adjust the plan?", "Want Shoko to propose a plan revision that works around this blocker?\n\n"
                               "(You'll review it before anything changes.)", parent=win):
            start_revision(mid, "Blocker: " + v["text"])

    def do_resolve_blocker(mid, b):
        v = _form_dialog(win, "Resolve blocker", [{"key": "res", "label": "How was it resolved? (optional)", "lines": 2}],
                         ok_text="Mark resolved", intro=b["text"])
        if v is not None:
            safe(resolve_blocker, mid, b["id"], v["res"])
            render()

    def do_note(mid, kind):
        label = {"progress": "Progress update", "decision": "Decision", "note": "Note"}[kind]
        v = _form_dialog(win, label, [{"key": "text", "label": label, "lines": 4}],
                         validate=lambda v: "" if v["text"] else "Write something first.", ok_text="Save")
        if v:
            safe(add_note, mid, v["text"], kind)
            render()

    def do_revision(mid):
        v = _form_dialog(win, "Propose plan revision", [
            {"key": "reason", "label": "What changed or isn't working?", "lines": 4}],
            validate=lambda v: "" if len(v["reason"].split()) >= 3 else "Say what changed in a few words.",
            ok_text="Ask Shoko", intro="Shoko will suggest changes. Completed tasks are never touched, and nothing "
                                       "is applied until you approve it.")
        if v:
            start_revision(mid, v["reason"])

    def start_revision(mid, reason):
        def done(res, err):
            if err:
                messagebox.showerror("Couldn't propose a revision", friendly_error(err), parent=win)
                return
            render()
            review_revision(mid)
        run_async("Shoko is reviewing the plan", lambda: propose_revision(mid, reason), done)

    def review_revision(mid):
        m = get_mission(mid)
        if m is None or not m.get("pending_revision"):
            render()
            return
        rev = m["pending_revision"]
        dlg = tk.Toplevel(win)
        dlg.title("Proposed plan revision")
        dlg.geometry("640x560")
        dlg.configure(bg=P["bg"])
        dlg.transient(win)
        lines = [f"WHY: {rev['reason']}", ""]
        if rev["add_tasks"]:
            lines.append("ADD TASKS")
            lines += [f"  + {a['title']}  (in \"{a['milestone_title']}\")" for a in rev["add_tasks"]]
            lines.append("")
        if rev["new_milestones"]:
            lines.append("ADD MILESTONES")
            for nm in rev["new_milestones"]:
                lines.append(f"  + {nm['title']}")
                lines += [f"      - {t['title']}" for t in nm["tasks"]]
            lines.append("")
        if rev["modify_tasks"]:
            lines.append("CHANGE TASKS")
            lines += [f"  ~ {a['old_title']}  ->  {a['new_title']}" for a in rev["modify_tasks"]]
            lines.append("")
        if rev["remove_tasks"]:
            lines.append("REMOVE TASKS")
            lines += [f"  - {r['title']}" + ("  (currently in progress!)" if r["status"] == "doing" else "")
                      for r in rev["remove_tasks"]]
            lines.append("")
        lines.append(f"KEPT AS-IS: all {rev['kept_done']} completed task(s) and everything not listed above.")
        if rev["risks"]:
            lines += ["", "NEW RISKS"] + [f"  ! {r['text']}" + (f" -> {r['handling']}" if r["handling"] else "") for r in rev["risks"]]
        txt = tk.Text(dlg, wrap="word", bg=P["card"], fg=P["text"], bd=0, relief="flat", font=(FONT, 10), padx=12, pady=10)
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        row = tk.Frame(dlg, bg=P["bg"])
        row.pack(fill="x", padx=16, pady=(0, 14))

        def apply():
            if safe(apply_revision, mid) is not None:
                dlg.destroy()
                render()

        def reject():
            safe(reject_revision, mid)
            dlg.destroy()
            render()
        _btn(row, "Apply these changes", apply, primary=True).pack(side="right")
        _btn(row, "Reject", reject, danger=True).pack(side="right", padx=8)
        _btn(row, "Decide later", dlg.destroy).pack(side="right")

    def do_continue(mid):
        m = get_mission(mid)
        if m is None:
            return
        if m["status"] == "draft":
            messagebox.showinfo("Approve the plan first", "Review and approve the plan before continuing the mission.", parent=win)
            return
        if m["status"] not in ("active", "paused"):
            return

        def work():
            summary = ""
            if get_chat_summary and m["chat_id"]:
                try:
                    summary = get_chat_summary(m["chat_id"]) or ""
                except Exception:
                    summary = ""
            return build_briefing(m, summary)

        def done(briefing, err):
            if err:
                messagebox.showerror("Couldn't build the briefing", friendly_error(err), parent=win)
                return
            if m["status"] == "paused":
                safe(set_mission_status, mid, "active", "Mission resumed.")
            chat_id = on_continue(mid, m["title"], briefing, m["chat_id"])
            if chat_id:
                safe(link_chat, mid, chat_id)
                try:
                    update_mission(mid, lambda mm: _log(mm, "continued", "Continued the mission in chat."))
                except Exception as e:
                    print("Mission log error:", e)
            render()
        run_async("Shoko is preparing your briefing", work, done)

    # ---------------- lifecycle ----------------
    def on_close():
        _ui["window"] = None
        _ui["refresh"] = None
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)
    render()