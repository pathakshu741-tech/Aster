"""
vibe_code.py  --  "Vibe Code" workspace for the AP/Nova assistant.

Put this file in the SAME folder as your main file (next to creator_studio.py).
Main file only needs:   from vibe_code import open_vibe_code
                        open_vibe_code(window, theme, image_fn=generate_scene_image)

What it gives you
  * project explorer, tabbed editor (line numbers, syntax colours, find, save)
  * Ollama coding chat that proposes changes as validated JSON
  * diff review -> Apply / Reject, checkpoints, restore
  * Run / Stop / Test, terminal, problems list, agent activity log
  * "Fix with AI" (max 3 attempts), static / node preview in your browser
  * per-project memory under ~/.ava_assistant/vibe_code/

Honest limits (also shown in the app):
  * Running code is NOT sandboxed. It runs with your Windows user rights.
    Secret-looking environment variables are hidden from child processes,
    but that is a courtesy, not isolation.
  * The agent does not call tools by itself. The app gathers context with its
    own tools (file list, read, search, dependency check) and logs each step in
    "Agent Activity"; the model only proposes changes, you approve them.
"""
import os
import re
import sys
import json
import uuid
import time
import ast
import bisect
import queue
import shutil
import signal
import socket
import difflib
import fnmatch
import zipfile
import threading
import subprocess
import posixpath
import webbrowser
from collections import deque, Counter
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

try:
    import ollama
except Exception:          # the app still opens; AI features show a clear error
    ollama = None

# =====================================================================
# Constants
# =====================================================================
BASE_DIR = os.path.join(os.path.expanduser("~"), ".ava_assistant")
VIBE_DIR = os.path.join(BASE_DIR, "vibe_code")
PROJECTS_INDEX = os.path.join(VIBE_DIR, "projects.json")
PROJECTS_DATA = os.path.join(VIBE_DIR, "projects")
DEFAULT_PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), "VibeProjects")

DEFAULT_MODEL = "qwen2.5-coder:3b"
VISION_MODEL = "moondream"            # same vision model your chat already uses
DEFAULT_SETTINGS = {"model": DEFAULT_MODEL, "num_ctx": 8192, "python": "",
                    "timeout": 120, "run_command": ""}

EXCLUDE_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
                "env", ".idea", ".vscode", "dist", "build", ".next", ".cache",
                ".pytest_cache", ".mypy_cache", ".tox", "site-packages", ".vibe"}
SECRET_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "id_rsa*", "*.pfx", "*.p12",
                   "secrets.*", "credentials*", "*.keystore"]
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp", ".exe", ".dll",
               ".zip", ".rar", ".7z", ".pdf", ".mp3", ".mp4", ".wav", ".mov", ".avi",
               ".woff", ".woff2", ".ttf", ".otf", ".pyc", ".so", ".bin", ".onnx", ".pt",
               ".db", ".sqlite"}
MAX_FILE_BYTES = 400_000          # biggest file we will read for AI context
MAX_EDIT_BYTES = 1_000_000        # biggest file we will open in the editor
MAX_NEW_FILE_CHARS = 200_000
MAX_CHANGE_FILES = 25
MAX_FIX_ATTEMPTS = 3

BLOCKED_CMD = [r"\brm\s+-\w*[rf]", r"\brmdir\s+/s", r"\brd\s+/s", r"\bdel\s+.*(/s|/q|/f)",
               r"\bformat\s+[a-z]:", r"\bshutdown\b", r"\bmkfs\b", r"\bdiskpart\b",
               r"\breg\s+delete\b", r"remove-item\b.*-recurse", r"git\s+reset\s+--hard",
               r"git\s+clean\s+-\w*f", r"\bcipher\s+/w", r":\(\)\s*\{"]
INSTALL_CMD = re.compile(r"\b(pip3?|npm|yarn|pnpm|winget|choco|scoop|apt|apt-get|brew)\b.*\b(install|add)\b")

P = dict(bg="#0B0F19", bg2="#101626", panel="#151D2D", panel2="#151D2D", accent="#657FFF",
         accent2="#8B7CFF", glow="#50C8E8", text="#E8EDF7", dim="#7786A3", border="#2A3650",
         ok="#43D9A3", err="#FF647C", warn="#F5BD62", sel="#1C263A", side="#101626")
UI_FONT = "Segoe UI"
MONO = "Consolas"
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

P.update(card="#131B2E", cardh="#1A2540")
KIND_ICON = {"python": ("Py", "#657FFF"), "node": ("JS", "#43D9A3"),
             "web": ("</>", "#8B7CFF"), "other": ("{ }", "#4A5878")}


def time_ago(ts):
    s = max(0, time.time() - ts)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        h = int(s // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if s < 86400 * 30:
        d = int(s // 86400)
        return f"{d} day{'s' if d != 1 else ''} ago"
    return datetime.fromtimestamp(ts).strftime("%d %b %Y")


def latest_mtime(root, cap=400):
    """Newest file time in a project (looks at <= cap files, skips generated folders)."""
    best, n = 0.0, 0
    try:
        best = os.path.getmtime(root)
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
            for f in files:
                try:
                    best = max(best, os.path.getmtime(os.path.join(d, f)))
                except OSError:
                    pass
                n += 1
                if n >= cap:
                    return best
    except OSError:
        pass
    return best


def _is_inside(child, parent):
    try:
        c = os.path.normcase(os.path.realpath(child))
        p = os.path.normcase(os.path.realpath(parent))
        return os.path.commonpath([c, p]) == p
    except ValueError:
        return False


# =====================================================================
# Small helpers
# =====================================================================
def _now():
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write(path, text):
    """Write to a temp file, then replace. A crash never leaves a half-written file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".vibe_tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".vibe_tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, data):
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))


def safe_join(root, rel):
    """Turn a project-relative path into a real path, or raise ValueError if it
    is absolute, uses '..', points at a protected folder, or escapes the root."""
    if not isinstance(rel, str) or not rel.strip() or "\x00" in rel:
        raise ValueError("empty or invalid path")
    rel = rel.replace("\\", "/").strip()
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValueError(f"absolute paths are not allowed: {rel}")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts:
        raise ValueError("empty path")
    if any(p == ".." for p in parts):
        raise ValueError(f"'..' is not allowed in paths: {rel}")
    if parts[0] in (".git", ".vibe"):
        raise ValueError(f"protected folder: {parts[0]}")
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, *parts))
    try:
        inside = os.path.commonpath([full, root_real]) == root_real
    except ValueError:
        inside = False
    if not inside:
        raise ValueError(f"path escapes the project folder: {rel}")
    return full


def rel_of(root, full):
    return os.path.relpath(full, os.path.realpath(root)).replace("\\", "/")


def is_secret(rel):
    name = os.path.basename(rel).lower()
    return any(fnmatch.fnmatch(name, p) for p in SECRET_PATTERNS)


def looks_text(path):
    if os.path.splitext(path)[1].lower() in BINARY_EXTS:
        return False
    try:
        with open(path, "rb") as f:
            return b"\x00" not in f.read(2048)
    except OSError:
        return False


def read_text_nl(path, limit=MAX_FILE_BYTES):
    """Returns (text with \\n newlines, original newline style)."""
    size = os.path.getsize(path)
    if size > limit:
        raise ValueError(f"file too large ({size // 1024} KB)")
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), nl


def write_text_nl(path, text, nl="\n"):
    _atomic_write(path, text.replace("\n", nl) if nl != "\n" else text)


def list_dir_entries(root, rel=""):
    base = safe_join(root, rel) if rel else os.path.realpath(root)
    try:
        entries = list(os.scandir(base))
    except OSError:
        return [], []
    dirs = sorted((e.name for e in entries
                   if e.is_dir(follow_symlinks=False) and e.name not in EXCLUDE_DIRS), key=str.lower)
    files = sorted((e.name for e in entries if e.is_file(follow_symlinks=False)), key=str.lower)
    return dirs, files


def iter_files(root, max_files=3000):
    out = []
    root = os.path.realpath(root)
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs
                         if x not in EXCLUDE_DIRS and not os.path.islink(os.path.join(d, x)))
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(d, f), root).replace("\\", "/"))
            if len(out) >= max_files:
                return out
    return out


def build_tree_text(root, limit=150):
    files = iter_files(root, limit + 1)
    text = "\n".join(files[:limit])
    if len(files) > limit:
        text += "\n...(more files not shown)"
    return text or "(empty project)"


def search_code(root, query, max_hits=25):
    """Tool: search source text. Returns [(rel, line_no, line_text)]."""
    q = query.lower()
    hits = []
    for rel in iter_files(root, 1500):
        full = os.path.join(root, rel)
        try:
            if is_secret(rel) or os.path.getsize(full) > MAX_FILE_BYTES or not looks_text(full):
                continue
            text, _ = read_text_nl(full)
        except Exception:
            continue
        for i, line in enumerate(text.split("\n"), 1):
            if q in line.lower():
                hits.append((rel, i, line.strip()[:160]))
                if len(hits) >= max_hits:
                    return hits
    return hits


def rank_files(root, terms, limit=4):
    """Which files mention the request's keywords the most."""
    score = Counter()
    for rel in iter_files(root, 1200):
        full = os.path.join(root, rel)
        try:
            if is_secret(rel) or os.path.getsize(full) > MAX_FILE_BYTES or not looks_text(full):
                continue
            text, _ = read_text_nl(full)
        except Exception:
            continue
        low = text.lower() + " " + rel.lower()
        for t in terms:
            c = low.count(t)
            if c:
                score[rel] += min(c, 5)
    return [r for r, _ in score.most_common(limit)]


def detect_project(root):
    info = {"kind": "other", "language": "", "framework": ""}

    def has(n):
        return os.path.isfile(os.path.join(root, n))

    try:
        top = os.listdir(root)
    except OSError:
        return info
    if has("package.json"):
        pkg = _read_json(os.path.join(root, "package.json"), {})
        deps = {}
        for k in ("dependencies", "devDependencies"):
            if isinstance(pkg.get(k), dict):
                deps.update(pkg[k])
        info["kind"] = "node"
        info["language"] = "TypeScript" if ("typescript" in deps or has("tsconfig.json")) else "JavaScript"
        for fw in ("next", "react", "vue", "svelte", "express", "vite"):
            if fw in deps:
                info["framework"] = fw.capitalize()
                break
    elif has("requirements.txt") or has("pyproject.toml") or has("setup.py") or any(f.endswith(".py") for f in top):
        info["kind"] = "python"
        info["language"] = "Python"
        req = ""
        if has("requirements.txt"):
            try:
                req = read_text_nl(os.path.join(root, "requirements.txt"))[0].lower()
            except Exception:
                req = ""
        for fw in ("flask", "django", "fastapi", "streamlit"):
            if fw in req:
                info["framework"] = fw.capitalize()
                break
        if not info["framework"]:
            for f in [x for x in top if x.endswith(".py")][:20]:
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fh:
                        if re.search(r"^\s*(import|from)\s+tkinter", fh.read(4000), re.M):
                            info["framework"] = "Tkinter"
                            break
                except OSError:
                    pass
    elif has("index.html"):
        info["kind"] = "web"
        info["language"] = "HTML/CSS/JS"
    return info


# =====================================================================
# Project store (metadata, chat, history) - never touches normal chat files
# =====================================================================
class ProjectStore:
    def __init__(self):
        os.makedirs(PROJECTS_DATA, exist_ok=True)
        self.lock = threading.RLock()

    def _index(self):
        return _read_json(PROJECTS_INDEX, {"projects": []}).get("projects", [])

    def _save_index(self, items):
        _write_json(PROJECTS_INDEX, {"projects": items})

    def _dir(self, pid):
        return os.path.join(PROJECTS_DATA, pid)

    def register(self, root, name=None):
        """Returns a STABLE project id for this folder (creates one if new)."""
        root = os.path.abspath(root)
        key = os.path.normcase(root)
        with self.lock:
            items = self._index()
            for it in items:
                if os.path.normcase(it["root"]) == key:
                    it["last_opened"] = _now()
                    if name:
                        it["name"] = name
                    self._save_index(items)
                    return it["id"]
            pid = uuid.uuid4().hex[:12]
            items.append({"id": pid, "name": name or os.path.basename(root) or "project",
                          "root": root, "last_opened": _now()})
            self._save_index(items)
            return pid

    def entry(self, pid):
        for it in self._index():
            if it["id"] == pid:
                return it
        return None

    def recent(self):
        items = sorted(self._index(), key=lambda i: i.get("last_opened", ""), reverse=True)
        return [dict(i, exists=os.path.isdir(i["root"])) for i in items]

    def remove(self, pid):
        """Removes from the recent list only. Source files and saved context stay."""
        with self.lock:
            self._save_index([i for i in self._index() if i["id"] != pid])

    def rename(self, pid, name):
        with self.lock:
            items = self._index()
            for it in items:
                if it["id"] == pid:
                    it["name"] = name
            self._save_index(items)
        meta = self.meta(pid)
        meta["name"] = name
        self.save_meta(pid, meta)

    def meta(self, pid):
        path = os.path.join(self._dir(pid), "meta.json")
        meta = _read_json(path, None)
        if not isinstance(meta, dict):
            e = self.entry(pid) or {}
            meta = {"id": pid, "name": e.get("name", "project"), "root": e.get("root", ""),
                    "kind": "other", "language": "", "framework": "", "created": _now(),
                    "requirements": [], "decisions": [], "recent_changes": [],
                    "last_open_files": []}
        s = dict(DEFAULT_SETTINGS)
        s.update(meta.get("settings") or {})
        meta["settings"] = s
        for k in ("requirements", "decisions", "recent_changes", "last_open_files"):
            meta.setdefault(k, [])
        return meta

    def save_meta(self, pid, meta):
        _write_json(os.path.join(self._dir(pid), "meta.json"), meta)

    def chat(self, pid):
        return _read_json(os.path.join(self._dir(pid), "chat.json"), [])

    def save_chat(self, pid, log):
        _write_json(os.path.join(self._dir(pid), "chat.json"), log[-200:])

    def history(self, pid):
        return _read_json(os.path.join(self._dir(pid), "history.json"), [])

    def add_history(self, pid, entry):
        h = self.history(pid)
        h.append(dict(entry, time=_now()))
        _write_json(os.path.join(self._dir(pid), "history.json"), h[-200:])


# =====================================================================
# Checkpoints
# =====================================================================
def _cp_root(pid):
    return os.path.join(PROJECTS_DATA, pid, "checkpoints")


def create_checkpoint(pid, root, rel_paths, label):
    cp_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    cp_dir = os.path.join(_cp_root(pid), cp_id)
    entries = []
    for rel in rel_paths:
        full = safe_join(root, rel)
        if os.path.isfile(full):
            dest = os.path.join(cp_dir, "files", *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(full, dest)
            entries.append({"path": rel, "existed": True})
        else:
            entries.append({"path": rel, "existed": False})
    _write_json(os.path.join(cp_dir, "manifest.json"),
                {"id": cp_id, "time": _now(), "label": label, "entries": entries})
    return cp_id


def list_checkpoints(pid):
    out = []
    base = _cp_root(pid)
    if os.path.isdir(base):
        for name in os.listdir(base):
            man = _read_json(os.path.join(base, name, "manifest.json"), None)
            if man:
                out.append(man)
    return sorted(out, key=lambda m: m.get("id", ""), reverse=True)


def restore_checkpoint(pid, root, cp_id, remove_new=False, make_backup=True):
    cp_dir = os.path.join(_cp_root(pid), cp_id)
    man = _read_json(os.path.join(cp_dir, "manifest.json"), None)
    if not man:
        raise FileNotFoundError("checkpoint not found")
    if make_backup:      # restoring is itself undoable
        create_checkpoint(pid, root, [e["path"] for e in man["entries"]], f"before restoring {cp_id}")
    for e in man["entries"]:
        full = safe_join(root, e["path"])
        if e["existed"]:
            src = os.path.join(cp_dir, "files", *e["path"].split("/"))
            with open(src, "rb") as f:
                data = f.read()
            _atomic_write_bytes(full, data)
        elif remove_new and os.path.isfile(full):
            os.remove(full)
    return man


# =====================================================================
# Project tools (the agent's real operations - no UI code in here)
# =====================================================================
class ProjectTools:
    def __init__(self, root, store, pid, log=None):
        self.root = os.path.realpath(root)
        self.store = store
        self.pid = pid
        self.log = log or (lambda m: None)

    def list_files(self, limit=150):
        return build_tree_text(self.root, limit)

    def read_file(self, rel, limit=6000):
        full = safe_join(self.root, rel)
        if is_secret(rel):
            raise ValueError("secret-looking files are never sent to the model")
        text, _ = read_text_nl(full)
        return text if len(text) <= limit else text[:limit] + "\n...[truncated]"

    def search(self, query):
        return search_code(self.root, query)

    def inspect_dependencies(self):
        lines = []
        rp = os.path.join(self.root, "requirements.txt")
        if os.path.isfile(rp):
            lines.append("requirements.txt:\n" + read_text_nl(rp)[0][:1500])
        pj = os.path.join(self.root, "package.json")
        if os.path.isfile(pj):
            pkg = _read_json(pj, {})
            lines.append("package.json dependencies: " +
                         json.dumps({**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})})[:1500])
            lines.append("scripts: " + json.dumps(pkg.get("scripts", {})))
        return "\n".join(lines) or "(no dependency files)"

    def validate_changes(self, raw):
        """Checks every proposed change. Returns (good_changes, error_messages).
        Nothing is written here."""
        changes, errors, seen = [], [], set()
        if not isinstance(raw, list):
            return [], ["'changes' must be a list"]
        if len(raw) > MAX_CHANGE_FILES:
            errors.append(f"Too many files ({len(raw)}); only the first {MAX_CHANGE_FILES} are considered")
            raw = raw[:MAX_CHANGE_FILES]
        for i, ch in enumerate(raw):
            if not isinstance(ch, dict):
                errors.append(f"change {i + 1}: not an object")
                continue
            rel = str(ch.get("path", "")).strip()
            try:
                full = safe_join(self.root, rel)
            except ValueError as e:
                errors.append(f"{rel or '(empty path)'}: {e}")
                continue
            rel = rel_of(self.root, full)
            if rel in seen:
                errors.append(f"{rel}: listed twice, ignored the duplicate")
                continue
            if is_secret(rel):
                errors.append(f"{rel}: secret-looking files are never written by the agent")
                continue
            action = str(ch.get("action", "modify")).lower()
            if action not in ("create", "modify"):
                errors.append(f"{rel}: unsupported action '{action}' (deleting is not allowed)")
                continue
            if os.path.isdir(full):
                errors.append(f"{rel}: is a folder")
                continue
            exists = os.path.isfile(full)
            old, nl = "", "\n"
            if exists:
                try:
                    old, nl = read_text_nl(full)
                except Exception as e:
                    errors.append(f"{rel}: cannot read the existing file ({e})")
                    continue
            edits = ch.get("edits")
            if isinstance(edits, list) and edits:
                if not exists:
                    errors.append(f"{rel}: 'edits' need an existing file")
                    continue
                new, bad = old, False
                for e in edits:
                    f_ = e.get("find") if isinstance(e, dict) else None
                    r_ = e.get("replace") if isinstance(e, dict) else None
                    if not isinstance(f_, str) or not f_ or not isinstance(r_, str):
                        errors.append(f"{rel}: an edit is missing 'find' or 'replace'")
                        bad = True
                        break
                    n = new.count(f_)
                    if n != 1:
                        errors.append(f"{rel}: 'find' text matched {n} times (needs exactly 1): {f_[:50]!r}")
                        bad = True
                        break
                    new = new.replace(f_, r_, 1)
                if bad:
                    continue
            elif isinstance(ch.get("content"), str):
                new = ch["content"].replace("\r\n", "\n")
            else:
                errors.append(f"{rel}: needs 'content' or 'edits'")
                continue
            if len(new) > MAX_NEW_FILE_CHARS:
                errors.append(f"{rel}: file is too large")
                continue
            if exists and new == old:
                continue
            seen.add(rel)
            changes.append({"path": rel, "exists": exists, "old": old, "new": new, "nl": nl})
        return changes, errors

    def apply_changes(self, changes, label):
        """Checkpoint first, then write each file atomically. If any write
        fails, everything is rolled back from the checkpoint."""
        cp = create_checkpoint(self.pid, self.root, [c["path"] for c in changes], label)
        try:
            for c in changes:
                write_text_nl(safe_join(self.root, c["path"]), c["new"], c["nl"])
        except Exception as e:
            try:
                restore_checkpoint(self.pid, self.root, cp, remove_new=True, make_backup=False)
            except Exception:
                pass
            raise RuntimeError(f"Write failed ({e}). Changes were rolled back.") from e
        return cp

    def rename_file(self, old_rel, new_rel):
        a, b = safe_join(self.root, old_rel), safe_join(self.root, new_rel)
        if not os.path.exists(a):
            raise FileNotFoundError(old_rel)
        if os.path.exists(b):
            raise FileExistsError(f"{new_rel} already exists")
        os.makedirs(os.path.dirname(b), exist_ok=True)
        os.rename(a, b)


# =====================================================================
# Agent: context building + Ollama calls (no UI code in here either)
# =====================================================================
SYSTEM_PROMPT = """You are Vibe Code, a careful coding agent inside a desktop IDE. You work ONLY inside the user's project folder.
You get the project file tree and the contents of some files. You have NOT run anything, so never claim you ran, tested or verified code.
If you need a file that was not provided, say so in "explanation" and ask the user to open it. Never invent file contents.
Reply with ONE JSON object and nothing else:
{"explanation": "short plain-English summary of what you did or found",
 "plan": ["short step", "..."],
 "changes": [
   {"path": "relative/path.ext", "action": "create", "content": "full file text"},
   {"path": "relative/path.ext", "action": "modify", "edits": [{"find": "exact existing text, unique in the file", "replace": "new text"}]}
 ]}
Rules:
- Paths are relative to the project root, use forward slashes, never absolute, never "..".
- Prefer small "edits" on existing files. "find" must be copied EXACTLY from the provided file and appear exactly once. Use full "content" only for new files.
- Do not delete files. Do not touch unrelated files. Never write secrets or .env files.
- For a new project: create a sensible folder structure, a README.md, and requirements.txt or package.json if needed.
- If the user only asks a question, return "changes": [].
- Write complete, runnable code. No placeholders like "..." inside files."""

STOP_WORDS = set("the and for with that this from into make add fix find build please then have what when how why can "
                 "you your are not but all any new use using want need file files code project app page".split())


def build_context(tools, request, active_rel, buffers, focus_files, budget, log):
    """Pick a small, relevant slice of the project instead of sending everything."""
    parts, used, done = [], 0, set()
    root = tools.root

    def add(rel, why):
        nonlocal used
        if rel in done or is_secret(rel):
            return
        try:
            if rel in buffers:
                text = buffers[rel]
            else:
                full = safe_join(root, rel)
                if not os.path.isfile(full) or not looks_text(full):
                    return
                text, _ = read_text_nl(full)
        except Exception:
            return
        if len(text) > 6000:
            text = text[:6000] + "\n...[truncated]"
        block = f"--- FILE: {rel} ---\n{text}\n"
        if used + len(block) > budget:
            return
        parts.append(block)
        used += len(block)
        done.add(rel)
        log(f"Inspected {rel} ({why})")

    for rel in focus_files:
        add(rel, "mentioned in the error")
    if active_rel:
        add(active_rel, "open in the editor")
    low = request.lower()
    for rel in iter_files(root, 1500):
        if os.path.basename(rel).lower() in low or rel.lower() in low:
            add(rel, "named in the request")
    for rel in ("README.md", "requirements.txt", "package.json"):
        add(rel, "project overview")
    terms = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", low) if w not in STOP_WORDS]
    terms = sorted(set(terms), key=len, reverse=True)[:5]
    if terms:
        for rel in rank_files(root, terms):
            add(rel, "matches your keywords")
    return "\n".join(parts) if parts else "(no file contents were selected)"


def parse_model_json(raw):
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("The model did not return JSON")
        obj = json.loads(s[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("The model's JSON was not an object")
    obj["explanation"] = str(obj.get("explanation", "") or "")
    plan = obj.get("plan")
    obj["plan"] = [str(x) for x in plan if x] if isinstance(plan, list) else []
    if not isinstance(obj.get("changes"), list):
        obj["changes"] = []
    return obj


def list_models():
    if ollama is None:
        raise RuntimeError("The 'ollama' Python package is not installed.")
    r = ollama.list()
    models = getattr(r, "models", None)
    if models is None:
        models = r.get("models", []) if isinstance(r, dict) else []
    names = []
    for m in models:
        n = (m.get("model") or m.get("name")) if isinstance(m, dict) else (getattr(m, "model", None) or getattr(m, "name", None))
        if n and "embed" not in n.lower():
            names.append(n)
    return sorted(names)


def model_installed(model, names):
    return model in names or (":" not in model and model + ":latest" in names)


def friendly_ollama_error(e, model):
    s = str(e).lower()
    if "not found" in s or "pull" in s:
        return (f"The model '{model}' is not installed in Ollama.\n"
                f"Open a terminal and run:  ollama pull {model}\n"
                f"or pick another installed model from the dropdown above.")
    if "connect" in s or "refused" in s or "10061" in s or "failed to connect" in s:
        return "I can't reach Ollama. Start the Ollama app (or run 'ollama serve') and try again."
    return f"Ollama error: {e}"


def call_model(model, messages, num_ctx, cancelled, on_progress):
    stream = ollama.chat(model=model, messages=messages, stream=True, format="json",
                         keep_alive="5m", options={"num_ctx": num_ctx, "temperature": 0.2})
    out, last = [], 0.0
    for chunk in stream:
        if cancelled():
            return None
        piece = chunk.get("message", {}).get("content", "") or ""
        out.append(piece)
        if time.time() - last > 0.4:
            last = time.time()
            on_progress(sum(len(p) for p in out))
    return "".join(out)


def describe_images(paths, names):
    """Screenshots -> text description using the vision model."""
    notes = []
    if VISION_MODEL not in [n.split(":")[0] for n in names]:
        return f"(A screenshot was attached but the vision model is missing. Run: ollama pull {VISION_MODEL})"
    for p in paths:
        r = ollama.chat(model=VISION_MODEL, keep_alive="2m", messages=[{
            "role": "user", "images": [p],
            "content": "Describe this UI screenshot for a developer: layout, colours, components, visible text."}])
        notes.append(f"Screenshot '{os.path.basename(p)}': " + r["message"]["content"].strip()[:1500])
    return "\n".join(notes)


# =====================================================================
# Process runner (worker threads; never blocks Tk)
# =====================================================================
def clean_env():
    env = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if any(s in ku for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")):
            continue
        env[k] = v
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class ProcRunner:
    def __init__(self):
        self.proc = None
        self.active = False
        self.stopped = False
        self.timed_out = False

    def start(self, cmd, cwd, on_line, on_done, timeout=120, shell=False):
        self.active, self.stopped, self.timed_out = True, False, False

        def worker():
            try:
                kw = dict(cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8",
                          errors="replace", bufsize=1, env=clean_env(), shell=shell)
                if os.name == "nt":
                    kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | _NOWIN
                else:
                    kw["start_new_session"] = True
                self.proc = subprocess.Popen(cmd, **kw)
            except Exception as e:
                on_line("err", f"Could not start the process: {e}\n")
                self.active = False
                on_done(-1, False, False)
                return
            timer = None
            if timeout:
                timer = threading.Timer(timeout, self._timeout)
                timer.daemon = True
                timer.start()

            def pump(stream, tag):
                try:
                    for line in stream:
                        on_line(tag, line)
                except Exception:
                    pass

            t1 = threading.Thread(target=pump, args=(self.proc.stdout, "out"), daemon=True)
            t2 = threading.Thread(target=pump, args=(self.proc.stderr, "err"), daemon=True)
            t1.start()
            t2.start()
            code = self.proc.wait()
            t1.join(2)
            t2.join(2)
            if timer:
                timer.cancel()
            self.proc = None
            self.active = False
            on_done(code, self.timed_out, self.stopped)

        threading.Thread(target=worker, daemon=True).start()

    def _kill(self, p):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True,
                               creationflags=_NOWIN, timeout=10)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _timeout(self):
        p = self.proc
        if p is not None and p.poll() is None:
            self.timed_out = True
            self._kill(p)

    def stop(self):
        p = self.proc
        if p is not None and p.poll() is None:
            self.stopped = True
            self._kill(p)


def find_python(root, settings):
    """Pick the interpreter for the PROJECT (not the one running AP)."""
    custom = (settings.get("python") or "").strip()
    if custom and os.path.isfile(custom):
        return custom, "custom interpreter from Project Settings"
    for d in (".venv", "venv", "env"):
        for sub in (("Scripts", "python.exe"), ("bin", "python")):
            c = os.path.join(root, d, *sub)
            if os.path.isfile(c):
                return c, f"project virtual environment ({d})"
    for name in ("python", "python3", "py"):
        c = shutil.which(name)
        if c:
            return c, f"'{name}' found on PATH"
    return None, "no Python interpreter found"


def detect_run_command(root, meta):
    """Returns a dict {mode, cmd, display, shell} or None."""
    s = meta.get("settings", {})
    custom = (s.get("run_command") or "").strip()
    if custom:
        return {"mode": "process", "cmd": custom, "display": custom, "shell": True}
    kind = meta.get("kind")
    if kind == "python":
        py, _ = find_python(root, s)
        if not py:
            return {"mode": "error", "message": "No Python interpreter found. Install Python or set the path in Project Settings."}
        for entry in ("main.py", "app.py", "run.py", "__main__.py"):
            if os.path.isfile(os.path.join(root, entry)):
                return {"mode": "process", "cmd": [py, entry], "display": f"python {entry}", "shell": False}
        pys = [f for f in os.listdir(root) if f.endswith(".py")]
        if len(pys) == 1:
            return {"mode": "process", "cmd": [py, pys[0]], "display": f"python {pys[0]}", "shell": False}
        return None
    if kind == "node":
        npm = shutil.which("npm")
        if not npm:
            return {"mode": "error", "message": "Node.js / npm was not found. Install it from nodejs.org, then restart AP."}
        pkg = _read_json(os.path.join(root, "package.json"), {})
        scripts = pkg.get("scripts", {}) if isinstance(pkg, dict) else {}
        for name in ("dev", "start"):
            if name in scripts:
                return {"mode": "process", "cmd": [npm, "run", name], "display": f"npm run {name}", "shell": False}
        return None
    if kind == "web":
        return {"mode": "preview"}
    return None


_SYNTAX_CHECK = r'''
import os, sys
bad = 0; n = 0
skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", ".vibe"}
for d, dirs, files in os.walk("."):
    dirs[:] = [x for x in dirs if x not in skip]
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(d, f); n += 1
            try:
                compile(open(p, encoding="utf-8").read(), p, "exec")
            except SyntaxError as e:
                bad += 1
                print('File "%s", line %s' % (p, e.lineno))
                print("SyntaxError: %s" % e.msg, file=sys.stderr)
            except Exception as e:
                bad += 1
                print("%s: %r" % (p, e), file=sys.stderr)
print("Checked %d Python file(s): %s" % (n, "OK" if not bad else "%d with errors" % bad))
sys.exit(1 if bad else 0)
'''

_DEP_CHECK = ("import importlib.util, json, sys\n"
              "def miss(x):\n"
              "    try: return importlib.util.find_spec(x) is None\n"
              "    except Exception: return True\n"
              "print(json.dumps([x for x in json.loads(sys.argv[1]) if miss(x)]))\n")


def python_import_names(root):
    names, local = set(), set()
    files = [r for r in iter_files(root, 1500) if r.endswith(".py")]
    for r in files:
        local.add(r.split("/")[0].replace(".py", ""))
        local.add(os.path.splitext(os.path.basename(r))[0])
    for r in files:
        try:
            tree = ast.parse(read_text_nl(os.path.join(root, r))[0])
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return sorted(n for n in names - local if n)


def parse_problems(output, root):
    """Find 'File "x", line N' (Python) and 'x.js:N' (Node) in output."""
    probs, pending = [], []
    root_real = os.path.realpath(root)
    for line in output.splitlines():
        m = re.search(r'File "(.+?)", line (\d+)', line)
        m2 = re.search(r'([\w./\\:-]+\.(?:js|jsx|ts|tsx|mjs)):(\d+)', line) if not m else None
        mm = m or m2
        if mm:
            p = mm.group(1)
            full = os.path.realpath(p if os.path.isabs(p) else os.path.join(root_real, p))
            try:
                inside = os.path.commonpath([full, root_real]) == root_real
            except ValueError:
                inside = False
            if inside and os.path.isfile(full):
                prob = {"file": rel_of(root_real, full), "line": int(mm.group(2)), "msg": ""}
                probs.append(prob)
                pending.append(prob)
            continue
        if re.match(r"^\s*\w*(Error|Exception)\b", line) or re.match(r"^\s*(Uncaught|TypeError|ReferenceError)", line):
            for pr in pending:
                pr["msg"] = line.strip()[:200]
            pending = []
    seen, out = set(), []
    for pr in probs:
        k = (pr["file"], pr["line"])
        if k not in seen:
            seen.add(k)
            out.append(pr)
    return out[-30:]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def open_in_os(path):
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Open", str(e))


# =====================================================================
# Syntax highlighting rules
# =====================================================================
def _kw(words):
    return re.compile(r"\b(?:" + "|".join(words.split()) + r")\b")


_PY_KW = ("False None True and as assert async await break class continue def del elif else except finally "
          "for from global if import in is lambda nonlocal not or pass raise return try while with yield self")
_JS_KW = ("break case catch class const continue debugger default delete do else export extends finally for "
          "function if import in instanceof let new return super switch this throw try typeof var void while "
          "with yield async await of null undefined true false")
_TS_KW = _JS_KW + " interface type enum implements namespace public private protected readonly abstract keyof"
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_STR_PY = re.compile(r"""'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*\"""")
_STR_JS = re.compile(r"""`(?:\\.|[^`\\])*`|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*\"""")
_STR_Q = re.compile(r"""'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*\"""")

SYN_TAGS = ["keyword", "number", "string", "attr", "func", "tag", "comment"]
SYN_COLORS = {"keyword": "#c4b5ff", "number": "#ffb86c", "string": "#8be9a8", "attr": "#ffd479",
              "func": "#7cc7ff", "tag": "#ff7eb6", "comment": "#6a6f9a"}

LANG_RULES = {
    "python": [("keyword", _kw(_PY_KW)), ("number", _NUM), ("string", _STR_PY),
               ("func", re.compile(r"(?<=def )\w+|(?<=class )\w+")),
               ("comment", re.compile(r"#[^\n]*"))],
    "javascript": [("keyword", _kw(_JS_KW)), ("number", _NUM), ("string", _STR_JS),
                   ("func", re.compile(r"(?<=function )\w+|(?<=class )\w+")),
                   ("comment", re.compile(r"//[^\n]*|/\*[\s\S]*?\*/"))],
    "typescript": [("keyword", _kw(_TS_KW)), ("number", _NUM), ("string", _STR_JS),
                   ("func", re.compile(r"(?<=function )\w+|(?<=class )\w+|(?<=interface )\w+")),
                   ("comment", re.compile(r"//[^\n]*|/\*[\s\S]*?\*/"))],
    "html": [("string", _STR_Q), ("attr", re.compile(r"\b[\w:-]+(?==)")),
             ("tag", re.compile(r"</?[A-Za-z][\w:-]*|/?>")),
             ("comment", re.compile(r"<!--[\s\S]*?-->"))],
    "css": [("number", re.compile(r"\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms)?\b")), ("string", _STR_Q),
            ("attr", re.compile(r"[\w-]+(?=\s*:)")), ("func", re.compile(r"[.#][\w-]+")),
            ("comment", re.compile(r"/\*[\s\S]*?\*/"))],
    "json": [("keyword", _kw("true false null")), ("number", re.compile(r"-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")),
             ("string", re.compile(r'"(?:\\.|[^"\\\n])*"')),
             ("attr", re.compile(r'"(?:\\.|[^"\\\n])*"(?=\s*:)'))],
    "markdown": [("string", re.compile(r"`[^`\n]+`|```[\s\S]*?```")), ("keyword", re.compile(r"\*\*[^*\n]+\*\*")),
                 ("func", re.compile(r"^#{1,6}\s.*$", re.M))],
}
EXT_LANG = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
            ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript", ".html": "html",
            ".htm": "html", ".css": "css", ".json": "json", ".md": "markdown"}


# =====================================================================
# UI helpers
# =====================================================================
class OutputPane:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=P["bg"])
        self.text = tk.Text(self.frame, bg=P["bg"], fg=P["text"], font=(MONO, 10), bd=0, wrap="word",
                            state="disabled", padx=8, pady=6, insertbackground=P["accent2"])
        sb = tk.Scrollbar(self.frame, orient="vertical", command=self.text.yview)
        self.sb = sb
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        for tag, col in (("out", P["text"]), ("err", P["err"]), ("cmd", P["accent2"]),
                         ("info", P["dim"]), ("ok", P["ok"]), ("warn", P["warn"])):
            self.text.tag_configure(tag, foreground=col)

    def append(self, s, tag="out"):
        t = self.text
        t.config(state="normal")
        t.insert("end", s, tag)
        n = int(t.index("end-1c").split(".")[0])
        if n > 6000:
            t.delete("1.0", f"{n - 5000}.0")
        t.config(state="disabled")
        t.see("end")

    def clear(self):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")


class EditorTab:
    pass


def _safe_grab(top):
    try:
        top.after(150, lambda: _safe_grab(top))
    except tk.TclError:
        pass


class ReviewDialog:
    """Shows proposed changes as diffs. Nothing is written until Apply."""

    def __init__(self, ws, changes, summary):
        self.ws, self.changes = ws, changes
        self.checked = [True] * len(changes)
        self.top = tk.Toplevel(ws.win)
        top = self.top
        top.title("Review changes")
        top.geometry("1020x640")
        top.configure(bg=P["bg"])
        top.transient(ws.win)
        top.protocol("WM_DELETE_WINDOW", self.reject)
        tk.Label(top, text="Review proposed changes - nothing has been written yet", bg=P["bg"], fg=P["accent2"],
                 font=(UI_FONT, 11, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(top, text=(summary or "")[:300], bg=P["bg"], fg=P["dim"], font=(UI_FONT, 9),
                 wraplength=960, justify="left").pack(anchor="w", padx=14)
        tk.Label(top, text="Double-click a file to include / exclude it.", bg=P["bg"], fg=P["dim"],
                 font=(UI_FONT, 8)).pack(anchor="w", padx=14, pady=(0, 6))
        body = tk.Frame(top, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=14)
        self.lb = tk.Listbox(body, width=44, bg=P["panel2"], fg=P["text"], bd=0, highlightthickness=1,
                             highlightbackground=P["border"], selectbackground=P["sel"],
                             selectforeground=P["accent2"], font=(MONO, 9), activestyle="none", exportselection=False)
        self.lb.pack(side="left", fill="y")
        right = tk.Frame(body, bg=P["panel2"])
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.diff = tk.Text(right, bg=P["panel2"], fg=P["text"], font=(MONO, 10), bd=0, wrap="none",
                            state="disabled", padx=8, pady=6)
        sy = tk.Scrollbar(right, orient="vertical", command=self.diff.yview)
        sx = tk.Scrollbar(right, orient="horizontal", command=self.diff.xview)
        self.diff.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.diff.pack(side="left", fill="both", expand=True)
        for tag, col in (("add", "#4ade80"), ("del", "#ff7a93"), ("hunk", P["accent2"]), ("meta", P["dim"])):
            self.diff.tag_configure(tag, foreground=col)
        self.warn = tk.Label(top, text="", bg=P["bg"], fg=P["warn"], font=(UI_FONT, 9), wraplength=960, justify="left")
        self.warn.pack(anchor="w", padx=14, pady=(6, 0))
        btns = tk.Frame(top, bg=P["bg"])
        btns.pack(fill="x", padx=14, pady=12)
        ws.mkbtn(btns, "Apply Changes", self.apply, primary=True).pack(side="right", padx=(6, 0))
        ws.mkbtn(btns, "Reject", self.reject).pack(side="right")
        self.counts = []
        for c in changes:
            a = d = 0
            for ln in self._udiff(c):
                if ln.startswith("+") and not ln.startswith("+++"):
                    a += 1
                elif ln.startswith("-") and not ln.startswith("---"):
                    d += 1
            self.counts.append((a, d))
        self.refresh_list()
        conflicts = [c["path"] for c in changes if c["path"] in ws.tabs and ws.tabs[c["path"]].modified]
        if conflicts:
            self.warn.config(text="Unsaved edits in the editor for: " + ", ".join(conflicts) +
                                  " - you will be asked before they are overwritten.")
        self.lb.bind("<<ListboxSelect>>", lambda e: self.show())
        self.lb.bind("<Double-Button-1>", self.toggle)
        self.lb.selection_set(0)
        self.show()
        top.after(150, lambda: _safe_grab(top))

    def _udiff(self, c):
        return list(difflib.unified_diff(c["old"].splitlines(), c["new"].splitlines(),
                                         "a/" + c["path"], "b/" + c["path"], lineterm="", n=3))

    def refresh_list(self):
        sel = self.lb.curselection()
        self.lb.delete(0, "end")
        for i, c in enumerate(self.changes):
            a, d = self.counts[i]
            box = "[x]" if self.checked[i] else "[ ]"
            kind = "MOD" if c["exists"] else "NEW"
            self.lb.insert("end", f"{box} {kind} {c['path']}  +{a}/-{d}")
        if sel:
            self.lb.selection_set(sel[0])

    def toggle(self, event=None):
        sel = self.lb.curselection()
        if sel:
            self.checked[sel[0]] = not self.checked[sel[0]]
            self.refresh_list()

    def show(self):
        sel = self.lb.curselection()
        if not sel:
            return
        c = self.changes[sel[0]]
        lines = self._udiff(c) if c["exists"] else ["--- new file: " + c["path"]] + ["+" + l for l in c["new"].splitlines()]
        self.diff.config(state="normal")
        self.diff.delete("1.0", "end")
        for ln in lines:
            tag = "meta"
            if ln.startswith("@@"):
                tag = "hunk"
            elif ln.startswith("+") and not ln.startswith("+++"):
                tag = "add"
            elif ln.startswith("-") and not ln.startswith("---"):
                tag = "del"
            elif not (ln.startswith("---") or ln.startswith("+++")):
                tag = None
            self.diff.insert("end", ln + "\n", tag or ())
        self.diff.config(state="disabled")

    def apply(self):
        chosen = [c for c, k in zip(self.changes, self.checked) if k]
        self.top.destroy()
        if chosen:
            self.ws.apply_changes(chosen, "AI change")
        else:
            self.ws.on_rejected()

    def reject(self):
        self.top.destroy()
        self.ws.on_rejected()


# =====================================================================
# The workspace
# =====================================================================
class VibeWorkspace:
    def __init__(self, parent, theme=None, image_fn=None):
        # Vibe Code keeps its own JARVIS palette; the main app's theme is ignored on purpose.
        self.parent, self.image_fn = parent, image_fn
        self.store = ProjectStore()
        self.pid = self.root = self.meta = self.tools = None
        self.tabs, self.tab_order, self.active = {}, [], None
        self.q = queue.Queue()
        self.closed = False
        self.busy = False
        self.token = 0
        self.attachments, self.chat_log = [], []
        self.models, self.models_loaded = [], False
        self.confirmed_cmds, self.terminal_ok = set(), False
        self.runner, self.preview_runner = ProcRunner(), ProcRunner()
        self.preview_url, self.preview_opened = None, False
        self.last_run, self.fix_attempts, self.pending_verify = None, 0, None
        self.problems, self.problem_lines = [], {}
        self.expanded, self.tree_items = set(), []

        self.win = tk.Toplevel(parent)
        self.win.title("AP - Vibe Code")
        self.win.geometry("1380x860")
        self.win.minsize(900, 600)
        self.win.configure(bg=P["bg"])
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self._poll()
        self.refresh_models()
        self.win.bind("<Control-s>", lambda e: (self.save_current(), "break")[1])
        self.win.bind("<Control-f>", lambda e: (self.toggle_find(), "break")[1])
        self.win.bind("<Control-b>", lambda e: (self.toggle_pane("left"), "break")[1])
        self.win.bind("<Control-l>", lambda e: (self.toggle_pane("right"), self.input.focus_set(), "break")[2])
        self.show_home()

    # ---------------- thread-safe UI queue ----------------
    def ui(self, fn, *args):
        self.q.put((None, fn, args))

    def _poll(self):
        if self.closed:
            return
        for _ in range(80):
            try:
                tok, fn, args = self.q.get_nowait()
            except queue.Empty:
                break
            if tok is not None and tok != self.token:
                continue                      # result of a task that was stopped
            try:
                fn(*args)
            except tk.TclError:
                pass
            except Exception as e:
                print("Vibe Code UI error:", e)
        try:
            self.win.after(40, self._poll)
        except tk.TclError:
            pass

    # ---------------- widgets ----------------
    def mkbtn(self, parent, text, cmd, primary=False):
        bg = P["accent"] if primary else P["sel"]
        hov = P["accent2"] if primary else "#25324A"
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg="white" if primary else P["text"], bd=0,
                      relief="flat", cursor="hand2", font=(UI_FONT, 10, "bold" if primary else "normal"),
                      activebackground=hov, activeforeground="white", disabledforeground=P["dim"], padx=12, pady=5)
        b.bind("<Enter>", lambda e: b.config(bg=hov) if str(b["state"]) != "disabled" else None)
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _menu(self, parent=None, **kw):
        return tk.Menu(parent or self.win, tearoff=0, bg=P["panel"], fg=P["text"],
                       activebackground=P["accent"], activeforeground="white", bd=0, **kw)

    def _entry(self, parent, **kw):
        return tk.Entry(parent, bg=P["panel2"], fg=P["text"], insertbackground=P["accent2"], bd=0,
                        relief="flat", highlightthickness=1, highlightbackground=P["border"],
                        highlightcolor=P["accent"], font=(UI_FONT, 9), **kw)

    # =================================================================
    # NEW UI: Projects home  +  AI-first workspace
    # =================================================================
    def _build(self):
        self.home_page = tk.Frame(self.win, bg=P["bg"])
        self.work_page = tk.Frame(self.win, bg=P["bg"])
        self._mt_cache = {}
        self._build_home(self.home_page)
        self._build_workspace(self.work_page)
        self._bind_keys()
        self.win.after(100, self._prog_tick)

    def _bind_keys(self):
        w = self.win
        pass
        pass
        w.bind("<F6>", lambda e: self.toggle_editor())
        w.bind("<F7>", lambda e: self.toggle_drawer())
        w.bind("<MouseWheel>", self._home_wheel, add="+")

    def show_shortcuts(self):
        messagebox.showinfo("Keyboard shortcuts",
                            "Enter            Send to the AI\n"
                            "Shift+Enter      New line in the prompt\n"
                            "Ctrl+L           Jump to the prompt box\n"
                            "Ctrl+B           Show / hide Files\n"
                            "F6               Show / hide the code editor\n"
                            "F7               Show / hide the output panel\n"
                            "F5               Run\n"
                            "Shift+F5         Stop\n"
                            "Ctrl+S           Save file\n"
                            "Ctrl+F           Find in file", parent=self.win)

    # ---------------- Projects home ----------------
    def _build_home(self, page):
        head = tk.Frame(page, bg=P["bg"])
        head.pack(fill="x", padx=48, pady=(38, 0))
        tk.Label(head, text="</>", bg=P["bg"], fg=P["accent2"], font=(MONO, 22, "bold")).pack(side="left")
        tk.Label(head, text="Vibe Code", bg=P["bg"], fg=P["text"], font=(UI_FONT, 24, "bold")).pack(side="left", padx=(12, 0))
        tk.Label(page, text="Start with a project, then describe what you want and let the AI build it.",
                 bg=P["bg"], fg=P["dim"], font=(UI_FONT, 11)).pack(anchor="w", padx=50, pady=(4, 22))
        bar = tk.Frame(page, bg=P["bg"])
        bar.pack(fill="x", padx=48)
        b = self.mkbtn(bar, "+  New Project", self.new_project_dialog, primary=True)
        b.config(padx=20, pady=9, font=(UI_FONT, 11, "bold"))
        b.pack(side="right")
        b2 = self.mkbtn(bar, "Open Project", self.open_project_dialog)
        b2.config(padx=16, pady=9, font=(UI_FONT, 11))
        b2.pack(side="right", padx=10)
        self.home_search = tk.StringVar()
        tk.Label(bar, text="Search", bg=P["bg"], fg=P["dim"], font=(UI_FONT, 10)).pack(side="left", padx=(0, 8))
        se = self._entry(bar, textvariable=self.home_search, width=32)
        se.config(font=(UI_FONT, 11))
        se.pack(side="left", ipady=7)
        self.home_search.trace_add("write", lambda *a: self.render_home())
        tk.Label(page, text="RECENT PROJECTS", bg=P["bg"], fg=P["glow"], font=(MONO, 9, "bold")).pack(
            anchor="w", padx=50, pady=(26, 4))
        wrap = tk.Frame(page, bg=P["bg"])
        wrap.pack(fill="both", expand=True, padx=(40, 30), pady=(0, 16))
        self.home_canvas = tk.Canvas(wrap, bg=P["bg"], highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=self.home_canvas.yview)
        self.home_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.home_canvas.pack(side="left", fill="both", expand=True)
        self.home_inner = tk.Frame(self.home_canvas, bg=P["bg"])
        self._home_win = self.home_canvas.create_window((0, 0), window=self.home_inner, anchor="nw")
        self.home_inner.bind("<Configure>", lambda e: self.home_canvas.configure(scrollregion=self.home_canvas.bbox("all")))
        self.home_canvas.bind("<Configure>", self._home_resize)

    def _home_resize(self, e):
        self.home_canvas.itemconfigure(self._home_win, width=e.width)
        cols = max(1, min(4, (e.width - 10) // 340))
        if cols != getattr(self, "_home_cols", None):
            self._home_cols = cols
            self.render_home()

    def _home_wheel(self, e):
        try:
            if self.home_page.winfo_ismapped():
                c = self.home_canvas
                bb = c.bbox("all")
                if bb and bb[3] > c.winfo_height():
                    c.yview_scroll(int(-e.delta / 120), "units")
        except tk.TclError:
            pass

    def _mtime(self, it):
        hit = self._mt_cache.get(it["root"])
        if hit and time.time() - hit[0] < 30:
            return hit[1]
        ts = latest_mtime(it["root"])
        if ts <= 0:
            try:
                ts = datetime.fromisoformat(it.get("last_opened", "")).timestamp()
            except Exception:
                ts = time.time()
        self._mt_cache[it["root"]] = (time.time(), ts)
        return ts

    def render_home(self):
        inner = self.home_inner
        for w in inner.winfo_children():
            w.destroy()
        cols = getattr(self, "_home_cols", 3)
        for c in range(6):
            inner.grid_columnconfigure(c, weight=1 if c < cols else 0, uniform="cards" if c < cols else "")
        items = self.store.recent()
        q = self.home_search.get().strip().lower()
        shown = [i for i in items if q in i["name"].lower() or q in i["root"].lower()]
        if not items:
            self._home_empty(inner, cols)
            return
        if not shown:
            tk.Label(inner, text=f'No projects match "{q}".', bg=P["bg"], fg=P["dim"],
                     font=(UI_FONT, 11)).grid(row=0, column=0, columnspan=cols, pady=40)
            return
        for n, it in enumerate(shown):
            self._project_card(inner, it).grid(row=n // cols, column=n % cols, sticky="nsew", padx=8, pady=8)

    def _home_empty(self, inner, cols):
        box = tk.Frame(inner, bg=P["bg"])
        box.grid(row=0, column=0, columnspan=cols, pady=(40, 0))
        tk.Label(box, text="</>", bg=P["bg"], fg=P["glow"], font=(MONO, 40, "bold")).pack()
        tk.Label(box, text="Create your first project", bg=P["bg"], fg=P["text"],
                 font=(UI_FONT, 18, "bold")).pack(pady=(8, 4))
        tk.Label(box, text="A project is a folder on your computer. Create one, then describe what you\n"
                           "want to build and the AI writes the files for you. You review every change.",
                 bg=P["bg"], fg=P["dim"], font=(UI_FONT, 11), justify="center").pack()
        row = tk.Frame(box, bg=P["bg"])
        row.pack(pady=18)
        b = self.mkbtn(row, "+  New Project", self.new_project_dialog, primary=True)
        b.config(padx=20, pady=9, font=(UI_FONT, 11, "bold"))
        b.pack(side="left", padx=6)
        b2 = self.mkbtn(row, "Open existing folder", self.open_project_dialog)
        b2.config(padx=16, pady=9, font=(UI_FONT, 11))
        b2.pack(side="left", padx=6)
        tk.Label(box, text='Try: "Build a modern student management app with login, a dashboard and a database."',
                 bg=P["bg"], fg=P["accent2"], font=(UI_FONT, 10, "italic")).pack()

    def _project_card(self, parent, it):
        ok = it["exists"]
        meta = self.store.meta(it["id"])
        txt, col = KIND_ICON.get(meta.get("kind"), KIND_ICON["other"])
        card = tk.Frame(parent, bg=P["card"], cursor="hand2", highlightthickness=1, highlightbackground=P["border"])
        top = tk.Frame(card, bg=P["card"])
        top.pack(fill="x", padx=16, pady=(16, 0))
        icon = tk.Label(top, text=txt, bg=col if ok else P["border"], fg="white", font=(MONO, 12, "bold"),
                        width=4, height=2)
        icon.pack(side="left")
        mb = tk.Menubutton(top, text="•••", bg=P["card"], fg=P["dim"], bd=0, relief="flat",
                           font=(UI_FONT, 11, "bold"), activebackground=P["cardh"], activeforeground=P["text"],
                           cursor="hand2", padx=6)
        mm = self._menu(mb)
        mm.add_command(label="Open", command=lambda: self.open_recent(it))
        mm.add_command(label="Rename...", command=lambda: self.rename_recent(it))
        mm.add_command(label="Show folder", command=lambda: open_in_os(it["root"]),
                       state="normal" if ok else "disabled")
        mm.add_separator()
        mm.add_command(label="Remove from list", command=lambda: self.remove_from_home(it))
        mm.add_command(label="Delete project folder...", command=lambda: self.delete_project_folder(it))
        mb.config(menu=mm)
        mb.pack(side="right", anchor="n")
        name = tk.Label(card, text=it["name"], bg=P["card"], fg=P["text"], font=(UI_FONT, 12, "bold"), anchor="w")
        name.pack(fill="x", padx=16, pady=(14, 0))
        root = it["root"]
        loc = tk.Label(card, text=root if len(root) <= 42 else "..." + root[-39:], bg=P["card"], fg=P["dim"],
                       font=(MONO, 8), anchor="w")
        loc.pack(fill="x", padx=16, pady=(2, 0))
        if ok:
            when_txt, when_col = "Edited " + time_ago(self._mtime(it)), P["dim"]
        else:
            when_txt, when_col = "Folder missing - click to remove", P["warn"]
        when = tk.Label(card, text=when_txt, bg=P["card"], fg=when_col, font=(UI_FONT, 9), anchor="w")
        when.pack(fill="x", padx=16, pady=(8, 16))
        tint = [card, top, name, loc, when, mb]

        def hv(on):
            c = P["cardh"] if on else P["card"]
            for wdg in tint:
                wdg.config(bg=c)
            card.config(highlightbackground=P["accent"] if on else P["border"])

        def leave(e):
            try:
                x, y = card.winfo_pointerxy()
                w = card.winfo_containing(x, y)
                inside = w is not None and str(w).startswith(str(card))
            except Exception:
                inside = False
            if not inside:
                hv(False)
        for wdg in tint + [icon]:
            wdg.bind("<Enter>", lambda e: hv(True), add="+")
            wdg.bind("<Leave>", leave, add="+")
        for wdg in (card, top, icon, name, loc, when):
            wdg.bind("<Button-1>", lambda e: self.open_recent(it))
        return card

    def rename_recent(self, it):
        name = simpledialog.askstring("Rename project", "New name (the folder is not renamed):",
                                      initialvalue=it["name"], parent=self.win)
        if name and name.strip():
            self.store.rename(it["id"], name.strip())
            self.render_home()

    def remove_from_home(self, it):
        if messagebox.askyesno("Remove from list", f"Remove '{it['name']}' from the project list?\n"
                                                    "Your files are NOT deleted.", parent=self.win):
            self.store.remove(it["id"])
            self.render_home()

    def delete_project_folder(self, it):
        root = os.path.abspath(it["root"])
        home = os.path.expanduser("~")
        here = os.path.dirname(os.path.abspath(__file__))
        if (os.path.dirname(root) == root or _is_inside(home, root) or _is_inside(BASE_DIR, root)
                or _is_inside(here, root)):
            messagebox.showerror("Delete project",
                                 "This folder contains your user profile, a drive root or AP itself, so Vibe Code "
                                 "will never delete it.\nUse 'Remove from list' instead.", parent=self.win)
            return
        if not os.path.isdir(root):
            self.store.remove(it["id"])
            self.render_home()
            return
        typed = simpledialog.askstring(
            "Delete project folder",
            f"This deletes the folder and ALL files in it:\n{root}\n\nType the project name to confirm:\n{it['name']}",
            parent=self.win)
        if typed is None:
            return
        if typed.strip() != it["name"]:
            messagebox.showinfo("Delete project", "The name did not match. Nothing was deleted.", parent=self.win)
            return

        def _unlock(func, path, exc):
            try:
                os.chmod(path, 0o700)
                func(path)
            except OSError:
                pass
        try:
            try:
                from send2trash import send2trash        # optional: pip install send2trash
                send2trash(root)
            except ImportError:
                shutil.rmtree(root, onerror=_unlock)
        except Exception as e:
            messagebox.showerror("Delete project", str(e), parent=self.win)
            return
        self.store.remove(it["id"])
        shutil.rmtree(os.path.join(PROJECTS_DATA, it["id"]), ignore_errors=True)
        self.render_home()

    # ---------------- page switching ----------------
    def show_home(self):
        self._reset_workspace_view()
        self.work_page.pack_forget()
        self.home_page.pack(fill="both", expand=True)
        self.win.title("AP - Vibe Code")
        self.render_home()

    def show_workspace(self):
        self.home_page.pack_forget()
        self.work_page.pack(fill="both", expand=True)
        self.show_editor_pane(False)
        self.win.title(f"AP - Vibe Code - {self.meta['name']}" if self.meta else "AP - Vibe Code")
        self.win.after(100, self.input.focus_set)

    def _reset_workspace_view(self):
        self.close_drawer()
        self.show_editor_pane(False)
        if self._pane_shown(self.pane_left):
            self.hpane.forget(self.pane_left)
        for k in ("run", "test", "activity"):
            self.panes[k].clear()
        self.btn_fix.pack_forget()

    def go_home(self):
        if self.busy:
            if not messagebox.askyesno("AI is working", "The AI is still working. Stop it and go back to Projects?",
                                       parent=self.win):
                return
            self.stop_agent()
        if self.runner.active or self.preview_runner.active:
            if not messagebox.askyesno("Process running", "Running processes will be stopped. Continue?",
                                       parent=self.win):
                return
        self.close_project()

    # ---------------- workspace ----------------
    def _build_workspace(self, page):
        bar = tk.Frame(page, bg=P["bg"], height=50)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        self.mkbtn(bar, "←  Projects", self.go_home).pack(side="left", padx=(14, 10), pady=9)
        self.proj_label = tk.Label(bar, text="No project open", bg=P["bg"], fg=P["text"], font=(UI_FONT, 11, "bold"))
        self.proj_label.pack(side="left")
        self.btn_fix = self.mkbtn(bar, "Fix with AI", self.fix_with_ai, primary=True)   # appears only after a failure

        more = tk.Menubutton(bar, text="•••", bg=P["sel"], fg=P["text"], bd=0, relief="flat",
                             font=(UI_FONT, 10, "bold"), padx=12, pady=5, cursor="hand2",
                             activebackground="#25324A", activeforeground="white")
        mm = self._menu(more)
        pv = self._menu(mm)
        pv.add_command(label="Start preview", command=self.start_preview)
        pv.add_command(label="Stop preview", command=self.stop_preview)
        pv.add_command(label="Reopen in browser", command=self.refresh_preview)
        pv.add_command(label="Copy preview address", command=self.copy_preview_url)
        pass
        pass
        mm.add_command(label="Show / hide output  (F7)", command=self.toggle_drawer)
        mm.add_separator()
        mm.add_command(label="Save  (Ctrl+S)", command=self.save_current)
        mm.add_command(label="Save As...", command=self.save_as)
        mm.add_command(label="Find in file  (Ctrl+F)", command=self.toggle_find)
        mm.add_separator()
        mm.add_command(label="History & Restore...", command=self.history_dialog)
        mm.add_command(label="Past conversations...", command=self.past_conversations)
        mm.add_command(label="Check dependencies", command=self.check_dependencies)
        mm.add_command(label="Install requirements...", command=self.install_requirements)
        mm.add_command(label="Git status", command=self.git_status)
        mm.add_separator()
        mm.add_command(label="Project settings...", command=self.open_settings)
        mm.add_command(label="Project details", command=self.project_details)
        mm.add_command(label="Rename project...", command=self.rename_project)
        mm.add_command(label="Open README", command=self.open_readme)
        mm.add_command(label="Open project folder", command=lambda: self.root and open_in_os(self.root))
        mm.add_command(label="Copy project path", command=self.copy_path)
        mm.add_command(label="Export as ZIP...", command=self.export_zip)
        mm.add_separator()
        mm.add_command(label="Reload model list", command=self.refresh_models)
        mm.add_command(label="Keyboard shortcuts", command=self.show_shortcuts)
        mm.add_command(label="Close project", command=self.go_home)
        more.config(menu=mm)
        more.pack(side="right", padx=(4, 14))
        self.btn_run = self.mkbtn(bar, "▶  Run", self.run_project, primary=True)
        pass
        self.btn_stop = self.mkbtn(bar, "Stop", self.stop_process)
        self.btn_stop.config(state="disabled")
        pass
        self.btn_code = self.mkbtn(bar, "Code", self.toggle_editor)
        self.btn_code.pack(side="right", padx=3)
        self.btn_files = self.mkbtn(bar, "Files", lambda: self.toggle_pane("left"))
        self.btn_files.pack(side="right", padx=3)
        self.btn_preview = self.mkbtn(bar, "Preview", self.start_preview)      # shown only for web / node projects
        self.status_label = tk.Label(bar, text="Idle", bg=P["bg"], fg=P["dim"], font=(MONO, 9), anchor="e")
        self.status_label.pack(side="right", padx=12)

        self.body = tk.Frame(page, bg=P["bg"])
        self.body.pack(fill="both", expand=True)
        self.hpane = tk.PanedWindow(self.body, orient="horizontal", bg=P["bg"], sashwidth=3, bd=0)
        self.hpane.pack(fill="both", expand=True)
        self.pane_left = tk.Frame(self.hpane, bg=P["side"], width=260)
        self.pane_center = tk.Frame(self.hpane, bg=P["bg"])
        self.pane_right = tk.Frame(self.hpane, bg=P["bg"], width=640)
        self.hpane.add(self.pane_center, minsize=340, stretch="always")   # files + editor start hidden
        self._build_files(self.pane_left)
        self._build_chat(self.pane_center)
        self._build_editor(self.pane_right)
        self._build_drawer(page)

    def _sync_toolbar(self):
        return

    def _old_sync_toolbar(self):
        kind = (self.meta or {}).get("kind")
        can = kind in ("web", "node") or bool(self.root and os.path.isfile(os.path.join(self.root, "index.html")))
        if can and not self.btn_preview.winfo_manager():
            self.btn_preview.pack(side="right", padx=3, before=self.btn_code)
        elif not can:
            self.btn_preview.pack_forget()

    def _pane_shown(self, w):
        return str(w) in [str(p) for p in self.hpane.panes()]

    def show_editor_pane(self, show=True):
        shown = self._pane_shown(self.pane_right)
        if show and not shown:
            self.hpane.add(self.pane_right, after=self.pane_center, minsize=360, width=640)
        elif not show and shown:
            self.hpane.forget(self.pane_right)

    def toggle_editor(self):
        self.show_editor_pane(not self._pane_shown(self.pane_right))

    def open_drawer(self):
        if not self.out_drawer.winfo_manager():
            self.out_drawer.pack(side="bottom", fill="x", before=self.body)

    def close_drawer(self):
        self.out_drawer.pack_forget()

    def toggle_drawer(self):
        if self.out_drawer.winfo_manager():
            self.close_drawer()
        else:
            self.open_drawer()

    # ---- file explorer (hidden until opened) ----
    def _build_files(self, left):
        left.pack_propagate(False)
        hdr = tk.Frame(left, bg=P["side"])
        hdr.pack(fill="x", padx=10, pady=(12, 0))
        tk.Label(hdr, text="Files", bg=P["side"], fg=P["text"], font=(UI_FONT, 11, "bold")).pack(side="left")
        mb = tk.Menubutton(hdr, text="•••", bg=P["sel"], fg=P["text"], bd=0, relief="flat", font=(UI_FONT, 9, "bold"),
                           padx=8, pady=2, cursor="hand2", activebackground="#25324A", activeforeground="white")
        fm = self._menu(mb)
        for label, cmd in (("New file...", self.new_file), ("New folder...", self.new_folder),
                           ("Rename...", self.rename_selected), ("Delete file...", self.delete_selected),
                           ("Refresh", self.refresh_tree)):
            fm.add_command(label=label, command=cmd)
        mb.config(menu=fm)
        mb.pack(side="right")
        self.mkbtn(hdr, "+", self.new_file).pack(side="right", padx=4)
        self.name_label = tk.Label(left, text="", bg=P["side"])          # kept so older code can update it
        self.path_label = tk.Label(left, text="", bg=P["side"], fg=P["dim"], font=(MONO, 8), anchor="w")
        self.path_label.pack(fill="x", padx=12, pady=(2, 0))
        self.search_var = tk.StringVar()
        se = self._entry(left, textvariable=self.search_var)
        se.pack(fill="x", padx=10, pady=(8, 6), ipady=4)
        self.search_var.trace_add("write", lambda *a: self.refresh_tree())
        self.empty_tree = tk.Label(left, text="No project open.", bg=P["side"], fg=P["dim"], font=(UI_FONT, 9))
        self.tree_frame = tk.Frame(left, bg=P["side"])
        self.tree_frame.pack(fill="both", expand=True, padx=(6, 0))
        sb = tk.Scrollbar(self.tree_frame, orient="vertical")
        self.tree = tk.Listbox(self.tree_frame, bg=P["side"], fg=P["text"], bd=0, highlightthickness=0,
                               selectbackground=P["sel"], selectforeground=P["accent2"], font=(MONO, 10),
                               activestyle="none", exportselection=False, yscrollcommand=sb.set)
        sb.config(command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-Button-1>", self.on_tree_activate)
        self.tree.bind("<Return>", self.on_tree_activate)
        self.tree.bind("<Button-3>", self.tree_context)

    def tree_context(self, e):
        if not self.root or not self.tree_items:
            return
        idx = self.tree.nearest(e.y)
        if 0 <= idx < len(self.tree_items):
            self.tree.selection_clear(0, "end")
            self.tree.selection_set(idx)
        m = self._menu(self.tree)
        m.add_command(label="Open / expand", command=self.on_tree_activate)
        m.add_separator()
        m.add_command(label="New file here...", command=self.new_file)
        m.add_command(label="New folder here...", command=self.new_folder)
        m.add_command(label="Rename...", command=self.rename_selected)
        m.add_command(label="Delete file...", command=self.delete_selected)
        try:
            m.tk_popup(e.x_root, e.y_root)
        finally:
            m.grab_release()

    # ---- centre: Build with AI ----
    def _build_chat(self, pane):
        holder = tk.Frame(pane, bg=P["bg"])
        holder.pack(fill="both", expand=True)
        col = tk.Frame(holder, bg=P["bg"])
        col.place(relx=0.5, rely=0, anchor="n", relheight=1, width=820)
        holder.bind("<Configure>", lambda e: col.place_configure(width=max(340, min(e.width - 40, 860))))

        hdr = tk.Frame(col, bg=P["bg"])
        hdr.pack(fill="x", pady=(14, 6))
        tk.Label(hdr, text="Build with AI", bg=P["bg"], fg=P["text"], font=(UI_FONT, 13, "bold")).pack(side="left")
        self.mkbtn(hdr, "New chat", self.clear_chat).pack(side="right")
        am = tk.Menubutton(hdr, text="Quick actions  ▾", bg=P["sel"], fg=P["text"], bd=0, relief="flat",
                           font=(UI_FONT, 10), padx=12, pady=5, cursor="hand2",
                           activebackground="#25324A", activeforeground="white")
        amm = self._menu(am)
        for lbl in ("Explain", "Review", "Find bugs", "Write tests", "Refactor", "Optimize",
                    "Document", "Create README", "Summarize progress"):
            amm.add_command(label=lbl, command=lambda l=lbl: self.quick_action(l))
        am.config(menu=amm)
        am.pack(side="right", padx=6)

        cf = tk.Frame(col, bg=P["bg"])
        cf.pack(fill="both", expand=True)
        csb = tk.Scrollbar(cf, orient="vertical")
        self.chat = tk.Text(cf, bg=P["bg"], fg=P["text"], font=(UI_FONT, 11), bd=0, wrap="word", state="disabled",
                            height=1, padx=4, pady=8, yscrollcommand=csb.set, spacing1=2, spacing3=3, highlightthickness=0,
                            cursor="arrow")
        csb.config(command=self.chat.yview)
        csb.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)
        for tag, kw in (("you", dict(foreground=P["accent2"], font=(UI_FONT, 9, "bold"))),
                        ("bot", dict(foreground="#7cc7ff", font=(UI_FONT, 9, "bold"))),
                        ("msg", dict(foreground=P["text"])),
                        ("sys", dict(foreground=P["dim"], font=(UI_FONT, 9, "italic"))),
                        ("err", dict(foreground=P["err"])), ("ok", dict(foreground=P["ok"])),
                        ("link", dict(foreground=P["glow"], underline=True))):
            self.chat.tag_configure(tag, **kw)
        self.chat.tag_bind("link", "<Enter>", lambda e: self.chat.config(cursor="hand2"))
        self.chat.tag_bind("link", "<Leave>", lambda e: self.chat.config(cursor="arrow"))

        self.prog = tk.Canvas(col, height=3, bg=P["bg"], highlightthickness=0)
        self.prog.pack(fill="x", pady=(6, 0))
        self.agent_label = tk.Label(col, text="Idle", bg=P["bg"], fg=P["dim"], font=(MONO, 9), anchor="w")
        self.agent_label.pack(fill="x")
        self.model_hint = tk.Label(col, text="", bg=P["bg"], fg=P["warn"], font=(UI_FONT, 9), wraplength=700,
                                   justify="left", anchor="w")
        self.model_hint.pack(fill="x")

        comp = tk.Frame(col, bg=P["panel"], highlightthickness=1, highlightbackground=P["border"],
                        highlightcolor=P["accent"])
        comp.pack(fill="x", pady=(6, 16))
        self.input = tk.Text(comp, height=3, bg=P["panel"], fg=P["text"], insertbackground=P["accent2"], bd=0,
                             wrap="word", font=(UI_FONT, 11), padx=12, pady=10, highlightthickness=0, undo=True)
        self.input.pack(fill="x")
        self.ph = tk.Label(comp, text="Describe what you want to build or change...", bg=P["panel"], fg=P["dim"],
                           font=(UI_FONT, 11))
        self.ph.place(x=14, y=10)
        self.ph.bind("<Button-1>", lambda e: self.input.focus_set())
        self.input.bind("<<Modified>>", self._ph_sync)
        self.input.bind("<Return>", lambda e: (self.send_from_input(), "break")[1])
        self.input.bind("<Control-Return>", lambda e: (self.send_from_input(), "break")[1])
        self.input.bind("<Shift-Return>", lambda e: None)

        foot = tk.Frame(comp, bg=P["panel"])
        foot.pack(fill="x", padx=8, pady=(0, 8))
        slot = tk.Frame(foot, bg=P["panel"])
        slot.pack(side="right")
        self.btn_send = self.mkbtn(slot, "Build  ➤", self.send_from_input, primary=True)
        self.btn_send.config(padx=16, pady=6)
        self.btn_send.pack(side="left")
        self.btn_agent_stop = self.mkbtn(slot, "■  Stop", self.stop_agent)
        self.btn_agent_stop.config(padx=16, pady=6)
        plus = tk.Menubutton(foot, text="+", bg=P["sel"], fg=P["text"], bd=0, relief="flat",
                             font=(UI_FONT, 11, "bold"), padx=10, pady=2, cursor="hand2",
                             activebackground="#25324A", activeforeground="white")
        pm = self._menu(plus)
        pm.add_command(label="Attach files / screenshots...", command=self.attach_files)
        pm.add_command(label="Clear attachments", command=self.clear_attachments)
        pm.add_command(label="Generate image asset...", command=self.generate_asset)
        pm.add_separator()
        pm.add_command(label="Retry last request", command=self.retry_last)
        pm.add_command(label="Copy last reply", command=self.copy_last_reply)
        plus.config(menu=pm)
        plus.pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_menu = tk.OptionMenu(foot, self.model_var, DEFAULT_MODEL)
        self.model_menu.config(bg=P["sel"], fg=P["text"], bd=0, highlightthickness=0, font=(UI_FONT, 9),
                               activebackground="#25324A", activeforeground="white", relief="flat", padx=8, pady=3)
        self.model_menu["menu"].config(bg=P["panel"], fg=P["text"])
        self.model_menu.pack(side="left", padx=6)
        self.btn_pull = self.mkbtn(foot, "Pull...", self.pull_model)
        self.btn_pull.pack(side="left")
        self.attach_label = tk.Label(foot, text="", bg=P["panel"], fg=P["accent2"], font=(UI_FONT, 8),
                                     anchor="w", cursor="hand2")
        self.attach_label.pack(side="left", fill="x", expand=True, padx=8)
        self.attach_label.bind("<Button-1>", lambda e: self.clear_attachments())

    def _ph_sync(self, e=None):
        if not self.input.edit_modified():
            return
        self.input.edit_modified(False)
        if self.input.get("1.0", "end-1c").strip():
            self.ph.place_forget()
        else:
            self.ph.place(x=14, y=10)
        lines = int(self.input.index("end-1c").split(".")[0])
        self.input.config(height=min(max(lines, 3), 9))

    def _prog_tick(self):
        if self.closed:
            return
        try:
            c = self.prog
            c.delete("all")
            if self.busy or self.runner.active:
                w = max(c.winfo_width(), 10)
                self._prog_x = (getattr(self, "_prog_x", 0) + 14) % (w + 160)
                c.create_rectangle(self._prog_x - 160, 0, self._prog_x, 3, fill=P["accent"], outline="")
            self.win.after(40, self._prog_tick)
        except tk.TclError:
            pass

    # ---- editor (hidden until a file is opened) ----
    def _build_editor(self, pane):
        pane.pack_propagate(False)
        head = tk.Frame(pane, bg=P["bg2"])
        head.pack(fill="x")
        tools = tk.Frame(head, bg=P["bg2"])
        tools.pack(side="right")
        self.pos_label = tk.Label(tools, text="", bg=P["bg2"], fg=P["dim"], font=(MONO, 9))
        self.pos_label.pack(side="left", padx=6)
        self.mkbtn(tools, "Save", self.save_current).pack(side="left", padx=2, pady=3)
        self.mkbtn(tools, "Find", self.toggle_find).pack(side="left", padx=2, pady=3)
        self.mkbtn(tools, "✕", lambda: self.show_editor_pane(False)).pack(side="left", padx=(2, 6), pady=3)
        self.tabbar = tk.Frame(head, bg=P["bg2"], height=32)
        self.tabbar.pack(side="left", fill="x", expand=True)
        self.find_frame = tk.Frame(pane, bg=P["panel"])
        self.find_var = tk.StringVar()
        fe = self._entry(self.find_frame, textvariable=self.find_var, width=30)
        fe.pack(side="left", padx=8, pady=4, ipady=3)
        self.find_entry = fe
        fe.bind("<Return>", lambda e: self.do_find(1))
        fe.bind("<Shift-Return>", lambda e: self.do_find(-1))
        fe.bind("<Escape>", lambda e: self.toggle_find(False))
        self.find_var.trace_add("write", lambda *a: self.do_find(1, from_entry=True))
        self.mkbtn(self.find_frame, "Next", lambda: self.do_find(1)).pack(side="left", padx=2)
        self.mkbtn(self.find_frame, "Prev", lambda: self.do_find(-1)).pack(side="left", padx=2)
        self.find_count = tk.Label(self.find_frame, text="", bg=P["panel"], fg=P["dim"], font=(UI_FONT, 9))
        self.find_count.pack(side="left", padx=8)
        self.mkbtn(self.find_frame, "x", lambda: self.toggle_find(False)).pack(side="right", padx=6)
        self.editor_container = tk.Frame(pane, bg=P["bg"])
        self.editor_container.pack(fill="both", expand=True)
        self.overview = tk.Frame(self.editor_container, bg=P["bg"])
        self.overview.pack(fill="both", expand=True)
        box = tk.Frame(self.overview, bg=P["bg"])
        box.pack(expand=True)
        tk.Label(box, text="No file open", bg=P["bg"], fg=P["text"], font=(UI_FONT, 12, "bold")).pack()
        tk.Label(box, text="Open Files (Ctrl+B) and double-click a file,\nor click a file name in the AI's replies.",
                 bg=P["bg"], fg=P["dim"], font=(UI_FONT, 10), justify="center").pack(pady=6)

    # ---- output drawer (hidden until something runs) ----
    def _build_drawer(self, page):
        self.out_drawer = tk.Frame(page, bg=P["bg2"], height=250)
        self.out_drawer.pack_propagate(False)
        hdr = tk.Frame(self.out_drawer, bg=P["bg2"])
        hdr.pack(fill="x")
        self.out_tabs, self.panes = {}, {}
        self.out_body = tk.Frame(self.out_drawer, bg=P["bg"])
        for key, label in (("terminal", "Terminal"), ("run", "Run Output"), ("problems", "Problems"),
                           ("test", "Test Results"), ("activity", "Agent Activity")):
            lb = tk.Label(hdr, text=label, bg=P["bg2"], fg=P["dim"], font=(UI_FONT, 9), padx=12, pady=5, cursor="hand2")
            lb.pack(side="left")
            lb.bind("<Button-1>", lambda e, k=key: self.show_out_tab(k))
            self.out_tabs[key] = lb
        self.mkbtn(hdr, "✕", self.close_drawer).pack(side="right", padx=6, pady=2)
        self.mkbtn(hdr, "Clear", self.clear_output).pack(side="right", padx=2, pady=2)
        self.out_body.pack(fill="both", expand=True)
        for key in ("terminal", "run", "test", "activity"):
            self.panes[key] = OutputPane(self.out_body)
        prob = tk.Frame(self.out_body, bg=P["bg"])
        self.prob_list = tk.Listbox(prob, bg=P["bg"], fg=P["text"], bd=0, highlightthickness=0, font=(MONO, 10),
                                    selectbackground=P["sel"], activestyle="none")
        self.prob_list.pack(fill="both", expand=True)
        self.prob_list.bind("<Double-Button-1>", self.open_problem)
        self.prob_frame = prob
        term_row = tk.Frame(self.panes["terminal"].frame, bg=P["bg2"])
        term_row.pack(side="bottom", fill="x", before=self.panes["terminal"].sb)
        tk.Label(term_row, text="SHELL $", bg=P["bg2"], fg=P["glow"], font=(MONO, 10, "bold")).pack(side="left", padx=(8, 2))
        self.term_entry = self._entry(term_row)
        self.term_entry.config(font=(MONO, 10))
        self.term_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4, ipady=3)
        self.term_entry.bind("<Return>", self.run_terminal_command)
        self.panes["terminal"].append("Terminal runs inside the project folder. Destructive commands are blocked; "
                                      "installs ask first.\n", "info")
        self.show_out_tab("terminal")
        self.close_drawer()

    # ---- chat extras: clickable files, undo, conversation history ----
    def chat_file_links(self, files, cp=None):
        t = self.chat
        t.config(state="normal")
        t.insert("end", "Open: ", "sys")
        for rel in files[:12]:
            tag = "lnk" + uuid.uuid4().hex[:8]
            t.insert("end", rel, (tag, "link"))
            t.tag_bind(tag, "<Button-1>", lambda e, r=rel: self.open_file(r))
            t.insert("end", "   ", "sys")
        if cp:
            tag = "lnk" + uuid.uuid4().hex[:8]
            t.insert("end", "  [Undo this change]", (tag, "link"))
            t.tag_bind(tag, "<Button-1>", lambda e, c=cp: self.undo_checkpoint(c))
        t.insert("end", "\n\n", "sys")
        t.config(state="disabled")
        t.see("end")

    def undo_checkpoint(self, cp):
        if not self.pid:
            return
        if self.busy:
            messagebox.showinfo("Busy", "Stop the current task first.", parent=self.win)
            return
        man = _read_json(os.path.join(_cp_root(self.pid), cp, "manifest.json"), None)
        if not man:
            messagebox.showerror("Undo", "That checkpoint no longer exists.", parent=self.win)
            return
        old = [en["path"] for en in man["entries"] if en["existed"]]
        new_files = [en["path"] for en in man["entries"]
                     if not en["existed"] and os.path.isfile(safe_join(self.root, en["path"]))]
        msg = "Undo this AI change and restore the earlier version of:\n" + ("\n".join(old) if old else "(nothing)")
        if new_files:
            msg += "\n\nFiles CREATED by the change will be removed:\n" + "\n".join(new_files)
        msg += "\n\nA checkpoint of the current state is saved first, so this can be undone too (History)."
        if not messagebox.askyesno("Undo AI change", msg, parent=self.win):
            return
        try:
            restore_checkpoint(self.pid, self.root, cp, remove_new=True)
        except Exception as e:
            messagebox.showerror("Undo", str(e), parent=self.win)
            return
        for en in man["entries"]:
            rel = en["path"]
            if rel in self.tabs:
                self.tabs[rel].modified = False
                if os.path.isfile(safe_join(self.root, rel)):
                    self.reload_tab(rel)
                else:
                    self.close_tab(rel)
        self.store.add_history(self.pid, {"type": "undo", "summary": f"undid {cp}", "checkpoint": cp})
        self.act_log(f"Undid checkpoint {cp}")
        self.refresh_tree()
        self.render_tabs()
        self.chat_add("", f"UNDONE: restored the files from checkpoint {cp}.", "ok", persist=False)

    def archive_chat(self):
        log = self.store.chat(self.pid)
        if any(m.get("role") == "user" for m in log):
            d = os.path.join(PROJECTS_DATA, self.pid, "conversations")
            _write_json(os.path.join(d, datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"), log)

    def past_conversations(self):
        if not self.require_project():
            return
        d = os.path.join(PROJECTS_DATA, self.pid, "conversations")
        names = sorted((f for f in os.listdir(d) if f.endswith(".json")), reverse=True) if os.path.isdir(d) else []
        logs = [_read_json(os.path.join(d, f), []) for f in names]
        top = tk.Toplevel(self.win)
        top.title("Past conversations")
        top.geometry("820x480")
        top.configure(bg=P["bg"])
        top.transient(self.win)
        tk.Label(top, text="Past conversations in this project", bg=P["bg"], fg=P["accent2"],
                 font=(UI_FONT, 11, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        body = tk.Frame(top, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=14)
        lb = tk.Listbox(body, width=44, bg=P["panel2"], fg=P["text"], bd=0, highlightthickness=0, font=(MONO, 9),
                        selectbackground=P["sel"], activestyle="none", exportselection=False)
        lb.pack(side="left", fill="y")
        det = tk.Text(body, bg=P["panel2"], fg=P["text"], bd=0, font=(UI_FONT, 10), wrap="word", state="disabled",
                      padx=8, pady=6)
        det.pack(side="left", fill="both", expand=True, padx=(8, 0))
        for f, log in zip(names, logs):
            first = next((m["content"] for m in log if m.get("role") == "user"), "(empty)")
            lb.insert("end", f"{f[:4]}-{f[4:6]}-{f[6:8]} {f[9:11]}:{f[11:13]}  " + first[:34].replace(chr(10), " "))
        if not names:
            lb.insert("end", "(none yet - use New chat to start one)")

        def show(e=None):
            sel = lb.curselection()
            if not sel or not names:
                return
            det.config(state="normal")
            det.delete("1.0", "end")
            for m in logs[sel[0]]:
                who = {"user": "YOU", "assistant": "AGENT"}.get(m.get("role"))
                if who:
                    det.insert("end", who + "\n" + m.get("content", "") + "\n\n")
            det.config(state="disabled")
        lb.bind("<<ListboxSelect>>", show)

        def resume():
            sel = lb.curselection()
            if not sel or not names:
                return
            if self.busy:
                messagebox.showinfo("Busy", "Stop the current task first.", parent=top)
                return
            self.archive_chat()
            self.store.save_chat(self.pid, logs[sel[0]])
            try:
                os.remove(os.path.join(d, names[sel[0]]))
            except OSError:
                pass
            self.load_chat_view()
            top.destroy()
        self.mkbtn(top, "Continue this conversation", resume, primary=True).pack(anchor="e", padx=14, pady=10)

    def _old_build(self):
        w = self.win
        top = tk.Frame(w, bg=P["bg2"], height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="</>  VIBE CODE", bg=P["bg2"], fg=P["accent2"], font=(MONO, 13, "bold")).pack(side="left", padx=14)
        self.proj_label = tk.Label(top, text="No project open", bg=P["bg2"], fg=P["text"], font=(UI_FONT, 10))
        self.proj_label.pack(side="left", padx=10)
        self.status_label = tk.Label(top, text="Idle", bg=P["bg2"], fg=P["dim"], font=(MONO, 9), anchor="e")
        self.status_label.pack(side="right", padx=14)
        tk.Frame(w, bg=P["border"], height=1).pack(fill="x")

        bar = tk.Frame(w, bg=P["bg"])
        bar.pack(fill="x", padx=8, pady=6)
        self.btn_run = self.mkbtn(bar, "Run", self.run_project, primary=True)
        self.btn_stop = self.mkbtn(bar, "Stop", self.stop_process)
        self.btn_test = self.mkbtn(bar, "Test", self.run_tests)
        self.btn_stop.config(state="disabled")
        for b in (self.btn_run, self.btn_stop, self.btn_test):
            b.pack(side="left", padx=3)
        pv = tk.Menubutton(bar, text="Preview", bg=P["sel"], fg=P["text"], bd=0, relief="flat",
                           font=(UI_FONT, 9), padx=10, pady=3, cursor="hand2", activebackground="#262b5c",
                           activeforeground="white")
        pm = self._menu(pv)
        pm.add_command(label="Start Preview", command=self.start_preview)
        pm.add_command(label="Stop Preview", command=self.stop_preview)
        pm.add_command(label="Refresh (reopen in browser)", command=self.refresh_preview)
        pm.add_command(label="Copy preview address", command=self.copy_preview_url)
        pv.config(menu=pm)
        pv.pack(side="left", padx=3)
        tk.Frame(bar, bg=P["border"], width=1, height=22).pack(side="left", padx=8)
        self.mkbtn(bar, "Save", self.save_current).pack(side="left", padx=3)
        self.mkbtn(bar, "Save As", self.save_as).pack(side="left", padx=3)
        self.mkbtn(bar, "Find", self.toggle_find).pack(side="left", padx=3)
        self.mkbtn(bar, "History", self.history_dialog).pack(side="left", padx=3)
        tk.Frame(bar, bg=P["border"], width=1, height=22).pack(side="left", padx=8)
        self.mkbtn(bar, "Explorer", lambda: self.toggle_pane("left")).pack(side="left", padx=3)
        self.mkbtn(bar, "AI Panel", lambda: self.toggle_pane("right")).pack(side="left", padx=3)
        self.btn_fix = self.mkbtn(bar, "Fix with AI", self.fix_with_ai, primary=True)   # shown only after a failure
        self.pos_label = tk.Label(bar, text="", bg=P["bg"], fg=P["dim"], font=(MONO, 9))
        self.pos_label.pack(side="right", padx=8)

        self.hpane = tk.PanedWindow(w, orient="horizontal", bg=P["bg"], sashwidth=4, bd=0)
        self.hpane.pack(fill="both", expand=True)
        left = tk.Frame(self.hpane, bg=P["side"], width=270)
        left.pack_propagate(False)
        center = tk.Frame(self.hpane, bg=P["bg"])
        right = tk.Frame(self.hpane, bg=P["panel2"], width=400)
        right.pack_propagate(False)
        self.hpane.add(left, minsize=220)
        self.hpane.add(center, minsize=420, stretch="always")
        self.hpane.add(right, minsize=320)
        self.pane_left, self.pane_center, self.pane_right = left, center, right
        self._build_left(left)
        self._build_center(center)
        self._build_right(right)

    # ---------------- left: project panel ----------------
    def _build_left(self, left):
        tk.Label(left, text="// PROJECT", bg=P["side"], fg=P["glow"], font=(MONO, 8, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        row = tk.Frame(left, bg=P["side"])
        row.pack(fill="x", padx=8)
        self.mkbtn(row, "+ New", self.new_project_dialog, primary=True).pack(side="left", padx=2)
        self.mkbtn(row, "Open", self.open_project_dialog).pack(side="left", padx=2)
        rb = tk.Menubutton(row, text="Recent v", bg=P["sel"], fg=P["text"], bd=0, relief="flat", font=(UI_FONT, 9),
                           padx=8, pady=3, cursor="hand2", activebackground="#262b5c", activeforeground="white")
        self.recent_menu = self._menu(rb, postcommand=self.build_recent_menu)
        rb.config(menu=self.recent_menu)
        rb.pack(side="left", padx=2)
        self.name_label = tk.Label(left, text="No project", bg=P["side"], fg=P["text"], font=(UI_FONT, 10, "bold"), anchor="w")
        self.name_label.pack(fill="x", padx=12, pady=(10, 0))
        self.path_label = tk.Label(left, text="", bg=P["side"], fg=P["dim"], font=(MONO, 8), anchor="w")
        self.path_label.pack(fill="x", padx=12)
        self.search_var = tk.StringVar()
        se = self._entry(left, textvariable=self.search_var)
        se.pack(fill="x", padx=10, pady=(8, 4), ipady=4)
        self.search_var.trace_add("write", lambda *a: self.refresh_tree())
        tools = tk.Frame(left, bg=P["side"])
        tools.pack(fill="x", padx=8, pady=(0, 4))
        for text, cmd in (("+File", self.new_file), ("+Dir", self.new_folder), ("Rename", self.rename_selected),
                          ("Delete", self.delete_selected), ("Refresh", self.refresh_tree)):
            self.mkbtn(tools, text, cmd).pack(side="left", padx=2)
        pm = tk.Menubutton(left, text="Project settings / tools v", bg=P["sel"], fg=P["text"], bd=0, relief="flat",
                           font=(UI_FONT, 9), pady=3, cursor="hand2", activebackground="#262b5c", activeforeground="white")
        m = self._menu(pm)
        for item in (("Project Settings...", self.open_settings), ("Project Details", self.project_details),
                           ("Rename Project...", self.rename_project), ("Open README", self.open_readme),
                           ("Open Project Folder", lambda: self.root and open_in_os(self.root)),
                           ("Copy Project Path", self.copy_path), None,
                           ("Check Dependencies", self.check_dependencies),
                           ("Install Requirements...", self.install_requirements),
                           ("Git Status", self.git_status), None,
                           ("Export as ZIP...", self.export_zip), ("History & Restore...", self.history_dialog), None,
                           ("Remove from Recent Projects", self.remove_current_from_recent),
                           ("Close Project", self.close_project)):
            if item is None:
                m.add_separator()
            else:
                m.add_command(label=item[0], command=item[1])
        pm.config(menu=m)
        pm.pack(fill="x", padx=10, pady=(0, 6))
        self.tree_frame = tk.Frame(left, bg=P["side"])
        self.tree_frame.pack(fill="both", expand=True, padx=(6, 0))
        sb = tk.Scrollbar(self.tree_frame, orient="vertical")
        self.tree = tk.Listbox(self.tree_frame, bg=P["side"], fg=P["text"], bd=0, highlightthickness=0,
                               selectbackground=P["sel"], selectforeground=P["accent2"], font=(MONO, 10),
                               activestyle="none", exportselection=False, yscrollcommand=sb.set)
        sb.config(command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-Button-1>", self.on_tree_activate)
        self.tree.bind("<Return>", self.on_tree_activate)
        self.empty_tree = tk.Label(left, text="No project open.\n\nCreate one with + New\nor open an existing folder.",
                                   bg=P["side"], fg=P["dim"], font=(UI_FONT, 9), justify="center")

    # ---------------- centre: editor + output ----------------
    def _build_center(self, center):
        self.vpane = tk.PanedWindow(center, orient="vertical", bg=P["bg"], sashwidth=4, bd=0)
        self.vpane.pack(fill="both", expand=True)
        editor_area = tk.Frame(self.vpane, bg=P["bg"])
        out_area = tk.Frame(self.vpane, bg=P["bg2"])
        self.vpane.add(editor_area, minsize=200, stretch="always")
        self.vpane.add(out_area, minsize=120, height=250)

        self.tabbar = tk.Frame(editor_area, bg=P["bg2"], height=30)
        self.tabbar.pack(fill="x")
        self.editor_container = tk.Frame(editor_area, bg=P["bg"])
        self.editor_container.pack(fill="both", expand=True)
        self.overview = tk.Frame(self.editor_container, bg=P["bg"])
        self.find_frame = tk.Frame(editor_area, bg=P["panel"])
        self.find_var = tk.StringVar()
        fe = self._entry(self.find_frame, textvariable=self.find_var, width=30)
        fe.pack(side="left", padx=8, pady=4, ipady=3)
        self.find_entry = fe
        fe.bind("<Return>", lambda e: self.do_find(1))
        fe.bind("<Shift-Return>", lambda e: self.do_find(-1))
        fe.bind("<Escape>", lambda e: self.toggle_find(False))
        self.find_var.trace_add("write", lambda *a: self.do_find(1, from_entry=True))
        self.mkbtn(self.find_frame, "Next", lambda: self.do_find(1)).pack(side="left", padx=2)
        self.mkbtn(self.find_frame, "Prev", lambda: self.do_find(-1)).pack(side="left", padx=2)
        self.find_count = tk.Label(self.find_frame, text="", bg=P["panel"], fg=P["dim"], font=(UI_FONT, 9))
        self.find_count.pack(side="left", padx=8)
        self.mkbtn(self.find_frame, "x", lambda: self.toggle_find(False)).pack(side="right", padx=6)

        # output panel
        hdr = tk.Frame(out_area, bg=P["bg2"])
        hdr.pack(fill="x")
        self.out_tabs, self.panes = {}, {}
        self.out_body = tk.Frame(out_area, bg=P["bg"])
        for key, label in (("terminal", "Terminal"), ("run", "Run Output"), ("problems", "Problems"),
                           ("test", "Test Results"), ("activity", "Agent Activity")):
            lb = tk.Label(hdr, text=label, bg=P["bg2"], fg=P["dim"], font=(UI_FONT, 9), padx=12, pady=5, cursor="hand2")
            lb.pack(side="left")
            lb.bind("<Button-1>", lambda e, k=key: self.show_out_tab(k))
            self.out_tabs[key] = lb
        self.mkbtn(hdr, "Clear Output", self.clear_output).pack(side="right", padx=6, pady=2)
        self.out_body.pack(fill="both", expand=True)
        for key in ("terminal", "run", "test", "activity"):
            self.panes[key] = OutputPane(self.out_body)
        prob = tk.Frame(self.out_body, bg=P["bg"])
        self.prob_list = tk.Listbox(prob, bg=P["bg"], fg=P["text"], bd=0, highlightthickness=0, font=(MONO, 10),
                                    selectbackground=P["sel"], activestyle="none")
        self.prob_list.pack(fill="both", expand=True)
        self.prob_list.bind("<Double-Button-1>", self.open_problem)
        self.prob_frame = prob
        term_row = tk.Frame(self.panes["terminal"].frame, bg=P["bg2"])
        term_row.pack(side="bottom", fill="x", before=self.panes["terminal"].sb)
        tk.Label(term_row, text="SHELL $", bg=P["bg2"], fg=P["glow"], font=(MONO, 10, "bold")).pack(side="left", padx=(8, 2))
        self.term_entry = self._entry(term_row)
        self.term_entry.config(font=(MONO, 10))
        self.term_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4, ipady=3)
        self.term_entry.bind("<Return>", self.run_terminal_command)
        self.panes["terminal"].append("Terminal runs inside the project folder. Destructive commands are blocked; "
                                      "installs ask first.\n", "info")
        self.show_out_tab("terminal")

    def show_out_tab(self, key):
        self.open_drawer()
        for k, lb in self.out_tabs.items():
            lb.config(fg=P["accent2"] if k == key else P["dim"], bg=P["sel"] if k == key else P["bg2"])
        for k, pane in self.panes.items():
            pane.frame.pack_forget()
        self.prob_frame.pack_forget()
        if key == "problems":
            self.prob_frame.pack(fill="both", expand=True)
        else:
            self.panes[key].frame.pack(fill="both", expand=True)
        self.cur_out = key

    def clear_output(self):
        if self.cur_out == "problems":
            self.problems = []
            self.problem_lines = {}
            self.prob_list.delete(0, "end")
            self.redraw_all_gutters()
        else:
            self.panes[self.cur_out].clear()

    # ---------------- right: AI panel ----------------
    def _build_right(self, right):
        hdr = tk.Frame(right, bg=P["panel2"])
        hdr.pack(fill="x", padx=10, pady=(10, 2))
        tk.Label(hdr, text="// AI CODING AGENT", bg=P["panel2"], fg=P["glow"], font=(MONO, 8, "bold")).pack(side="left")
        self.mkbtn(hdr, "New chat", self.clear_chat).pack(side="right")
        am = tk.Menubutton(hdr, text="Actions v", bg=P["sel"], fg=P["text"], bd=0, relief="flat",
                           font=(UI_FONT, 10), padx=12, pady=5, cursor="hand2",
                           activebackground="#25324A", activeforeground="white")
        amm = self._menu(am)
        for lbl in ("Explain", "Review", "Find bugs", "Write tests", "Refactor", "Optimize",
                    "Document", "Create README", "Summarize progress"):
            amm.add_command(label=lbl, command=lambda l=lbl: self.quick_action(l))
        am.config(menu=amm)
        am.pack(side="right", padx=6)
        mrow = tk.Frame(right, bg=P["panel2"])
        mrow.pack(fill="x", padx=10, pady=4)
        tk.Label(mrow, text="Model", bg=P["panel2"], fg=P["dim"], font=(UI_FONT, 9)).pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_menu = tk.OptionMenu(mrow, self.model_var, DEFAULT_MODEL)
        self.model_menu.config(bg=P["panel"], fg=P["text"], bd=0, highlightthickness=0, font=(UI_FONT, 9),
                               activebackground=P["sel"], activeforeground="white", relief="flat")
        self.model_menu["menu"].config(bg=P["panel"], fg=P["text"])
        self.model_menu.pack(side="left", padx=6)
        self.mkbtn(mrow, "Reload", self.refresh_models).pack(side="left")
        self.btn_pull = self.mkbtn(mrow, "Pull...", self.pull_model)
        self.btn_pull.pack(side="left", padx=4)
        self.model_hint = tk.Label(right, text="", bg=P["panel2"], fg=P["warn"], font=(UI_FONT, 8),
                                   wraplength=360, justify="left", anchor="w")
        self.model_hint.pack(fill="x", padx=10)
        cf = tk.Frame(right, bg=P["panel2"])
        cf.pack(fill="both", expand=True, padx=10, pady=4)
        csb = tk.Scrollbar(cf, orient="vertical")
        self.chat = tk.Text(cf, bg=P["bg"], fg=P["text"], font=(UI_FONT, 10), bd=0, wrap="word", state="disabled",
                            padx=8, pady=8, yscrollcommand=csb.set, spacing3=2)
        csb.config(command=self.chat.yview)
        csb.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)
        for tag, kw in (("you", dict(foreground=P["accent2"], font=(UI_FONT, 9, "bold"))),
                        ("bot", dict(foreground="#7cc7ff", font=(UI_FONT, 9, "bold"))),
                        ("msg", dict(foreground=P["text"])),
                        ("sys", dict(foreground=P["dim"], font=(UI_FONT, 9, "italic"))),
                        ("err", dict(foreground=P["err"])), ("ok", dict(foreground=P["ok"]))):
            self.chat.tag_configure(tag, **kw)
        self.agent_label = tk.Label(right, text="Idle", bg=P["panel2"], fg=P["dim"], font=(MONO, 9), anchor="w")
        self.agent_label.pack(fill="x", padx=10)
        self.attach_label = tk.Label(right, text="", bg=P["panel2"], fg=P["accent2"], font=(UI_FONT, 8),
                                     anchor="w", cursor="hand2")
        self.attach_label.pack(fill="x", padx=10)
        self.attach_label.bind("<Button-1>", lambda e: self.clear_attachments())
        tk.Label(right, text="ASK THE AI  -  Enter = send, Shift+Enter = new line (this is not the terminal)",
                 bg=P["panel2"], fg=P["glow"], font=(MONO, 8, "bold"), anchor="w").pack(fill="x", padx=10, pady=(6, 0))
        self.input = tk.Text(right, height=4, bg=P["panel"], fg=P["text"], insertbackground=P["accent2"], bd=0,
                             wrap="word", font=(UI_FONT, 10), padx=8, pady=6, highlightthickness=1,
                             highlightbackground=P["border"], highlightcolor=P["accent"])
        self.input.pack(fill="x", padx=10, pady=4)
        self.input.bind("<Return>", lambda e: (self.send_from_input(), "break")[1])
        self.input.bind("<Control-Return>", lambda e: (self.send_from_input(), "break")[1])
        self.input.bind("<Shift-Return>", lambda e: None)   # default behaviour: new line
        brow = tk.Frame(right, bg=P["panel2"])
        brow.pack(fill="x", padx=10, pady=(0, 10))
        slot = tk.Frame(brow, bg=P["panel2"])
        slot.pack(side="left")
        self.btn_send = self.mkbtn(slot, "Send", self.send_from_input, primary=True)
        self.btn_send.pack(side="left")
        self.btn_agent_stop = self.mkbtn(slot, "Stop", self.stop_agent)
        self.mkbtn(brow, "Attach", self.attach_files).pack(side="left", padx=4)
        self.mkbtn(brow, "Image asset", self.generate_asset).pack(side="left")
        self.mkbtn(brow, "Retry", self.retry_last).pack(side="left", padx=4)
        self.mkbtn(brow, "Copy", self.copy_last_reply).pack(side="left")

    # =================================================================
    # status / activity / chat
    # =================================================================
    def set_status(self, text, color=None):
        icon = "• "
        if color == P["ok"]:
            icon = "✓ "
        elif color == P["err"]:
            icon = "✕ "
        elif color == P["warn"]:
            icon = "! "
        try:
            self.status_label.config(text=icon + text, fg=color or P["dim"])
            self.agent_label.config(text=icon + text, fg=color or P["dim"])
        except tk.TclError:
            pass

    def toggle_pane(self, which):
        if which == "right":                 # Ctrl+L: jump to the prompt box
            self.input.focus_set()
            return
        if self._pane_shown(self.pane_left):
            self.hpane.forget(self.pane_left)
        else:
            self.hpane.add(self.pane_left, before=self.pane_center, minsize=200, width=260)
            self.refresh_tree()

    def _old_toggle_pane(self, which):
        w = self.pane_left if which == "left" else self.pane_right
        if str(w) in [str(p) for p in self.hpane.panes()]:
            self.hpane.forget(w)
        elif which == "left":
            self.hpane.add(w, before=self.pane_center, minsize=220)
        else:
            self.hpane.add(w, after=self.pane_center, minsize=320)

    def quick_action(self, label):
        if not self.require_project():
            return
        tab = self.cur_tab()
        sel = ""
        if tab:
            try:
                sel = tab.text.get("sel.first", "sel.last")
            except tk.TclError:
                sel = ""
        if sel.strip():
            target, extra = "the selected code", f"\n\nSELECTED CODE ({tab.rel}):\n{sel[:4000]}"
        elif tab:
            target, extra = f"the file {tab.rel}", ""
        else:
            target, extra = "this project", ""
        prompts = {
            "Explain": f"Explain {target}. Do not change any files.",
            "Review": f"Review {target} for bugs, unclear code and risky patterns. List findings. Do not change files.",
            "Find bugs": f"Find likely bugs in {target} and explain each. Only propose a fix if it is small and certain.",
            "Write tests": f"Write or improve tests for {target}. Create or edit test files only.",
            "Refactor": f"Refactor {target} for clarity without changing behaviour. Use small edits.",
            "Optimize": f"Suggest and apply safe performance improvements to {target}. Use small edits.",
            "Document": f"Add clear docstrings and comments to {target}. Do not change behaviour.",
            "Create README": "Create or update README.md describing this project, how to install and how to run it.",
            "Summarize progress": "Summarize the current progress of this project from earlier requests and recent "
                                  "changes, and list what is probably left. Do not change files.",
        }
        self.send_request(prompts[label] + extra, display_text=f"{label}: {target}")

    def retry_last(self):
        lr = getattr(self, "last_request", None)
        if not lr:
            messagebox.showinfo("Retry", "There is no previous request to retry.", parent=self.win)
            return
        self.send_request(*lr)

    def copy_last_reply(self):
        for m in reversed(self.chat_log):
            if m.get("role") == "assistant":
                self.win.clipboard_clear()
                self.win.clipboard_append(m["content"])
                self.set_status("Reply copied")
                return
        messagebox.showinfo("Copy", "There is no AI reply to copy yet.", parent=self.win)

    def pull_model(self):
        if getattr(self, "pulling", False):
            self.pull_cancel = True
            return
        if ollama is None:
            messagebox.showerror("Pull model", "The 'ollama' Python package is not installed.", parent=self.win)
            return
        start = self.meta["settings"]["model"] if self.meta else DEFAULT_MODEL
        name = simpledialog.askstring("Pull model", "Model to download with Ollama (e.g. qwen2.5-coder:3b):",
                                      initialvalue=start, parent=self.win)
        if not name or not name.strip():
            return
        name = name.strip()
        if not messagebox.askyesno("Download model?", f"Download '{name}' with Ollama?\n"
                                   "This can be several GB and needs an internet connection.", parent=self.win):
            return
        self.pulling, self.pull_cancel = True, False
        self.btn_pull.config(text="Cancel")

        def g(o, k):
            return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

        def work():
            try:
                for ev in ollama.pull(name, stream=True):
                    if self.pull_cancel:
                        self.ui(lambda: self.model_hint.config(text="Download cancelled."))
                        break
                    done, total, st = g(ev, "completed"), g(ev, "total"), g(ev, "status") or "downloading"
                    msg = f"{name}: {st}" + (f" {done * 100 // total}%" if done and total else "")
                    self.ui(lambda m=msg: self.model_hint.config(text=m))
                else:
                    self.ui(lambda: self.model_hint.config(text=f"'{name}' downloaded."))
            except Exception as e:
                self.ui(lambda m=friendly_ollama_error(e, name): self.model_hint.config(text=m))
            finally:
                self.pulling = False
                self.ui(lambda: self.btn_pull.config(text="Pull..."))
                self.ui(self.refresh_models)
        threading.Thread(target=work, daemon=True).start()

    def act_log(self, msg):
        self.panes["activity"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n", "info")

    def chat_add(self, who, text, tag="msg", persist=True):
        t = self.chat
        t.config(state="normal")
        if who:
            t.insert("end", ("YOU" if who == "you" else "AGENT") + "\n", who)
        t.insert("end", text.strip() + "\n\n", tag)
        t.config(state="disabled")
        t.see("end")
        if persist and self.pid:
            role = {"you": "user", "bot": "assistant"}.get(who, "sys")
            self.chat_log.append({"role": role, "content": text.strip()[:4000], "tag": tag, "time": _now()})
            self.store.save_chat(self.pid, self.chat_log)

    def load_chat_view(self):
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self.chat_log = self.store.chat(self.pid) if self.pid else []
        saved = list(self.chat_log)
        self.chat_log = []
        for m in saved[-60:]:
            who = {"user": "you", "assistant": "bot"}.get(m.get("role"), "")
            t = self.chat
            t.config(state="normal")
            if who:
                t.insert("end", ("YOU" if who == "you" else "AGENT") + "\n", who)
            t.insert("end", m.get("content", "") + "\n\n", m.get("tag", "msg") if not who else "msg")
            t.config(state="disabled")
            self.chat_log.append(m)
        self.chat.see("end")
        if not self.chat_log:
            self.chat_add("", "Describe what you want, e.g. \"Build a modern login page\" or \"Find and fix the bug "
                              "in the open file\". I only propose changes - you review and apply them. "
                              "I can't see anything I haven't been given, and I never run your code.",
                          "sys", persist=False)

    def clear_chat(self):
        if not self.pid:
            return
        if self.busy:
            messagebox.showinfo("Busy", "Stop the current task first.", parent=self.win)
            return
        if messagebox.askyesno("New coding chat", "Clear this project's chat? Project memory and checkpoints are kept.",
                               parent=self.win):
            self.chat_log = []
            self.archive_chat()
            self.store.save_chat(self.pid, [])
            self.load_chat_view()

    # =================================================================
    # models
    # =================================================================
    def refresh_models(self):
        def work():
            try:
                names = list_models()
                self.ui(self._set_models, names, None)
            except Exception as e:
                self.ui(self._set_models, [], friendly_ollama_error(e, "") if ollama else str(e))
        threading.Thread(target=work, daemon=True).start()

    def _set_models(self, names, err):
        """Called when the model list arrives from Ollama."""
        self.models, self.models_loaded = names, True
        self._render_models(err)

    def _render_models(self, err=None):
        names = self.models
        menu = self.model_menu["menu"]
        menu.delete(0, "end")
        for n in names:
            menu.add_command(label=n, command=lambda v=n: self.set_model(v))
        cur = self.meta["settings"]["model"] if self.meta else (self.model_var.get() or DEFAULT_MODEL)
        self.model_var.set(cur)
        if err:
            self.model_hint.config(text=err + "\nPress Reload to retry.")
        elif not names and self.models_loaded:
            self.model_hint.config(text="Ollama is running but has no models. Press Pull... to download one.")
        elif names and not model_installed(cur, names):
            if self.meta:
                pick = next((n for n in names if "coder" in n.lower()), names[0])
                self.meta["settings"]["model"] = pick
                self.store.save_meta(self.pid, self.meta)
                self.model_var.set(pick)
                self.model_hint.config(text=f"'{cur}' is not installed, so installed model '{pick}' is selected. "
                                            f"Press Pull... if you want to download '{cur}'.")
            else:
                self.model_hint.config(text=f"'{cur}' is not installed. Open a project to pick an installed model, "
                                            f"or press Pull....")
        else:
            self.model_hint.config(text="")

    def set_model(self, name):
        self.model_var.set(name)
        if self.meta:
            self.meta["settings"]["model"] = name
            self.store.save_meta(self.pid, self.meta)
        self._render_models()

    # =================================================================
    # projects
    # =================================================================
    def require_project(self):
        if not self.pid:
            messagebox.showinfo("Vibe Code", "Open or create a project first.", parent=self.win)
            return False
        return True

    def build_recent_menu(self):
        m = self.recent_menu
        m.delete(0, "end")
        items = self.store.recent()[:12]
        if not items:
            m.add_command(label="(no recent projects)", state="disabled")
            return
        for it in items:
            m.add_command(label=it["name"] + ("" if it["exists"] else "  (folder missing)"),
                          command=lambda i=it: self.open_recent(i))
        m.add_separator()
        for it in items:
            m.add_command(label="Remove from list: " + it["name"], command=lambda i=it: self.remove_recent(i["id"]))

    def open_recent(self, item):
        if not item["exists"]:
            if messagebox.askyesno("Folder missing", f"{item['root']}\nno longer exists. Remove it from the recent list?",
                                   parent=self.win):
                self.remove_recent(item["id"])
            return
        self.open_project_path(item["root"])

    def remove_recent(self, pid):
        self.store.remove(pid)
        if pid == self.pid:
            self.close_project(force=True)
        self.render_home()

    def remove_current_from_recent(self):
        if self.require_project() and messagebox.askyesno(
                "Remove from recent", "Remove this project from the recent list?\nYour files are NOT deleted.",
                parent=self.win):
            self.remove_recent(self.pid)

    def open_project_dialog(self):
        d = filedialog.askdirectory(title="Open project folder", parent=self.win)
        if d:
            self.open_project_path(d)

    def open_project_path(self, root, quiet=False):
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            if not quiet:
                messagebox.showerror("Open project", f"Folder not found:\n{root}", parent=self.win)
            return False
        if self.runner.active and not messagebox.askyesno("Process running", "A process is still running. Stop it and switch project?",
                                                         parent=self.win):
            return False
        if not self.close_all_tabs():
            return False
        self.runner.stop()
        self.preview_runner.stop()
        pid = self.store.register(root)
        self.pid, self.root = pid, root
        self.meta = self.store.meta(pid)
        self.meta["root"] = root
        self.meta.update(detect_project(root))
        self.store.save_meta(pid, self.meta)
        self.tools = ProjectTools(root, self.store, pid, self.act_log)
        self.expanded = set()
        self.problems = []
        self.problem_lines = {}
        self.fix_attempts, self.last_run, self.pending_verify = 0, None, None
        self.btn_fix.pack_forget()
        self.prob_list.delete(0, "end")
        self.name_label.config(text=self.meta["name"])
        self.path_label.config(text=root if len(root) < 34 else "..." + root[-31:])
        self.proj_label.config(text=f"{self.meta['name']}  -  {self.meta.get('language') or 'project'}")
        self.model_var.set(self.meta["settings"]["model"])
        self._render_models()
        self.load_chat_view()
        self.refresh_tree()
        for rel in self.meta.get("last_open_files", [])[:5]:
            try:
                if os.path.isfile(safe_join(root, rel)):
                    self.open_file(rel)
            except ValueError:
                pass
        if not self.tabs:
            self.show_overview()
        self.set_status("Idle")
        self.act_log(f"Opened project {self.meta['name']} ({root})")
        self.show_workspace()
        self._sync_toolbar()
        return True

    def close_project(self, force=False):
        if not self.close_all_tabs():
            return
        self.runner.stop()
        self.preview_runner.stop()
        self.pid = self.root = self.meta = self.tools = None
        self.name_label.config(text="No project")
        self.path_label.config(text="")
        self.proj_label.config(text="No project open")
        self.load_chat_view()
        self.refresh_tree()
        self.show_overview()

    def new_project_dialog(self):
        top = tk.Toplevel(self.win)
        top.title("New Project")
        top.geometry("540x470")
        top.configure(bg=P["bg"])
        top.transient(self.win)
        top.after(150, lambda: _safe_grab(top))
        tk.Label(top, text="New project", bg=P["bg"], fg=P["accent2"], font=(UI_FONT, 12, "bold")).pack(anchor="w", padx=16, pady=(14, 6))
        tk.Label(top, text="Name", bg=P["bg"], fg=P["dim"], font=(UI_FONT, 9)).pack(anchor="w", padx=16)
        name_e = self._entry(top)
        name_e.pack(fill="x", padx=16, ipady=4)
        tk.Label(top, text="Location (a new folder with the project name is created inside)", bg=P["bg"], fg=P["dim"],
                 font=(UI_FONT, 9)).pack(anchor="w", padx=16, pady=(10, 0))
        lrow = tk.Frame(top, bg=P["bg"])
        lrow.pack(fill="x", padx=16)
        loc_e = self._entry(lrow)
        loc_e.insert(0, DEFAULT_PROJECTS_ROOT)
        loc_e.pack(side="left", fill="x", expand=True, ipady=4)

        def browse():
            d = filedialog.askdirectory(parent=top)
            if d:
                loc_e.delete(0, "end")
                loc_e.insert(0, d)
        self.mkbtn(lrow, "Browse", browse).pack(side="left", padx=(6, 0))
        tk.Label(top, text="What should the AI build? (optional - leave empty for an empty project)", bg=P["bg"],
                 fg=P["dim"], font=(UI_FONT, 9)).pack(anchor="w", padx=16, pady=(10, 0))
        desc = tk.Text(top, height=7, bg=P["panel2"], fg=P["text"], insertbackground=P["accent2"], bd=0, wrap="word",
                       font=(UI_FONT, 10), padx=8, pady=6)
        desc.pack(fill="x", padx=16)
        err = tk.Label(top, text="", bg=P["bg"], fg=P["err"], font=(UI_FONT, 9), wraplength=500, justify="left")
        err.pack(anchor="w", padx=16, pady=4)

        def create():
            name = re.sub(r"[^\w\- ]", "", name_e.get()).strip().replace(" ", "-")
            if not name:
                err.config(text="Enter a project name (letters, numbers, - and _).")
                return
            path = os.path.join(os.path.abspath(loc_e.get().strip() or DEFAULT_PROJECTS_ROOT), name)
            if os.path.exists(path) and os.listdir(path):
                err.config(text="That folder already exists and is not empty. Pick another name or use Open.")
                return
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                err.config(text=f"Could not create the folder: {e}")
                return
            text = desc.get("1.0", "end").strip()
            top.destroy()
            if self.open_project_path(path):
                self.store.rename(self.pid, name)
                self.meta["name"] = name
                self.name_label.config(text=name)
                self.proj_label.config(text=name)
                if text:
                    self.send_request("Create a new project from this description:\n" + text,
                                      display_text="Create project: " + text[:300])
        self.mkbtn(top, "Create", create, primary=True).pack(anchor="e", padx=16, pady=8)
        name_e.focus_set()

    # =================================================================
    # file tree + file operations
    # =================================================================
    def refresh_tree(self):
        lb = self.tree
        sel = self.selected()
        yview = lb.yview()
        lb.delete(0, "end")
        self.tree_items = []
        if not self.root:
            self.tree_frame.pack_forget()
            self.empty_tree.pack(fill="both", expand=True)
            return
        self.empty_tree.pack_forget()
        self.tree_frame.pack(fill="both", expand=True, padx=(6, 0))
        q = self.search_var.get().strip().lower()
        if q:
            for rel in iter_files(self.root, 3000):
                if q in rel.lower():
                    self.tree_items.append((rel, False, 0))
                    if len(self.tree_items) >= 300:
                        break
        else:
            self._walk("", 0)
        for i, (rel, isdir, depth) in enumerate(self.tree_items):
            tab = self.tabs.get(rel)
            mark = "  *" if tab is not None and tab.modified else ""
            if q:
                label = rel + mark
            else:
                icon = ("v " if rel in self.expanded else "> ") if isdir else "  "
                label = "   " * depth + icon + posixpath.basename(rel) + mark
            lb.insert("end", label)
            if tab is not None and tab.modified:
                lb.itemconfig(i, fg=P["warn"])
            elif isdir:
                lb.itemconfig(i, fg=P["accent2"])
        if sel:
            for i, (rel, _, _) in enumerate(self.tree_items):
                if rel == sel[0]:
                    lb.selection_set(i)
                    break
        lb.yview_moveto(yview[0])

    def _walk(self, rel, depth):
        dirs, files = list_dir_entries(self.root, rel)
        for d in dirs:
            r = posixpath.join(rel, d) if rel else d
            self.tree_items.append((r, True, depth))
            if r in self.expanded:
                self._walk(r, depth + 1)
        for f in files:
            self.tree_items.append((posixpath.join(rel, f) if rel else f, False, depth))

    def selected(self):
        sel = self.tree.curselection()
        if not sel or sel[0] >= len(self.tree_items):
            return None
        rel, isdir, _ = self.tree_items[sel[0]]
        return rel, isdir

    def on_tree_activate(self, event=None):
        if event is not None and event.type == tk.EventType.ButtonPress:
            idx = self.tree.nearest(event.y)
            self.tree.selection_clear(0, "end")
            self.tree.selection_set(idx)
        sel = self.selected()
        if not sel:
            return
        rel, isdir = sel
        if isdir:
            if rel in self.expanded:
                self.expanded.discard(rel)
            else:
                self.expanded.add(rel)
            self.refresh_tree()
        else:
            self.open_file(rel)

    def _expand_parents(self, rel):
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            self.expanded.add("/".join(parts[:i]))

    def _target_dir(self):
        sel = self.selected()
        if not sel:
            return ""
        rel, isdir = sel
        return rel if isdir else posixpath.dirname(rel)

    def new_file(self):
        if not self.require_project():
            return
        base = self._target_dir()
        name = simpledialog.askstring("New file", "File name (e.g. app.py or src/utils.js):", parent=self.win)
        if not name:
            return
        rel = (base + "/" + name.strip()) if base else name.strip()
        try:
            full = safe_join(self.root, rel)
        except ValueError as e:
            messagebox.showerror("New file", str(e), parent=self.win)
            return
        if os.path.exists(full):
            messagebox.showerror("New file", "That file already exists.", parent=self.win)
            return
        _atomic_write(full, "")
        rel = rel_of(self.root, full)
        self._expand_parents(rel)
        self.refresh_tree()
        self.open_file(rel)

    def new_folder(self):
        if not self.require_project():
            return
        base = self._target_dir()
        name = simpledialog.askstring("New folder", "Folder name:", parent=self.win)
        if not name:
            return
        rel = (base + "/" + name.strip()) if base else name.strip()
        try:
            full = safe_join(self.root, rel)
            os.makedirs(full, exist_ok=True)
        except (ValueError, OSError) as e:
            messagebox.showerror("New folder", str(e), parent=self.win)
            return
        self._expand_parents(rel_of(self.root, full) + "/x")
        self.refresh_tree()

    def rename_selected(self):
        sel = self.selected()
        if not sel:
            messagebox.showinfo("Rename", "Select a file or folder first.", parent=self.win)
            return
        rel, isdir = sel
        new = simpledialog.askstring("Rename", "New name:", initialvalue=posixpath.basename(rel), parent=self.win)
        if not new or new == posixpath.basename(rel):
            return
        if "/" in new or "\\" in new:
            messagebox.showerror("Rename", "Use just a name, not a path.", parent=self.win)
            return
        new_rel = posixpath.join(posixpath.dirname(rel), new)
        try:
            self.tools.rename_file(rel, new_rel)
        except Exception as e:
            messagebox.showerror("Rename", str(e), parent=self.win)
            return
        for old in list(self.tab_order):
            if old == rel or old.startswith(rel + "/"):
                self.rekey_tab(old, new_rel + old[len(rel):])
        if rel in self.expanded:
            self.expanded.discard(rel)
            self.expanded.add(new_rel)
        self.store.add_history(self.pid, {"type": "rename", "summary": f"{rel} -> {new_rel}"})
        self.act_log(f"Renamed {rel} -> {new_rel}")
        self.refresh_tree()
        self.render_tabs()

    def delete_selected(self):
        sel = self.selected()
        if not sel:
            messagebox.showinfo("Delete", "Select a file first.", parent=self.win)
            return
        rel, isdir = sel
        if isdir:
            messagebox.showinfo("Delete", "For safety, folders can't be deleted here.\n"
                                          "Delete files individually, or remove the folder in File Explorer.", parent=self.win)
            return
        if not messagebox.askyesno("Delete file", f"Delete {rel}?\nA copy is saved to a checkpoint first, so you can restore it from History.",
                                   parent=self.win):
            return
        try:
            cp = create_checkpoint(self.pid, self.root, [rel], f"before deleting {rel}")
            os.remove(safe_join(self.root, rel))
        except Exception as e:
            messagebox.showerror("Delete", str(e), parent=self.win)
            return
        if rel in self.tabs:
            self.tabs[rel].modified = False
            self.close_tab(rel)
        self.store.add_history(self.pid, {"type": "delete", "summary": f"deleted {rel}", "files": [rel], "checkpoint": cp})
        self.act_log(f"Deleted {rel} (checkpoint {cp})")
        self.refresh_tree()

    # =================================================================
    # editor tabs
    # =================================================================
    def cur_tab(self):
        return self.tabs.get(self.active) if self.active else None

    def open_file(self, rel, line=None):
        if rel in self.tabs:
            self.activate(rel)
            if line:
                self.goto_line(self.tabs[rel], line)
            return
        try:
            full = safe_join(self.root, rel)
        except ValueError as e:
            messagebox.showerror("Open", str(e), parent=self.win)
            return
        if not os.path.isfile(full):
            return
        if not looks_text(full):
            messagebox.showinfo("Can't open", "This looks like a binary file, so it can't be edited here.", parent=self.win)
            return
        try:
            text, nl = read_text_nl(full, MAX_EDIT_BYTES)
        except UnicodeDecodeError:
            messagebox.showinfo("Can't open", "This file isn't UTF-8 text. Vibe Code won't open it (saving could corrupt it).",
                                parent=self.win)
            return
        except Exception as e:
            messagebox.showinfo("Can't open", str(e), parent=self.win)
            return
        tab = EditorTab()
        tab.rel, tab.nl, tab.saved_text, tab.modified = rel, nl, text, False
        tab.lang = EXT_LANG.get(os.path.splitext(rel)[1].lower())
        tab.mtime, tab.hl_job = os.path.getmtime(full), None
        fr = tk.Frame(self.editor_container, bg=P["bg"])
        tab.frame = fr
        tab.gutter = tk.Canvas(fr, width=54, bg=P["side"], highlightthickness=0)
        vsb = tk.Scrollbar(fr, orient="vertical")
        hsb = tk.Scrollbar(fr, orient="horizontal")
        font = (MONO, 11)
        t = tk.Text(fr, bg=P["bg2"], fg=P["text"], insertbackground=P["accent2"], font=font, bd=0, wrap="none",
                    undo=True, padx=8, pady=6, selectbackground="#2a2f6a", highlightthickness=0,
                    yscrollcommand=lambda a, b: (vsb.set(a, b), self.redraw_gutter(tab)), xscrollcommand=hsb.set)
        tab.text = t
        vsb.config(command=t.yview)
        hsb.config(command=t.xview)
        tab.gutter.grid(row=0, column=0, sticky="ns")
        t.grid(row=0, column=1, sticky="nsew")
        vsb.grid(row=0, column=2, sticky="ns")
        hsb.grid(row=1, column=1, sticky="ew")
        fr.grid_rowconfigure(0, weight=1)
        fr.grid_columnconfigure(1, weight=1)
        import tkinter.font as tkfont
        t.configure(tabs=(tkfont.Font(font=font).measure("    "),))
        for tag in SYN_TAGS:                       # creation order = priority
            kw = dict(foreground=SYN_COLORS[tag])
            if tag == "comment":
                kw["font"] = (MONO, 11, "italic")
            t.tag_configure(tag, **kw)
        t.tag_configure("errline", background="#3a1530")
        t.tag_configure("find", background="#3d3a8a")
        t.tag_configure("findcur", background="#7c6cff", foreground="white")
        t.tag_raise("sel")
        t.insert("1.0", text)
        t.edit_reset()
        t.edit_modified(False)
        t.bind("<<Modified>>", lambda e: self.on_modified(tab))
        t.bind("<KeyRelease>", lambda e: self.update_pos(tab))
        t.bind("<ButtonRelease-1>", lambda e: self.update_pos(tab))
        t.bind("<Configure>", lambda e: self.redraw_gutter(tab))
        t.bind("<Return>", lambda e: self.on_return(tab))
        t.bind("<Tab>", lambda e: (t.insert("insert", "    "), "break")[1])
        self.tabs[rel] = tab
        self.tab_order.append(rel)
        self.activate(rel)
        self.highlight(tab)
        if line:
            self.goto_line(tab, line)

    def on_return(self, tab):
        t = tab.text
        indent = re.match(r"[ \t]*", t.get("insert linestart", "insert")).group(0)
        if t.get("insert linestart", "insert lineend").rstrip().endswith((":", "{", "(", "[")):
            indent += "    "
        t.insert("insert", "\n" + indent)
        t.see("insert")
        return "break"

    def goto_line(self, tab, line):
        tab.text.mark_set("insert", f"{line}.0")
        tab.text.see(f"{line}.0")
        tab.text.focus_set()

    def update_pos(self, tab):
        try:
            ln, col = tab.text.index("insert").split(".")
            self.pos_label.config(text=f"Ln {ln}, Col {int(col) + 1}")
        except tk.TclError:
            pass

    def on_modified(self, tab):
        t = tab.text
        if not t.edit_modified():
            return
        t.edit_modified(False)
        cur = t.get("1.0", "end-1c")
        m = cur != tab.saved_text
        if m != tab.modified:
            tab.modified = m
            self.render_tabs()
            self.refresh_tree()
        self.schedule_highlight(tab)
        self.redraw_gutter(tab)

    def activate(self, rel):
        self.show_editor_pane(True)
        self._activate_inner(rel)

    def _activate_inner(self, rel):
        if self.active in self.tabs and self.active != rel:
            self.tabs[self.active].frame.pack_forget()
        self.overview.pack_forget()
        self.active = rel
        tab = self.tabs[rel]
        tab.frame.pack(fill="both", expand=True)
        self.render_tabs()
        tab.text.focus_set()
        self.win.after(30, lambda: self.redraw_gutter(tab))
        self.update_pos(tab)
        if self.meta:
            self.meta["last_open_files"] = list(self.tab_order)[-8:]

    def render_tabs(self):
        for w in self.tabbar.winfo_children():
            w.destroy()
        for rel in self.tab_order:
            tab = self.tabs[rel]
            active = rel == self.active
            fr = tk.Frame(self.tabbar, bg=P["bg"] if active else P["bg2"], cursor="hand2")
            fr.pack(side="left", padx=(0, 1))
            name = posixpath.basename(rel) + ("  *" if tab.modified else "")
            lb = tk.Label(fr, text=name, bg=fr["bg"], fg=P["accent2"] if active else P["dim"], font=(UI_FONT, 9), padx=10, pady=6)
            lb.pack(side="left")
            x = tk.Label(fr, text="x", bg=fr["bg"], fg=P["dim"], font=(UI_FONT, 9), padx=4, cursor="hand2")
            x.pack(side="left")
            for wdg in (fr, lb):
                wdg.bind("<Button-1>", lambda e, r=rel: self.activate(r))
            x.bind("<Button-1>", lambda e, r=rel: self.close_tab(r))
            if active:
                tk.Frame(fr, bg=P["accent"], height=2).pack(side="bottom", fill="x")

    def close_tab(self, rel):
        tab = self.tabs.get(rel)
        if tab is None:
            return True
        if tab.modified:
            ans = messagebox.askyesnocancel("Unsaved changes", f"Save changes to {rel}?", parent=self.win)
            if ans is None:
                return False
            if ans and not self.save_tab(tab):
                return False
        tab.frame.destroy()
        del self.tabs[rel]
        self.tab_order.remove(rel)
        if self.active == rel:
            self.active = None
            if self.tab_order:
                self.activate(self.tab_order[-1])
            else:
                self.show_overview()
        self.render_tabs()
        self.refresh_tree()
        return True

    def close_all_tabs(self):
        for rel in list(self.tab_order):
            if not self.close_tab(rel):
                return False
        return True

    def rekey_tab(self, old, new):
        tab = self.tabs.pop(old)
        tab.rel = new
        self.tabs[new] = tab
        self.tab_order[self.tab_order.index(old)] = new
        if self.active == old:
            self.active = new

    def save_tab(self, tab):
        try:
            full = safe_join(self.root, tab.rel)
            if os.path.isfile(full) and abs(os.path.getmtime(full) - tab.mtime) > 0.01:
                if not messagebox.askyesno("File changed on disk",
                                           f"{tab.rel} was changed outside the editor since you opened it.\nOverwrite it anyway?",
                                           parent=self.win):
                    return False
            text = tab.text.get("1.0", "end-1c")
            write_text_nl(full, text, tab.nl)
            tab.saved_text, tab.modified, tab.mtime = text, False, os.path.getmtime(full)
        except Exception as e:
            messagebox.showerror("Save failed", f"{e}\nYour text is still in the editor.", parent=self.win)
            return False
        self.render_tabs()
        self.refresh_tree()
        self.act_log(f"Saved {tab.rel}")
        return True

    def save_current(self):
        tab = self.cur_tab()
        if tab:
            self.save_tab(tab)

    def save_as(self):
        tab = self.cur_tab()
        if not tab:
            return
        path = filedialog.asksaveasfilename(initialdir=self.root, initialfile=posixpath.basename(tab.rel), parent=self.win)
        if not path:
            return
        try:
            rel = rel_of(self.root, safe_join(self.root, rel_of(self.root, os.path.abspath(path))))
        except ValueError:
            messagebox.showwarning("Outside project", "Vibe Code only saves inside the project folder.\n"
                                                      "Choose a location inside it.", parent=self.win)
            return
        if os.path.exists(path) and os.path.normcase(os.path.realpath(path)) != os.path.normcase(safe_join(self.root, tab.rel)) and not messagebox.askyesno(
                "Overwrite", f"{rel} exists. Overwrite?", parent=self.win):
            return
        try:
            text = tab.text.get("1.0", "end-1c")
            write_text_nl(safe_join(self.root, rel), text, tab.nl)
        except Exception as e:
            messagebox.showerror("Save As", str(e), parent=self.win)
            return
        if rel != tab.rel and rel not in self.tabs:
            tab.text.edit_modified(False)
            self.rekey_tab(tab.rel, rel)
            tab.lang = EXT_LANG.get(os.path.splitext(rel)[1].lower())
        tab.saved_text, tab.modified = text, False
        tab.mtime = os.path.getmtime(safe_join(self.root, rel))
        self._expand_parents(rel)
        self.refresh_tree()
        self.render_tabs()
        self.highlight(tab)

    def reload_tab(self, rel):
        tab = self.tabs.get(rel)
        if tab is None:
            return
        full = safe_join(self.root, rel)
        text, nl = read_text_nl(full, MAX_EDIT_BYTES)
        tab.text.delete("1.0", "end")
        tab.text.insert("1.0", text)
        tab.text.edit_reset()
        tab.text.edit_modified(False)
        tab.saved_text, tab.nl, tab.modified, tab.mtime = text, nl, False, os.path.getmtime(full)
        self.highlight(tab)
        self.redraw_gutter(tab)

    # ---------------- gutter / highlight / find ----------------
    def redraw_gutter(self, tab):
        g, t = tab.gutter, tab.text
        try:
            g.delete("all")
            bad = self.problem_lines.get(tab.rel, set())
            i, last = t.index("@0,0"), None
            while True:
                d = t.dlineinfo(i)
                if d is None:
                    break
                ln = int(i.split(".")[0])
                if ln == last:
                    break
                last = ln
                g.create_text(46, d[1], anchor="ne", text=str(ln), fill=P["dim"], font=(MONO, 10))
                if ln in bad:
                    g.create_oval(6, d[1] + 4, 14, d[1] + 12, fill=P["err"], outline="")
                i = t.index(f"{ln + 1}.0")
                if int(i.split(".")[0]) == ln:
                    break
        except tk.TclError:
            pass

    def redraw_all_gutters(self):
        for tab in self.tabs.values():
            tab.text.tag_remove("errline", "1.0", "end")
            for ln in self.problem_lines.get(tab.rel, ()):
                tab.text.tag_add("errline", f"{ln}.0", f"{ln}.end")
            self.redraw_gutter(tab)

    def schedule_highlight(self, tab):
        if tab.hl_job is not None:
            try:
                self.win.after_cancel(tab.hl_job)
            except Exception:
                pass
        tab.hl_job = self.win.after(250, lambda: self.highlight(tab))

    def highlight(self, tab):
        tab.hl_job = None
        if tab.rel not in self.tabs:
            return
        t = tab.text
        for tg in SYN_TAGS:
            t.tag_remove(tg, "1.0", "end")
        text = t.get("1.0", "end-1c")
        if not tab.lang or len(text) > 150000:
            return
        starts = [0]
        for m in re.finditer("\n", text):
            starts.append(m.end())

        def pos(off):
            i = bisect.bisect_right(starts, off) - 1
            return f"{i + 1}.{off - starts[i]}"
        for tag, rx in LANG_RULES[tab.lang]:
            for m in rx.finditer(text):
                if m.end() > m.start():
                    t.tag_add(tag, pos(m.start()), pos(m.end()))

    def toggle_find(self, show=None):
        visible = self.find_frame.winfo_ismapped()
        if show is None:
            show = not visible
        if show:
            if not self.cur_tab():
                return
            self.find_frame.pack(fill="x", before=self.editor_container)
            self.find_entry.focus_set()
            self.find_entry.select_range(0, "end")
        else:
            self.find_frame.pack_forget()
            tab = self.cur_tab()
            if tab:
                tab.text.tag_remove("find", "1.0", "end")
                tab.text.tag_remove("findcur", "1.0", "end")
                tab.text.focus_set()

    def do_find(self, direction=1, from_entry=False):
        tab = self.cur_tab()
        if not tab:
            return
        t, pat = tab.text, self.find_var.get()
        t.tag_remove("find", "1.0", "end")
        t.tag_remove("findcur", "1.0", "end")
        if not pat:
            self.find_count.config(text="")
            return
        start, n = "1.0", 0
        while n < 2000:
            idx = t.search(pat, start, "end", nocase=True)
            if not idx:
                break
            end = f"{idx}+{len(pat)}c"
            t.tag_add("find", idx, end)
            start, n = end, n + 1
        if direction > 0:
            idx = (t.search(pat, "insert" if from_entry else "insert+1c", "end", nocase=True)
                   or t.search(pat, "1.0", "end", nocase=True))
        else:
            idx = (t.search(pat, "insert", "1.0", backwards=True, nocase=True)
                   or t.search(pat, "end", "1.0", backwards=True, nocase=True))
        if idx:
            t.mark_set("insert", idx)
            t.see(idx)
            t.tag_add("findcur", idx, f"{idx}+{len(pat)}c")
        self.find_count.config(text=f"{n} match(es)" if n else "no matches")

    # =================================================================
    # overview / welcome screen
    # =================================================================
    def show_overview(self):
        for tab in self.tabs.values():
            tab.frame.pack_forget()
        if not self.pid:
            self.show_home()
            return
        self.overview.pack(fill="both", expand=True)
        self.show_editor_pane(False)

    def _old_show_overview(self):
        for tab in self.tabs.values():
            tab.frame.pack_forget()
        ov = self.overview
        for w in ov.winfo_children():
            w.destroy()
        ov.pack(fill="both", expand=True)
        box = tk.Frame(ov, bg=P["bg"])
        box.pack(expand=True)
        if not self.pid:
            tk.Label(box, text="</>", bg=P["bg"], fg=P["glow"], font=(MONO, 36, "bold")).pack()
            tk.Label(box, text="Welcome to Vibe Code", bg=P["bg"], fg=P["accent2"], font=(UI_FONT, 16, "bold")).pack(pady=(4, 2))
            tk.Label(box, text="Describe what you want and the AI proposes the code.\nYou review every change before it is written.",
                     bg=P["bg"], fg=P["dim"], font=(UI_FONT, 10), justify="center").pack(pady=6)
            row = tk.Frame(box, bg=P["bg"])
            row.pack(pady=8)
            self.mkbtn(row, "+ New Project", self.new_project_dialog, primary=True).pack(side="left", padx=4)
            self.mkbtn(row, "Open Project", self.open_project_dialog).pack(side="left", padx=4)
            for it in [r for r in self.store.recent() if r["exists"]][:5]:
                self.mkbtn(box, "Recent: " + it["name"], lambda i=it: self.open_recent(i)).pack(pady=2)
            return
        m = self.meta
        tk.Label(box, text=m["name"], bg=P["bg"], fg=P["accent2"], font=(UI_FONT, 16, "bold")).pack()
        tk.Label(box, text=self.root, bg=P["bg"], fg=P["dim"], font=(MONO, 9)).pack()
        info = " / ".join(x for x in (m.get("language"), m.get("framework")) if x) or "language not detected yet"
        tk.Label(box, text=f"{info}   |   model: {m['settings']['model']}", bg=P["bg"], fg=P["text"],
                 font=(UI_FONT, 10)).pack(pady=6)
        row = tk.Frame(box, bg=P["bg"])
        row.pack(pady=6)
        self.mkbtn(row, "Explain this project", lambda: self.send_request(
            "Explain how this project works: structure, main files, how to run it. Do not change any files.",
            display_text="Explain this project")).pack(side="left", padx=4)
        self.mkbtn(row, "Open README", self.open_readme).pack(side="left", padx=4)
        if m.get("requirements"):
            tk.Label(box, text="Recent requests", bg=P["bg"], fg=P["glow"], font=(MONO, 8, "bold")).pack(pady=(14, 2))
            for r in m["requirements"][-5:]:
                tk.Label(box, text="- " + r[:90].replace("\n", " "), bg=P["bg"], fg=P["dim"], font=(UI_FONT, 9)).pack(anchor="w")
        if m.get("recent_changes"):
            tk.Label(box, text="Recent changes", bg=P["bg"], fg=P["glow"], font=(MONO, 8, "bold")).pack(pady=(10, 2))
            for r in m["recent_changes"][-5:]:
                tk.Label(box, text="- " + r[:90], bg=P["bg"], fg=P["dim"], font=(UI_FONT, 9)).pack(anchor="w")

    def open_readme(self):
        if not self.require_project():
            return
        for n in os.listdir(self.root):
            if n.lower().startswith("readme") and os.path.isfile(os.path.join(self.root, n)):
                self.open_file(n)
                return
        messagebox.showinfo("README", "This project has no README yet. Ask the AI: \"Create a README\".", parent=self.win)

    # =================================================================
    # project settings / details / misc menu actions
    # =================================================================
    def open_settings(self):
        if not self.require_project():
            return
        s = self.meta["settings"]
        top = tk.Toplevel(self.win)
        top.title("Project Settings")
        top.geometry("560x400")
        top.configure(bg=P["bg"])
        top.transient(self.win)
        top.after(150, lambda: _safe_grab(top))
        fields = (("Model", "model"), ("Context window (tokens, 2048-32768)", "num_ctx"),
                  ("Python interpreter (blank = auto-detect)", "python"),
                  ("Run / test timeout in seconds", "timeout"),
                  ("Custom run command (blank = auto-detect)", "run_command"))
        entries = {}
        for label, key in fields:
            tk.Label(top, text=label, bg=P["bg"], fg=P["dim"], font=(UI_FONT, 9)).pack(anchor="w", padx=16, pady=(8, 0))
            e = self._entry(top)
            e.insert(0, str(s.get(key, "")))
            e.pack(fill="x", padx=16, ipady=4)
            entries[key] = e
        py, why = find_python(self.root, s)
        tk.Label(top, text=f"Detected interpreter: {py or 'none'}  ({why})", bg=P["bg"], fg=P["dim"],
                 font=(UI_FONT, 8), wraplength=520, justify="left").pack(anchor="w", padx=16, pady=(8, 0))
        err = tk.Label(top, text="", bg=P["bg"], fg=P["err"], font=(UI_FONT, 9))
        err.pack(anchor="w", padx=16)

        def save():
            try:
                ctx = int(entries["num_ctx"].get())
                to = int(entries["timeout"].get())
                if not (2048 <= ctx <= 32768) or not (5 <= to <= 3600):
                    raise ValueError
            except ValueError:
                err.config(text="Context must be 2048-32768 and timeout 5-3600.")
                return
            s.update(model=entries["model"].get().strip() or DEFAULT_MODEL, num_ctx=ctx, timeout=to,
                     python=entries["python"].get().strip(), run_command=entries["run_command"].get().strip())
            self.store.save_meta(self.pid, self.meta)
            self.model_var.set(s["model"])
            self._render_models()
            top.destroy()
        self.mkbtn(top, "Save", save, primary=True).pack(anchor="e", padx=16, pady=10)

    def project_details(self):
        if not self.require_project():
            return
        m = self.meta
        py, why = find_python(self.root, m["settings"])
        n = len(iter_files(self.root, 5000))
        git = "yes" if os.path.isdir(os.path.join(self.root, ".git")) else "no"
        messagebox.showinfo("Project Details",
                            f"Name: {m['name']}\nID: {self.pid}\nFolder: {self.root}\n"
                            f"Type: {m.get('kind')}  ({m.get('language')} {m.get('framework')})\n"
                            f"Files (excluding generated folders): {n}\nGit repository: {git}\n"
                            f"Model: {m['settings']['model']}\nPython: {py or 'none'} ({why})\nCreated: {m.get('created')}",
                            parent=self.win)

    def rename_project(self):
        if not self.require_project():
            return
        name = simpledialog.askstring("Rename project", "New project name (the folder is not renamed):",
                                      initialvalue=self.meta["name"], parent=self.win)
        if name and name.strip():
            self.store.rename(self.pid, name.strip())
            self.meta["name"] = name.strip()
            self.name_label.config(text=name.strip())
            self.proj_label.config(text=name.strip())

    def copy_path(self):
        if self.root:
            self.win.clipboard_clear()
            self.win.clipboard_append(self.root)
            self.set_status("Project path copied")

    # =================================================================
    # AI agent
    # =================================================================
    def attach_files(self):
        paths = filedialog.askopenfilenames(parent=self.win, title="Attach files or screenshots")
        if paths:
            self.attachments.extend(paths)
            self.attach_label.config(text="Attached: " + ", ".join(os.path.basename(p) for p in self.attachments) + "   (click to clear)")

    def clear_attachments(self):
        self.attachments = []
        self.attach_label.config(text="")

    def send_from_input(self):
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        if self.send_request(text):
            self.input.delete("1.0", "end")

    def history_messages(self):
        msgs = [{"role": m["role"], "content": m["content"][:1200]} for m in self.chat_log
                if m.get("role") in ("user", "assistant")]
        return msgs[-6:]

    def send_request(self, text, display_text=None, focus_files=None, is_fix=False):
        if not self.require_project():
            return False
        if self.busy:
            self.chat_add("", "Still working on the previous request. Wait for it or press Stop.", "sys", persist=False)
            return False
        if ollama is None:
            self.chat_add("", "The 'ollama' Python package isn't installed (pip install ollama).", "err", persist=False)
            return False
        model = self.meta["settings"]["model"]
        self.last_request = (text, display_text, focus_files, is_fix)
        hist = self.history_messages()
        self.chat_add("you", display_text or text)
        if not is_fix:
            self.meta["requirements"] = (self.meta["requirements"] + [text[:200]])[-30:]
            self.store.save_meta(self.pid, self.meta)
        buffers = {rel: tab.text.get("1.0", "end-1c") for rel, tab in self.tabs.items()
                   if tab.modified or rel == self.active}
        attachments = list(self.attachments)
        self.clear_attachments()
        self.busy = True
        self.token += 1
        tok = self.token
        self.set_status("Understanding request", P["accent2"])
        self.btn_send.pack_forget()
        self.btn_agent_stop.pack(side="left")
        threading.Thread(target=self._agent_worker,
                         args=(tok, text, model, attachments, buffers, self.active, focus_files or [], hist, is_fix),
                         daemon=True).start()
        return True

    def _agent_worker(self, tok, text, model, attachments, buffers, active, focus, hist, is_fix):
        def ui(fn, *a):
            self.q.put((tok, fn, a))

        cancelled = lambda: self.token != tok
        try:
            ui(self.set_status, "Inspecting project", P["accent2"])
            names = list_models()
            if not model_installed(model, names):
                raise RuntimeError(friendly_ollama_error("model not found", model))
            s = self.meta["settings"]
            budget = max(6000, int(s["num_ctx"] * 1.5))
            ctx = build_context(self.tools, text, active, buffers, focus, budget,
                                lambda m: ui(self.act_log, m))
            extra = ""
            if attachments:
                img_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
                imgs = [p for p in attachments if p.lower().endswith(img_ext)]
                for p in attachments:
                    if p in imgs:
                        continue
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            extra += f"\n--- ATTACHED FILE: {os.path.basename(p)} ---\n{f.read(5000)}\n"
                        ui(self.act_log, f"Read attachment {os.path.basename(p)}")
                    except OSError as e:
                        extra += f"\n(Could not read attachment {p}: {e})\n"
                if imgs:
                    ref = os.path.join(PROJECTS_DATA, self.pid, "references")
                    os.makedirs(ref, exist_ok=True)
                    for p in imgs:
                        shutil.copy2(p, os.path.join(ref, os.path.basename(p)))
                    ui(self.set_status, "Reading screenshot", P["accent2"])
                    extra += "\n" + describe_images(imgs, names) + "\n"
                    ui(self.act_log, f"Described {len(imgs)} screenshot(s); copies kept in the project's references folder")
            m = self.meta
            reqs = "\n".join("- " + r for r in m["requirements"][-8:]) or "(none yet)"
            changes_txt = "\n".join("- " + r for r in m["recent_changes"][-5:]) or "(none yet)"
            user_msg = (f"PROJECT: {m['name']} | {m.get('language') or 'unknown language'} | {m.get('framework') or 'no framework'}\n"
                        f"EARLIER REQUESTS:\n{reqs}\nRECENT CHANGES:\n{changes_txt}\n"
                        f"FILE TREE:\n{self.tools.list_files()}\n\nFILES:\n{ctx}\n{extra}\nREQUEST:\n{text}")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + hist + [{"role": "user", "content": user_msg}]
            result = None
            for attempt in range(2):
                ui(self.set_status, "Planning changes" if attempt == 0 else "Retrying (invalid JSON)", P["accent2"])
                raw = call_model(model, messages, s["num_ctx"], cancelled,
                                 lambda n: ui(self.set_status, f"Generating files ({n} chars)", P["accent2"]))
                if raw is None:
                    return
                try:
                    result = parse_model_json(raw)
                    break
                except ValueError as e:
                    messages = messages + [{"role": "assistant", "content": raw[:2000]},
                                           {"role": "user", "content": f"That was not valid ({e}). Reply again with ONLY the JSON object."}]
            if result is None:
                raise RuntimeError("The model could not produce valid JSON twice. Try a bigger model or a smaller request.")
            if cancelled():
                return
            changes, errors = self.tools.validate_changes(result["changes"])
            ui(self._agent_done, result, changes, errors, is_fix)
        except Exception as e:
            if not cancelled():
                msg = str(e) if "ollama pull" in str(e) else friendly_ollama_error(e, model)
                ui(self._agent_failed, msg)

    def _agent_done(self, result, changes, errors, is_fix):
        self.busy = False
        self.btn_agent_stop.pack_forget()
        self.btn_send.pack(side="left")
        text = result["explanation"] or "(no explanation given)"
        if result["plan"]:
            text += "\n\nPlan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(result["plan"], 1))
        self.chat_add("bot", text)
        if errors:
            self.chat_add("", "Skipped invalid change(s):\n" + "\n".join("- " + e for e in errors), "err")
            self.act_log("Rejected invalid changes: " + "; ".join(errors)[:300])
        if changes:
            self.act_log("Proposed changes: " + ", ".join(c["path"] for c in changes))
            self.set_status("Reviewing changes", P["warn"])
            self.chat_add("", f"PROPOSED: {len(changes)} file change(s). Nothing has been written yet - review them in the window.", "sys")
            ReviewDialog(self, changes, result["explanation"])
        else:
            self.set_status("Completed", P["ok"])
            self.pending_verify = None
            if not errors:
                self.chat_add("", "No file changes were proposed.", "sys")

    def _agent_failed(self, msg):
        self.busy = False
        self.btn_agent_stop.pack_forget()
        self.btn_send.pack(side="left")
        self.pending_verify = None
        self.set_status("Failed", P["err"])
        self.chat_add("", msg, "err")

    def stop_agent(self):
        if not self.busy:
            return
        self.token += 1                 # any late result from the old task is ignored
        self.busy = False
        self.pending_verify = None
        self.btn_agent_stop.pack_forget()
        self.btn_send.pack(side="left")
        self.set_status("Stopped", P["warn"])
        self.chat_add("", "Stopped. Nothing was changed.", "sys")

    def on_rejected(self):
        self.pending_verify = None
        self.set_status("Idle")
        self.chat_add("", "REJECTED: no files were changed.", "sys")
        self.act_log("User rejected the proposed changes")

    def apply_changes(self, changes, label):
        self.set_status("Applying changes", P["accent2"])
        final = []
        for c in changes:
            tab = self.tabs.get(c["path"])
            if tab and tab.modified and not messagebox.askyesno(
                    "Unsaved edits", f"You have unsaved edits in {c['path']}.\nOverwrite them with the AI version?\n"
                                     "(The file on disk is checkpointed, but unsaved editor text is not.)", parent=self.win):
                self.act_log(f"Skipped {c['path']} (unsaved edits)")
                continue
            final.append(c)
        if not final:
            self.on_rejected()
            return
        try:
            cp = self.tools.apply_changes(final, label)
        except Exception as e:
            messagebox.showerror("Apply failed", str(e), parent=self.win)
            self.set_status("Failed", P["err"])
            self.chat_add("", str(e), "err")
            self.pending_verify = None
            return
        for c in final:
            self._expand_parents(c["path"])
            if c["path"] in self.tabs:
                try:
                    self.reload_tab(c["path"])
                except Exception:
                    pass
        files = [c["path"] for c in final]
        self.store.add_history(self.pid, {"type": "apply", "summary": label, "files": files, "checkpoint": cp})
        self.meta["recent_changes"] = (self.meta["recent_changes"] + [f"{_now()[:16]}: " + ", ".join(files)[:120]])[-20:]
        self.meta.update(detect_project(self.root))
        self.store.save_meta(self.pid, self.meta)
        self.refresh_tree()
        self.act_log(f"Applied {len(final)} file(s), checkpoint {cp}")
        self.set_status("Completed", P["ok"])
        self.chat_add("", f"APPLIED: {', '.join(files)}\nCheckpoint {cp} saved (History > Restore). "
                          f"These changes are not tested - run the code yourself to check.", "ok")
        if not self.tabs:
            self.show_overview()
        self._sync_toolbar()
        self.chat_file_links(files, cp)
        if self.pending_verify:
            kind = self.pending_verify
            self.win.after(300, self.run_project if kind == "run" else self.run_tests)

    # ---------------- fix with AI ----------------
    def fix_with_ai(self):
        lr = self.last_run
        if not lr or lr["ok"]:
            return
        if self.fix_attempts >= MAX_FIX_ATTEMPTS:
            self.chat_add("", f"I already tried {MAX_FIX_ATTEMPTS} fixes for this failure, so I'm stopping to avoid a loop. "
                              "Read the error yourself or send a more specific request.", "err")
            return
        self.fix_attempts += 1
        why = "timed out" if lr["timed_out"] else f"exit code {lr['code']}"
        req = (f"Fix attempt {self.fix_attempts}/{MAX_FIX_ATTEMPTS}. The command `{lr['display']}` failed ({why}).\n"
               f"Output (last part):\n{lr['output'][-5000:]}\n\n"
               "Find the root cause and propose the smallest fix. Do not rewrite unrelated files.")
        self.pending_verify = lr["kind"]
        if not self.send_request(req, display_text=f"Fix the failure from: {lr['display']}  (attempt {self.fix_attempts}/{MAX_FIX_ATTEMPTS})",
                                 focus_files=[p["file"] for p in self.problems][:4], is_fix=True):
            self.pending_verify = None
            self.fix_attempts -= 1

    # ---------------- image assets ----------------
    def generate_asset(self):
        if not self.require_project():
            return
        if self.image_fn is None:
            messagebox.showinfo("Image asset", "Image generation isn't connected in this build.", parent=self.win)
            return
        desc = simpledialog.askstring("Image asset", "Describe the image (saved into the project's assets folder):", parent=self.win)
        if not desc:
            return
        self.set_status("Generating image", P["accent2"])

        def work():
            try:
                src = self.image_fn(desc)
                slug = re.sub(r"[^a-z0-9]+", "-", desc.lower()).strip("-")[:30] or "image"
                rel = f"assets/{slug}-{uuid.uuid4().hex[:4]}.png"
                dest = safe_join(self.root, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                self.ui(self._asset_done, rel)
            except Exception as e:
                self.ui(self._asset_failed, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _asset_done(self, rel):
        self.set_status("Completed", P["ok"])
        self._expand_parents(rel)
        self.refresh_tree()
        self.chat_add("", f"Image saved to {rel}. (It's a portrait scene image; use it in your code via that path.)", "ok")

    def _asset_failed(self, msg):
        self.set_status("Failed", P["err"])
        self.chat_add("", f"Image generation failed: {msg}\nCoding features still work normally.", "err")

    # =================================================================
    # running things
    # =================================================================
    def confirm_run(self, display):
        if (self.pid, display) in self.confirmed_cmds:
            return True
        ok = messagebox.askokcancel(
            "Run this command?",
            f"Command:  {display}\nFolder:   {self.root}\n\n"
            "Generated or downloaded code is NOT sandboxed. It runs with your Windows user's permissions "
            "(it can read or delete your files and use the network). Secret-looking environment variables are "
            "hidden from it, but that is not real isolation.\n\nOnly run code you trust.", parent=self.win)
        if ok:
            self.confirmed_cmds.add((self.pid, display))
        return ok

    def ask_save_all(self):
        dirty = [t for t in self.tabs.values() if t.modified]
        if not dirty:
            return True
        ans = messagebox.askyesnocancel("Unsaved changes", f"Save {len(dirty)} modified file(s) before running?", parent=self.win)
        if ans is None:
            return False
        if ans:
            return all(self.save_tab(t) for t in dirty)
        return True

    def start_process(self, cmd, display, tab, kind, shell=False, timeout=None):
        if self.runner.active:
            messagebox.showinfo("Busy", "A process is already running. Stop it first.", parent=self.win)
            return False
        pane = self.panes[tab]
        if tab in ("run", "test"):
            pane.clear()
        self.show_out_tab(tab)
        pane.append(f"$ {display}\n", "cmd")
        self.act_log(f"Ran command: {display}")
        self.set_status("Running checks" if kind == "test" else "Running", P["accent2"])
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        buf, t0 = deque(maxlen=3000), time.time()

        def on_line(tag, line):
            buf.append(line)
            self.ui(pane.append, line, tag)

        def on_done(code, timed_out, stopped):
            self.ui(self._process_done, kind, display, tab, "".join(buf), code, timed_out, stopped, time.time() - t0)
        self.runner.start(cmd, self.root, on_line, on_done, timeout or int(self.meta["settings"]["timeout"]), shell)
        return True

    def _process_done(self, kind, display, tab, output, code, timed_out, stopped, secs):
        pane = self.panes[tab]
        ok = code == 0 and not timed_out and not stopped
        if timed_out:
            pane.append(f"\n[timed out after {secs:.0f}s and was stopped]\n", "err")
        elif stopped:
            pane.append("\n[stopped by you]\n", "warn")
        else:
            pane.append(f"\n[exit code {code} after {secs:.1f}s]\n", "ok" if ok else "err")
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        if kind not in ("run", "test"):
            self.set_status("Completed" if ok else "Failed", P["ok"] if ok else P["err"])
            return
        self.last_run = {"display": display, "kind": kind, "ok": ok, "code": code, "timed_out": timed_out,
                         "output": output}
        self.problems = parse_problems(output, self.root) if not ok else []
        self.problem_lines = {}
        self.prob_list.delete(0, "end")
        for p in self.problems:
            self.problem_lines.setdefault(p["file"], set()).add(p["line"])
            self.prob_list.insert("end", f"{p['file']}:{p['line']}   {p['msg']}")
        self.redraw_all_gutters()
        if ok:
            self.btn_fix.pack_forget()
            if self.pending_verify:
                self.chat_add("", f"VERIFIED: `{display}` finished with exit code 0 after the fix.", "ok")
            self.pending_verify = None
            self.fix_attempts = 0
            self.set_status("Completed", P["ok"])
        else:
            if not stopped:
                self.btn_fix.pack(side="left", padx=8)
                if self.problems:
                    self.show_out_tab("problems")
            if self.pending_verify and not stopped:
                self.chat_add("", f"NOT FIXED: `{display}` still fails (attempt {self.fix_attempts}/{MAX_FIX_ATTEMPTS}). "
                                  "Press Fix with AI to try again, or fix it yourself.", "err")
            self.pending_verify = None
            self.set_status("Failed" if not stopped else "Stopped", P["err"] if not stopped else P["warn"])

    def open_problem(self, event=None):
        sel = self.prob_list.curselection()
        if sel and sel[0] < len(self.problems):
            p = self.problems[sel[0]]
            self.open_file(p["file"], p["line"])

    def run_project(self):
        if not self.require_project():
            return
        spec = detect_run_command(self.root, self.meta)
        if spec is None:
            messagebox.showinfo("Run", "I couldn't work out how to run this project.\nSet a custom run command in "
                                       "Project settings, or ask the AI to add a main.py / start script.", parent=self.win)
            return
        if spec["mode"] == "error":
            messagebox.showerror("Run", spec["message"], parent=self.win)
            return
        if spec["mode"] == "preview":
            self.start_preview()
            return
        if not self.ask_save_all() or not self.confirm_run(spec["display"]):
            return
        self.start_process(spec["cmd"], spec["display"], "run", "run", shell=spec.get("shell", False))

    def run_tests(self):
        if not self.require_project():
            return
        kind, s = self.meta.get("kind"), self.meta["settings"]
        if kind == "python":
            py, _ = find_python(self.root, s)
            if not py:
                messagebox.showerror("Test", "No Python interpreter found.", parent=self.win)
                return
            has_tests = os.path.isdir(os.path.join(self.root, "tests")) or any(
                f.startswith("test_") and f.endswith(".py") for f in os.listdir(self.root))
            if has_tests:
                cmd, disp = [py, "-m", "pytest", "-q"], "python -m pytest -q"
            else:
                cmd, disp = [py, "-c", _SYNTAX_CHECK], "syntax check of all .py files (no tests found)"
        elif kind == "node":
            pkg = _read_json(os.path.join(self.root, "package.json"), {})
            npm = shutil.which("npm")
            if not npm or "test" not in (pkg.get("scripts") or {}):
                messagebox.showinfo("Test", "No 'test' script in package.json (or npm not installed).", parent=self.win)
                return
            cmd, disp = [npm, "test"], "npm test"
        else:
            messagebox.showinfo("Test", "There are no automatic checks for this project type yet.", parent=self.win)
            return
        if not self.ask_save_all() or not self.confirm_run(disp):
            return
        self.start_process(cmd, disp, "test", "test")

    def stop_process(self):
        self.runner.stop()

    def run_terminal_command(self, event=None):
        cmd = self.term_entry.get().strip()
        if not cmd or not self.require_project():
            return
        low = cmd.lower()
        words = cmd.split()
        shell_words = {"cd", "dir", "echo", "set", "type", "cls", "copy", "move", "md", "mkdir", "ren",
                       "ver", "where", "start", "python", "py", "pip", "git", "node", "npm", "dir", "del"}
        if len(words) >= 3 and words[0].lower() not in shell_words and shutil.which(words[0]) is None:
            self.panes["terminal"].append(
                "That looks like a message to the AI, not a shell command, so it was NOT run.\n"
                "Type it in the AI panel on the right instead.\n", "warn")
            return
        if any(re.search(p, low) for p in BLOCKED_CMD):
            self.panes["terminal"].append("Blocked: that command looks destructive. Run it in your own terminal if you "
                                          "really mean it.\n", "err")
            return
        if INSTALL_CMD.search(low) and not messagebox.askokcancel(
                "Install packages?", f"{cmd}\n\nThis downloads and installs software into the environment it points to. Continue?",
                parent=self.win):
            return
        if not self.terminal_ok:
            if not messagebox.askokcancel("Terminal", "Terminal commands run with your user permissions and are not sandboxed.\nContinue?",
                                          parent=self.win):
                return
            self.terminal_ok = True
        self.term_entry.delete(0, "end")
        self.start_process(cmd, cmd, "terminal", "terminal", shell=True, timeout=300)

    # ---------------- dependencies / git ----------------
    def check_dependencies(self):
        if not self.require_project():
            return
        pane = self.panes["terminal"]
        self.show_out_tab("terminal")
        kind = self.meta.get("kind")
        if kind == "python":
            py, why = find_python(self.root, self.meta["settings"])
            if not py:
                pane.append("No Python interpreter found.\n", "err")
                return
            names = python_import_names(self.root)
            pane.append(f"Checking {len(names)} imported module(s) with: {py}  ({why})\n", "info")

            def work():
                try:
                    r = subprocess.run([py, "-c", _DEP_CHECK, json.dumps(names)], capture_output=True, text=True,
                                       timeout=60, cwd=self.root, env=clean_env(), creationflags=_NOWIN)
                    missing = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else []
                    if missing:
                        msg = ("Missing: " + ", ".join(missing) + "\nInstall with:  python -m pip install <name>  "
                               "(pip names can differ from import names, e.g. cv2 = opencv-python).\n")
                        self.ui(pane.append, msg, "warn")
                    else:
                        self.ui(pane.append, "All imported modules are available to that interpreter.\n", "ok")
                    if os.path.isfile(os.path.join(self.root, "requirements.txt")):
                        self.ui(pane.append, "requirements.txt found - use 'Install Requirements...' to install it.\n", "info")
                except Exception as e:
                    self.ui(pane.append, f"Dependency check failed: {e}\n", "err")
            threading.Thread(target=work, daemon=True).start()
        elif kind == "node":
            node, npm = shutil.which("node"), shutil.which("npm")
            pane.append(f"node: {node or 'NOT FOUND'}\nnpm: {npm or 'NOT FOUND'}\n", "info" if node and npm else "err")
            pane.append(self.tools.inspect_dependencies() + "\n", "info")
            if not os.path.isdir(os.path.join(self.root, "node_modules")):
                pane.append("node_modules is missing. Run:  npm install\n", "warn")
        else:
            pane.append("No dependency files to check for this project type.\n", "info")

    def install_requirements(self):
        if not self.require_project():
            return
        kind = self.meta.get("kind")
        if kind == "python" and os.path.isfile(os.path.join(self.root, "requirements.txt")):
            py, _ = find_python(self.root, self.meta["settings"])
            if not py:
                messagebox.showerror("Install", "No Python interpreter found.", parent=self.win)
                return
            cmd, disp = [py, "-m", "pip", "install", "-r", "requirements.txt"], "python -m pip install -r requirements.txt"
            target = f"into the interpreter:\n{py}"
        elif kind == "node" and shutil.which("npm"):
            cmd, disp, target = [shutil.which("npm"), "install"], "npm install", "into this project's node_modules"
        else:
            messagebox.showinfo("Install", "No requirements.txt / package.json (or npm) found.", parent=self.win)
            return
        if messagebox.askokcancel("Install dependencies?", f"{disp}\n\nThis downloads and installs packages {target}\n"
                                                           "(not into AP's own environment unless that interpreter is the same one).",
                                  parent=self.win):
            self.start_process(cmd, disp, "terminal", "install", timeout=900)

    def git_status(self):
        if not self.require_project():
            return
        if not os.path.isdir(os.path.join(self.root, ".git")) or not shutil.which("git"):
            messagebox.showinfo("Git", "This project isn't a Git repository (or Git isn't installed).", parent=self.win)
            return
        self.start_process(["git", "status", "--short", "--branch"], "git status --short --branch", "terminal", "git", timeout=30)

    # ---------------- preview ----------------
    def start_preview(self):
        if not self.require_project():
            return
        if self.preview_runner.active:
            self.refresh_preview()
            return
        kind, root = self.meta.get("kind"), self.root
        pane = self.panes["run"]
        if kind == "node":
            spec = detect_run_command(root, self.meta)
            if not spec or spec["mode"] != "process":
                messagebox.showinfo("Preview", "Add a 'dev' or 'start' script to package.json to preview.", parent=self.win)
                return
            cmd, disp, url = spec["cmd"], spec["display"], None
        elif kind == "web" or os.path.isfile(os.path.join(root, "index.html")):
            py, _ = find_python(root, self.meta["settings"])
            if not py:
                messagebox.showerror("Preview", "No Python interpreter found (used for a tiny local web server).", parent=self.win)
                return
            port = free_port()
            cmd, disp, url = [py, "-m", "http.server", str(port), "--bind", "127.0.0.1"], f"python -m http.server {port}", f"http://127.0.0.1:{port}/"
        else:
            messagebox.showinfo("Preview", "Preview works for static web projects (index.html) and Node projects with a dev/start script.",
                                parent=self.win)
            return
        if not self.confirm_run(disp):
            return
        self.preview_url, self.preview_opened = url, False
        self.show_out_tab("run")
        pane.append(f"[preview] $ {disp}\n", "cmd")
        self.set_status("Starting preview", P["accent2"])

        def on_line(tag, line):
            self.ui(pane.append, "[preview] " + line, tag)
            if url is None and not self.preview_opened:
                m = re.search(r"https?://(?:localhost|127\.0\.0\.1):\d+[^\s]*", line)
                if m:
                    self.preview_opened = True
                    self.ui(self._preview_found, m.group(0))

        def on_done(code, to, stopped):
            self.ui(self._preview_done, code, stopped)
        self.preview_runner.start(cmd, root, on_line, on_done, timeout=0)
        if url:
            self.win.after(1000, self._preview_check)

    def _preview_check(self):
        if self.preview_runner.active and self.preview_url:
            self.preview_opened = True
            webbrowser.open(self.preview_url)
            self.set_status(f"Preview: {self.preview_url}", P["ok"])

    def _preview_found(self, url):
        self.preview_url = url
        webbrowser.open(url)
        self.set_status(f"Preview: {url}", P["ok"])

    def _preview_done(self, code, stopped):
        self.preview_url = None
        if stopped:
            self.panes["run"].append("[preview stopped]\n", "warn")
            self.set_status("Preview stopped")
        else:
            self.panes["run"].append(f"[preview exited with code {code}] - see errors above\n", "err")
            self.set_status("Preview failed", P["err"])
            self.chat_add("", "The preview server stopped unexpectedly. Check Run Output for the error.", "err", persist=False)

    def stop_preview(self):
        self.preview_runner.stop()

    def refresh_preview(self):
        if self.preview_runner.active and self.preview_url:
            webbrowser.open(self.preview_url)
        else:
            messagebox.showinfo("Preview", "No preview is running. Use Preview > Start Preview.", parent=self.win)

    def copy_preview_url(self):
        if self.preview_url:
            self.win.clipboard_clear()
            self.win.clipboard_append(self.preview_url)
            self.set_status("Preview address copied")
        else:
            messagebox.showinfo("Preview", "No preview address yet. Start the preview first.", parent=self.win)

    # ---------------- export ----------------
    def export_zip(self):
        if not self.require_project():
            return
        files, skipped = [], []
        for rel in iter_files(self.root, 20000):
            full = os.path.join(self.root, rel)
            if is_secret(rel):
                skipped.append(rel)
            elif rel.endswith(".pyc") or os.path.getsize(full) > 25_000_000:
                continue
            else:
                files.append(rel)
        total = sum(os.path.getsize(os.path.join(self.root, r)) for r in files)
        sample = "\n".join(files[:25]) + ("\n..." if len(files) > 25 else "")
        msg = (f"{len(files)} file(s), {total / 1024:.0f} KB will be exported.\n\n{sample}\n\n"
               f"Always excluded: .git, virtual environments, node_modules, caches, build output.\n"
               f"Secret-looking files skipped: {', '.join(skipped) if skipped else 'none'}\n\nContinue?")
        if not messagebox.askyesno("Export project as ZIP", msg, parent=self.win):
            return
        dest = filedialog.asksaveasfilename(parent=self.win, defaultextension=".zip", initialfile=self.meta["name"] + ".zip",
                                            filetypes=[("ZIP", "*.zip")])
        if not dest:
            return
        self.set_status("Exporting ZIP", P["accent2"])

        def work():
            try:
                with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
                    for rel in files:
                        z.write(os.path.join(self.root, rel), rel)
                self.ui(self.set_status, "Completed", P["ok"])
                self.ui(self.chat_add, "", f"Exported {len(files)} file(s) to {dest}", "ok", False)
            except Exception as e:
                self.ui(self.set_status, "Failed", P["err"])
                self.ui(self.chat_add, "", f"Export failed: {e}", "err", False)
        threading.Thread(target=work, daemon=True).start()

    # ---------------- history / restore ----------------
    def history_dialog(self):
        if not self.require_project():
            return
        cps = list_checkpoints(self.pid)
        top = tk.Toplevel(self.win)
        top.title("History & Restore")
        top.geometry("760x460")
        top.configure(bg=P["bg"])
        top.transient(self.win)
        tk.Label(top, text="Checkpoints (saved before every AI change, delete and restore)", bg=P["bg"], fg=P["accent2"],
                 font=(UI_FONT, 11, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        body = tk.Frame(top, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=14)
        lb = tk.Listbox(body, width=48, bg=P["panel2"], fg=P["text"], bd=0, highlightthickness=0, font=(MONO, 9),
                        selectbackground=P["sel"], activestyle="none", exportselection=False)
        lb.pack(side="left", fill="y")
        det = tk.Text(body, bg=P["panel2"], fg=P["text"], bd=0, font=(MONO, 9), wrap="word", state="disabled", padx=8, pady=6)
        det.pack(side="left", fill="both", expand=True, padx=(8, 0))
        for c in cps:
            lb.insert("end", f"{c['time'].replace('T', ' ')}  {c['label'][:30]}  ({len(c['entries'])})")
        if not cps:
            lb.insert("end", "(no checkpoints yet)")

        def show(e=None):
            sel = lb.curselection()
            if not sel or not cps:
                return
            c = cps[sel[0]]
            det.config(state="normal")
            det.delete("1.0", "end")
            det.insert("end", f"{c['label']}\n{c['time']}\n\n")
            for en in c["entries"]:
                det.insert("end", ("saved copy of   " if en["existed"] else "did not exist   ") + en["path"] + "\n")
            det.config(state="disabled")
        lb.bind("<<ListboxSelect>>", show)

        def restore():
            sel = lb.curselection()
            if not sel or not cps:
                return
            c = cps[sel[0]]
            new_files = [en["path"] for en in c["entries"]
                         if not en["existed"] and os.path.isfile(safe_join(self.root, en["path"]))]
            old = [en["path"] for en in c["entries"] if en["existed"]]
            msg = "Restore " + (", ".join(old) if old else "nothing") + " to their earlier content?"
            if new_files:
                msg += "\n\nThese files were CREATED by that change and will be REMOVED:\n" + "\n".join(new_files)
            msg += "\n\nA new checkpoint of the current state is saved first, so this can be undone."
            if not messagebox.askyesno("Restore checkpoint", msg, parent=top):
                return
            try:
                restore_checkpoint(self.pid, self.root, c["id"], remove_new=True)
            except Exception as e:
                messagebox.showerror("Restore", str(e), parent=top)
                return
            for en in c["entries"]:
                rel = en["path"]
                if rel in self.tabs:
                    if os.path.isfile(safe_join(self.root, rel)):
                        self.tabs[rel].modified = False
                        self.reload_tab(rel)
                    else:
                        self.tabs[rel].modified = False
                        self.close_tab(rel)
            self.store.add_history(self.pid, {"type": "restore", "summary": f"restored {c['id']}", "checkpoint": c["id"]})
            self.act_log(f"Restored checkpoint {c['id']}")
            self.refresh_tree()
            self.render_tabs()
            self.chat_add("", f"RESTORED checkpoint {c['id']}.", "ok", persist=False)
            top.destroy()
        self.mkbtn(top, "Restore selected", restore, primary=True).pack(anchor="e", padx=14, pady=10)

    # =================================================================
    def on_close(self):
        if not self.close_all_tabs():
            return
        if self.runner.active and not messagebox.askyesno("Quit", "A process is still running. Stop it and close?", parent=self.win):
            return
        self.runner.stop()
        self.preview_runner.stop()
        self.token += 1
        if self.pid and self.meta:
            try:
                self.store.save_meta(self.pid, self.meta)
            except Exception:
                pass
        self.closed = True
        _STATE["ws"] = None
        self.win.destroy()


# =====================================================================
# Public entry point (the only thing the main app needs)
# =====================================================================
_STATE = {"ws": None}


def open_vibe_code(parent, theme=None, image_fn=None):
    ws = _STATE["ws"]
    if ws is not None:
        try:
            if ws.win.winfo_exists():
                ws.win.deiconify()
                ws.win.lift()
                ws.win.focus_force()
                return ws.win
        except tk.TclError:
            pass
    ws = VibeWorkspace(parent, theme, image_fn)
    _STATE["ws"] = ws
    return ws.win


if __name__ == "__main__":          # quick stand-alone test:  python vibe_code.py
    _root = tk.Tk()
    _root.withdraw()
    _w = open_vibe_code(_root)
    _w.protocol("WM_DELETE_WINDOW", lambda: (_STATE["ws"].on_close(), _root.destroy()))
    _root.mainloop()