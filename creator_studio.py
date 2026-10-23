import os, re, json, uuid, shutil, threading, time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import ollama

BASE_DIR = os.path.join(os.path.expanduser("~"), ".ava_assistant")
CREATOR_FILE = os.path.join(BASE_DIR, "creator.json")
SCENE_IMG_DIR = os.path.join(BASE_DIR, "scene_images")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
AUDIO_DIR = os.path.join(BASE_DIR, "narration")
for _d in (SCENE_IMG_DIR, EXPORT_DIR, AUDIO_DIR):
    os.makedirs(_d, exist_ok=True)

ACTIVE_MODEL = {"name": "llama3.1:8b"}   # changed live from the Settings tab
CREATOR_MODEL = "llama3.1:8b"   # run: ollama pull llama3.1:8b  (3b is too weak for good scripts)
STATUSES = ["IDEA", "RESEARCH", "SCRIPT", "PRODUCTION", "READY", "PUBLISHED", "ANALYZING"]
HOOK_TYPES = ["Curiosity", "Shock", "Question", "Story", "Contrarian", "Mystery", "Fast-paced"]
PLATFORMS = ["YouTube Shorts", "Instagram Reels", "Facebook Reels"]  # add more later

DEFAULT_SETTINGS = {
    "niche": "", "audience": "", "platform": "YouTube Shorts", "duration_sec": "45",
    "language": "English", "tone": "curious, confident", "storytelling_style": "story-driven",
    "intensity": "medium", "goal": "entertainment + education",
    "voice_style": "calm documentary narrator", "visual_style": "cinematic realistic",
    "posting_frequency": "1 per day",
    "model": CREATOR_MODEL,
}

SYSTEM_BASE = (
    "You are a blunt, experienced short-form video strategist and script editor for faceless "
    "voice-over content. You challenge weak ideas, explain WHY something is weak, and keep the "
    "creator's original concept instead of rewriting it. No hype, no filler.\n"
)
FACT_RULES = (
    "FACT RULES (never break):\n"
    "- Never invent statistics, quotes, dates, names, studies or events.\n"
    "- Tag factual claims: VERIFIED (only if it appears in the SOURCES text provided), "
    "INFERENCE (reasonable but unconfirmed), STORY (creative framing), FICTION (invented on purpose).\n"
    "- With no sources provided, nothing may be VERIFIED; use 'UNVERIFIED - check before publishing'.\n"
    "- Name a specific real person/place/species/event only if you are confident it is real; "
    "otherwise say it needs checking.\n"
)

# ---------- storage ----------
def _load():
    try:
        with open(CREATOR_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    d.setdefault("settings", dict(DEFAULT_SETTINGS))
    for k, v in DEFAULT_SETTINGS.items():
        d["settings"].setdefault(k, v)
    d.setdefault("videos", [])
    d.setdefault("experiments", [])
    return d

_save_lock = threading.Lock()

def _save(d):
    # write to a temp file first so a crash mid-save can't corrupt creator.json
    with _save_lock:
        try:
            tmp = CREATOR_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, CREATOR_FILE)
        except Exception as e:
            print("Creator save error:", e)

def new_video(title):
    return {"id": uuid.uuid4().hex[:8], "title": title, "topic": title, "status": "IDEA",
            "idea": "", "script": "", "hook_type": "", "hook_text": "", "sources": "",
            "research": None, "scenes": [], "voice": None, "package": None,
            "metrics": None, "experiment_id": None, "fact_check": None, "score": None,
            "thumbnails": None, "audio_path": "",
            "created_at": datetime.now().isoformat(timespec="seconds")}

def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0

# ---------- helpers ----------
def _parse_json(text):
    text = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise

def _chat(system, user, as_json=False, temperature=0.7, retries=2):
    for attempt in range(retries + 1):
        try:
            kwargs = {"format": "json"} if as_json else {}
            r = ollama.chat(
                model=ACTIVE_MODEL["name"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": max(0.2, temperature - 0.2 * attempt),
                         "num_ctx": 8192, "num_predict": 4000},
                keep_alive="30m", **kwargs)
            text = r["message"]["content"]
            if not as_json:
                return text
            parsed = _parse_json(text)
            if isinstance(parsed, dict):
                return parsed
        except ollama.ResponseError as e:
            raise RuntimeError(f"Model problem: {e}. Try: ollama pull {ACTIVE_MODEL['name']}")
        except ConnectionError:
            raise RuntimeError("Ollama isn't running. Start it and try again.")
        except (ValueError, KeyError):
            pass  # bad JSON, retry
    raise RuntimeError("The model returned unusable output 3 times. Try again.")

def _settings_block(s):
    return "\n".join(f"- {k}: {v}" for k, v in s.items() if str(v).strip() and k != "model")

def _target_words(s):
    return int(_num(s.get("duration_sec", 45)) * 2.5) or 110

def pretty(obj, level=0):
    pad = "  " * level
    if isinstance(obj, dict):
        return "\n".join(
            f"{pad}{k}:" + ("\n" + pretty(v, level + 1) if isinstance(v, (dict, list)) else f" {v}")
            for k, v in obj.items())
    if isinstance(obj, list):
        return "\n".join(
            f"{pad}- " + (pretty(i, level + 1).lstrip() if isinstance(i, (dict, list)) else str(i))
            for i in obj)
    return f"{pad}{obj}"

# ---------- AI steps ----------
def generate_ideas(settings, existing_titles, n=8):
    user = (f"Creator settings:\n{_settings_block(settings)}\n\n"
            f"Already made/saved (do NOT repeat or reword): {existing_titles[-40:]}\n\n"
            f"Give {n} short-form ideas, each with a DIFFERENT angle (untold story, myth-bust, "
            "comparison, 'what if', counterintuitive fact...). No generic listicles. You have NO "
            "live trend data: never claim something is trending. Be honest about each weakness.\n"
            'Return ONLY JSON: {"ideas":[{"title":"","angle":"","why_it_could_work":"",'
            '"weakness":"","needs_fact_check":true}]}')
    return _chat(SYSTEM_BASE + FACT_RULES, user, True, 0.9)

def research_notes(topic, sources):
    user = (f"Topic: {topic}\n\nSOURCES (the ONLY allowed basis for VERIFIED facts):\n"
            f"{sources or '(none provided)'}\n\n"
            "Organise what can and cannot be claimed.\n"
            'Return ONLY JSON: {"verified_facts":[{"claim":"","where_in_sources":""}],'
            '"inferences":[],"speculation":[],"open_questions_to_check":[],"do_not_claim":[]}')
    return _chat(SYSTEM_BASE + FACT_RULES, user, True, 0.2)

def upgrade_script(rough, settings, sources):
    user = (f"Creator settings:\n{_settings_block(settings)}\n"
            f"Target length: about {_target_words(settings)} spoken words.\n\n"
            f"SOURCES:\n{sources or '(none provided)'}\n\nROUGH SCRIPT/IDEA:\n{rough}\n\n"
            "Task: keep the creator's concept. Improve hook, pacing, storytelling, curiosity gaps, "
            "transitions; cut boring parts; make it natural spoken narration; add pattern interrupts "
            "and mark where visuals should change with [CUT]. First judge the original honestly.\n"
            'Return ONLY JSON: {"verdict":"","weak_points":[],"upgraded_script":"","changes_made":[],'
            '"pattern_interrupts":[],"fact_flags":[{"claim":"","tag":""}]}')
    return _chat(SYSTEM_BASE + FACT_RULES, user, True, 0.6)

def generate_hooks(script, settings):
    user = (f"Settings:\n{_settings_block(settings)}\n\nScript:\n{script}\n\n"
            f"Write one strong hook (max ~15 spoken words, no misleading claims) for each type: "
            f"{', '.join(HOOK_TYPES)}.\n"
            'Return ONLY JSON: {"hooks":[{"type":"","text":"","why":""}]}')
    return _chat(SYSTEM_BASE, user, True, 0.9)

def apply_hook(script, hook, settings):
    user = (f"Script:\n{script}\n\nAdapt it so it opens with this {hook['type']} hook: \"{hook['text']}\". "
            "Smooth the lines right after it and make sure the hook's promise is paid off later. "
            "Change the rest only if needed. Return ONLY the full script as plain text.")
    return _chat(SYSTEM_BASE, user, False, 0.5)

def script_variants(script, types, settings):
    user = (f"Script:\n{script}\n\nCreate one version per hook type: {', '.join(types)}. Same topic and "
            "body, only the hook (and the lines right after it) change.\n"
            'Return ONLY JSON: {"variants":[{"type":"","hook":"","script":""}]}')
    return _chat(SYSTEM_BASE, user, True, 0.8)

def plan_scenes(script, settings):
    user = (f"Settings:\n{_settings_block(settings)}\n\nSCRIPT:\n{script}\n\n"
            "Make a scene-by-scene plan (scenes of ~2-6s, total ≈ target duration, vertical 9:16). "
            "First write a character_bible and style_bible. In EVERY image_prompt, copy the relevant "
            "character/location description and the style_bible word for word so visuals stay "
            "consistent, and keep the time period accurate. Be specific; no placeholders like "
            "'a mysterious person'. Suggest real footage (archives, Wikimedia Commons, NASA...) only "
            "if you're confident it exists, otherwise say 'search for ...'. Tag each scene's "
            "fact_tag as VERIFIED/INFERENCE/STORY/FICTION.\n"
            'Return ONLY JSON: {"character_bible":"","style_bible":"","scenes":[{"narration":"",'
            '"visual":"","image_prompt":"","real_footage":"","on_screen_text":"","camera":"",'
            '"duration_sec":0,"transition":"","sfx":"","music_mood":"","fact_tag":""}]}')
    return _chat(SYSTEM_BASE + FACT_RULES, user, True, 0.6)

def prep_voice(script, settings):
    user = (f"Voice style: {settings.get('voice_style')}\nScript:\n{script}\n\n"
            "Prepare for AI voice generation: put '...' for pauses, CAPS for stressed words, split "
            "into short breaths, add pronunciation guidance for hard names/terms, emotional delivery "
            "notes per section and a speed recommendation.\n"
            'Return ONLY JSON: {"voice_script":"","pronunciations":[{"word":"","say_as":""}],'
            '"delivery_notes":[],"speed_wpm":0}')
    return _chat(SYSTEM_BASE, user, True, 0.4)

def make_package(script, settings):
    user = (f"Settings:\n{_settings_block(settings)}\n\nFinal script:\n{script}\n\n"
            "Create the publish package. Titles: 5 curiosity-driven options that are NOT misleading "
            "(the video must deliver what the title promises). Captions: platform-specific. "
            "Hashtags: 3-6 specific ones per platform, no generic spam like #fyp.\n"
            'Return ONLY JSON: {"titles":[],"captions":{"YouTube Shorts":"","Instagram Reels":"",'
            '"Facebook Reels":""},"hashtags":{"YouTube Shorts":[],"Instagram Reels":[],'
            '"Facebook Reels":[]},"title_honesty_check":""}')
    return _chat(SYSTEM_BASE, user, True, 0.7)

# ---------- script analysis (numbers come from code, opinions from the model) ----------
def script_stats(script, settings):
    body = re.sub(r"\[CUT\]", " ", script or "")
    words = re.findall(r"[\w'’-]+", body)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s.strip()]
    first = sentences[0] if sentences else ""
    return {
        "words": len(words),
        "est_seconds": round(len(words) / 2.5),
        "target_seconds": int(_num(settings.get("duration_sec", 45))),
        "cuts_marked": (script or "").count("[CUT]"),
        "hook_words": len(re.findall(r"[\w'’-]+", first)),
        "avg_sentence_words": round(len(words) / len(sentences), 1) if sentences else 0,
    }

def score_script(script, settings):
    stats = script_stats(script, settings)
    user = (f"Creator settings:\n{_settings_block(settings)}\n\n"
            f"Measured by code (trust these): {json.dumps(stats)}\n\nSCRIPT:\n{script}\n\n"
            "Judge this like a hard-to-impress editor. Scores are your OPINION, not a prediction "
            "of views - say so in the verdict. Quote the exact lines that will lose viewers and "
            "give a concrete rewrite for each.\n"
            'Return ONLY JSON: {"hook_score":0,"pacing_score":0,"clarity_score":0,"curiosity_score":0,'
            '"verdict":"","retention_risks":[{"line":"","why":"","rewrite":""}],"keep_as_is":[]}')
    return _chat(SYSTEM_BASE, user, True, 0.4)

def fact_check_pass(script, sources):
    user = (f"SOURCES:\n{sources or '(none provided)'}\n\nSCRIPT:\n{script}\n\n"
            "List every factual claim in the script (names, numbers, dates, causes, 'firsts', "
            "superlatives). Tag VERIFIED only if the SOURCES text supports it, otherwise INFERENCE, "
            "STORY, FICTION or UNVERIFIED. Risk is high for numbers, dates, quotes, accusations and "
            "anything a viewer could repeat as fact.\n"
            'Return ONLY JSON: {"claims":[{"claim":"","tag":"","risk":"","how_to_verify":"",'
            '"safer_wording":""}],"publish_blockers":[]}')
    return _chat(SYSTEM_BASE + FACT_RULES, user, True, 0.2)

def thumbnail_ideas(script, settings):
    user = (f"Settings:\n{_settings_block(settings)}\n\nScript:\n{script}\n\n"
            "Give 4 cover-frame ideas for a vertical short. Text overlay max 4 words, must not "
            "promise anything the video doesn't deliver.\n"
            'Return ONLY JSON: {"covers":[{"text_overlay":"","image_prompt":"","why":""}]}')
    return _chat(SYSTEM_BASE, user, True, 0.8)

def series_plan(topic, settings, existing_titles, n=7):
    user = (f"Creator settings:\n{_settings_block(settings)}\n\nSeries theme / seed idea: {topic}\n"
            f"Already made (do NOT repeat): {existing_titles[-40:]}\n\n"
            f"Plan {n} connected episodes. Each must work for a new viewer but tease or build on the "
            "previous one. Vary hook types and angles. No fake trend claims. Flag episodes needing fact-checks.\n"
            'Return ONLY JSON: {"series_name":"","episodes":[{"title":"","angle":"","hook_type":"",'
            '"builds_on":"","needs_fact_check":true}]}')
    return _chat(SYSTEM_BASE + FACT_RULES, user, True, 0.8)

def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_srt(scenes):
    """Pure code, no AI. Timings are estimates from each scene's duration_sec."""
    t, idx, lines = 0.0, 1, []
    for sc in scenes:
        dur = _num(sc.get("duration_sec")) or 3.0
        text = (sc.get("narration") or "").strip()
        if text:
            lines += [str(idx), f"{_srt_time(t)} --> {_srt_time(t + dur)}", text, ""]
            idx += 1
        t += dur
    return "\n".join(lines)

# ---------- analytics ----------
def summarize_performance(videos):
    rows = [v for v in videos if v.get("metrics")]
    out = {"videos_with_data": len(rows), "by_hook_type": {}, "by_topic": {}}
    for v in rows:
        m = v["metrics"]
        for key, name in (("by_hook_type", v.get("hook_type") or "unknown"),
                          ("by_topic", v.get("topic") or "unknown")):
            b = out[key].setdefault(name, {"n": 0, "views": [], "watch": []})
            b["n"] += 1
            b["views"].append(m.get("views", 0))
            b["watch"].append(m.get("avg_watch_pct", 0))
    for key in ("by_hook_type", "by_topic"):
        for b in out[key].values():
            views, watch = b.pop("views"), b.pop("watch")
            b["avg_views"] = round(sum(views) / len(views), 1)
            b["avg_watch_pct"] = round(sum(watch) / len(watch), 1)
    return out

def analyze_performance(videos, settings):
    rows = [{"title": v["title"], "topic": v.get("topic"), "hook_type": v.get("hook_type"),
             "hook": v.get("hook_text"), "metrics": v["metrics"]} for v in videos if v.get("metrics")]
    if not rows:
        return "No performance data yet. Enter metrics for at least one published video."
    user = (f"Creator settings:\n{_settings_block(settings)}\n\n"
            f"Numbers computed by code (trust these):\n{pretty(summarize_performance(videos))}\n\n"
            f"Raw rows:\n{json.dumps(rows, indent=1)}\n\n"
            "Analyse. Rules: with fewer than ~10 videos treat everything as weak evidence and say so; "
            "separate what the data shows from your guesses; never invent numbers. Give: what probably "
            "worked, what probably failed, hook comparison, topic comparison, where viewers probably "
            "dropped off (use the drop-off notes), and 3 next tests, each changing ONE variable.")
    return _chat(SYSTEM_BASE, user, False, 0.4)

def experiment_result(exp, videos, metric="avg_watch_pct"):
    by_id = {v["id"]: v for v in videos}
    rows = []
    for var in exp["variants"]:
        v = by_id.get(var["video_id"])
        if v and v.get("metrics"):
            rows.append((var["label"], v["metrics"].get(metric, 0)))
    return sorted(rows, key=lambda r: r[1], reverse=True)

# ---------- scene images / export ----------
def generate_scene_images(video, image_fn, progress=None):
    folder = os.path.join(SCENE_IMG_DIR, video["id"])
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, s in enumerate(video.get("scenes", []), 1):
        src = image_fn(s.get("image_prompt") or s.get("visual", ""))
        dest = os.path.join(folder, f"scene_{i:02d}_{uuid.uuid4().hex[:6]}.png")
        shutil.copy(src, dest)
        s["image"] = dest
        paths.append(dest)
        if progress:
            progress(i)
    return paths

def export_package(v):
    name = re.sub(r"[^\w\-]+", "_", v["title"])[:40]
    path = os.path.join(EXPORT_DIR, f"{name}_{v['id']}.md")
    parts = [f"# {v['title']}", f"Status: {v['status']}",
             f"Hook ({v.get('hook_type')}): {v.get('hook_text')}",
             "\n## Script\n" + (v.get("script") or ""),
             "\n## Scenes\n" + pretty(v.get("scenes") or []),
             "\n## Voice\n" + pretty(v.get("voice") or {}),
             "\n## Publish package\n" + pretty(v.get("package") or {}),
             "\n## Research\n" + pretty(v.get("research") or {}),
             "\n## Fact check\n" + pretty(v.get("fact_check") or {}),
             "\n## Script score\n" + pretty(v.get("score") or {}),
             "\n## Cover ideas\n" + pretty(v.get("thumbnails") or {}),
             "\n## Narration audio\n" + (v.get("audio_path") or "(not generated)")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path
# ---------- design system: every colour/font lives here (dark-blue only, no light mode) ----------
CS = {
    "bg": "#0b1226", "sidebar": "#080e1f", "panel": "#111c3a", "card": "#15234a",
    "border": "#243463", "accent": "#4f8cff", "accent_dark": "#3569d6",
    "fg": "#e8edff", "muted": "#8fa0cc",
    "ok": "#22c55e", "warn": "#f59e0b", "bad": "#ef4444", "font": "Segoe UI",
}
STATUS_COLORS = {"IDEA": "#6366f1", "RESEARCH": "#0ea5e9", "SCRIPT": "#8b5cf6",
                 "PRODUCTION": "#f59e0b", "READY": "#22c55e", "PUBLISHED": "#14b8a6",
                 "ANALYZING": "#ec4899"}

def video_progress(v):
    """(steps done, total steps) for the 8-step pipeline. Pure code, no AI."""
    checks = [bool(v.get("script")), bool(v.get("hook_text")), bool(v.get("fact_check")),
              bool(v.get("scenes")), bool(v.get("voice")),
              any(s.get("image") for s in v.get("scenes", [])),
              bool(v.get("audio_path")), bool(v.get("package"))]
    return sum(checks), len(checks)

def open_creator_studio(parent, theme, image_fn=None, audio_fn=None):
    # `theme` is ignored on purpose: Creator Studio always uses the CS dark-blue palette.
    BG, FG, ACC, PANEL = CS["bg"], CS["fg"], CS["accent"], CS["panel"]
    SIDEBAR, CARD, BORDER, MUTED = CS["sidebar"], CS["card"], CS["border"], CS["muted"]
    OK, WARN, BAD, FONT = CS["ok"], CS["warn"], CS["bad"], CS["font"]
    win = tk.Toplevel(parent)
    win.title("Creator Studio")
    win.geometry("1280x800")
    win.minsize(1100, 680)
    win.configure(bg=BG)
    data = _load()
    ACTIVE_MODEL["name"] = data["settings"].get("model") or CREATOR_MODEL
    win.protocol("WM_DELETE_WINDOW", lambda: on_close())
    cur = {"video": None}
    busy = {"on": False}
    pick = {"kind": None, "items": []}

    bottom_bar = tk.Frame(win, bg=BG)
    bottom_bar.pack(side="bottom", fill="x", padx=10, pady=4)
    progress = ttk.Progressbar(bottom_bar, mode="indeterminate", length=120)
    progress.pack(side="right")
    status = tk.Label(bottom_bar, bg=BG, fg=ACC, anchor="w",
                      text="Ready. Press ＋ New video, or open Script Studio and paste a script.")
    status.pack(side="left", fill="x", expand=True)
    style = ttk.Style(win)
    style.theme_use("clam")
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL, foreground=FG, padding=(14, 6))
    style.map("TNotebook.Tab", background=[("selected", ACC)], foreground=[("selected", "black")])
    style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG, arrowcolor=ACC)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
    win.option_add("*TCombobox*Listbox.background", PANEL)
    win.option_add("*TCombobox*Listbox.foreground", FG)
    style.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=ACC,
                    bordercolor=BG, lightcolor=ACC, darkcolor=ACC)
    style.configure("Vertical.TScrollbar", background=PANEL, troughcolor=BG, arrowcolor=ACC,
                    bordercolor=BG, lightcolor=PANEL, darkcolor=PANEL)
    style.layout("Hidden.TNotebook.Tab", [])          # tabs are hidden; the sidebar navigates
    style.configure("Hidden.TNotebook", background=BG, borderwidth=0, tabmargins=0)
    sidebar = tk.Frame(win, bg=SIDEBAR, width=210)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)
    main = tk.Frame(win, bg=BG)
    main.pack(side="left", fill="both", expand=True)
    topbar = tk.Frame(main, bg=BG, height=60)
    topbar.pack(side="top", fill="x")
    topbar.pack_propagate(False)
    nb = ttk.Notebook(main, style="Hidden.TNotebook")
    nb.pack(fill="both", expand=True)
    PAGES = [("Home", "🏠", "Home"), ("Library", "🎞", "My Videos"), ("Studio", "✍", "Script Studio"),
             ("Scenes", "🎬", "Scenes & Visuals"), ("Analytics", "📊", "Analytics"),
             ("Experiments", "🧪", "Experiments"), ("Settings", "⚙", "Settings")]
    tabs = {}
    for key, _icon, _title in PAGES:
        f = tk.Frame(nb, bg=BG)
        nb.add(f, text=key)
        tabs[key] = f

    all_buttons = []

    def set_busy_ui(on):
        for b in all_buttons:
            try:
                b.config(state="disabled" if on else "normal")
            except tk.TclError:
                pass

    def tip(widget, text):
        tw = {"w": None}
        def enter(e):
            t = tk.Toplevel(widget); tw["w"] = t
            t.wm_overrideredirect(True)
            t.geometry(f"+{e.x_root + 12}+{e.y_root + 16}")
            tk.Label(t, text=text, bg="#2b2b3a", fg="white", padx=8, pady=4,
                     wraplength=280, justify="left").pack()
        def leave(_e):
            if tw["w"]:
                tw["w"].destroy(); tw["w"] = None
        widget.bind("<Enter>", enter, add="+")
        widget.bind("<Leave>", leave, add="+")

    # ---------- reusable components ----------
    def label(p, t): return tk.Label(p, text=t, bg=BG, fg=MUTED, anchor="w", font=(FONT, 9, "bold"))

    def button(p, t, cmd, hint=None, kind="primary", track=True):
        solid = kind == "primary"
        b = tk.Button(p, text=t, command=cmd, bd=0, padx=12, pady=6, cursor="hand2", relief="flat",
                      bg=ACC if solid else PANEL, fg="white" if solid else FG,
                      activebackground=CS["accent_dark"] if solid else BORDER, activeforeground="white",
                      disabledforeground="#51618f", font=(FONT, 9, "bold"))
        if hint:
            tip(b, hint)
        if track:
            all_buttons.append(b)
        return b

    def textbox(p, h): return tk.Text(p, height=h, bg=PANEL, fg=FG, insertbackground=FG, wrap="word",
                                      bd=0, padx=8, pady=6, undo=True, font=(FONT, 10),
                                      highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACC)
    def entry(p, var): return tk.Entry(p, textvariable=var, bg=PANEL, fg=FG, insertbackground=FG, bd=0,
                                       font=(FONT, 10), highlightthickness=1,
                                       highlightbackground=BORDER, highlightcolor=ACC)

    def card(p, **kw):
        return tk.Frame(p, bg=CARD, highlightbackground=BORDER, highlightthickness=1, **kw)

    def badge(p, text, color=None):
        return tk.Label(p, text=text.title(), bg=color or STATUS_COLORS.get(text.upper(), ACC),
                        fg="white", font=(FONT, 8, "bold"), padx=8, pady=1)

    def heading(p, text, sub=None):
        f = tk.Frame(p, bg=BG)
        tk.Label(f, text=text, bg=BG, fg="white", font=(FONT, 13, "bold"), anchor="w").pack(fill="x")
        if sub:
            tk.Label(f, text=sub, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        return f

    def progress_bar(p, frac, color=None):
        trough = tk.Frame(p, bg=PANEL, height=6)
        tk.Frame(trough, bg=color or ACC).place(relx=0, rely=0, relheight=1,
                                                relwidth=max(0.0, min(1.0, frac)))
        return trough

    def empty_state(p, icon, title, text, action_text=None, action=None):
        f = tk.Frame(p, bg=BG)
        tk.Label(f, text=icon, bg=BG, fg=ACC, font=(FONT, 30)).pack(pady=(10, 4))
        tk.Label(f, text=title, bg=BG, fg="white", font=(FONT, 13, "bold")).pack()
        tk.Label(f, text=text, bg=BG, fg=MUTED, wraplength=380, justify="center").pack(pady=(4, 10))
        if action:
            button(f, action_text, action, track=False).pack()
        return f

    def bind_click(widget, cmd, skip=()):
        if widget in skip:
            return
        widget.bind("<Button-1>", lambda e: cmd())
        for ch in widget.winfo_children():
            bind_click(ch, cmd, skip)

    # ---------- sidebar ----------
    titles = {k: t for k, _i, t in PAGES}
    nav_rows = {}
    tk.Label(sidebar, text="✦ Creator Studio", bg=SIDEBAR, fg="white", anchor="w",
             font=(FONT, 14, "bold")).pack(fill="x", padx=18, pady=(22, 2))
    tk.Label(sidebar, text="Faceless short-video workspace", bg=SIDEBAR, fg=MUTED, anchor="w",
             font=(FONT, 8)).pack(fill="x", padx=18, pady=(0, 18))

    def nav_to(key): nb.select(tabs[key])

    for key, icon, title in PAGES:
        row = tk.Frame(sidebar, bg=SIDEBAR, cursor="hand2")
        row.pack(fill="x", padx=10, pady=2)
        accent_bar = tk.Frame(row, bg=SIDEBAR, width=4)
        accent_bar.pack(side="left", fill="y")
        lbl = tk.Label(row, text=f"{icon}  {title}", bg=SIDEBAR, fg=MUTED, anchor="w",
                       font=(FONT, 10, "bold"), padx=10, pady=9)
        lbl.pack(side="left", fill="x", expand=True)
        for w in (row, accent_bar, lbl):
            w.bind("<Button-1>", lambda e, k=key: nav_to(k))
        nav_rows[key] = (row, accent_bar, lbl)
    model_label = tk.Label(sidebar, text="", bg=SIDEBAR, fg=MUTED, anchor="w", font=(FONT, 8))
    model_label.pack(side="bottom", fill="x", padx=18, pady=14)

    # ---------- top bar ----------
    page_title = tk.Label(topbar, text="", bg=BG, fg="white", font=(FONT, 16, "bold"))
    page_title.pack(side="left", padx=24)
    button(topbar, "＋ New video", lambda: new_idea(), "Start a blank video in Script Studio").pack(side="right", padx=24)
    video_chip = tk.Label(topbar, text="No video open", bg=CARD, fg=MUTED, padx=12, pady=5, font=(FONT, 9))
    video_chip.pack(side="right", padx=4)

    def refresh_topbar():
        v = cur["video"]
        video_chip.config(text=f"🎞  {v['title'][:34]}" if v else "No video open", fg=FG if v else MUTED)

    def on_nav_change(_e=None):
        key = next((k for k in tabs if str(tabs[k]) == str(nb.select())), "Home")
        for k, (row, bar_, lbl) in nav_rows.items():
            on = k == key
            row.config(bg=CARD if on else SIDEBAR)
            bar_.config(bg=ACC if on else SIDEBAR)
            lbl.config(bg=CARD if on else SIDEBAR, fg="white" if on else MUTED)
        page_title.config(text=titles[key])
        model_label.config(text=f"🤖 {ACTIVE_MODEL['name']}")
        if key == "Home":
            refresh_home()
    win.bind("<<NotebookTabChanged>>", on_nav_change, add="+")

    def run(text, work, apply):
        """Runs slow AI work in a thread so the window never freezes."""
        t0 = time.time()
        if busy["on"]:
            status.config(text="Still working on the previous step - wait for it to finish.")
            return
        busy["on"] = True
        set_busy_ui(True)
        progress.start(12)
        alive = {"on": True}

        def tick():
            if alive["on"]:
                status.config(text=f"{text}... {time.time() - t0:.0f}s", fg=ACC)
                win.after(500, tick)
        tick()

        def worker():
            try:
                res = work()
            except Exception as e:
                res = e
            win.after(0, lambda: finish(res))

        def finish(res):
            alive["on"] = False
            busy["on"] = False
            progress.stop()
            set_busy_ui(False)
            if isinstance(res, Exception):
                status.config(text=f"⚠ Error: {res}", fg=BAD)
                return
            try:
                apply(res)
                status.config(text=f"Done ✓ ({time.time() - t0:.0f}s)", fg=OK)
            except Exception as e:
                status.config(text=f"Error showing result: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def save_all():
        update_pipeline()
        _save(data)
        refresh_library()
        refresh_combos()
        refresh_home()

    def combo_video(cb):
        if not cb.get():
            return None
        vid = cb.get().rsplit(" | ", 1)[-1]
        return next((v for v in data["videos"] if v["id"] == vid), None)

    # ================= MY VIDEOS (project cards) =================
    L = tabs["Library"]
    bar = tk.Frame(L, bg=BG); bar.pack(fill="x", padx=24, pady=(16, 8))
    filter_var = tk.StringVar(value="ALL")
    filter_combo = ttk.Combobox(bar, textvariable=filter_var, values=["ALL"] + STATUSES,
                                state="readonly", width=14)
    filter_combo.pack(side="left")
    search_var = tk.StringVar()
    tk.Label(bar, text="🔍", bg=BG, fg=MUTED).pack(side="left", padx=(14, 4))
    search_entry = entry(bar, search_var)
    search_entry.pack(side="left", ipady=4)
    search_var.trace_add("write", lambda *_: refresh_library())
    filter_combo.bind("<<ComboboxSelected>>", lambda e: refresh_library())
    count_label = tk.Label(bar, text="", bg=BG, fg=MUTED)
    count_label.pack(side="right")

    lib_state = {"cols": 0}
    lib_scroll = ttk.Scrollbar(L, orient="vertical")
    lib_scroll.pack(side="right", fill="y")
    lib_canvas = tk.Canvas(L, bg=BG, highlightthickness=0, yscrollcommand=lib_scroll.set)
    lib_canvas.pack(side="left", fill="both", expand=True, padx=(24, 0), pady=(0, 10))
    lib_scroll.config(command=lib_canvas.yview)
    lib_grid = tk.Frame(lib_canvas, bg=BG)
    lib_win = lib_canvas.create_window((0, 0), window=lib_grid, anchor="nw")
    lib_grid.bind("<Configure>", lambda e: lib_canvas.configure(scrollregion=lib_canvas.bbox("all")))

    def on_lib_resize(e):
        lib_canvas.itemconfigure(lib_win, width=e.width)
        if max(1, e.width // 300) != lib_state["cols"]:
            refresh_library()
    lib_canvas.bind("<Configure>", on_lib_resize)

    def _wheel(e):
        if str(nb.select()) == str(L):
            lib_canvas.yview_scroll(int(-e.delta / 120), "units")
    win.bind_all("<MouseWheel>", _wheel, add="+")

    def set_video_status(v, s):
        v["status"] = s
        save_all()

    def delete_video(v):
        if messagebox.askyesno("Delete", f'Delete "{v["title"]}"?', parent=win):
            data["videos"].remove(v)
            if cur["video"] is v:
                cur["video"] = None
            save_all()

    def video_card(parent, v):
        done, total = video_progress(v)
        c = card(parent, padx=14, pady=12, cursor="hand2")
        top = tk.Frame(c, bg=CARD); top.pack(fill="x")
        badge(top, v["status"]).pack(side="left")
        more = tk.Label(top, text="⋯", bg=CARD, fg=MUTED, font=(FONT, 14, "bold"), cursor="hand2")
        more.pack(side="right")
        tk.Label(c, text=v["title"], bg=CARD, fg="white", font=(FONT, 11, "bold"), anchor="w",
                 justify="left", wraplength=250).pack(fill="x", pady=(8, 2))
        tk.Label(c, text=f'Hook: {v.get("hook_type") or "not chosen yet"}', bg=CARD, fg=MUTED,
                 anchor="w").pack(fill="x")
        progress_bar(c, done / total, OK if done == total else ACC).pack(fill="x", pady=(10, 2))
        tk.Label(c, text=f"{done}/{total} steps done", bg=CARD, fg=MUTED, anchor="w",
                 font=(FONT, 8)).pack(fill="x")
        menu = tk.Menu(c, tearoff=0, bg=PANEL, fg=FG, activebackground=ACC)
        for s in STATUSES:
            menu.add_command(label=f"Set status: {s.title()}", command=lambda s=s: set_video_status(v, s))
        menu.add_separator()
        menu.add_command(label="Delete video…", command=lambda: delete_video(v))

        def popup(e):
            try:
                menu.tk_popup(e.x_root, e.y_root)
            finally:
                menu.grab_release()
        more.bind("<Button-1>", popup)
        c.bind("<Button-3>", popup)
        bind_click(c, lambda: load_into_studio(v), skip=(more,))
        return c

    def refresh_library(*_):
        for w in lib_grid.winfo_children():
            w.destroy()
        q = search_var.get().strip().lower()
        vids = [v for v in reversed(data["videos"])
                if (not q or q in v["title"].lower() or q in (v.get("topic") or "").lower())
                and filter_var.get() in ("ALL", v["status"])]
        count_label.config(text=f'{len(vids)} video{"" if len(vids) == 1 else "s"}')
        cols = max(1, lib_canvas.winfo_width() // 300)
        lib_state["cols"] = cols
        for c in range(8):
            lib_grid.columnconfigure(c, weight=0, uniform="")
        if not vids:
            empty_state(lib_grid, "🎞", "No videos here yet",
                        "Create your first video, or change the filter / search.",
                        "＋ New video", lambda: new_idea()).grid(row=0, column=0, columnspan=cols, pady=40, sticky="ew")
            return
        for c in range(cols):
            lib_grid.columnconfigure(c, weight=1, uniform="vc")
        for i, v in enumerate(vids):
            video_card(lib_grid, v).grid(row=i // cols, column=i % cols, sticky="nsew", padx=8, pady=8)

    # ================= STUDIO =================
    S = tabs["Studio"]
    idea_var = tk.StringVar()
    pipe_frame = tk.Frame(S, bg=BG)
    pipe_frame.pack(fill="x", padx=10, pady=(10, 0))
    pipe_chips = {}
    for n in ("Research", "Script", "Hook", "Fact-check", "Scenes", "Voice", "Images", "Audio", "Package"):
        chip = tk.Label(pipe_frame, text=n, bg=PANEL, fg=MUTED, padx=10, pady=3, font=(FONT, 8, "bold"))
        chip.pack(side="left", padx=(0, 6))
        pipe_chips[n] = chip
    pipe_label = tk.Label(S, text="", bg=BG, fg=ACC, anchor="w", font=(FONT, 10, "bold"))
    pipe_label.pack(fill="x", padx=10, pady=(6, 0))

    def update_pipeline():
        refresh_topbar()
        v = cur["video"]
        if not v:
            for n, chip in pipe_chips.items():
                chip.config(text=n, bg=PANEL, fg=MUTED)
            pipe_label.config(text="→ Next: type a title and press 💡 Ideas, or paste a rough script")
            return
        steps = [("Research", v.get("research")), ("Script", v.get("script")),
                 ("Hook", v.get("hook_text")), ("Fact-check", v.get("fact_check")),
                 ("Scenes", v.get("scenes")), ("Voice", v.get("voice")),
                 ("Images", any(s.get("image") for s in v.get("scenes", []))),
                 ("Audio", v.get("audio_path")), ("Package", v.get("package"))]
        done = dict(steps)
        order = [("Script", "paste a rough script, then press ✍ Upgrade script"),
                 ("Hook", "press 🪝 Hooks and double-click one"),
                 ("Fact-check", "press 🧪 Fact-check"),
                 ("Scenes", "press 🎬 Scenes"),
                 ("Voice", "press 🎙 Voice prep"),
                 ("Images", "press 🖼 Scene images"),
                 ("Audio", "press 🔊 Narration"),
                 ("Package", "press 📦 Titles & tags")]
        nxt = next((h for n, h in order if not done.get(n)), None)
        for n, ok in steps:
            pipe_chips[n].config(text=("✓ " if ok else "") + n, bg="#12351f" if ok else PANEL,
                                 fg="#86efac" if ok else MUTED)
        pipe_label.config(text=f"→ Next: {nxt}" if nxt else "→ All steps done - press 📤 Export")

    paned = tk.PanedWindow(S, orient="horizontal", bg=BG, sashwidth=6, bd=0)
    paned.pack(fill="both", expand=True, padx=8, pady=6)
    left = tk.Frame(paned, bg=BG)
    right = tk.Frame(paned, bg=BG)
    paned.add(left, minsize=460, width=600)
    paned.add(right, minsize=340)

    label(left, "① Idea / working title").pack(fill="x", pady=(4, 0))
    entry(left, idea_var).pack(fill="x", ipady=4)
    label(left, "② Script - paste a rough one (upgraded versions appear here)").pack(fill="x", pady=(8, 0))
    script_box = textbox(left, 10); script_box.pack(fill="both", expand=True)
    stats_label = tk.Label(left, text="", bg=BG, fg=ACC, anchor="w", font=("Consolas", 9))
    stats_label.pack(fill="x")

    def update_stats(*_):
        st = script_stats(script_box.get("1.0", "end"), data["settings"])
        warn = ""
        if st["words"] and st["est_seconds"] > st["target_seconds"] * 1.15:
            warn = "  ⚠ too long for target"
        elif st["words"] and st["hook_words"] > 18:
            warn = "  ⚠ hook sentence is long"
        stats_label.config(text=f'{st["words"]} words · ~{st["est_seconds"]}s '
                                f'(target {st["target_seconds"]}s) · hook {st["hook_words"]} words · '
                                f'{st["cuts_marked"]} cuts{warn}')

    script_box.bind("<KeyRelease>", update_stats)
    label(left, "③ Sources (optional) - text you trust; only this can be marked VERIFIED").pack(fill="x", pady=(8, 0))
    sources_box = textbox(left, 3); sources_box.pack(fill="x")
    actions = tk.Frame(left, bg=BG); actions.pack(fill="x", pady=(8, 0))
    actions.columnconfigure((0, 1), weight=1)
    groups = {}
    for n, g in enumerate(("1 · Develop", "2 · Check", "3 · Produce", "4 · Publish")):
        gf = tk.LabelFrame(actions, text=g, bg=BG, fg=ACC, bd=1, padx=6, pady=4)
        gf.grid(row=n // 2, column=n % 2, sticky="nsew", padx=3, pady=3)
        groups[g] = gf

    label(right, "Suggestions - double-click one to use it").pack(fill="x", pady=(4, 0))
    pick_list = tk.Listbox(right, height=7, bg=PANEL, fg=FG, selectbackground=ACC, bd=0,
                           font=("Segoe UI", 10), exportselection=False)
    pick_list.pack(fill="x")
    label(right, "Results").pack(fill="x", pady=(8, 0))
    out = textbox(right, 14); out.pack(fill="both", expand=True)

    def show(text):
        out.delete("1.0", "end"); out.insert("1.0", text)

    def set_pick(kind, items, texts):
        pick["kind"], pick["items"] = kind, items
        pick_list.delete(0, "end")
        for t in texts:
            pick_list.insert("end", t)

    def ensure_video():
        if cur["video"] is None:
            v = new_video(idea_var.get().strip() or "Untitled")
            data["videos"].append(v)
            cur["video"] = v
        v = cur["video"]
        v["title"] = idea_var.get().strip() or v["title"]
        v["sources"] = sources_box.get("1.0", "end").strip()
        return v

    def current_script():
        v = ensure_video()
        v["script"] = script_box.get("1.0", "end").strip()
        return v["script"]

    def set_script(text):
        script_box.delete("1.0", "end"); script_box.insert("1.0", text)
        update_stats()

    def load_into_studio(v):
        cur["video"] = v
        idea_var.set(v["title"])
        set_script(v.get("script") or "")
        sources_box.delete("1.0", "end"); sources_box.insert("1.0", v.get("sources") or "")
        shown = {k: v[k] for k in ("research", "fact_check", "score", "scenes", "voice",
                                   "thumbnails", "package") if v.get(k)}
        show(pretty(shown) if shown else "")
        refresh_scene_list()
        update_stats()
        update_pipeline()
        nb.select(S)

    def do_ideas():
        existing = [v["title"] for v in data["videos"]]
        def apply(res):
            ideas = res.get("ideas", [])
            set_pick("ideas", ideas, [f'{i.get("title","")}  -  {i.get("angle","")}' for i in ideas])
            show(pretty(ideas))
        run("Generating ideas", lambda: generate_ideas(data["settings"], existing), apply)

    def do_research():
        v = ensure_video()
        def apply(res):
            v["research"] = res; v["status"] = "RESEARCH"; show(pretty(res)); save_all()
        run("Structuring research", lambda: research_notes(v["title"], v["sources"]), apply)

    def do_upgrade():
        script = current_script(); v = ensure_video()
        if not script:
            status.config(text="Paste a rough script or idea first."); return
        def apply(res):
            v["script"] = res.get("upgraded_script", script); v["status"] = "SCRIPT"
            set_script(v["script"])
            show(pretty({k: res.get(k) for k in ("verdict", "weak_points", "changes_made",
                                                   "pattern_interrupts", "fact_flags")}))
            save_all()
        run("Upgrading script", lambda: upgrade_script(script, data["settings"], v["sources"]), apply)

    def do_hooks():
        script = current_script()
        def apply(res):
            hooks = res.get("hooks", [])
            set_pick("hooks", hooks, [f'[{h.get("type")}] {h.get("text")}' for h in hooks])
            show(pretty(hooks))
        run("Writing hooks", lambda: generate_hooks(script, data["settings"]), apply)

    def use_selected():
        sel = pick_list.curselection()
        if not sel or not pick["kind"]:
            return
        item = pick["items"][sel[0]]
        if pick["kind"] == "ideas":
            cur["video"] = None
            idea_var.set(item.get("title", ""))
            set_script("")
            sources_box.delete("1.0", "end")
            v = ensure_video(); v["idea"] = item.get("angle", "")
            save_all()
        else:
            script = current_script(); v = ensure_video()
            def apply(res):
                v["script"] = res.strip(); v["hook_type"] = item["type"]; v["hook_text"] = item["text"]
                set_script(v["script"]); save_all()
            run("Adapting script to hook", lambda: apply_hook(script, item, data["settings"]), apply)

    def do_scenes():
        script = current_script(); v = ensure_video()
        def apply(res):
            v["scenes"] = res.get("scenes", [])
            v["character_bible"] = res.get("character_bible", "")
            v["style_bible"] = res.get("style_bible", "")
            v["status"] = "PRODUCTION"; show(pretty(res)); save_all(); refresh_scene_list()
        run("Planning scenes", lambda: plan_scenes(script, data["settings"]), apply)

    def do_voice():
        script = current_script(); v = ensure_video()
        def apply(res):
            v["voice"] = res; show(pretty(res)); save_all()
        run("Preparing narration", lambda: prep_voice(script, data["settings"]), apply)

    def do_package():
        script = current_script(); v = ensure_video()
        def apply(res):
            v["package"] = res; v["status"] = "READY"; show(pretty(res)); save_all()
        run("Writing titles/captions/hashtags", lambda: make_package(script, data["settings"]), apply)

    def show_gallery(paths):
        from PIL import Image, ImageTk
        g = tk.Toplevel(win)
        g.title("Scene images")
        g.configure(bg=BG)
        g.photos = []
        for i, p in enumerate(paths):
            img = Image.open(p)
            img.thumbnail((150, 270))
            ph = ImageTk.PhotoImage(img)
            g.photos.append(ph)
            tk.Label(g, image=ph, bg=BG).grid(row=(i // 6) * 2, column=i % 6, padx=4, pady=4)
            tk.Label(g, text=f"Scene {i + 1}", bg=BG, fg=FG).grid(row=(i // 6) * 2 + 1, column=i % 6)

    def do_images():
        v = ensure_video()
        if image_fn is None or not v.get("scenes"):
            status.config(text="Plan scenes first (and wait for the image model to load)."); return
        def work():
            return generate_scene_images(
                v, image_fn,
                lambda i: win.after(0, lambda: status.config(text=f"Rendering image {i}/{len(v['scenes'])}...")))
        def after_render(paths):
            show("Saved:\n" + "\n".join(paths))
            save_all(); refresh_scene_list()
            show_gallery(paths)
        run("Rendering scene images", work, after_render)

    def do_copy():
        win.clipboard_clear()
        win.clipboard_append(out.get("1.0", "end").strip())
        status.config(text="Output copied ✓")

    def do_export():
        v = ensure_video(); current_script()
        save_all()
        show(f"Exported to:\n{export_package(v)}")

    # (action buttons are created below, after every do_* function exists)

    def do_score():
        script = current_script(); v = ensure_video()
        if not script:
            status.config(text="Write or paste a script first."); return
        def apply(res):
            v["score"] = res
            show(pretty(script_stats(script, data["settings"])) + "\n\n" + pretty(res)); save_all()
        run("Scoring script", lambda: score_script(script, data["settings"]), apply)

    def do_factcheck():
        script = current_script(); v = ensure_video()
        if not script:
            status.config(text="Write or paste a script first."); return
        def apply(res):
            v["fact_check"] = res; show(pretty(res)); save_all()
        run("Fact-checking", lambda: fact_check_pass(script, v["sources"]), apply)

    def do_thumbs():
        script = current_script(); v = ensure_video()
        if not script:
            status.config(text="Write or paste a script first."); return
        def apply(res):
            v["thumbnails"] = res; show(pretty(res)); save_all()
        run("Cover ideas", lambda: thumbnail_ideas(script, data["settings"]), apply)

    def do_series():
        topic = idea_var.get().strip()
        if not topic:
            status.config(text="Type a series theme in the idea box first."); return
        existing = [v["title"] for v in data["videos"]]
        def apply(res):
            eps = res.get("episodes", [])
            set_pick("ideas", eps, [f'{e.get("title","")}  -  {e.get("angle","")}' for e in eps])
            show(f'Series: {res.get("series_name","")}\n\n' + pretty(eps))
        run("Planning series", lambda: series_plan(topic, data["settings"], existing), apply)

    def do_srt():
        v = ensure_video()
        if not v.get("scenes"):
            status.config(text="Plan scenes first."); return
        path = os.path.join(EXPORT_DIR, f"{v['id']}.srt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_srt(v["scenes"]))
        show(f"Subtitles saved (timings are estimates from scene durations):\n{path}")

    def do_audio():
        v = ensure_video(); current_script()
        if audio_fn is None:
            status.config(text="Narration audio isn't hooked up (see Part B)."); return
        text = (v.get("voice") or {}).get("voice_script") or v.get("script") or ""
        text = re.sub(r"\[CUT\]", " ", text).strip()
        if not text:
            status.config(text="Nothing to narrate yet."); return
        path = os.path.join(AUDIO_DIR, f"{v['id']}.wav")
        def apply(p):
            v["audio_path"] = p; show(f"Narration saved:\n{p}"); save_all()
        run("Generating narration audio", lambda: audio_fn(text, path), apply)

    def do_save():
        current_script(); save_all(); status.config(text="Saved ✓")

    def do_open_folder():
        try:
            os.startfile(EXPORT_DIR)
        except Exception as e:
            status.config(text=f"Couldn't open folder: {e}")

    GROUPS = {
        "1 · Develop": [
            ("💡 Ideas", do_ideas, "Brainstorm 8 fresh angles for your niche"),
            ("🗓 Series plan", do_series, "Plan 7 connected episodes from the title box"),
            ("🔎 Research", do_research, "Organise what you can and can't claim (uses Sources)"),
            ("✍ Upgrade script", do_upgrade, "Judge and improve your rough script"),
            ("🪝 Hooks", do_hooks, "Write one hook per hook type"),
            ("✔ Use selected", use_selected, "Use the highlighted suggestion (or double-click it)")],
        "2 · Check": [
            ("📊 Score script", do_score, "Hard-editor review with exact lines to fix"),
            ("🧪 Fact-check", do_factcheck, "List every claim and how risky it is")],
        "3 · Produce": [
            ("🎬 Scenes", do_scenes, "Scene-by-scene plan with image prompts"),
            ("🎙 Voice prep", do_voice, "Pauses, stress and pronunciation for AI voice"),
            ("🖼 Scene images", do_images, "Render an image for every scene"),
            ("🔊 Narration", do_audio, "Generate the voice-over audio file"),
            ("🎞 Subtitles", do_srt, "Save a .srt file (timings are estimates)")],
        "4 · Publish": [
            ("📦 Titles & tags", do_package, "Titles, captions and hashtags per platform"),
            ("🖼 Cover ideas", do_thumbs, "4 cover-frame ideas"),
            ("📤 Export .md", do_export, "Save everything as one Markdown file"),
            ("📂 Open folder", do_open_folder, "Open the exports folder"),
            ("📋 Copy output", do_copy, "Copy the Results box to the clipboard"),
            ("💾 Save", do_save, "Save now (Ctrl+S)")],
    }
    for g, items in GROUPS.items():
        gf = groups[g]
        gf.columnconfigure((0, 1), weight=1)
        for i, (text, fn, hint) in enumerate(items):
            button(gf, text, fn, hint).grid(row=i // 2, column=i % 2, sticky="ew", padx=3, pady=2)

    pick_list.bind("<Double-Button-1>", lambda e: use_selected())
    win.bind("<Control-s>", lambda e: do_save())

    def new_idea():
        cur["video"] = None
        idea_var.set(""); set_script("")
        sources_box.delete("1.0", "end")
        show(""); set_pick(None, [], [])
        refresh_scene_list(); update_pipeline()
        nb.select(S)
    # (video cards open in Script Studio when clicked; "⋯" or right-click = status / delete)

    # ================= SCENES (edit, reorder, re-render one scene) =================
    C = tabs["Scenes"]
    sc_left = tk.Frame(C, bg=BG); sc_left.pack(side="left", fill="y", padx=(10, 6), pady=10)
    sc_list = tk.Listbox(sc_left, bg=PANEL, fg=FG, selectbackground=ACC, bd=0, width=46,
                         exportselection=False)
    sc_list.pack(fill="y", expand=True)
    sc_total = tk.Label(sc_left, text="", bg=BG, fg=ACC, anchor="w")
    sc_total.pack(fill="x", pady=(6, 0))
    sc_right = tk.Frame(C, bg=BG); sc_right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
    sc_boxes = {}
    for key, title, h in (("narration", "Narration", 3), ("visual", "Visual", 2),
                          ("image_prompt", "Image prompt", 4),
                          ("on_screen_text", "On-screen text", 1), ("duration_sec", "Duration (sec)", 1)):
        label(sc_right, title).pack(fill="x")
        b = textbox(sc_right, h); b.pack(fill="x", pady=(0, 6))
        sc_boxes[key] = b
    sc_preview = tk.Label(sc_right, bg=BG, fg=FG, text="(no image yet)")
    sc_preview.pack(pady=4)
    sc_state = {"idx": None}

    def show_scene_image(path):
        if not path or not os.path.exists(path):
            sc_preview.config(image="", text="(no image yet)"); sc_preview.image = None
            return
        from PIL import Image, ImageTk
        img = Image.open(path); img.thumbnail((240, 360))
        ph = ImageTk.PhotoImage(img)
        sc_preview.config(image=ph, text=""); sc_preview.image = ph

    def load_scene_fields():
        v, i = cur["video"], sc_state["idx"]
        s = v["scenes"][i] if (v and i is not None and i < len(v.get("scenes", []))) else {}
        for k, b in sc_boxes.items():
            b.delete("1.0", "end"); b.insert("1.0", str(s.get(k, "")))
        show_scene_image(s.get("image"))

    def store_scene_fields():
        v, i = cur["video"], sc_state["idx"]
        if not v or i is None or i >= len(v.get("scenes", [])):
            return
        s = v["scenes"][i]
        for k, b in sc_boxes.items():
            val = b.get("1.0", "end").strip()
            s[k] = _num(val) if k == "duration_sec" else val

    def refresh_scene_list(keep=None):
        sc_list.delete(0, "end")
        v = cur["video"]
        scenes = v.get("scenes", []) if v else []
        total = 0.0
        for i, s in enumerate(scenes):
            if not s.get("image"):   # pick up images made before this version existed
                guess = os.path.join(SCENE_IMG_DIR, v["id"], f"scene_{i + 1:02d}.png")
                if os.path.exists(guess):
                    s["image"] = guess
            total += _num(s.get("duration_sec"))
            mark = "🖼" if s.get("image") else "  "
            sc_list.insert("end", f'{mark} {i + 1:02d}  {(s.get("narration") or "")[:42]}')
        target = _num(data["settings"].get("duration_sec", 45))
        sc_total.config(text=f"{len(scenes)} scenes · {total:.0f}s planned · target {target:.0f}s")
        if keep is not None and 0 <= keep < len(scenes):
            sc_list.selection_set(keep); sc_list.see(keep)
            sc_state["idx"] = keep
        else:
            sc_state["idx"] = None
        load_scene_fields()

    def on_scene_pick(_e=None):
        sel = sc_list.curselection()
        if not sel:
            return
        store_scene_fields()          # keep edits to the scene we're leaving
        sc_state["idx"] = sel[0]
        load_scene_fields()
    sc_list.bind("<<ListboxSelect>>", on_scene_pick)

    def save_scene():
        store_scene_fields(); i = sc_state["idx"]; save_all(); refresh_scene_list(i)

    def move_scene(d):
        v, i = cur["video"], sc_state["idx"]
        if not v or i is None:
            return
        store_scene_fields()
        j = i + d
        if 0 <= j < len(v["scenes"]):
            v["scenes"][i], v["scenes"][j] = v["scenes"][j], v["scenes"][i]
            save_all(); refresh_scene_list(j)

    def delete_scene():
        v, i = cur["video"], sc_state["idx"]
        if not v or i is None:
            return
        if not messagebox.askyesno("Delete scene", f"Delete scene {i + 1}?", parent=win):
            return
        del v["scenes"][i]
        save_all()
        refresh_scene_list(min(i, len(v["scenes"]) - 1) if v["scenes"] else None)

    def rerender_scene():
        v, i = cur["video"], sc_state["idx"]
        if image_fn is None or not v or i is None:
            status.config(text="Pick a scene first (and wait for the image model to load)."); return
        store_scene_fields()
        s = v["scenes"][i]
        prompt = s.get("image_prompt") or s.get("visual", "")
        def work():
            src = image_fn(prompt)
            folder = os.path.join(SCENE_IMG_DIR, v["id"]); os.makedirs(folder, exist_ok=True)
            dest = os.path.join(folder, f"scene_{uuid.uuid4().hex[:6]}.png")
            shutil.copy(src, dest)
            return dest
        def apply(dest):
            s["image"] = dest; save_all(); refresh_scene_list(i)
        run("Rendering scene image", work, apply)

    sc_btns = tk.Frame(sc_right, bg=BG); sc_btns.pack(fill="x", pady=(4, 0))
    for text, fn in (("💾 Save scene", save_scene), ("🔄 Re-render image", rerender_scene),
                     ("⬆", lambda: move_scene(-1)), ("⬇", lambda: move_scene(1)),
                     ("🗑 Delete", delete_scene)):
        button(sc_btns, text, fn).pack(side="left", padx=(0, 6))

    def on_tab_change(_e=None):
        if nb.select() == str(C):
            refresh_scene_list(sc_state["idx"])
        else:
            store_scene_fields()
    nb.bind("<<NotebookTabChanged>>", on_tab_change)

    # ================= SCENES (edit, reorder, re-render one scene) =================
    C = tabs["Scenes"]
    sc_left = tk.Frame(C, bg=BG); sc_left.pack(side="left", fill="y", padx=(10, 6), pady=10)
    sc_list = tk.Listbox(sc_left, bg=PANEL, fg=FG, selectbackground=ACC, bd=0, width=46,
                         exportselection=False)
    sc_list.pack(fill="y", expand=True)
    sc_total = tk.Label(sc_left, text="", bg=BG, fg=ACC, anchor="w")
    sc_total.pack(fill="x", pady=(6, 0))
    sc_right = tk.Frame(C, bg=BG); sc_right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
    sc_boxes = {}
    for key, title, h in (("narration", "Narration", 3), ("visual", "Visual", 2),
                          ("image_prompt", "Image prompt", 4),
                          ("on_screen_text", "On-screen text", 1), ("duration_sec", "Duration (sec)", 1)):
        label(sc_right, title).pack(fill="x")
        b = textbox(sc_right, h); b.pack(fill="x", pady=(0, 6))
        sc_boxes[key] = b
    sc_preview = tk.Label(sc_right, bg=BG, fg=FG, text="(no image yet)")
    sc_preview.pack(pady=4)
    sc_state = {"idx": None}

    def show_scene_image(path):
        if not path or not os.path.exists(path):
            sc_preview.config(image="", text="(no image yet)"); sc_preview.image = None
            return
        from PIL import Image, ImageTk
        img = Image.open(path); img.thumbnail((240, 360))
        ph = ImageTk.PhotoImage(img)
        sc_preview.config(image=ph, text=""); sc_preview.image = ph

    def load_scene_fields():
        v, i = cur["video"], sc_state["idx"]
        s = v["scenes"][i] if (v and i is not None and i < len(v.get("scenes", []))) else {}
        for k, b in sc_boxes.items():
            b.delete("1.0", "end"); b.insert("1.0", str(s.get(k, "")))
        show_scene_image(s.get("image"))

    def store_scene_fields():
        v, i = cur["video"], sc_state["idx"]
        if not v or i is None or i >= len(v.get("scenes", [])):
            return
        s = v["scenes"][i]
        for k, b in sc_boxes.items():
            val = b.get("1.0", "end").strip()
            s[k] = _num(val) if k == "duration_sec" else val

    def refresh_scene_list(keep=None):
        sc_list.delete(0, "end")
        v = cur["video"]
        scenes = v.get("scenes", []) if v else []
        total = 0.0
        for i, s in enumerate(scenes):
            if not s.get("image"):   # pick up images made before this version existed
                guess = os.path.join(SCENE_IMG_DIR, v["id"], f"scene_{i + 1:02d}.png")
                if os.path.exists(guess):
                    s["image"] = guess
            total += _num(s.get("duration_sec"))
            mark = "🖼" if s.get("image") else "  "
            sc_list.insert("end", f'{mark} {i + 1:02d}  {(s.get("narration") or "")[:42]}')
        target = _num(data["settings"].get("duration_sec", 45))
        sc_total.config(text=f"{len(scenes)} scenes · {total:.0f}s planned · target {target:.0f}s")
        if keep is not None and 0 <= keep < len(scenes):
            sc_list.selection_set(keep); sc_list.see(keep)
            sc_state["idx"] = keep
        else:
            sc_state["idx"] = None
        load_scene_fields()

    def on_scene_pick(_e=None):
        sel = sc_list.curselection()
        if not sel:
            return
        store_scene_fields()          # keep edits to the scene we're leaving
        sc_state["idx"] = sel[0]
        load_scene_fields()
    sc_list.bind("<<ListboxSelect>>", on_scene_pick)

    def save_scene():
        store_scene_fields(); i = sc_state["idx"]; save_all(); refresh_scene_list(i)

    def move_scene(d):
        v, i = cur["video"], sc_state["idx"]
        if not v or i is None:
            return
        store_scene_fields()
        j = i + d
        if 0 <= j < len(v["scenes"]):
            v["scenes"][i], v["scenes"][j] = v["scenes"][j], v["scenes"][i]
            save_all(); refresh_scene_list(j)

    def delete_scene():
        v, i = cur["video"], sc_state["idx"]
        if not v or i is None:
            return
        if not messagebox.askyesno("Delete scene", f"Delete scene {i + 1}?", parent=win):
            return
        del v["scenes"][i]
        save_all()
        refresh_scene_list(min(i, len(v["scenes"]) - 1) if v["scenes"] else None)

    def rerender_scene():
        v, i = cur["video"], sc_state["idx"]
        if image_fn is None or not v or i is None:
            status.config(text="Pick a scene first (and wait for the image model to load)."); return
        store_scene_fields()
        s = v["scenes"][i]
        prompt = s.get("image_prompt") or s.get("visual", "")
        def work():
            src = image_fn(prompt)
            folder = os.path.join(SCENE_IMG_DIR, v["id"]); os.makedirs(folder, exist_ok=True)
            dest = os.path.join(folder, f"scene_{uuid.uuid4().hex[:6]}.png")
            shutil.copy(src, dest)
            return dest
        def apply(dest):
            s["image"] = dest; save_all(); refresh_scene_list(i)
        run("Rendering scene image", work, apply)

    sc_btns = tk.Frame(sc_right, bg=BG); sc_btns.pack(fill="x", pady=(4, 0))
    for text, fn in (("💾 Save scene", save_scene), ("🔄 Re-render image", rerender_scene),
                     ("⬆", lambda: move_scene(-1)), ("⬇", lambda: move_scene(1)),
                     ("🗑 Delete", delete_scene)):
        button(sc_btns, text, fn).pack(side="left", padx=(0, 6))

    def on_tab_change(_e=None):
        if nb.select() == str(C):
            refresh_scene_list(sc_state["idx"])
        else:
            store_scene_fields()
    nb.bind("<<NotebookTabChanged>>", on_tab_change)

    # ================= ANALYTICS =================
    A = tabs["Analytics"]
    an_combo = ttk.Combobox(A, state="readonly", width=80)
    an_combo.pack(padx=10, pady=10, anchor="w")
    METRICS = [("views", "Views"), ("likes", "Likes"), ("comments", "Comments"), ("shares", "Shares"),
               ("saves", "Saves"), ("avg_watch_pct", "Avg % watched"),
               ("retention_3s", "% still watching at 3s"), ("ctr", "CTR %"),
               ("followers_gained", "Followers gained")]
    mvars = {}
    grid = tk.Frame(A, bg=BG); grid.pack(anchor="w", padx=10)
    for i, (k, t) in enumerate(METRICS):
        mvars[k] = tk.StringVar()
        label(grid, t).grid(row=i // 3, column=(i % 3) * 2, sticky="w", padx=(0, 6), pady=3)
        entry(grid, mvars[k]).grid(row=i // 3, column=(i % 3) * 2 + 1, padx=(0, 18), ipady=3)
    drop_var = tk.StringVar()
    label(A, "Where did people drop off? (timestamp / what was on screen)").pack(fill="x", padx=10, pady=(10, 0))
    entry(A, drop_var).pack(fill="x", padx=10, ipady=4)
    an_out = textbox(A, 18)

    def save_metrics():
        v = combo_video(an_combo)
        if not v:
            status.config(text="Pick a video first."); return
        m = {k: _num(mvars[k].get()) for k, _ in METRICS}
        m["dropoff_note"] = drop_var.get().strip()
        v["metrics"] = m; v["status"] = "ANALYZING"
        save_all(); status.config(text="Metrics saved ✓")

    def load_metrics(*_):
        v = combo_video(an_combo)
        m = (v or {}).get("metrics") or {}
        for k, _t in METRICS:
            mvars[k].set(str(m.get(k, "")))
        drop_var.set(m.get("dropoff_note", ""))
    an_combo.bind("<<ComboboxSelected>>", load_metrics)

    def do_analyze():
        def apply(text):
            an_out.delete("1.0", "end"); an_out.insert("1.0", text)
        run("Analysing performance", lambda: analyze_performance(data["videos"], data["settings"]), apply)

    arow = tk.Frame(A, bg=BG); arow.pack(fill="x", padx=10, pady=8)
    button(arow, "Save metrics", save_metrics).pack(side="left")
    button(arow, "Analyse all my videos", do_analyze).pack(side="left", padx=8)
    an_out.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    # ================= EXPERIMENTS =================
    E = tabs["Experiments"]
    exp_name = tk.StringVar(); exp_types = tk.StringVar(value="Shock, Mystery, Question")
    label(E, "Experiment name").pack(fill="x", padx=10, pady=(10, 0))
    entry(E, exp_name).pack(fill="x", padx=10, ipady=4)
    label(E, "Hook types to compare (uses the idea + script currently in Studio)").pack(fill="x", padx=10, pady=(8, 0))
    entry(E, exp_types).pack(fill="x", padx=10, ipady=4)
    exp_out = textbox(E, 18)

    def create_experiment():
        script = script_box.get("1.0", "end").strip(); topic = idea_var.get().strip()
        types = [t.strip() for t in exp_types.get().split(",") if t.strip()]
        if not script or not topic or len(types) < 2:
            status.config(text="Need a topic, a script in Studio, and 2+ hook types."); return
        def apply(res):
            exp = {"id": uuid.uuid4().hex[:8], "name": exp_name.get().strip() or topic,
                   "topic": topic, "variants": []}
            for var in res.get("variants", []):
                v = new_video(f'{topic} [{var.get("type")}]')
                v.update(topic=topic, script=var.get("script", ""), hook_type=var.get("type", ""),
                         hook_text=var.get("hook", ""), status="SCRIPT", experiment_id=exp["id"])
                data["videos"].append(v)
                exp["variants"].append({"label": var.get("type", ""), "hook_type": var.get("type", ""),
                                        "video_id": v["id"]})
            data["experiments"].append(exp); save_all()
            exp_out.delete("1.0", "end")
            exp_out.insert("1.0", f"Created {len(exp['variants'])} variants. Publish each one, then enter "
                                  "their metrics in Analytics and press Evaluate.")
        run("Writing hook variants", lambda: script_variants(script, types, data["settings"]), apply)

    exp_combo = ttk.Combobox(E, state="readonly", width=60)
    metric_combo = ttk.Combobox(E, values=["avg_watch_pct", "retention_3s", "views", "shares", "saves"],
                                state="readonly", width=16)
    metric_combo.set("avg_watch_pct")

    def evaluate_experiment():
        if not exp_combo.get():
            return
        exp = next((e for e in data["experiments"] if e["id"] == exp_combo.get().rsplit(" | ", 1)[-1]), None)
        rows = experiment_result(exp, data["videos"], metric_combo.get()) if exp else []
        if len(rows) < 2:
            msg = "Need metrics for at least 2 variants first."
        else:
            msg = "\n".join(f"{i+1}. {lab}: {val}" for i, (lab, val) in enumerate(rows))
            msg += ("\n\nOne run per variant is weak evidence - a small gap can be luck. "
                    "Repeat with a different topic before calling a winner.")
        exp_out.delete("1.0", "end"); exp_out.insert("1.0", msg)

    erow = tk.Frame(E, bg=BG); erow.pack(fill="x", padx=10, pady=8)
    button(erow, "Create experiment", create_experiment).pack(side="left")
    exp_combo.pack(in_=erow, side="left", padx=8)
    metric_combo.pack(in_=erow, side="left")
    button(erow, "Evaluate", evaluate_experiment).pack(side="left", padx=8)
    exp_out.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    # ================= SETTINGS =================
    T = tabs["Settings"]
    CHOICES = {
        "platform": PLATFORMS,
        "duration_sec": ["20", "30", "45", "60", "90"],
        "intensity": ["low", "medium", "high"],
        "language": ["English", "Hindi", "Spanish", "French", "German"],
        "storytelling_style": ["story-driven", "fact-driven", "list-style", "mystery", "documentary"],
        "posting_frequency": ["1 per day", "3 per week", "1 per week"],
    }
    try:
        installed = [m.get("model") or m.get("name") for m in ollama.list().get("models", [])]
    except Exception:
        installed = []
    CHOICES["model"] = installed or [CREATOR_MODEL]

    T.columnconfigure((0, 1), weight=1)
    tk.Label(T, text="Set your niche and audience first - everything the AI writes is tuned to these. "
                     "Dropdowns also accept typed values.", bg=BG, fg=ACC, anchor="w"
             ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))
    svars = {}
    for n, (k, v) in enumerate(data["settings"].items()):
        svars[k] = tk.StringVar(value=v)
        cell = tk.Frame(T, bg=BG)
        cell.grid(row=n // 2 + 1, column=n % 2, sticky="ew", padx=10, pady=4)
        label(cell, k.replace("_", " ").title()).pack(fill="x")
        if k in CHOICES:
            ttk.Combobox(cell, textvariable=svars[k], values=CHOICES[k]).pack(fill="x", ipady=2)
        else:
            entry(cell, svars[k]).pack(fill="x", ipady=4)

    def save_settings():
        for k, var in svars.items():
            data["settings"][k] = var.get().strip()
        ACTIVE_MODEL["name"] = data["settings"].get("model") or CREATOR_MODEL
        _save(data); status.config(text="Settings saved ✓")
    button(T, "💾 Save settings", save_settings).grid(row=20, column=0, sticky="w", padx=10, pady=14)

    # ================= HOME (dashboard) =================
    H = tabs["Home"]
    home = tk.Frame(H, bg=BG); home.pack(fill="both", expand=True, padx=28, pady=20)
    tk.Label(home, text="Welcome back 👋", bg=BG, fg="white", font=(FONT, 20, "bold"), anchor="w").pack(fill="x")
    tk.Label(home, text="What would you like to create today?", bg=BG, fg=MUTED,
             font=(FONT, 11), anchor="w").pack(fill="x", pady=(0, 16))

    stats_row = tk.Frame(home, bg=BG); stats_row.pack(fill="x")
    stat_vals = {}
    for i, (key, title, color) in enumerate((("total", "Total videos", ACC), ("wip", "In progress", WARN),
                                             ("ready", "Ready to publish", OK), ("pub", "Published", "#14b8a6"))):
        stats_row.columnconfigure(i, weight=1, uniform="stat")
        sc = card(stats_row, padx=16, pady=12)
        sc.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
        num = tk.Label(sc, text="0", bg=CARD, fg=color, font=(FONT, 24, "bold"), anchor="w")
        num.pack(fill="x")
        tk.Label(sc, text=title, bg=CARD, fg=MUTED, anchor="w").pack(fill="x")
        stat_vals[key] = num

    heading(home, "Quick start").pack(fill="x", pady=(22, 8))
    qa = tk.Frame(home, bg=BG); qa.pack(fill="x")

    def continue_last():
        if data["videos"]:
            load_into_studio(data["videos"][-1])
        else:
            new_idea()

    def idea_session():
        new_idea(); do_ideas()

    for i, (icon, title, desc, cmd) in enumerate((
            ("✍", "Start a new video", "Type a title or paste a rough script", new_idea),
            ("💡", "Get video ideas", "The AI suggests 8 fresh angles", idea_session),
            ("▶", "Continue last video", "Pick up your most recent video", continue_last),
            ("📊", "Add results", "Log views and see what worked", lambda: nb.select(tabs["Analytics"])))):
        qa.columnconfigure(i, weight=1, uniform="qa")
        ac = card(qa, padx=16, pady=14, cursor="hand2")
        ac.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
        tk.Label(ac, text=icon, bg=CARD, fg=ACC, font=(FONT, 20), anchor="w").pack(fill="x")
        tk.Label(ac, text=title, bg=CARD, fg="white", font=(FONT, 11, "bold"), anchor="w").pack(fill="x", pady=(6, 0))
        tk.Label(ac, text=desc, bg=CARD, fg=MUTED, anchor="w", wraplength=190, justify="left").pack(fill="x")
        bind_click(ac, cmd)

    rec_head = tk.Frame(home, bg=BG); rec_head.pack(fill="x", pady=(22, 8))
    heading(rec_head, "Recent videos").pack(side="left")
    see_all = tk.Label(rec_head, text="See all →", bg=BG, fg=ACC, cursor="hand2", font=(FONT, 9, "bold"))
    see_all.pack(side="right")
    see_all.bind("<Button-1>", lambda e: nav_to("Library"))
    recent_frame = tk.Frame(home, bg=BG); recent_frame.pack(fill="both", expand=True)

    def refresh_home():
        vids = data["videos"]
        stat_vals["total"].config(text=str(len(vids)))
        stat_vals["wip"].config(text=str(sum(v["status"] in ("IDEA", "RESEARCH", "SCRIPT", "PRODUCTION") for v in vids)))
        stat_vals["ready"].config(text=str(sum(v["status"] == "READY" for v in vids)))
        stat_vals["pub"].config(text=str(sum(v["status"] in ("PUBLISHED", "ANALYZING") for v in vids)))
        for w in recent_frame.winfo_children():
            w.destroy()
        if not vids:
            empty_state(recent_frame, "🎬", "Nothing here yet",
                        "Your videos will show up here. Start with an idea - the app will guide you step by step.",
                        "＋ Start your first video", new_idea).pack(pady=16)
            return
        for v in list(reversed(vids))[:4]:
            done, total = video_progress(v)
            row = card(recent_frame, padx=14, pady=10, cursor="hand2")
            row.pack(fill="x", pady=4)
            badge(row, v["status"]).pack(side="right", padx=(10, 0))
            info = tk.Frame(row, bg=CARD); info.pack(side="left", fill="x", expand=True)
            tk.Label(info, text=v["title"], bg=CARD, fg="white", font=(FONT, 11, "bold"), anchor="w").pack(fill="x")
            tk.Label(info, text=f"{done}/{total} steps done", bg=CARD, fg=MUTED, anchor="w",
                     font=(FONT, 8)).pack(fill="x")
            pb = progress_bar(row, done / total, OK if done == total else ACC)
            pb.pack(side="right", fill="x", padx=14, ipadx=70)
            bind_click(row, lambda v=v: load_into_studio(v))

    # ================= final wiring =================
    def on_close():
        try:
            if cur["video"] is not None:
                cur["video"]["script"] = script_box.get("1.0", "end").strip()
            _save(data)
        finally:
            win.destroy()
    def refresh_combos():
        vals = [f'{v["title"]} | {v["id"]}' for v in data["videos"]]
        an_combo["values"] = vals
        exp_combo["values"] = [f'{e["name"]} | {e["id"]}' for e in data["experiments"]]

    def autosave():
        try:
            if not win.winfo_exists():
                return
            if cur["video"] is not None and not busy["on"]:
                cur["video"]["script"] = script_box.get("1.0", "end").strip()
                _save(data)
            win.after(30000, autosave)
        except tk.TclError:
            pass

    refresh_library()
    refresh_combos()
    update_pipeline()
    refresh_home()
    on_nav_change()
    win.after(30000, autosave)