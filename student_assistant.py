import os
import threading
import tkinter as tk
from tkinter import filedialog
import ollama

STUDENT_MODEL = "llama3.2:3b"   # change to a bigger model for better answers
MAX_HISTORY = 12
MAX_FILE_CHARS = 8000          # per file kept in context
MAX_TOTAL_FILE_CHARS = 12000   # all attached files together

# ---------------- palette (matches AP's neon-purple look) ----------------
BG = "#090B19"
SIDEBAR = "#0D1024"
CARD = "#13172D"
SURFACE = "#191D3A"
PURPLE = "#7965FF"
PURPLE_SOFT = "#9B8CFF"
PURPLE_DEEP = "#5B4FD6"
TEXT = "#F3F2FF"
MUTED = "#939BC6"
BORDER = "#252B4B"
GREEN = "#36D982"
RED = "#FF777F"
AMBER = "#C98F2B"
BUBBLE_USER = "#3A2F9E"
BUBBLE_AI = "#14172E"

# ---------------- same prompts / modes as before ----------------
BASE_RULES = (
    "You are AP's Student Assistant, helping a college student. "
    "Use simple English. Keep simple questions simple. Do not invent textbook "
    "content, citations, formulas, or official exam patterns; if you are unsure, say so. "
    "Never claim a question will appear in an exam. No filler intros or repeated conclusions."
)

MODES = {
    "Study & Learn": {
        "label": "Style",
        "options": ["Beginner-friendly", "Step-by-step", "Exam-oriented",
                    "Concept with examples", "Quick revision", "Practical application"],
        "prompt": (
            "Explain the topic at the chosen style. When the topic is big enough, use: "
            "simple definition, easy explanation, important points, one example or analogy, "
            "key terms, a short memory trick, and one quick question to check understanding. "
            "Do not force this structure on small questions. If asked to re-explain, "
            "use a DIFFERENT explanation, not the same one."
        ),
    },
    "Exam Answers": {
        "label": "Marks",
        "options": ["1 mark", "2 marks", "4 marks", "5 marks", "8 marks"],
        "prompt": (
            "Write an exam-ready answer sized to the marks. 1 mark: one direct line "
            "(definition, fact or formula), nothing else. 2 marks: definition plus 2-3 "
            "essential points. 4+ marks: headings or numbered points, a formula, example "
            "or labelled diagram description where relevant. No long textbook explanations."
        ),
    },
    "Programming Tutor": {
        "label": "Task",
        "options": ["Explain line by line", "Find the bug", "Hints only (progressive)",
                    "Review my code", "Generate test cases", "Explain the output",
                    "Practice problem"],
        "prompt": (
            "You are a patient programming tutor. If the student shares code, do NOT replace "
            "it with a different solution. First explain the issue, point to the relevant "
            "lines, then show the corrected version. Explain unfamiliar syntax for beginners. "
            "In 'Hints only' mode give ONE small hint and stop; give the next hint only when "
            "asked, and never the full solution unless the student asks for it. "
            "Put code in plain indented blocks."
        ),
    },
    "Quick Revision": {
        "label": "Time",
        "options": ["15 minutes", "30 minutes", "60 minutes"],
        "prompt": (
            "Create last-minute revision material that fits the time limit: key definitions, "
            "formulas with meanings, key concepts, commonly confused points, memory tricks, "
            "then a 3-question self-test WITHOUT answers (give answers only if asked)."
        ),
    },
}

FOLLOW_UPS = {
    "Simpler": "Explain that more simply, using a different approach.",
    "Example": "Give a practical example.",
    "Step by step": "Explain it step by step.",
    "Exam-ready": "Rewrite it as an exam-ready answer.",
    "Memory trick": "Give me a memory trick for this.",
    "Test me": "Ask me one question to test my understanding. Do not give the answer yet.",
}

_singleton = {"win": None}


# =====================================================================
# Reusable UI pieces
# =====================================================================
def _round_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
           x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
           x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


class RoundButton(tk.Canvas):
    """Flat rounded button drawn on a canvas, with hover + disabled states."""

    def __init__(self, parent, text, command, width=120, height=34, radius=12,
                 fill=PURPLE, hover=PURPLE_SOFT, fg="white", bg=SIDEBAR,
                 font=("Segoe UI", 9, "bold"), anchor="center"):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._btn_text, self._cmd = text, command
        self._btn_w, self._btn_h, self._btn_r = width, height, radius
        self._btn_fill, self._btn_hover, self._btn_fg = fill, hover, fg
        self._btn_font, self._btn_anchor = font, anchor
        self._over, self._enabled = False, True
        self.bind("<Enter>", lambda e: self._set_over(True))
        self.bind("<Leave>", lambda e: self._set_over(False))
        self.bind("<Button-1>", self._click)
        self.bind("<Configure>", self._resize)
        self._draw()

    def _resize(self, event):
        self._btn_w, self._btn_h = event.width, event.height
        self._draw()

    def _set_over(self, value):
        self._over = value
        self._draw()

    def _click(self, _event):
        if self._enabled and self._cmd:
            self._cmd()

    def set_style(self, fill, hover, fg):
        self._btn_fill, self._btn_hover, self._btn_fg = fill, hover, fg
        self._draw()

    def set_text(self, text):
        self._btn_text = text
        self._draw()

    def set_enabled(self, enabled):
        self._enabled = enabled
        self.config(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _draw(self):
        self.delete("all")
        fill = self._btn_hover if (self._over and self._enabled) else self._btn_fill
        fg = self._btn_fg if self._enabled else MUTED
        if not self._enabled:
            fill = SURFACE
        _round_rect(self, 1, 1, self._btn_w - 1, self._btn_h - 1, self._btn_r,
                    fill=fill, outline="")
        if self._btn_anchor == "w":
            self.create_text(16, self._btn_h / 2, text=self._btn_text, fill=fg,
                             font=self._btn_font, anchor="w")
        else:
            self.create_text(self._btn_w / 2, self._btn_h / 2, text=self._btn_text,
                             fill=fg, font=self._btn_font)


def _make_entry(parent):
    """Dark entry with a thin border that turns purple when focused."""
    box = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                   highlightbackground=BORDER, highlightcolor=PURPLE)
    entry = tk.Entry(box, bg=SURFACE, fg=TEXT, insertbackground=PURPLE_SOFT,
                     bd=0, relief=tk.FLAT, font=("Segoe UI", 9))
    entry.pack(fill="x", ipady=5, padx=8)
    return box, entry


# =====================================================================
# Main window
# =====================================================================
def open_student_assistant(parent, theme=None, save_note=None, extract_file=None):
    """save_note(title, content) is supplied by the main app (uses create_note)."""
    w = _singleton["win"]
    if w is not None:
        try:
            if w.winfo_exists():
                w.deiconify(); w.lift(); w.focus_force()
                return w
        except tk.TclError:
            pass

    win = tk.Toplevel(parent)
    win.title("AP — Student Assistant")
    win.geometry("1000x720")
    win.minsize(820, 560)
    win.configure(bg=BG)
    _singleton["win"] = win

    state = {"mode": "Study & Learn", "history": [], "last_answer": "", "busy": False,
             "files": []}
    anim = {"job": None, "n": 0}

    def safe_after(fn):
        try:
            win.after(0, fn)
        except (tk.TclError, RuntimeError):
            pass

    # ------------------------------------------------------------------
    # Left sidebar: logo, modes, context
    # ------------------------------------------------------------------
    side = tk.Frame(win, bg=SIDEBAR, width=250)
    side.pack(side="left", fill="y")
    side.pack_propagate(False)
    tk.Frame(win, bg=BORDER, width=1).pack(side="left", fill="y")

    logo_row = tk.Frame(side, bg=SIDEBAR)
    logo_row.pack(fill="x", padx=16, pady=(18, 10))
    logo = tk.Canvas(logo_row, width=40, height=40, bg=SIDEBAR, highlightthickness=0)
    logo.pack(side="left")
    logo.create_oval(2, 2, 38, 38, outline=PURPLE_DEEP, width=2)
    logo.create_oval(7, 7, 33, 33, fill=CARD, outline=PURPLE)
    logo.create_text(20, 20, text="S", fill=PURPLE_SOFT, font=("Consolas", 14, "bold"))
    name_box = tk.Frame(logo_row, bg=SIDEBAR)
    name_box.pack(side="left", padx=10)
    tk.Label(name_box, text="AP", bg=SIDEBAR, fg=PURPLE_SOFT,
             font=("Consolas", 13, "bold")).pack(anchor="w")
    tk.Label(name_box, text="Student Assistant", bg=SIDEBAR, fg=MUTED,
             font=("Segoe UI", 8)).pack(anchor="w")

    tk.Label(side, text="// STUDY MODES", bg=SIDEBAR, fg=PURPLE_DEEP,
             font=("Consolas", 8, "bold")).pack(anchor="w", padx=18, pady=(10, 6))

    mode_buttons = {}
    for mode_name in MODES:
        btn = RoundButton(side, mode_name, lambda n=mode_name: select_mode(n),
                          height=36, fill=SIDEBAR, hover=SURFACE, fg=MUTED,
                          bg=SIDEBAR, font=("Segoe UI", 10), anchor="w")
        btn.pack(fill="x", padx=12, pady=2)
        mode_buttons[mode_name] = btn

    tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=14)
    tk.Label(side, text="// CONTEXT", bg=SIDEBAR, fg=PURPLE_DEEP,
             font=("Consolas", 8, "bold")).pack(anchor="w", padx=18, pady=(0, 6))

    def field_label(text):
        tk.Label(side, text=text, bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=18, pady=(4, 2))

    field_label("Subject")
    subject_box, subject_entry = _make_entry(side)
    subject_box.pack(fill="x", padx=16)

    field_label("Topic / chapter")
    topic_box, topic_entry = _make_entry(side)
    topic_box.pack(fill="x", padx=16)

    opt_label = tk.Label(side, text="Style", bg=SIDEBAR, fg=MUTED, font=("Segoe UI", 8))
    opt_label.pack(anchor="w", padx=18, pady=(10, 2))
    opt_var = tk.StringVar()
    opt_menu = tk.OptionMenu(side, opt_var, "")
    opt_menu.config(bg=SURFACE, fg=TEXT, activebackground=PURPLE_DEEP,
                    activeforeground="white", highlightthickness=1,
                    highlightbackground=BORDER, bd=0, relief=tk.FLAT,
                    font=("Segoe UI", 9), anchor="w", cursor="hand2")
    opt_menu["menu"].config(bg=SURFACE, fg=TEXT, activebackground=PURPLE,
                            activeforeground="white", bd=0, font=("Segoe UI", 9))
    opt_menu.pack(fill="x", padx=16)

    def set_options(options):
        menu = opt_menu["menu"]
        menu.delete(0, "end")
        for o in options:
            menu.add_command(label=o, command=lambda v=o: opt_var.set(v))
        opt_var.set(options[0])

    # ------------------------------------------------------------------
    # Main column
    # ------------------------------------------------------------------
    main = tk.Frame(win, bg=BG)
    main.pack(side="left", fill="both", expand=True)

    # ----- header -----
    header = tk.Frame(main, bg=SIDEBAR, height=64)
    header.pack(side="top", fill="x")
    header.pack_propagate(False)
    title_box = tk.Frame(header, bg=SIDEBAR)
    title_box.pack(side="left", padx=20, pady=10)
    mode_title = tk.Label(title_box, text="Study & Learn", bg=SIDEBAR, fg=TEXT,
                          font=("Segoe UI", 13, "bold"))
    mode_title.pack(anchor="w")
    tk.Label(title_box, text=". //", bg=SIDEBAR, fg=MUTED,
             font=("Consolas", 7)).pack(anchor="w")

    pill = tk.Canvas(header, width=150, height=26, bg=SIDEBAR, highlightthickness=0)
    pill.pack(side="right", padx=20)

    def set_pill(kind):
        pill.delete("all")
        colors = {
            "checking": (AMBER, "CHECKING"),
            "ready": (GREEN, "MODEL READY"),
            "busy": (PURPLE_SOFT, "THINKING"),
            "offline": (RED, "OLLAMA OFFLINE"),
            "nomodel": (RED, "MODEL MISSING"),
        }
        color, label = colors.get(kind, colors["checking"])
        _round_rect(pill, 0, 1, 149, 25, 12, fill=CARD, outline=color)
        pill.create_oval(11, 9, 19, 17, fill=color, outline="")
        pill.create_text(86, 13, text=label, fill=color, font=("Consolas", 8, "bold"))

    set_pill("checking")
    tk.Frame(main, bg=BORDER, height=1).pack(side="top", fill="x")

    # ----- bottom pieces (packed before the chat view so the view gets the rest) -----
    actions = tk.Frame(main, bg=BG)
    actions.pack(side="bottom", fill="x", padx=20, pady=(2, 12))
    status = tk.Label(actions, text="", bg=BG, fg=MUTED, font=("Segoe UI", 8, "italic"))
    status.pack(side="left")

    input_card = tk.Frame(main, bg=CARD, highlightthickness=1,
                          highlightbackground=BORDER, highlightcolor=PURPLE)
    input_card.pack(side="bottom", fill="x", padx=20, pady=(4, 6))
    attach_btn = RoundButton(input_card, "📎", lambda: attach_file(), width=44, height=38,
                             radius=12, fill=SURFACE, hover=PURPLE_DEEP, fg=PURPLE_SOFT,
                             bg=CARD, font=("Segoe UI", 12))
    attach_btn.pack(side="left", padx=(10, 0), pady=10)
    entry = tk.Text(input_card, height=3, bg=CARD, fg=TEXT, insertbackground=PURPLE_SOFT,
                    bd=0, relief=tk.FLAT, font=("Consolas", 10), wrap="word",
                    padx=12, pady=10)
    entry.pack(side="left", fill="x", expand=True)
    send_btn = RoundButton(input_card, "Send", None, width=76, height=38, bg=CARD)
    send_btn.pack(side="right", padx=10, pady=10)

    chips = tk.Frame(main, bg=BG)
    chips.pack(side="bottom", fill="x", padx=20, pady=(6, 0))
    chip_buttons = []

    # ----- attachment bar (shown only while a file is attached) -----
    pending = {"name": None, "content": None, "reading": False}
    attach_bar = tk.Frame(main, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
    attach_label = tk.Label(attach_bar, text="", bg=SURFACE, fg=PURPLE_SOFT,
                            font=("Segoe UI", 9))
    attach_label.pack(side="left", padx=(10, 6), pady=4)
    attach_remove = tk.Label(attach_bar, text="✕", bg=SURFACE, fg=RED,
                             font=("Segoe UI", 9, "bold"), cursor="hand2")
    attach_remove.pack(side="right", padx=10)
    attach_remove.bind("<Button-1>", lambda e: clear_pending())

    # ----- chat view -----
    view_frame = tk.Frame(main, bg=BG)
    view_frame.pack(side="top", fill="both", expand=True, padx=20, pady=(14, 4))
    view = tk.Text(view_frame, bg=BG, fg=TEXT, wrap="word", bd=0, relief=tk.FLAT,
                   font=("Segoe UI", 10), padx=4, pady=6, state="disabled",
                   cursor="arrow", spacing1=2, spacing3=2)
    scroll = tk.Scrollbar(view_frame, command=view.yview, bd=0, relief=tk.FLAT,
                          troughcolor=BG, bg=SURFACE, activebackground=PURPLE_DEEP)
    view.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    view.pack(side="left", fill="both", expand=True)

    view.tag_config("user_name", justify="right", foreground=MUTED,
                    font=("Segoe UI", 8), rmargin=14, spacing1=10)
    view.tag_config("user", justify="right", background=BUBBLE_USER, foreground=TEXT,
                    lmargin1=140, lmargin2=140, rmargin=14, spacing1=3, spacing3=3)
    view.tag_config("ai_name", justify="left", foreground=PURPLE_SOFT,
                    font=("Segoe UI", 8, "bold"), lmargin1=14, spacing1=10)
    view.tag_config("ai", background=BUBBLE_AI, foreground=TEXT,
                    lmargin1=14, lmargin2=14, rmargin=90, spacing1=3, spacing3=3)
    view.tag_config("sys", foreground=MUTED, font=("Segoe UI", 9, "italic"),
                    lmargin1=14, lmargin2=14, rmargin=40)
    view.tag_config("err", foreground=RED, font=("Segoe UI", 9),
                    lmargin1=14, lmargin2=14, rmargin=40)

    def add_message(role, text):
        view.config(state="normal")
        if role == "user":
            view.insert("end", "You\n", "user_name")
            view.insert("end", " " + text + " ", "user")
            view.insert("end", "\n")
        elif role == "ai":
            view.insert("end", "AP\n", "ai_name")
            view.insert("end", " " + text + " ", "ai")
            view.insert("end", "\n")
        elif role == "err":
            view.insert("end", text + "\n", "err")
        else:
            view.insert("end", text + "\n", "sys")
        view.config(state="disabled")
        view.see("end")

    WELCOME = ("Pick a mode on the left, add a subject or topic if you like, and ask anything.\n"
               "Try: \"Explain the Kronig–Penney model\"  •  \"Define normalisation\"  •  "
               "paste a C program and choose 'Explain line by line'.")

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def select_mode(name):
        state["mode"] = name
        state["history"].clear()          # new mode = fresh context
        cfg = MODES[name]
        opt_label.config(text=cfg["label"])
        set_options(cfg["options"])
        mode_title.config(text=name)
        for n, b in mode_buttons.items():
            if n == name:
                b.set_style(PURPLE, PURPLE_SOFT, "white")
            else:
                b.set_style(SIDEBAR, SURFACE, MUTED)
        status.config(text="")

    def build_system():
        system = BASE_RULES + "\n\n" + MODES[state["mode"]]["prompt"]
        if state["files"]:
            parts = [f"--- File: {f['name']} ---\n{f['content']}" for f in state["files"]]
            system += (
                "\n\nThe student attached the files below. Treat them as the primary source "
                "for questions about them. If the answer is not in the file, say so clearly, "
                "then give any extra general knowledge separately. Do not invent content "
                "that is not in the file.\n\n" + "\n\n".join(parts)
            )
        return system

    def context_tag():
        parts = []
        if subject_entry.get().strip():
            parts.append(f"Subject: {subject_entry.get().strip()}")
        if topic_entry.get().strip():
            parts.append(f"Topic: {topic_entry.get().strip()}")
        parts.append(f"{MODES[state['mode']]['label']}: {opt_var.get()}")
        return "[" + " | ".join(parts) + "]"

    def set_busy(busy):
        state["busy"] = busy
        send_btn.set_enabled(not busy)
        attach_btn.set_enabled(not busy)
        for c in chip_buttons:
            c.set_enabled(not busy)

    def thinking_tick():
        dots = "." * (anim["n"] % 4)
        status.config(text="AP is thinking" + dots)
        anim["n"] += 1
        anim["job"] = win.after(400, thinking_tick)

    def start_thinking():
        anim["n"] = 0
        thinking_tick()
        set_pill("busy")

    def stop_thinking():
        if anim["job"] is not None:
            try:
                win.after_cancel(anim["job"])
            except tk.TclError:
                pass
            anim["job"] = None
        status.config(text="")

    def check_ollama():
        def work():
            try:
                data = ollama.list()
                models = getattr(data, "models", None)
                if models is None:
                    models = data.get("models", [])
                names = []
                for m in models:
                    n = (getattr(m, "model", None)
                         or (m.get("model") if isinstance(m, dict) else None)
                         or getattr(m, "name", None) or "")
                    names.append(n)
                ok = any(n == STUDENT_MODEL or n.startswith(STUDENT_MODEL) for n in names)
                result = "ready" if ok else "nomodel"
            except Exception:
                result = "offline"
            safe_after(lambda: set_pill(result))
        threading.Thread(target=work, daemon=True).start()

    def send(user_text, show_text=None):
        if state["busy"] or not user_text.strip():
            return
        set_busy(True)
        start_thinking()
        add_message("user", show_text or user_text)
        state["history"].append({"role": "user", "content": f"{context_tag()}\n{user_text}"})
        state["history"] = state["history"][-MAX_HISTORY:]
        messages = [{"role": "system", "content": build_system()}] + state["history"]

        def worker():
            try:
                resp = ollama.chat(model=STUDENT_MODEL, messages=messages, keep_alive="30m",
                                   options={"num_predict": 700, "temperature": 0.4,
                                            "num_ctx": 8192})
                reply = (resp["message"]["content"] or "").strip() or "(no response)"
                err = None
            except Exception as e:
                reply, err = "", str(e)

            def done():
                stop_thinking()
                set_busy(False)
                if err:
                    state["history"].pop()   # don't keep a failed turn
                    set_pill("offline")
                    add_message("err", f"Couldn't get an answer: {err}\n"
                                       "Check that Ollama is running and the model is pulled "
                                       f"(ollama pull {STUDENT_MODEL}).")
                    return
                set_pill("ready")
                state["history"].append({"role": "assistant", "content": reply})
                state["last_answer"] = reply
                add_message("ai", reply)
            safe_after(done)

        threading.Thread(target=worker, daemon=True).start()

    IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

    def clear_pending():
        pending.update(name=None, content=None, reading=False)
        attach_bar.pack_forget()

    def show_chip(text):
        attach_label.config(text="📎  " + text)
        attach_bar.pack(side="bottom", fill="x", padx=20, pady=(6, 0), before=chips)

    def attach_file():
        if state["busy"] or pending["reading"]:
            return
        if extract_file is None:
            status.config(text="File reading isn't connected.")
            return
        path = filedialog.askopenfilename(
            parent=win, title="Attach a file",
            filetypes=[("Study files", "*.pdf *.docx *.xlsx *.xlsm *.csv *.tsv *.txt *.md "
                                       "*.py *.c *.cpp *.java *.js *.json"),
                       ("All files", "*.*")])
        if not path:
            return
        name = os.path.basename(path)
        if os.path.splitext(name)[1].lower() in IMAGE_EXT:
            status.config(text="Images aren't supported here yet. Try PDF, DOCX or text.")
            return
        pending.update(name=name, content=None, reading=True)
        show_chip(f"Reading {name}...")

        def work():
            try:
                text, err = extract_file(path, name)
            except Exception as e:
                text, err = None, str(e)

            def done():
                if pending["name"] != name:      # removed while reading
                    return
                pending["reading"] = False
                if err or not text:
                    clear_pending()
                    add_message("err", f"Couldn't read '{name}': {err or 'no text found'}")
                    return
                pending["content"] = text
                show_chip(name)
            safe_after(done)

        threading.Thread(target=work, daemon=True).start()

    def commit_pending():
        name = pending["name"]
        content = pending["content"].strip()
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n[...file truncated...]"
        state["files"] = [f for f in state["files"] if f["name"] != name]
        state["files"].append({"name": name, "content": content})
        while (sum(len(f["content"]) for f in state["files"]) > MAX_TOTAL_FILE_CHARS
               and len(state["files"]) > 1):
            state["files"].pop(0)
        clear_pending()
        return name

    def on_ask(event=None):
        text = entry.get("1.0", "end-1c").strip()
        if state["busy"]:
            return "break"
        if pending["reading"]:
            status.config(text="Still reading the file, one moment...")
            return "break"
        has_file = bool(pending["name"] and pending["content"])
        if not text and not has_file:
            return "break"
        sent, shown = text, text
        if has_file:
            name = commit_pending()
            question = text or "Summarize this file in simple language."
            sent = f"[Attached file: '{name}'] {question}"
            shown = f"📎 {name}\n{question}"
        entry.delete("1.0", "end")
        send(sent, show_text=shown)
        return "break"

    send_btn._cmd = on_ask
    entry.bind("<Return>", on_ask)                  # Enter sends
    entry.bind("<Shift-Return>", lambda e: None)    # Shift+Enter = newline

    for label, instruction in FOLLOW_UPS.items():
        chip = RoundButton(chips, label,
                           lambda i=instruction, l=label: send(i, show_text=l),
                           width=96, height=28, radius=10, fill=SURFACE, hover=PURPLE_DEEP,
                           fg=PURPLE_SOFT, bg=BG, font=("Segoe UI", 8, "bold"))
        chip.pack(side="left", padx=(0, 6))
        chip_buttons.append(chip)

    def do_save():
        if not state["last_answer"]:
            status.config(text="Nothing to save yet.")
            return
        if save_note is None:
            status.config(text="Notes not connected.")
            return
        subj = subject_entry.get().strip()
        topic = topic_entry.get().strip()
        title = " - ".join(p for p in [subj, topic] if p) or "Study note"
        try:
            save_note(title, state["last_answer"])
            status.config(text="Saved to Notes.")
        except Exception as e:
            status.config(text=f"Couldn't save: {e}")

    def do_copy():
        if not state["last_answer"]:
            status.config(text="Nothing to copy yet.")
            return
        win.clipboard_clear()
        win.clipboard_append(state["last_answer"])
        status.config(text="Copied.")

    def do_clear():
        if state["busy"]:
            return
        state["history"].clear()
        state["files"].clear()
        clear_pending()
        state["last_answer"] = ""
        view.config(state="normal"); view.delete("1.0", "end"); view.config(state="disabled")
        add_message("sys", "Context cleared.\n" + WELCOME)

    RoundButton(actions, "Clear", do_clear, width=70, height=28, radius=10,
                fill=SURFACE, hover=PURPLE_DEEP, fg=RED, bg=BG,
                font=("Segoe UI", 8, "bold")).pack(side="right", padx=(6, 0))
    RoundButton(actions, "Copy answer", do_copy, width=100, height=28, radius=10,
                fill=SURFACE, hover=PURPLE_DEEP, fg=PURPLE_SOFT, bg=BG,
                font=("Segoe UI", 8, "bold")).pack(side="right", padx=(6, 0))
    RoundButton(actions, "Save to Notes", do_save, width=110, height=28, radius=10,
                fill=PURPLE, hover=PURPLE_SOFT, fg="white", bg=BG,
                font=("Segoe UI", 8, "bold")).pack(side="right", padx=(6, 0))

    def on_close():
        stop_thinking()
        _singleton["win"] = None
        try:
            win.destroy()
        except tk.TclError:
            pass

    win.protocol("WM_DELETE_WINDOW", on_close)

    select_mode("Study & Learn")
    add_message("sys", WELCOME)
    check_ollama()
    entry.focus_set()
    return win