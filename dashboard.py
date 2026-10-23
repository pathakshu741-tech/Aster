import calendar as _cal
import json
import math
import os
import tkinter as tk
import uuid
from datetime import datetime, date, timedelta

P = {
    "bg": "#050923", "rail": "#080D25", "rail_active": "#1f1b55",
    "card": "#101735", "edge": "#1d2757", "row": "#151d42",
    "text": "#F4F5FF", "muted": "#A5ACD6", "accent": "#7957FF",
    "accent2": "#A794FF", "cyan": "#54E5F5", "green": "#28DDA5",
    "green_bg": "#0f2a2e", "red": "#ff8f8f", "past": "#2a2f5c",
}
FONT = "Segoe UI"
TODO_FILE = os.path.join(os.path.expanduser("~"), ".ava_assistant", "todos.json")


# ---------------------------------------------------------- to-do storage
def load_todos():
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("todos", [])
    except Exception:
        return []


def save_todos(todos):
    try:
        os.makedirs(os.path.dirname(TODO_FILE), exist_ok=True)
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            json.dump({"todos": todos}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Save todos error:", e)


# --------------------------------------------------------------- helpers
def lerp(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def rr(c, x1, y1, x2, y2, r=16, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


# ---------------------------------------------------------------- rail
_HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = next((p for p in (
    os.path.join(_HERE, "Aster_logo.jpeg"),
    os.path.join(os.path.expanduser("~"), ".ava_assistant", "Aster_logo.jpeg"),
) if os.path.isfile(p)), os.path.join(_HERE, "Aster_logo.jpeg"))
_logo_cache = {}


def load_logo(size):
    """Returns a Tk image of the Aster logo, or None if the file is missing."""
    if size in _logo_cache:
        return _logo_cache[size]
    try:
        from PIL import Image, ImageTk, ImageChops
        img = Image.open(LOGO_PATH).convert("RGB")
        img = img.crop((337, 245, 917, 825))            # flower + orbit dots only, no wordmark
        bg = img.getpixel((2, 2))                       # navy background colour
        diff = ImageChops.difference(img, Image.new("RGB", img.size, bg)).convert("L")
        alpha = diff.point(lambda v: min(255, v * 6))   # background becomes transparent
        img = img.convert("RGBA")
        img.putalpha(alpha)
        img = img.resize((size, size), Image.LANCZOS)
        _logo_cache[size] = ImageTk.PhotoImage(img)
    except Exception as e:
        print("Aster logo failed to load:", LOGO_PATH, "->", e)
        _logo_cache[size] = None
    return _logo_cache[size]


def _tip(widget, text):
    tip = {"w": None}

    def show(e):
        if tip["w"]:
            return
        w = tk.Toplevel(widget)
        w.wm_overrideredirect(True)
        w.wm_geometry(f"+{widget.winfo_rootx() + widget.winfo_width() + 8}+{widget.winfo_rooty() + 6}")
        tk.Label(w, text=text, bg=P["card"], fg=P["text"], font=(FONT, 8), padx=8, pady=3,
                 highlightthickness=1, highlightbackground=P["edge"]).pack()
        tip["w"] = w

    def hide(e):
        if tip["w"]:
            tip["w"].destroy()
            tip["w"] = None

    widget.bind("<Enter>", show, add="+")
    widget.bind("<Leave>", hide, add="+")


RAIL_TIPS = {
    "home": "Home", "chat": "AI Chat", "notes": "Notes", "reminders": "Reminders & Tasks",
    "studio": "Creator Studio", "student": "Study Assistant", "vibe": "Code Generation",
    "memory": "Memory", "missions": "Missions", "skills": "Skills",
}


def build_rail(parent, items):
    """items: list of (key, icon, command). Returns (rail_frame, set_active)."""
    rail = tk.Frame(parent, bg=P["rail"], width=72)
    rail.pack_propagate(False)
    _logo = load_logo(36)
    if _logo:
        _lbl = tk.Label(rail, image=_logo, bg=P["rail"])
        _lbl.image = _logo
    else:
        _lbl = tk.Label(rail, text="✦", fg=P["accent2"], bg=P["rail"],
                        font=("Segoe UI Symbol", 18))
    _lbl.pack(pady=(18, 22))
    btns, cur = {}, {"k": None}

    def paint(k):
        active = (k == cur["k"])
        btns[k].config(bg=P["accent"] if active else P["rail"],
                       fg="white" if active else P["muted"])

    def set_active(key):
        cur["k"] = key
        for k in btns:
            paint(k)

    for key, icon, cmd in items:
        b = tk.Label(rail, text=icon, font=("Segoe UI Emoji", 15), width=3, pady=8,
                     bg=P["rail"], fg=P["muted"], cursor="hand2")
        b.pack(pady=3)
        btns[key] = b
        b.bind("<Button-1>", lambda e, c=cmd: c())
        b.bind("<Enter>", lambda e, k=key: btns[k].config(bg=P["row"], fg=P["text"]))
        b.bind("<Leave>", lambda e, k=key: paint(k))
        _tip(b, RAIL_TIPS.get(key, key.title()))
    return rail, set_active


# ---------------------------------------------------------------- card
class Card(tk.Canvas):
    def __init__(self, parent, title, icon="", **kw):
        super().__init__(parent, bg=P["bg"], highlightthickness=0, height=200, **kw)
        self.body = tk.Frame(self, bg=P["card"])
        self._win = self.create_window(18, 14, window=self.body, anchor="nw")
        head = tk.Frame(self.body, bg=P["card"])
        head.pack(fill="x", pady=(0, 8))
        tk.Label(head, text=f"{icon}  {title}", bg=P["card"], fg=P["text"],
                 font=(FONT, 11, "bold")).pack(side="left")
        self.badge = tk.Label(head, text="", bg=P["card"], fg=P["accent2"], font=(FONT, 8, "bold"))
        self.badge.pack(side="right")
        self.content = tk.Frame(self.body, bg=P["card"])
        self.content.pack(fill="both", expand=True)
        self.bind("<Configure>", self._draw)

    def _draw(self, e):
        self.delete("bg")
        rr(self, 1, 1, e.width - 1, e.height - 1, 18, fill=P["card"], outline=P["edge"], tags="bg")
        self.tag_lower("bg")
        self.itemconfig(self._win, width=e.width - 36, height=e.height - 28)

    def clear(self):
        for w in self.content.winfo_children():
            w.destroy()


def item_row(parent, title, sub="", right=None, command=None, dot=True):
    r = tk.Frame(parent, bg=P["row"], cursor="hand2" if command else "")
    r.pack(fill="x", pady=2)
    if dot:
        tk.Label(r, text="●", fg=P["accent"], bg=P["row"], font=(FONT, 10)).pack(side="left", padx=(10, 6), pady=4)
    col = tk.Frame(r, bg=P["row"])
    col.pack(side="left", fill="x", expand=True, pady=2, padx=(10 if not dot else 0, 0))
    tk.Label(col, text=title, fg=P["text"], bg=P["row"], font=(FONT, 9, "bold"), anchor="w").pack(fill="x")
    if sub:
        tk.Label(col, text=sub, fg=P["muted"], bg=P["row"], font=(FONT, 8), anchor="w").pack(fill="x")
    if right is not None:
        right(r).pack(side="right", padx=10)
    if command:
        for w in [r, col] + list(col.winfo_children()):
            w.bind("<Button-1>", lambda e: command())
    return r


def status_pill(parent, active, command=None):
    txt, fg, bg = ("●  On", P["green"], P["green_bg"]) if active else ("○  Off", P["muted"], "#1d2040")
    lbl = tk.Label(parent, text=txt, fg=fg, bg=bg, font=(FONT, 8, "bold"), padx=10, pady=3,
                   cursor="hand2" if command else "")
    if command:
        lbl.bind("<Button-1>", lambda e: command())
    return lbl


# ------------------------------------------------------- day popup
def open_day_popup(parent, day, get_reminders, add_reminder, delete_reminder, on_change):
    win = tk.Toplevel(parent)
    win.title(day.strftime("%A, %d %B %Y"))
    win.geometry("380x470")
    win.configure(bg=P["bg"])
    win.transient(parent.winfo_toplevel())

    tk.Label(win, text=day.strftime("%A, %d %B %Y"), bg=P["bg"], fg=P["text"],
             font=(FONT, 13, "bold")).pack(anchor="w", padx=18, pady=(16, 8))

    form = tk.Frame(win, bg=P["bg"])
    form.pack(fill="x", padx=18)
    tk.Label(form, text="What do you want to remember?", bg=P["bg"], fg=P["muted"],
             font=(FONT, 8)).pack(anchor="w")
    task = tk.Entry(form, bg=P["card"], fg=P["text"], insertbackground=P["text"],
                    bd=0, relief="flat", font=(FONT, 10))
    task.pack(fill="x", ipady=6, pady=(2, 8))

    opts = tk.Frame(form, bg=P["bg"])
    opts.pack(fill="x")
    tk.Label(opts, text="Time (HH:MM)", bg=P["bg"], fg=P["muted"], font=(FONT, 8)).pack(side="left")
    tm = tk.Entry(opts, width=7, bg=P["card"], fg=P["text"], insertbackground=P["text"],
                  bd=0, relief="flat", font=(FONT, 10))
    tm.insert(0, "09:00")
    tm.pack(side="left", padx=8, ipady=4)
    repeat = tk.BooleanVar(value=False)
    tk.Checkbutton(opts, text="Repeat every day", variable=repeat, bg=P["bg"], fg=P["text"],
                   selectcolor=P["card"], activebackground=P["bg"], activeforeground=P["text"],
                   font=(FONT, 8), bd=0, highlightthickness=0).pack(side="left", padx=(8, 0))

    msg = tk.Label(form, text="", bg=P["bg"], fg=P["red"], font=(FONT, 8))
    msg.pack(anchor="w", pady=(6, 0))

    lst = tk.Frame(win, bg=P["bg"])
    lst.pack(fill="both", expand=True, padx=18, pady=(4, 12))

    def draw():
        for w in lst.winfo_children():
            w.destroy()
        items = []
        for r in get_reminders():
            try:
                d = datetime.fromisoformat(r["due"])
            except Exception:
                continue
            if d.date() == day:
                items.append((d, r))
        items.sort(key=lambda x: x[0])
        if not items:
            tk.Label(lst, text="Nothing planned for this day.", bg=P["bg"], fg=P["muted"],
                     font=(FONT, 9)).pack(pady=16)
        for d, r in items:
            row = tk.Frame(lst, bg=P["row"])
            row.pack(fill="x", pady=3)
            done = bool(r.get("fired")) and not r.get("repeat")
            tk.Label(row, text=d.strftime("%I:%M %p"), bg=P["row"],
                     fg=P["muted"] if done else P["accent2"], font=(FONT, 8, "bold"),
                     width=9).pack(side="left", padx=(8, 0), pady=8)
            label = r["text"][:30] + ("  ↻" if r.get("repeat") else "")
            tk.Label(row, text=label, bg=P["row"], fg=P["muted"] if done else P["text"],
                     font=(FONT, 9), anchor="w").pack(side="left", fill="x", expand=True)
            x = tk.Label(row, text="✕", bg=P["row"], fg=P["red"], cursor="hand2", padx=10)
            x.pack(side="right")
            x.bind("<Button-1>", lambda e, rid=r["id"]: remove(rid))

    def remove(rid):
        delete_reminder(rid)
        draw()
        on_change()

    def add(event=None):
        text = task.get().strip()
        if not text:
            msg.config(text="Type something first.")
            return
        try:
            t = datetime.strptime(tm.get().strip(), "%H:%M").time()
        except ValueError:
            msg.config(text="Time must look like 18:30.")
            return
        due = datetime.combine(day, t)
        if due <= datetime.now():
            if not repeat.get():
                msg.config(text="That time has already passed.")
                return
            due += timedelta(days=1)
        add_reminder(text, due.isoformat(), "daily" if repeat.get() else None)
        task.delete(0, tk.END)
        msg.config(text="")
        draw()
        on_change()

    tk.Button(form, text="＋  Add reminder", command=add, bg=P["accent"], fg="white", bd=0,
              relief="flat", cursor="hand2", font=(FONT, 9, "bold"),
              activebackground=P["accent2"]).pack(fill="x", pady=(6, 0), ipady=5)
    task.bind("<Return>", add)
    task.focus_set()
    draw()


# ----------------------------------------------------------- dashboard
class HomeDashboard(tk.Frame):
    PLACEHOLDER = "Add a task and press Enter"

    def __init__(self, parent, user_name, get_chats, get_reminders, get_notes,
                 get_automations, on_open_chat, on_open_notes, add_reminder,
                 delete_reminder, on_open_reminders=None):
        super().__init__(parent, bg=P["bg"])
        self.user = user_name
        self.get_chats, self.get_reminders = get_chats, get_reminders
        self.get_notes, self.get_automations = get_notes, get_automations
        self.on_open_chat, self.on_open_notes = on_open_chat, on_open_notes
        self.add_reminder, self.delete_reminder = add_reminder, delete_reminder
        self.phase = 0.0
        self.subtitle = "Your AI assistant is ready to help you focus\non what matters most."
        today = date.today()
        self.cal_year, self.cal_month = today.year, today.month

        for c in range(3):
            self.columnconfigure(c, weight=1, uniform="c", pad=0)
        self.rowconfigure(1, weight=1, uniform="r")
        self.rowconfigure(2, weight=1, uniform="r")

        self.hero = tk.Canvas(self, bg=P["bg"], height=130, highlightthickness=0)
        self.hero.grid(row=0, column=0, columnspan=3, sticky="ew", padx=24, pady=(14, 0))

        pad = dict(padx=10, pady=10, sticky="nsew")
        self.chats = Card(self, "Recent Chats", "💬");   self.chats.grid(row=1, column=0, **pad)
        self.cal = Card(self, "Calendar", "📅");         self.cal.grid(row=1, column=1, **pad)
        self.notes = Card(self, "Notes Brief", "📄");    self.notes.grid(row=1, column=2, **pad)
        self.auto = Card(self, "Automations", "⚡");     self.auto.grid(row=2, column=0, columnspan=2, **pad)
        self.todo = Card(self, "To-Do", "✅");           self.todo.grid(row=2, column=2, **pad)

        self._build_todo_input()
        self.refresh()
        self._animate()
        self.after(30000, self._tick)

    # ----- hero: greeting + animated orb
    def _animate(self):
        if not self.winfo_ismapped():
            self.after(250, self._animate)
            return
        c, w = self.hero, max(self.hero.winfo_width(), 600)
        c.delete("all")
        h = datetime.now().hour
        greet = "Welcome back"
        c.create_text(4, 36, text=f"{greet}, {self.user}", anchor="w", fill=P["text"], font=(FONT, 26))
        c.create_text(4, 82, text=self.subtitle, anchor="w", fill=P["muted"], font=(FONT, 11))
        cx, cy, pulse = w - 110, 62, (math.sin(self.phase) + 1) / 2
        for i in range(10, 0, -1):                       # soft glow
            r = 34 + i * 4 + pulse * 3
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="",
                          fill=lerp(P["bg"], "#4b32c9", 0.05 * (11 - i) + 0.03 * pulse))
        for k in range(14):                              # lit sphere
            t = k / 13
            r, off = 34 * (1 - t * 0.72), -t * 9
            c.create_oval(cx + off - r, cy + off - r, cx + off + r, cy + off + r, outline="",
                          fill=lerp("#5237d6", "#9ff3fb", t))
        self.phase += 0.07
        self.after(50, self._animate)

    # ----- data
    def refresh(self):
        today = date.today()
        by_day = {}
        for r in self.get_reminders():
            try:
                d = datetime.fromisoformat(r["due"]).date()
            except Exception:
                continue
            tot = by_day.setdefault(d, [0, 0])
            tot[0] += 1
            if not r.get("fired"):
                tot[1] += 1

        n_today = by_day.get(today, [0, 0])[1]
        open_tasks = sum(1 for t in load_todos() if not t.get("done"))
        self.subtitle = (f"{n_today} reminder{'s' if n_today != 1 else ''} today  ·  "
                         f"{open_tasks} task{'s' if open_tasks != 1 else ''} open\n"
                         "What would you like to get done?")
        self._fill_chats()
        self._fill_calendar(by_day)
        self._fill_notes()
        self._fill_auto()
        self._fill_todo()

    def _tick(self):
        self.refresh()
        self.after(30000, self._tick)

    # ----- chats
    def _fill_chats(self):
        self.chats.clear()
        chats = self.get_chats()
        self.chats.badge.config(text=f"{len(chats)} chats")
        if not chats:
            tk.Label(self.chats.content, text="No chats yet.", fg=P["muted"], bg=P["card"]).pack(pady=20)
        for c in chats[:3]:
            msgs = c.get("messages", [])
            last = (msgs[-1]["content"] if msgs else "").replace("\n", " ")[:42]
            item_row(self.chats.content, (c.get("title") or "New chat")[:30], last,
                     command=lambda cid=c["id"]: self.on_open_chat(cid))

    # ----- calendar
    def _shift_month(self, delta):
        m, y = self.cal_month + delta, self.cal_year
        if m < 1:
            m, y = 12, y - 1
        elif m > 12:
            m, y = 1, y + 1
        self.cal_month, self.cal_year = m, y
        self.refresh()

    def _go_today(self):
        t = date.today()
        self.cal_year, self.cal_month = t.year, t.month
        self.refresh()

    def _open_day(self, day):
        open_day_popup(self, day, self.get_reminders, self.add_reminder,
                       self.delete_reminder, self.refresh)

    def _fill_calendar(self, by_day):
        self.cal.clear()
        today = date.today()
        n_today = by_day.get(today, [0, 0])[1]
        self.cal.badge.config(text=f"{n_today} today" if n_today else "")
        c = self.cal.content

        nav = tk.Frame(c, bg=P["card"])
        nav.pack(fill="x")
        prev = tk.Label(nav, text="‹", fg=P["muted"], bg=P["card"], font=(FONT, 12, "bold"),
                        cursor="hand2", padx=8)
        prev.pack(side="left")
        prev.bind("<Button-1>", lambda e: self._shift_month(-1))
        nxt = tk.Label(nav, text="›", fg=P["muted"], bg=P["card"], font=(FONT, 12, "bold"),
                       cursor="hand2", padx=8)
        nxt.pack(side="right")
        nxt.bind("<Button-1>", lambda e: self._shift_month(1))
        title = tk.Label(nav, text=f"{_cal.month_name[self.cal_month]} {self.cal_year}",
                         fg=P["text"], bg=P["card"], font=(FONT, 10, "bold"), cursor="hand2")
        title.pack(side="left", expand=True)
        title.bind("<Button-1>", lambda e: self._go_today())

        grid = tk.Frame(c, bg=P["card"])
        grid.pack(fill="both", expand=True, pady=(4, 0))
        for i in range(7):
            grid.columnconfigure(i, weight=1, uniform="d")
        for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            tk.Label(grid, text=name, fg=P["muted"], bg=P["card"], font=(FONT, 7)).grid(row=0, column=i)

        weeks = _cal.Calendar(firstweekday=0).monthdayscalendar(self.cal_year, self.cal_month)
        for r_i, week in enumerate(weeks, start=1):
            for c_i, dnum in enumerate(week):
                if dnum == 0:
                    continue
                d = date(self.cal_year, self.cal_month, dnum)
                info = by_day.get(d)
                fg, bg = P["text"], P["card"]
                if info and info[1] > 0:
                    fg, bg = "white", P["accent"]            # has upcoming reminders
                elif d == today:
                    fg, bg = P["accent2"], P["rail_active"]
                weight = "bold" if d == today else "normal"
                cell = tk.Label(grid, text=str(dnum), fg=fg, bg=bg, font=(FONT, 8, weight),
                                width=3, pady=1, cursor="hand2")
                if d == today and info and info[1] > 0:
                    cell.config(highlightthickness=1, highlightbackground=P["accent2"])
                cell.grid(row=r_i, column=c_i, padx=1, pady=1)
                cell.bind("<Button-1>", lambda e, dd=d: self._open_day(dd))

    # ----- notes
    def _fill_notes(self):
        self.notes.clear()
        notes = sorted(self.get_notes(), key=lambda n: n.get("updated_at", ""), reverse=True)
        self.notes.badge.config(text=f"{len(notes)} notes")
        banner = tk.Frame(self.notes.content, bg="#2a2370")
        banner.pack(fill="x")
        tk.Label(banner, text="📄", bg="#2a2370", font=(FONT, 16), pady=4).pack()
        for n in notes[:2]:
            title = (n.get("title") or n.get("content", "")[:30] or "Untitled")[:32]
            item_row(self.notes.content, title, command=self.on_open_notes)
        tk.Button(self.notes.content, text="View all notes  ›", command=self.on_open_notes,
                  bg=P["row"], fg=P["text"], bd=0, relief="flat", cursor="hand2",
                  font=(FONT, 9), activebackground=P["edge"]).pack(fill="x", pady=(6, 0), ipady=5)

    # ----- automations (clickable switches, two per row)
    def _run_toggle(self, toggle):
        try:
            toggle()
        except Exception as e:
            print("Automation toggle error:", e)
        self.after(300, self._fill_auto)

    def _fill_auto(self):
        self.auto.clear()
        items = self.get_automations()
        on = sum(1 for it in items if it[3])
        self.auto.badge.config(text=f"{on}/{len(items)} on")
        grid = tk.Frame(self.auto.content, bg=P["card"])
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1, uniform="a")
        grid.columnconfigure(1, weight=1, uniform="a")
        for i, (icon, title, sub, active, toggle) in enumerate(items):
            cell = tk.Frame(grid, bg=P["card"])
            cell.grid(row=i // 2, column=i % 2, sticky="ew", padx=4)
            item_row(cell, f"{icon}  {title}", sub,
                     right=lambda p, a=active, t=toggle: status_pill(p, a, lambda: self._run_toggle(t)),
                     dot=False)

    # ----- to-do list
    def _build_todo_input(self):
        box = tk.Frame(self.todo.content, bg=P["row"])
        box.pack(fill="x")
        e = tk.Entry(box, bg=P["row"], fg=P["muted"], insertbackground=P["text"], bd=0,
                     relief="flat", font=(FONT, 9))
        e.pack(side="left", fill="x", expand=True, ipady=5, padx=(8, 0))
        e.insert(0, self.PLACEHOLDER)
        self.todo_entry = e

        def focus_in(ev):
            if e.get() == self.PLACEHOLDER:
                e.delete(0, tk.END)
                e.config(fg=P["text"])

        def focus_out(ev):
            if not e.get().strip():
                e.delete(0, tk.END)
                e.insert(0, self.PLACEHOLDER)
                e.config(fg=P["muted"])

        e.bind("<FocusIn>", focus_in)
        e.bind("<FocusOut>", focus_out)
        e.bind("<Return>", self._add_todo)
        plus = tk.Label(box, text="＋", fg=P["accent2"], bg=P["row"], font=(FONT, 11, "bold"),
                        cursor="hand2", padx=8)
        plus.pack(side="right")
        plus.bind("<Button-1>", self._add_todo)
        wrap = tk.Frame(self.todo.content, bg=P["card"])
        wrap.pack(fill="both", expand=True, pady=(6, 0))
        cv = tk.Canvas(wrap, bg=P["card"], highlightthickness=0)
        cv.pack(side="left", fill="both", expand=True)
        self.todo_list = tk.Frame(cv, bg=P["card"])
        win = cv.create_window(0, 0, window=self.todo_list, anchor="nw")
        self.todo_list.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))

        def wheel(e):
            rx, ry = wrap.winfo_rootx(), wrap.winfo_rooty()
            px, py = wrap.winfo_pointerxy()
            inside = rx <= px <= rx + wrap.winfo_width() and ry <= py <= ry + wrap.winfo_height()
            if inside and self.todo_list.winfo_height() > cv.winfo_height():
                cv.yview_scroll(int(-e.delta / 120), "units")
        self.bind_all("<MouseWheel>", wheel, add="+")

    def _add_todo(self, event=None):
        text = self.todo_entry.get().strip()
        if not text or text == self.PLACEHOLDER:
            return
        todos = load_todos()
        todos.append({"id": uuid.uuid4().hex[:8], "text": text, "done": False,
                      "created_at": datetime.now().isoformat()})
        save_todos(todos)
        self.todo_entry.delete(0, tk.END)
        self._fill_todo()

    def _toggle_todo(self, tid):
        todos = load_todos()
        for t in todos:
            if t["id"] == tid:
                t["done"] = not t.get("done", False)
        save_todos(todos)
        self._fill_todo()

    def _delete_todo(self, tid):
        save_todos([t for t in load_todos() if t["id"] != tid])
        self._fill_todo()

    def _fill_todo(self):
        for w in self.todo_list.winfo_children():
            w.destroy()
        todos = load_todos()
        done_count = sum(1 for t in todos if t.get("done"))
        self.todo.badge.config(text=f"{done_count}/{len(todos)} done" if todos else "")
        if not todos:
            tk.Label(self.todo_list, text="No tasks yet. Add your first one above.",
                     fg=P["muted"], bg=P["card"], font=(FONT, 8)).pack(anchor="w", pady=6)
        ordered = sorted(todos, key=lambda t: bool(t.get("done")))
        for t in ordered:
            is_done = bool(t.get("done"))
            row = tk.Frame(self.todo_list, bg=P["card"])
            row.pack(fill="x", pady=1)
            box = tk.Label(row, text="☑" if is_done else "☐",
                           fg=P["green"] if is_done else P["accent2"], bg=P["card"],
                           font=("Segoe UI Symbol", 12), cursor="hand2")
            box.pack(side="left")
            txt = tk.Label(row, text=t["text"][:30],
                           fg=P["muted"] if is_done else P["text"], bg=P["card"], anchor="w",
                           font=(FONT, 9, "overstrike" if is_done else "normal"), cursor="hand2")
            txt.pack(side="left", fill="x", expand=True, padx=(4, 0))
            x = tk.Label(row, text="✕", fg=P["red"], bg=P["card"], cursor="hand2", padx=4)
            x.pack(side="right")
            for w in (box, txt):
                w.bind("<Button-1>", lambda e, tid=t["id"]: self._toggle_todo(tid))
            x.bind("<Button-1>", lambda e, tid=t["id"]: self._delete_todo(tid))
