#!/usr/bin/env python3
"""DocScan – system skanowania dokumentów i rozpoznawania tekstu (OCR).

Wymagania: customtkinter, opencv-python, pytesseract, pillow, numpy oraz zainstalowany program Tesseract OCR.
"""
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
from collections import Counter
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageTk

BG = "#17181c"
ACCENT = "#4f8cff"
IS_WIN = sys.platform.startswith("win")


def find_tesseract():
    base = getattr(sys, "_MEIPASS", None)
    if base:
        exe = os.path.join(base, "tesseract", "tesseract.exe" if IS_WIN else "tesseract")
        if os.path.exists(exe):
            os.environ["TESSDATA_PREFIX"] = os.path.join(base, "tesseract", "tessdata")
            return exe
    p = shutil.which("tesseract")
    if p:
        return p
    if IS_WIN:
        for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")):
            if os.path.exists(c):
                return c
    return None


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_document(img):
    """Zwraca cztery narożniki dokumentu we współrzędnych obrazu źródłowego."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    sh, sw = small.shape[:2]
    area = sh * sw
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [edges, otsu, cv2.bitwise_not(otsu)]

    best, best_area, fallback = None, 0, None
    for m in masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            a = cv2.contourArea(c)
            if a < 0.15 * area:
                break
            if a > 0.995 * area:
                continue
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx) == 4 and a > best_area:
                best, best_area = approx.reshape(4, 2), a
                break
            if fallback is None:
                fallback = cv2.boxPoints(cv2.minAreaRect(hull))
    if best is None:
        best = fallback
    if best is None:
        mx, my = 0.03 * sw, 0.03 * sh
        best = np.array([[mx, my], [sw - mx, my], [sw - mx, sh - my], [mx, sh - my]], np.float32)
    pts = order_points(np.asarray(best, np.float32) / scale)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def warp(img, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    W = max(int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))), 10)
    H = max(int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))), 10)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def apply_mode(img, mode):
    if mode == "Kolor":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Szary":
        return gray
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)


def clean_ocr_data(d, min_conf=55):
    """Odrzuca z wyniku OCR słowa i linie o niskiej wiarygodności rozpoznania."""
    lines, order = {}, []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            continue
        if conf < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append((t, conf))
    out, last_par = [], None
    for key in order:
        words = [(t, c) for t, c in lines[key]
                 if c >= min_conf and not (len(t) == 1 and (not t.isalnum() or c < 80))]
        if not words:
            continue
        line = " ".join(t for t, _ in words)
        compact = line.replace(" ", "")
        alnum = sum(ch.isalnum() for ch in compact)
        avg = sum(c for _, c in words) / len(words)
        if len(compact) < 2 or alnum / len(compact) < 0.6 or avg < min_conf + 5:
            continue
        par = key[:2]
        if last_par is not None and par != last_par:
            out.append("")
        out.append(line)
        last_par = par
    return "\n".join(out)


def ocr_text(img, lang, filt=True):
    if not filt:
        return pytesseract.image_to_string(img, lang=lang)
    d = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    return clean_ocr_data(d)


def to_pil(img):
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class ImageView(tk.Canvas):
    def __init__(self, master, placeholder="", **kw):
        super().__init__(master, highlightthickness=0, bd=0, bg=BG, **kw)
        self.img = None
        self.placeholder = placeholder
        self.scale = 1.0
        self.off = (0, 0)
        self._tk = None
        self.bind("<Configure>", lambda e: self.redraw())

    def set_image(self, img):
        self.img = img
        self.redraw()

    def to_canvas(self, p):
        return p[0] * self.scale + self.off[0], p[1] * self.scale + self.off[1]

    def to_image(self, x, y):
        return (x - self.off[0]) / self.scale, (y - self.off[1]) / self.scale

    def redraw(self):
        self.delete("all")
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 20 or ch < 20:
            return
        if self.img is None:
            self.create_text(cw // 2, ch // 2, text=self.placeholder, fill="#6b7280",
                             font=("Segoe UI", 14), justify="center")
            return
        h, w = self.img.shape[:2]
        s = min((cw - 24) / w, (ch - 24) / h)
        nw, nh = max(int(w * s), 1), max(int(h * s), 1)
        self.scale = nw / w
        self.off = ((cw - nw) // 2, (ch - nh) // 2)
        small = cv2.resize(self.img, (nw, nh), interpolation=cv2.INTER_AREA)
        self._tk = ImageTk.PhotoImage(to_pil(small))
        self.create_image(self.off[0], self.off[1], anchor="nw", image=self._tk)


class CornerView(ImageView):
    """Podgląd obrazu źródłowego z możliwością przesuwania narożników."""

    def __init__(self, master, on_change=None, **kw):
        super().__init__(master, **kw)
        self.corners = None
        self.on_change = on_change
        self.drag = None
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._release)

    def redraw(self):
        super().redraw()
        if self.img is None or self.corners is None:
            return
        pts = [self.to_canvas(p) for p in self.corners]
        order = [0, 1, 2, 3]
        flat = [c for i in order for c in pts[i]]
        self.create_polygon(*flat, outline=ACCENT, fill="", width=2)
        for x, y in pts:
            self.create_oval(x - 9, y - 9, x + 9, y + 9, fill=ACCENT, outline="white", width=2)

    def _press(self, e):
        if self.corners is None:
            return
        best, bd = None, 30
        for i, p in enumerate(self.corners):
            x, y = self.to_canvas(p)
            d = ((x - e.x) ** 2 + (y - e.y) ** 2) ** 0.5
            if d < bd:
                best, bd = i, d
        self.drag = best

    def _move(self, e):
        if self.drag is None or self.img is None:
            return
        h, w = self.img.shape[:2]
        x, y = self.to_image(e.x, e.y)
        self.corners[self.drag] = (min(max(x, 0), w - 1), min(max(y, 0), h - 1))
        self.redraw()

    def _release(self, e):
        if self.drag is not None and self.on_change:
            self.on_change()
        self.drag = None


class CameraWindow(ctk.CTkToplevel):
    def __init__(self, master, on_capture):
        super().__init__(master)
        self.title("Kamera")
        self.geometry("900x640")
        self.on_capture = on_capture
        self.cap = None
        self.frame = None
        self.job = None

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(12, 6))
        ctk.CTkLabel(top, text="Kamera:").pack(side="left")
        self.idx = ctk.CTkOptionMenu(top, values=["0", "1", "2", "3"], width=70, command=self.open_cam)
        self.idx.pack(side="left", padx=8)
        self.info = ctk.CTkLabel(top, text="", text_color="#f59e0b")
        self.info.pack(side="left", padx=8)

        self.view = ImageView(self, placeholder="Trwa uruchamianie kamery…")
        self.view.pack(fill="both", expand=True, padx=14, pady=6)

        bt = ctk.CTkFrame(self, fg_color="transparent")
        bt.pack(fill="x", padx=14, pady=(6, 14))
        ctk.CTkButton(bt, text="Wykonaj zdjęcie", height=40, command=self.capture).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(bt, text="Anuluj", height=40, fg_color="#2b2d34", hover_color="#383b44",
                      command=self.close).pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(150, lambda: self.open_cam("0"))

    def open_cam(self, v):
        self.release()
        i = int(v)
        self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if IS_WIN else cv2.VideoCapture(i)
        if not self.cap.isOpened():
            self.info.configure(text="Nie można uzyskać dostępu do kamery")
            self.view.set_image(None)
            return
        self.info.configure(text="")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.loop()

    def loop(self):
        if self.cap is None:
            return
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            self.view.set_image(f)
        self.job = self.after(30, self.loop)

    def release(self):
        if self.job:
            self.after_cancel(self.job)
            self.job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def capture(self):
        if self.frame is None:
            return
        f = self.frame.copy()
        self.close()
        self.on_capture(f)

    def close(self):
        self.release()
        self.destroy()


STOP = set("""
oraz który która które którzy jest są nie się tego tych przez jako będzie może dla przy czy lub ale gdy była było
zostało zostały także który jego jej ich nad pod przedized tylko jeszcze więc który sobie tego tym tak
that this with from have will your there their been were which about would could should than then them they
what when where into also more most some such only other over after before because while these those
""".split())


def extract_tags(text, n=12):
    words = re.findall(r"[^\W\d_]{4,}", text.lower())
    return [w for w, _ in Counter(w for w in words if w not in STOP).most_common(n)]


def parse_tags(s):
    out = []
    for t in re.split(r"[,;]", s):
        t = t.strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def enc(img, ext, q=92):
    params = [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    return buf.tobytes()


def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


def data_dir():
    if IS_WIN:
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, "DocScan")


def _esc(t):
    return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT, created TEXT, text TEXT, text_lc TEXT,
  original BLOB, scan BLOB, thumb BLOB,
  corners TEXT, mode TEXT, rot INTEGER);
CREATE TABLE IF NOT EXISTS tags(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS scan_tags(
  scan_id INTEGER REFERENCES scans(id) ON DELETE CASCADE,
  tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
  PRIMARY KEY(scan_id, tag_id));
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
"""

TAGS_SQL = ("(SELECT group_concat(g.name, ', ') FROM scan_tags st JOIN tags g ON g.id = st.tag_id "
            "WHERE st.scan_id = s.id)")


class Database:
    def __init__(self, path=None):
        path = path or os.path.join(data_dir(), "docscan.db")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)

    def _set_tags(self, sid, tags):
        c = self.con
        c.execute("DELETE FROM scan_tags WHERE scan_id = ?", (sid,))
        for t in tags:
            c.execute("INSERT OR IGNORE INTO tags(name) VALUES(?)", (t,))
            tid = c.execute("SELECT id FROM tags WHERE name = ?", (t,)).fetchone()[0]
            c.execute("INSERT OR IGNORE INTO scan_tags VALUES(?, ?)", (sid, tid))
        c.execute("DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM scan_tags)")

    def save(self, sid, title, tags, text, original, scan, thumb, corners, mode, rot):
        c = self.con
        if sid:
            cur = c.execute(
                "UPDATE scans SET title=?, text=?, text_lc=?, original=?, scan=?, thumb=?, corners=?, mode=?, rot=? "
                "WHERE id=?", (title, text, text.lower(), original, scan, thumb, corners, mode, rot, sid))
            if cur.rowcount == 0:
                sid = None
        if not sid:
            cur = c.execute(
                "INSERT INTO scans(title, created, text, text_lc, original, scan, thumb, corners, mode, rot) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (title, datetime.now().strftime("%Y-%m-%d %H:%M"), text, text.lower(),
                 original, scan, thumb, corners, mode, rot))
            sid = cur.lastrowid
        self._set_tags(sid, tags)
        c.commit()
        return sid

    def set_tags(self, sid, tags):
        self._set_tags(sid, tags)
        self.con.commit()

    def search(self, terms, in_text=False):
        sql = f"SELECT s.id, s.title, s.created, s.thumb, {TAGS_SQL} FROM scans s"
        conds, args = [], []
        for t in terms:
            c = ("EXISTS(SELECT 1 FROM scan_tags st JOIN tags g ON g.id = st.tag_id "
                 "WHERE st.scan_id = s.id AND g.name LIKE ? ESCAPE '\\')")
            args.append(_esc(t) + "%")
            if in_text:
                c = f"({c} OR s.text_lc LIKE ? ESCAPE '\\')"
                args.append("%" + _esc(t) + "%")
            conds.append(c)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY s.id DESC LIMIT 200"
        return self.con.execute(sql, args).fetchall()

    def get(self, sid):
        r = self.con.execute(
            f"SELECT s.title, s.created, s.text, s.original, s.scan, s.corners, s.mode, s.rot, {TAGS_SQL} "
            "FROM scans s WHERE s.id = ?", (sid,)).fetchone()
        if not r:
            return None
        return dict(title=r[0], created=r[1], text=r[2], original=r[3], scan=r[4], corners=r[5],
                    mode=r[6], rot=r[7] or 0, tags=[t for t in (r[8] or "").split(", ") if t])

    def delete(self, sid):
        self.con.execute("DELETE FROM scans WHERE id = ?", (sid,))
        self.con.execute("DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM scan_tags)")
        self.con.commit()

    def top_tags(self, n=12):
        return self.con.execute(
            "SELECT g.name, COUNT(*) c FROM tags g JOIN scan_tags st ON st.tag_id = g.id "
            "GROUP BY g.id ORDER BY c DESC, g.name LIMIT ?", (n,)).fetchall()


class SaveDialog(ctk.CTkToplevel):
    def __init__(self, master, title, tags, update=False):
        super().__init__(master)
        self.title("Aktualizacja wpisu w bazie danych" if update else "Zapis w bazie danych")
        self.geometry("680x270")
        self.resizable(False, False)
        self.transient(master)
        self.result = None
        ctk.CTkLabel(self, text="Tytuł").pack(anchor="w", padx=20, pady=(18, 4))
        self.t = ctk.CTkEntry(self, height=36)
        self.t.pack(fill="x", padx=20)
        self.t.insert(0, title)
        ctk.CTkLabel(self, text="Tagi (oddzielone przecinkami) – wygenerowano automatycznie na podstawie tekstu; można je edytować",
                     text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(14, 4))
        self.g = ctk.CTkEntry(self, height=36)
        self.g.pack(fill="x", padx=20)
        self.g.insert(0, ", ".join(tags))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=20)
        ctk.CTkButton(row, text="Zapisz", height=38, command=self.ok).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(row, text="Anuluj", height=38, fg_color="#2b2d34", hover_color="#383b44",
                      command=self.destroy).pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.bind("<Return>", lambda e: self.ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(200, lambda: (self.lift(), self.t.focus_force()))

    def ok(self):
        self.result = (self.t.get().strip(), parse_tags(self.g.get()))
        self.destroy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("DocScan")
        self.geometry("1280x800")
        self.minsize(960, 600)

        self.src = None
        self.warped = None
        self.result = None
        self.rot = 0

        self.tess = find_tesseract()
        if self.tess:
            pytesseract.pytesseract.tesseract_cmd = self.tess

        self.db = Database()
        self.current_id = None
        self.sel_id = None
        self.sel_imgs = None
        self._items = {}
        self._thumbs = []
        self._sjob = None
        self._build()
        self.refresh_db()
        if not self.tess:
            self.status("Nie znaleziono programu Tesseract OCR. Należy go zainstalować i dodać do zmiennej PATH.", err=True)

    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#1f2026")
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="DocScan", font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w", padx=20, pady=(24, 0))
        ctk.CTkLabel(side, text="Skanowanie dokumentów i OCR", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(0, 20))

        def btn(text, cmd, primary=False):
            b = ctk.CTkButton(side, text=text, command=cmd, height=40, anchor="w",
                              fg_color=ACCENT if primary else "#2b2d34",
                              hover_color="#3b73d9" if primary else "#383b44")
            b.pack(fill="x", padx=16, pady=4)
            return b

        btn("Wykonaj zdjęcie", self.open_camera, True)
        btn("Wczytaj z pliku", self.open_file, True)

        ctk.CTkLabel(side, text="Tryb obrazu", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(20, 4))
        self.mode = ctk.CTkSegmentedControl = ctk.CTkSegmentedButton(
            side, values=["Kolor", "Szary", "Czarno-biały"], command=lambda v: self.on_region_changed())
        self.mode.set("Szary")
        self.mode.pack(fill="x", padx=16)

        ctk.CTkLabel(side, text="Język rozpoznawania", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(16, 4))
        self.lang = ctk.CTkOptionMenu(side, values=self._languages())
        self.lang.pack(fill="x", padx=16)
        self.filter = ctk.CTkCheckBox(side, text="Filtruj błędnie rozpoznane znaki", command=self.schedule_ocr)
        self.filter.select()
        self.filter.pack(anchor="w", padx=20, pady=(10, 0))

        ctk.CTkLabel(side, text="Skan", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(20, 4))
        btn("Obróć o 90°", self.rotate)
        btn("Wykryj krawędzie ponownie", self.redetect)
        self.ocr_btn = btn("Rozpoznaj tekst", self.run_ocr)

        ctk.CTkLabel(side, text="Eksport", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(20, 4))
        btn("Zapisz skan (PNG/JPG/PDF)", self.save_image)
        btn("Zapisz tekst (.txt)", self.save_text)
        btn("Kopiuj tekst do schowka", self.copy_text)
        btn("Zapisz w bazie danych", self.save_to_db, True)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(main, command=self._on_tab)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        t1 = self.tabs.add("Oryginał")
        t2 = self.tabs.add("Skan")
        t3 = self.tabs.add("Tekst")
        t4 = self.tabs.add("Baza danych")
        self._build_db_tab(t4)
        self.tabs.set("Oryginał")

        self.orig_view = CornerView(t1, on_change=self.on_region_changed,
                                    placeholder="Wykonaj zdjęcie lub wczytaj plik z dysku.\n\nNarożniki dokumentu można korygować przeciągając je myszą.")
        self.orig_view.pack(fill="both", expand=True)
        self.scan_view = ImageView(t2, placeholder="Tutaj zostanie wyświetlony wyprostowany skan dokumentu")
        self.scan_view.pack(fill="both", expand=True)
        self.text = ctk.CTkTextbox(t3, font=ctk.CTkFont(size=15), wrap="word")
        self.text.pack(fill="both", expand=True)

        self.status_lbl = ctk.CTkLabel(main, text="Gotowy do pracy", anchor="w", text_color="#8b8f9a")
        self.status_lbl.grid(row=1, column=0, sticky="ew", padx=6, pady=(6, 0))

    def _languages(self):
        want = ["pol+eng", "pol", "eng"]
        try:
            avail = set(pytesseract.get_languages(config=""))
        except Exception:
            return want
        out = []
        if {"pol", "eng"} <= avail:
            out.append("pol+eng")
        out += [l for l in ("pol", "eng") if l in avail]
        if not out:
            out = [l for l in sorted(avail) if l != "osd"] or want
        return out

    def status(self, msg, err=False):
        self.status_lbl.configure(text=msg, text_color="#f87171" if err else "#8b8f9a")

    def open_camera(self):
        CameraWindow(self, self.load_image)

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("Wszystkie", "*.*")])
        if not path:
            return
        try:
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można otworzyć wskazanego pliku:\n{e}")
            return
        self.load_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    def load_image(self, bgr, corners=None, rot=0, mode=None, text=None, sid=None):
        self.src = bgr
        self.rot = rot
        self.current_id = sid
        if mode:
            self.mode.set(mode)
        if getattr(self, "_ocr_job", None):
            self.after_cancel(self._ocr_job)
            self._ocr_job = None
        self.orig_view.corners = corners or [tuple(p) for p in detect_document(bgr)]
        self.orig_view.set_image(bgr)
        self.process()
        self.tabs.set("Skan")
        if text is None:
            self.run_ocr(show=True)
        else:
            self._ocr_id = getattr(self, "_ocr_id", 0) + 1
            self.text.delete("1.0", "end")
            self.text.insert("1.0", text)

    def redetect(self):
        if self.src is None:
            return
        self.orig_view.corners = [tuple(p) for p in detect_document(self.src)]
        self.orig_view.redraw()
        self.on_region_changed()

    def process(self):
        if self.src is None or self.orig_view.corners is None:
            return
        self.warped = warp(self.src, np.array(self.orig_view.corners, np.float32))
        self.apply()

    def apply(self):
        if self.warped is None:
            return
        img = apply_mode(self.warped, self.mode.get())
        if self.rot:
            img = np.ascontiguousarray(np.rot90(img, -self.rot))
        self.result = img
        self.scan_view.set_image(img)
        self.status(f"Wymiary skanu: {img.shape[1]}×{img.shape[0]} px")

    def rotate(self):
        self.rot = (self.rot + 1) % 4
        self.apply()
        self.schedule_ocr()

    def on_region_changed(self):
        self.process()
        self.schedule_ocr()

    def schedule_ocr(self):
        if getattr(self, "_ocr_job", None):
            self.after_cancel(self._ocr_job)
        self._ocr_job = self.after(400, self.run_ocr)

    def run_ocr(self, show=False):
        self._ocr_job = None
        if self.result is None:
            return
        if not self.tess:
            self.status("Brak programu Tesseract OCR – rozpoznawanie tekstu jest niemożliwe.", err=True)
            return
        img = self.result.copy()
        h, w = img.shape[:2]
        if w < 1600:
            f = 1600 / w
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        lang = self.lang.get()
        filt = self.filter.get() == 1
        self.status("Trwa rozpoznawanie tekstu…")
        self._ocr_id = getattr(self, "_ocr_id", 0) + 1
        oid = self._ocr_id

        def work():
            try:
                txt = ocr_text(img, lang, filt)
                self.after(0, lambda: self.ocr_done(oid, txt, None, show))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self.ocr_done(oid, None, err, show))

        threading.Thread(target=work, daemon=True).start()

    def ocr_done(self, oid, txt, err, show):
        if oid != self._ocr_id:
            return
        if err:
            self.status(f"Błąd rozpoznawania tekstu: {err}", err=True)
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt.strip())
        self.status(f"Rozpoznawanie zakończone. Liczba słów: {len(txt.split())}")
        if show:
            self.tabs.set("Tekst")

    def _on_tab(self):
        if self.tabs.get() == "Baza danych":
            self.refresh_db()

    def _build_db_tab(self, p):
        p.grid_columnconfigure(0, weight=2, uniform="a")
        p.grid_columnconfigure(1, weight=3, uniform="a")
        p.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(p, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        bar.grid_columnconfigure(0, weight=1)
        self.q = ctk.CTkEntry(bar, height=38, placeholder_text="Wyszukiwanie według tagów, np. faktura 2024")
        self.q.grid(row=0, column=0, sticky="ew")
        self.q.bind("<KeyRelease>", lambda e: self._debounce_search())
        self.in_text = ctk.CTkCheckBox(bar, text="uwzględnij treść dokumentu", command=self.refresh_db)
        self.in_text.grid(row=0, column=1, padx=12)

        self.tagbar = ctk.CTkFrame(p, fg_color="transparent")
        self.tagbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self.list = ctk.CTkScrollableFrame(p)
        self.list.grid(row=2, column=0, sticky="nsew", padx=(0, 6))

        det = ctk.CTkFrame(p)
        det.grid(row=2, column=1, sticky="nsew", padx=(6, 0))
        det.grid_rowconfigure(1, weight=1)
        det.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(det, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        head.grid_columnconfigure(0, weight=1)
        self.det_title = ctk.CTkLabel(head, text="", anchor="w", font=ctk.CTkFont(size=16, weight="bold"))
        self.det_title.grid(row=0, column=0, sticky="ew")
        self.det_mode = ctk.CTkSegmentedButton(head, values=["Skan", "Oryginał"],
                                               command=lambda v: self._show_detail_img())
        self.det_mode.set("Skan")
        self.det_mode.grid(row=0, column=1)
        self.det_view = ImageView(det, placeholder="Wybierz dokument z listy")
        self.det_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        trow = ctk.CTkFrame(det, fg_color="transparent")
        trow.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        trow.grid_columnconfigure(0, weight=1)
        self.det_tags = ctk.CTkEntry(trow, placeholder_text="Tagi (oddzielone przecinkami)")
        self.det_tags.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(trow, text="Zapisz tagi", width=100, command=self.save_tags).grid(row=0, column=1, padx=(8, 0))
        brow = ctk.CTkFrame(det, fg_color="transparent")
        brow.grid(row=3, column=0, sticky="ew", padx=10, pady=(4, 10))
        ctk.CTkButton(brow, text="Otwórz w edytorze", command=self.open_selected).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(brow, text="Usuń", fg_color="#7f1d1d", hover_color="#991b1b",
                      command=self.delete_selected).pack(side="left", expand=True, fill="x", padx=(6, 0))

    def _debounce_search(self):
        if self._sjob:
            self.after_cancel(self._sjob)
        self._sjob = self.after(250, self.refresh_db)

    def pick_tag(self, name):
        cur = self.q.get().strip()
        if name not in cur.lower().split():
            self.q.delete(0, "end")
            self.q.insert(0, (cur + " " + name).strip())
        self.refresh_db()

    def refresh_db(self):
        self._sjob = None
        terms = [t for t in re.split(r"[,\s]+", self.q.get().lower()) if t]
        rows = self.db.search(terms, self.in_text.get() == 1)
        for f in self._items.values():
            f.destroy()
        self._items.clear()
        self._thumbs.clear()
        for r in rows:
            self._add_item(r)
        for w in self.tagbar.winfo_children():
            w.destroy()
        top = self.db.top_tags(12)
        if top:
            ctk.CTkLabel(self.tagbar, text="Tagi:", text_color="#8b8f9a").pack(side="left", padx=(0, 6))
        for name, n in top:
            ctk.CTkButton(self.tagbar, text=f"{name} ({n})", width=0, height=26, fg_color="#2b2d34",
                          hover_color="#383b44", command=lambda x=name: self.pick_tag(x)).pack(side="left", padx=2)
        self.status(f"Liczba znalezionych dokumentów: {len(rows)}")

    def _bind_click(self, w, cb):
        w.bind("<Button-1>", cb)
        for c in w.winfo_children():
            self._bind_click(c, cb)

    def _add_item(self, r):
        sid, title, created, thumb, tags = r
        fr = ctk.CTkFrame(self.list, fg_color="#2f3a52" if sid == self.sel_id else "#23252b", corner_radius=8)
        fr.pack(fill="x", pady=3, padx=2)
        try:
            pil = Image.open(io.BytesIO(thumb))
            pil.thumbnail((72, 72))
            ci = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
            self._thumbs.append(ci)
            ctk.CTkLabel(fr, image=ci, text="").pack(side="left", padx=8, pady=8)
        except Exception:
            pass
        box = ctk.CTkFrame(fr, fg_color="transparent")
        box.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=6)
        ctk.CTkLabel(box, text=title, anchor="w", justify="left", wraplength=240,
                     font=ctk.CTkFont(weight="bold")).pack(fill="x")
        ctk.CTkLabel(box, text=created, anchor="w", text_color="#8b8f9a").pack(fill="x")
        ctk.CTkLabel(box, text=tags or "—", anchor="w", justify="left", wraplength=240,
                     text_color=ACCENT).pack(fill="x")
        self._items[sid] = fr
        self._bind_click(fr, lambda e, i=sid: self.select(i))

    def select(self, sid):
        r = self.db.get(sid)
        if not r:
            return
        self.sel_id = sid
        self.sel_imgs = (dec(r["scan"]), dec(r["original"]))
        self.det_title.configure(text=r["title"])
        self.det_tags.delete(0, "end")
        self.det_tags.insert(0, ", ".join(r["tags"]))
        self._show_detail_img()
        for i, f in self._items.items():
            f.configure(fg_color="#2f3a52" if i == sid else "#23252b")

    def _show_detail_img(self):
        if self.sel_imgs:
            self.det_view.set_image(self.sel_imgs[0 if self.det_mode.get() == "Skan" else 1])

    def save_tags(self):
        if not self.sel_id:
            return
        self.db.set_tags(self.sel_id, parse_tags(self.det_tags.get()))
        self.refresh_db()
        self.status("Tagi zostały zapisane")

    def open_selected(self):
        if not self.sel_id:
            return
        r = self.db.get(self.sel_id)
        if not r:
            return
        corners = [tuple(c) for c in json.loads(r["corners"])]
        self.load_image(dec(r["original"]), corners=corners, rot=r["rot"], mode=r["mode"],
                        text=r["text"], sid=self.sel_id)

    def delete_selected(self):
        if not self.sel_id:
            return
        if not messagebox.askyesno("Potwierdzenie usunięcia", "Czy na pewno usunąć wybrany dokument z bazy danych?"):
            return
        self.db.delete(self.sel_id)
        if self.current_id == self.sel_id:
            self.current_id = None
        self.sel_id = None
        self.sel_imgs = None
        self.det_title.configure(text="")
        self.det_tags.delete(0, "end")
        self.det_view.set_image(None)
        self.refresh_db()

    def save_to_db(self):
        if self.src is None or self.result is None:
            self.status("Brak dokumentu do zapisania. Należy najpierw wczytać dokument.", err=True)
            return
        text = self.text.get("1.0", "end").strip()
        title, tags, upd = "", extract_tags(text), False
        if self.current_id:
            r = self.db.get(self.current_id)
            if r:
                title, tags, upd = r["title"], r["tags"], True
            else:
                self.current_id = None
        if not title:
            first = next((l.strip() for l in text.splitlines() if l.strip()), "")
            title = first[:60] or "Dokument " + datetime.now().strftime("%Y-%m-%d %H:%M")
        dlg = SaveDialog(self, title, tags, upd)
        self.wait_window(dlg)
        if not dlg.result:
            return
        title, tags = dlg.result
        h, w = self.result.shape[:2]
        k = min(1.0, 200 / max(h, w))
        thumb = cv2.resize(self.result, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else self.result
        corners = json.dumps([[float(x), float(y)] for x, y in self.orig_view.corners])
        scan_ext = ".png" if self.mode.get() == "Czarno-biały" else ".jpg"
        self.current_id = self.db.save(
            self.current_id, title or "Bez tytułu", tags, text,
            enc(self.src, ".jpg", 95), enc(self.result, scan_ext), enc(thumb, ".jpg", 80),
            corners, self.mode.get(), self.rot)
        self.refresh_db()
        self.status(f"Zapisano w bazie danych (nr {self.current_id})")

    def save_image(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")])
        if not path:
            return
        im = to_pil(self.result)
        if path.lower().endswith((".jpg", ".jpeg", ".pdf")):
            im = im.convert("RGB")
        im.save(path, resolution=200) if path.lower().endswith(".pdf") else im.save(path)
        self.status(f"Zapisano: {path}")

    def save_text(self):
        t = self.text.get("1.0", "end").strip()
        if not t:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Tekst", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(t)
            self.status(f"Zapisano: {path}")

    def copy_text(self):
        t = self.text.get("1.0", "end").strip()
        if t:
            self.clipboard_clear()
            self.clipboard_append(t)
            self.status("Tekst skopiowano do schowka")


if __name__ == "__main__":
    App().mainloop()
