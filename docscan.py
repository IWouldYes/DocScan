#!/usr/bin/env python3
"""DocScan – skaner dokumentów: zdjęcie/plik -> wykrycie krawędzi -> prostowanie -> OCR (Tesseract).

Działa na Linux/macOS/Windows.
Wymagania: pip install customtkinter opencv-python pytesseract pillow numpy
oraz zainstalowany Tesseract (z językiem polskim).
"""
import os
import shutil
import sys
import threading
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


# ───────────────────────── Tesseract ─────────────────────────
def find_tesseract():
    base = getattr(sys, "_MEIPASS", None)  # wersja spakowana (PyInstaller)
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


# ───────────────────────── Przetwarzanie obrazu ─────────────────────────
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl,tr,br,bl


def detect_document(img):
    """Zwraca 4 rogi kartki (tl,tr,br,bl) w współrzędnych oryginału."""
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


def to_pil(img):
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ───────────────────────── Widgety podglądu ─────────────────────────
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
    """Podgląd oryginału z przeciąganymi rogami."""

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


# ───────────────────────── Okno kamery ─────────────────────────
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

        self.view = ImageView(self, placeholder="Uruchamianie kamery…")
        self.view.pack(fill="both", expand=True, padx=14, pady=6)

        bt = ctk.CTkFrame(self, fg_color="transparent")
        bt.pack(fill="x", padx=14, pady=(6, 14))
        ctk.CTkButton(bt, text="📸  Zrób zdjęcie", height=40, command=self.capture).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(bt, text="Anuluj", height=40, fg_color="#2b2d34", hover_color="#383b44",
                      command=self.close).pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(150, lambda: self.open_cam("0"))

    def open_cam(self, v):
        self.release()
        i = int(v)
        self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if IS_WIN else cv2.VideoCapture(i)
        if not self.cap.isOpened():
            self.info.configure(text="Nie można otworzyć kamery")
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


# ───────────────────────── Aplikacja ─────────────────────────
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

        self._build()
        if not self.tess:
            self.status("⚠ Nie znaleziono Tesseract – zainstaluj go i dodaj do PATH.", err=True)

    # ── UI ──
    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#1f2026")
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="DocScan", font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w", padx=20, pady=(24, 0))
        ctk.CTkLabel(side, text="skaner dokumentów + OCR", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(0, 20))

        def btn(text, cmd, primary=False):
            b = ctk.CTkButton(side, text=text, command=cmd, height=40, anchor="w",
                              fg_color=ACCENT if primary else "#2b2d34",
                              hover_color="#3b73d9" if primary else "#383b44")
            b.pack(fill="x", padx=16, pady=4)
            return b

        btn("📷  Zrób zdjęcie", self.open_camera, True)
        btn("📁  Wczytaj plik", self.open_file, True)

        ctk.CTkLabel(side, text="Tryb obrazu", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(20, 4))
        self.mode = ctk.CTkSegmentedControl = ctk.CTkSegmentedButton(
            side, values=["Kolor", "Szary", "Czarno-biały"], command=lambda v: self.process())
        self.mode.set("Szary")
        self.mode.pack(fill="x", padx=16)

        ctk.CTkLabel(side, text="Język OCR", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(16, 4))
        self.lang = ctk.CTkOptionMenu(side, values=self._languages())
        self.lang.pack(fill="x", padx=16)

        ctk.CTkLabel(side, text="Skan", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(20, 4))
        btn("🔄  Obróć o 90°", self.rotate)
        btn("🔍  Wykryj ponownie", self.redetect)
        self.ocr_btn = btn("📝  Odczytaj tekst", self.run_ocr)

        ctk.CTkLabel(side, text="Eksport", text_color="#8b8f9a").pack(anchor="w", padx=20, pady=(20, 4))
        btn("💾  Zapisz skan (PNG/JPG/PDF)", self.save_image)
        btn("📄  Zapisz tekst (.txt)", self.save_text)
        btn("📋  Kopiuj tekst", self.copy_text)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(main)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        t1 = self.tabs.add("Oryginał")
        t2 = self.tabs.add("Skan")
        t3 = self.tabs.add("Tekst")
        self.tabs.set("Oryginał")

        self.orig_view = CornerView(t1, on_change=self.process,
                                    placeholder="Zrób zdjęcie lub wczytaj plik\n\nRogi możesz przeciągać myszą")
        self.orig_view.pack(fill="both", expand=True)
        self.scan_view = ImageView(t2, placeholder="Tu pojawi się wyprostowany skan")
        self.scan_view.pack(fill="both", expand=True)
        self.text = ctk.CTkTextbox(t3, font=ctk.CTkFont(size=15), wrap="word")
        self.text.pack(fill="both", expand=True)

        self.status_lbl = ctk.CTkLabel(main, text="Gotowe", anchor="w", text_color="#8b8f9a")
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

    # ── Wejście ──
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
            messagebox.showerror("Błąd", f"Nie można otworzyć pliku:\n{e}")
            return
        self.load_image(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

    def load_image(self, bgr):
        self.src = bgr
        self.rot = 0
        self.orig_view.corners = [tuple(p) for p in detect_document(bgr)]
        self.orig_view.set_image(bgr)
        self.process()
        self.tabs.set("Skan")
        self.run_ocr()

    def redetect(self):
        if self.src is None:
            return
        self.orig_view.corners = [tuple(p) for p in detect_document(self.src)]
        self.orig_view.redraw()
        self.process()

    # ── Przetwarzanie ──
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
        self.status(f"Skan: {img.shape[1]}×{img.shape[0]} px")

    def rotate(self):
        self.rot = (self.rot + 1) % 4
        self.apply()

    # ── OCR ──
    def run_ocr(self):
        if self.result is None:
            return
        if not self.tess:
            self.status("Brak Tesseract – nie można odczytać tekstu.", err=True)
            return
        img = self.result.copy()
        h, w = img.shape[:2]
        if w < 1600:
            f = 1600 / w
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        lang = self.lang.get()
        self.status("Rozpoznawanie tekstu…")
        self.ocr_btn.configure(state="disabled")

        def work():
            try:
                txt = pytesseract.image_to_string(img, lang=lang)
                self.after(0, lambda: self.ocr_done(txt, None))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self.ocr_done(None, err))

        threading.Thread(target=work, daemon=True).start()

    def ocr_done(self, txt, err):
        self.ocr_btn.configure(state="normal")
        if err:
            self.status(f"Błąd OCR: {err}", err=True)
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt.strip())
        self.status(f"Gotowe – rozpoznano {len(txt.split())} słów")
        self.tabs.set("Tekst")

    # ── Eksport ──
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
            self.status("Skopiowano do schowka")


if __name__ == "__main__":
    App().mainloop()
