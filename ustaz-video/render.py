#!/usr/bin/env python3
"""
«Ұстазым» — әннің 1-куплеті мен қайырмасына кинетикалық видео (лирик-видео).

Стиль: кремдік фон + мектеп тақырыбындағы дудл-өрнек,
  * куплет  — чаттағыдай сөздер жолаққа "теріледі", біткен жол хабарлама болып жоғары ұшады;
  * қайырма — үлкен қызыл кинетикалық типография, әр сөзге стикер ырғақпен "секіреді";
  * мектеп ғимараты — полароид-стикер ретінде (интро, куплет, қайырма, соңы).

Барлық уақыт пен мәтін lyrics.json ішінде. Іске қосу:
    python3 render.py --audio path/to/song.mp3 --out ustazym.mp4
    python3 render.py --audio song.mp3 --preview 9.5 32 43   # тек кадр-суреттер
"""
import argparse, json, math, os, random, subprocess, sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
W, H, FPS = 1920, 1080, 30

FONT_DIR = "/usr/share/fonts/opentype/inter"
F_BLACK = os.path.join(FONT_DIR, "Inter-Black.otf")
F_XBOLD = os.path.join(FONT_DIR, "Inter-ExtraBold.otf")
F_SEMI = os.path.join(FONT_DIR, "Inter-SemiBold.otf")
F_MED = os.path.join(FONT_DIR, "Inter-Medium.otf")

CREAM = (240, 233, 208)
DOODLE = (225, 215, 184)
RED = (228, 36, 40)
DARKRED = (160, 30, 36)
PINK = (242, 178, 168)
BLUE = (92, 111, 146)
BUBBLE_RED = (138, 38, 40)
INK = (70, 58, 45)
WHITE = (255, 255, 255)

# ----------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x

def ease_out_back(x, s=1.9):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2

def ease_out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3

def ease_in_cubic(x):
    x = clamp(x)
    return x ** 3

def ease_in_out(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x

# ----------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def font(path, size):
    return ImageFont.truetype(path, size)

@lru_cache(maxsize=512)
def text_img(text, path, size, fill, spacing=0):
    """Tight RGBA image of text (optional letter spacing)."""
    f = font(path, size)
    if spacing == 0:
        l, t, r, b = f.getbbox(text)
        asc, desc = f.getmetrics()
        im = Image.new("RGBA", (r - l + 4, asc + desc + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - l, 2), text, font=f, fill=fill)
        return im
    widths = [f.getlength(c) for c in text]
    total = int(sum(widths) + spacing * (len(text) - 1)) + 6
    asc, desc = f.getmetrics()
    im = Image.new("RGBA", (total, asc + desc + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 2
    for c, w in zip(text, widths):
        d.text((x, 2), c, font=f, fill=fill)
        x += w + spacing
    return im

def paste(canvas, img, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    """Alpha-composite img centred at (cx, cy) with scale/rotation/opacity."""
    if img is None or scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        img = img.resize((w, h), Image.BICUBIC)
    if abs(rot) > 0.05:
        img = img.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 0.999:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    x, y = int(cx - img.width / 2), int(cy - img.height / 2)
    sx0, sy0 = max(-x, 0), max(-y, 0)
    sx1 = min(img.width, canvas.width - x)
    sy1 = min(img.height, canvas.height - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    canvas.alpha_composite(img.crop((sx0, sy0, sx1, sy1)), (x + sx0, y + sy0))

def add_outline_shadow(img, border=12, shadow=(10, 14), blur=10):
    """Sticker look: white die-cut border + soft drop shadow."""
    pad = border + blur * 2 + max(shadow)
    base = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    base.paste(img, (pad, pad))
    a = base.getchannel("A").point(lambda v: 255 if v > 40 else 0)
    # grow alpha by `border` px (repeated small max filters are much faster)
    grown = a
    step = 5
    for _ in range(max(1, border // 2)):
        grown = grown.filter(ImageFilter.MaxFilter(step))
    grown = grown.filter(ImageFilter.GaussianBlur(1.2))
    white = Image.new("RGBA", base.size, WHITE + (0,))
    white.putalpha(grown)
    sh = Image.new("RGBA", base.size, (60, 40, 20, 0))
    sh.putalpha(grown.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * 0.45)))
    out = Image.new("RGBA", base.size, (0, 0, 0, 0))
    out.alpha_composite(sh, shadow)
    out.alpha_composite(white)
    out.alpha_composite(base)
    bb = out.getbbox()
    return out.crop(bb)

@lru_cache(maxsize=None)
def sticker(name):
    if name == "mektep":
        return polaroid()
    im = Image.open(os.path.join(ASSETS, "stickers", name + ".png")).convert("RGBA")
    im = im.resize((300, 300), Image.LANCZOS)
    return add_outline_shadow(im, border=14)

@lru_cache(maxsize=None)
def polaroid(width=760, caption="Біздің мектеп"):
    photo = Image.open(os.path.join(ASSETS, "mektep.jpg")).convert("RGB")
    # building + welcome sign; drop most of the sky and paving
    crop = photo.crop((0, 420, photo.width, 1215))
    h = int(crop.height * width / crop.width)
    crop = crop.resize((width, h), Image.LANCZOS)
    crop = ImageOps.autocontrast(crop, cutoff=0.5)
    m, mb = 26, 96
    card = Image.new("RGBA", (width + 2 * m, h + m + mb), (252, 250, 244, 255))
    card.paste(crop, (m, m))
    d = ImageDraw.Draw(card)
    f = font(F_XBOLD, 46)
    tw = f.getlength(caption)
    d.text(((card.width - tw) / 2, h + m + 20), caption, font=f, fill=DARKRED)
    # soft shadow
    pad = 40
    out = Image.new("RGBA", (card.width + pad * 2, card.height + pad * 2), (0, 0, 0, 0))
    sh = Image.new("L", out.size, 0)
    ImageDraw.Draw(sh).rectangle((pad + 8, pad + 14, pad + card.width + 8, pad + card.height + 14), fill=120)
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    shadow = Image.new("RGBA", out.size, (60, 40, 20, 0))
    shadow.putalpha(sh)
    out.alpha_composite(shadow)
    out.alpha_composite(card, (pad, pad))
    return out

def make_background(seed=7, strength=1.0):
    """Cream paper with faint school doodles (pencils, books, letters, stars...)."""
    rnd = random.Random(seed)
    bg = Image.new("RGBA", (W, H), CREAM + (255,))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = DOODLE + (int(255 * strength),)
    lw = 4

    def pencil(x, y, s, a):
        pts = [(-1, -0.18), (0.6, -0.18), (1, 0), (0.6, 0.18), (-1, 0.18)]
        ca, sa = math.cos(a), math.sin(a)
        P = [(x + s * (px * ca - py * sa), y + s * (px * sa + py * ca)) for px, py in pts]
        d.line(P + [P[0]], fill=col, width=lw, joint="curve")
        d.line([P[1], P[3]], fill=col, width=lw)

    def book(x, y, s, a):
        d.rounded_rectangle((x - s, y - s * 0.7, x + s, y + s * 0.7), radius=8, outline=col, width=lw)
        d.line((x, y - s * 0.7, x, y + s * 0.7), fill=col, width=lw)
        for k in (-0.3, 0, 0.3):
            d.line((x - s * 0.8, y + k * s, x - s * 0.2, y + k * s), fill=col, width=3)
            d.line((x + s * 0.2, y + k * s, x + s * 0.8, y + k * s), fill=col, width=3)

    def star(x, y, s, a):
        P = []
        for k in range(10):
            r = s if k % 2 == 0 else s * 0.45
            ang = a + k * math.pi / 5
            P.append((x + r * math.sin(ang), y - r * math.cos(ang)))
        d.line(P + [P[0]], fill=col, width=lw, joint="curve")

    def apple(x, y, s, a):
        d.ellipse((x - s, y - s * 0.85, x + s, y + s * 0.95), outline=col, width=lw)
        d.line((x, y - s * 0.85, x + s * 0.15, y - s * 1.35), fill=col, width=lw)
        d.ellipse((x + s * 0.15, y - s * 1.3, x + s * 0.65, y - s * 1.0), outline=col, width=3)

    def ruler(x, y, s, a):
        d.rectangle((x - s * 1.4, y - s * 0.3, x + s * 1.4, y + s * 0.3), outline=col, width=lw)
        for k in range(9):
            xx = x - s * 1.2 + k * s * 0.3
            d.line((xx, y - s * 0.3, xx, y - s * (0.05 if k % 2 else -0.05)), fill=col, width=3)

    def bell(x, y, s, a):
        d.arc((x - s * 0.7, y - s, x + s * 0.7, y + s * 0.6), 180, 360, fill=col, width=lw)
        d.line((x - s * 0.7, y - s * 0.2, x - s * 0.9, y + s * 0.6), fill=col, width=lw)
        d.line((x + s * 0.7, y - s * 0.2, x + s * 0.9, y + s * 0.6), fill=col, width=lw)
        d.line((x - s * 0.9, y + s * 0.6, x + s * 0.9, y + s * 0.6), fill=col, width=lw)
        d.ellipse((x - s * 0.15, y + s * 0.6, x + s * 0.15, y + s * 0.9), outline=col, width=3)

    def heart(x, y, s, a):
        P = [(x + s * 16 * math.sin(t) ** 3 / 16,
              y - s * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) / 16)
             for t in np.linspace(0, 2 * math.pi, 40)]
        d.line(P, fill=col, width=lw, joint="curve")

    def glyph(x, y, s, a):
        t = rnd.choice(["Әә", "Аа", "Ұұ", "2+2=4", "Бб", "5!", "Ққ", "abc", "Өө"])
        f = font(F_XBOLD, int(s * 1.2))
        d.text((x - f.getlength(t) / 2, y - s * 0.7), t, font=f, fill=col)

    shapes = [pencil, book, star, apple, ruler, bell, heart, glyph, glyph]
    cell = 150
    for gy in range(-1, H // cell + 2):
        for gx in range(-1, W // cell + 2):
            x = gx * cell + (gy % 2) * cell / 2 + rnd.uniform(-25, 25)
            y = gy * cell + rnd.uniform(-25, 25)
            rnd.choice(shapes)(x, y, rnd.uniform(28, 40), rnd.uniform(-0.6, 0.6))
    # faint watermark words, like the reference's channel stamp
    wf = font(F_BLACK, 70)
    for i, (x, y) in enumerate([(160, 120), (1180, 260), (520, 610), (1420, 820), (260, 900)]):
        word = ["МЕКТЕБІМ", "ҰСТАЗЫМ"][i % 2]
        d.text((x, y), word, font=wf, fill=DOODLE + (int(110 * strength),))
    layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    bg.alpha_composite(layer)
    return bg

# ----------------------------------------------------------------- timing
class Clock:
    def __init__(self, cfg):
        self.bp = 60.0 / cfg["bpm"]
        self.phase = cfg["beat_phase"]

    def since_beat(self, t):
        k = math.floor((t - self.phase) / self.bp)
        return t - (self.phase + k * self.bp), k

    def pulse(self, t, decay=7.0):
        dt, k = self.since_beat(t)
        strong = 1.0 if k % 4 == 0 else 0.6
        return strong * math.exp(-dt * decay)

    def snap(self, t, div=2):
        q = self.bp / div
        return self.phase + round((t - self.phase) / q) * q

# ----------------------------------------------------------------- sticker actor
def draw_sticker(canvas, name, x, y, size, t, t0, t_end, clock, rot0=0.0, label=None, seed=0):
    if t < t0:
        return
    p = (t - t0) / 0.38
    s = ease_out_back(p)
    out = clamp((t - (t_end - 0.18)) / 0.18)
    s *= 1 - ease_in_cubic(out)
    if s <= 0.01:
        return
    wob = 5 * math.sin(2 * math.pi * (t - t0) / (clock.bp * 4) + seed)
    bump = 1 + 0.06 * clock.pulse(t)
    img = sticker(name)
    base = size / max(img.width, img.height)
    rot = rot0 + wob + (1 - clamp(p)) * 25
    paste(canvas, img, x, y - 8 * clock.pulse(t), base * s * bump, rot)
    if label:
        tag = text_img(label, F_BLACK, int(size * 0.42), RED)
        paste(canvas, tag, x + size * 0.38, y + size * 0.32, s * bump, rot - 8)

# ----------------------------------------------------------------- scenes
def scene_intro(canvas, t, cfg, clock, bgs):
    sc = cfg["intro"]
    end = sc["end"]
    out = clamp((t - (end - 0.35)) / 0.35)
    zoom = 1 - 0.15 * ease_in_cubic(out)
    # building polaroid
    tp = (t - 0.43) / 0.6
    if tp > 0:
        s = ease_out_back(tp, 1.4) * zoom * (1 + 0.025 * clock.pulse(t))
        paste(canvas, polaroid(), 560, 560 + 30 * math.sin(t * 1.3), 0.82 * s,
              -5 + 3 * math.sin(t * 0.9), 1 - out)
    # title
    title = text_img(sc["title"], F_BLACK, 165, RED)
    tt = (t - 2.41) / 0.4
    if tt > 0:
        paste(canvas, title, 1430, 480, ease_out_back(tt) * zoom * (1 + 0.04 * clock.pulse(t)), 0, 1 - out)
    sub = text_img(sc["subtitle"], F_XBOLD, 54, DARKRED, 2)
    ts = (t - 3.40) / 0.35
    if ts > 0:
        paste(canvas, sub, 1430 + 80 * (1 - ease_out_cubic(ts)), 610, zoom, 0, clamp(ts) * (1 - out))
    for name, t0, x, y, sz, r in [("1f4da", 1.42, 1180, 200, 230, -10),
                                  ("1f34e", 2.90, 1760, 250, 190, 12),
                                  ("1f514", 3.90, 1700, 820, 210, -8),
                                  ("270f-fe0f", 4.89, 1150, 830, 210, 15),
                                  ("1f393", 5.39, 250, 210, 230, -12)]:
        draw_sticker(canvas, name, x, y, sz, t, t0, end, clock, r)


def bubble_img(text, color, fsize, pad_x, h):
    ti = text_img(text, F_MED, fsize, WHITE)
    w = ti.width + pad_x * 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=color + (255,))
    im.alpha_composite(ti, (pad_x, (h - ti.height) // 2 + 1))
    return im


VERSE_SLOTS = [(360, 340, -8), (900, 330, 5), (560, 600, -4)]
BAR = dict(x0=230, x1=1690, cy=850, h=118)


def scene_verse(canvas, t, cfg, clock, bgs):
    lines = cfg["verse"]
    bar_col = BLUE
    # which line is active
    idx = 0
    for i, ln in enumerate(lines):
        if t >= ln["start"]:
            idx = i
    ln = lines[idx]
    color = BUBBLE_RED if idx == 2 else BLUE

    # --- sent bubbles (top-right stack)
    stack_bottom = 690
    for j in range(idx):
        prev = lines[j]
        sent_t = prev["end"] - 0.12
        age_rank = idx - 1 - j          # 0 = newest
        # animate rank change when a new line gets sent
        move = ease_out_cubic((t - (lines[idx - 1]["end"] - 0.12)) / 0.3) if idx >= 1 else 1
        y_target = stack_bottom - 78 * age_rank
        y_prev = stack_bottom - 78 * (age_rank - 1) if age_rank > 0 else None
        if y_prev is None:
            # newest bubble flies from the bar
            fy = BAR["cy"] + (stack_bottom - BAR["cy"]) * move
            fs = 1.0 + 0.6 * (1 - move)
        else:
            fy = y_prev + (y_target - y_prev) * move
            fs = 1.0
        col = BUBBLE_RED if j == 2 else BLUE
        b = bubble_img(prev["text"], col, 30, 26, 58)
        alpha = clamp((fy - 140) / 140)
        paste(canvas, b, BAR["x1"] - b.width * fs / 2, fy, fs, 0, alpha)

    # --- input bar
    first = lines[0]["start"]
    grow = ease_out_back((t - first) / 0.5, 1.2)
    x0, x1, cy, h = BAR["x0"], BAR["x1"], BAR["cy"], BAR["h"]
    mid = (x0 + x1) / 2
    half = max(h / 2, (x1 - x0) / 2 * grow)
    bar = Image.new("RGBA", (int(half * 2) + 2, h), (0, 0, 0, 0))
    ImageDraw.Draw(bar).rounded_rectangle((0, 0, bar.width - 1, h - 1), radius=h // 2,
                                          fill=color + (255,))
    canvas.alpha_composite(bar, (int(mid - half), int(cy - h / 2)))

    # --- typed text
    words = ln["text"].split(" ")
    wt = ln.get("w") or []
    if len(wt) < len(words):
        # distribute missing word times evenly
        dur = (ln["end"] - ln["start"]) * 0.7
        wt = [ln["start"] + 0.15 + dur * k / len(words) for k in range(len(words))]
    shown = []
    for k, wd in enumerate(words):
        if t < wt[k]:
            break
        nxt = wt[k + 1] if k + 1 < len(words) else wt[k] + 0.6
        frac = clamp((t - wt[k]) / max(0.12, min(0.45, nxt - wt[k])))
        shown.append(wd[: max(1, int(round(len(wd) * frac)))])
    typed = " ".join(shown)
    send_out = t > ln["end"] - 0.12 and idx < len(lines)  # cleared after "send"
    if typed and not send_out and half > 200:
        f = text_img(typed, F_SEMI, 50, WHITE)
        canvas.alpha_composite(f, (x0 + 58, int(cy - f.height / 2) + 2))
        cx = x0 + 58 + f.width + 4
    else:
        cx = x0 + 58
    if half > 200 and (int(t * 2.2) % 2 == 0):
        ImageDraw.Draw(canvas).rectangle((cx, cy - 28, cx + 4, cy + 28), fill=WHITE + (230,))
    # send arrow icon on the right of the bar
    if half > 300:
        d = ImageDraw.Draw(canvas)
        ax, ay = x1 - 70, cy
        pop = 1 + 0.25 * math.exp(-max(0, t - (ln["end"] - 0.12)) * 10) if t > ln["end"] - 0.12 else 1
        r = 34 * pop
        d.ellipse((ax - r, ay - r, ax + r, ay + r), fill=(255, 255, 255, 60))
        d.polygon([(ax - 14 * pop, ay - 16 * pop), (ax + 18 * pop, ay), (ax - 14 * pop, ay + 16 * pop),
                   (ax - 8 * pop, ay)], fill=WHITE)

    # --- word stickers
    order = [0, 2, 1]
    slot_i = 0
    for st in ln.get("stickers", []):
        name, widx = st[0], st[1]
        label = st[2] if len(st) > 2 else None
        if name == "mektep":
            sx, sy, sr = VERSE_SLOTS[1]
            size = 560
        else:
            # skip slot 1 when the building takes it
            while True:
                s_idx = order[slot_i % 3]
                slot_i += 1
                if not (s_idx == 1 and any(s[0] == "mektep" for s in ln["stickers"])):
                    break
            sx, sy, sr = VERSE_SLOTS[s_idx]
            size = 270
        t0 = wt[min(widx, len(wt) - 1)]
        draw_sticker(canvas, name, sx, sy, size, t, t0, ln["end"], clock, sr, label, seed=widx)


CHORUS_SLOTS = [(250, 190, -10), (1670, 200, 9), (250, 830, 8), (1670, 820, -7), (960, 140, 4)]


def scene_chorus(canvas, t, cfg, clock, bgs):
    cards = cfg["chorus"]
    card = cards[0]
    for c in cards:
        if t >= c["start"]:
            card = c
    st, en = card["start"], card["end"]
    out = clamp((t - (en - 0.14)) / 0.14)
    shift = -70 * ease_in_cubic(out)
    fade = 1 - out

    # ticker of pink ghost text along the bottom
    ghost = text_img(card["hero"], F_BLACK, 150, PINK)
    step = ghost.width + 90
    x = -((t * 260) % step)
    while x < W:
        paste(canvas, ghost, x + ghost.width / 2, 905 + ghost.height / 2)
        x += step

    # hero words
    hero_words = card["hero"].split(" ")
    size = 220
    fh = font(F_BLACK, size)
    total = sum(fh.getlength(w) for w in hero_words) + 60 * (len(hero_words) - 1)
    if total > 1500:
        size = int(size * 1500 / total)
        fh = font(F_BLACK, size)
        total = sum(fh.getlength(w) for w in hero_words) + 60 * (len(hero_words) - 1)
    beat = 1 + 0.035 * clock.pulse(t)
    hx0 = W / 2 - total / 2
    hy = 520 + shift
    x = hx0
    for k, wd in enumerate(hero_words):
        img = text_img(wd, F_BLACK, size, RED)
        w = fh.getlength(wd)
        t0 = card["w_hero"][min(k, len(card["w_hero"]) - 1)]
        p = (t - t0) / 0.22
        if p > 0:
            s = (1.45 - 0.45 * ease_out_back(p, 1.2)) * beat
            paste(canvas, img, x + w / 2, hy, s, 0, clamp(p * 1.6) * fade)
        x += w + 60

    # small top / bottom lines (word by word)
    def small_line(text, times, y, align, fsize, col):
        if not text:
            return
        words = text.split(" ")
        f = font(F_XBOLD, fsize)
        gap = fsize * 0.45
        tw = sum(f.getlength(w) for w in words) + gap * (len(words) - 1)
        xx = hx0 if align == "left" else hx0 + total - tw
        for k, wd in enumerate(words):
            w = f.getlength(wd)
            t0 = times[min(k, len(times) - 1)] if times else st
            p = (t - t0) / 0.18
            if p > 0:
                img = text_img(wd, F_XBOLD, fsize, col)
                paste(canvas, img, xx + w / 2, y - 30 * (1 - ease_out_cubic(p)), 1, 0, clamp(p) * fade)
            xx += w + gap

    small_line(card.get("top", ""), card.get("w_top", []), hy - size * 0.68 - 10, "left", 60, DARKRED)
    small_line(card.get("bottom", ""), card.get("w_bottom", []), hy + size * 0.70 + 10, "right", 66, DARKRED)

    # stickers
    for i, stck in enumerate(card.get("stickers", [])):
        name, t0 = stck[0], stck[1]
        if name == "mektep":
            draw_sticker(canvas, name, 1480, 230, 640, t, t0, en, clock, 6, seed=i)
        else:
            sx, sy, sr = CHORUS_SLOTS[i % len(CHORUS_SLOTS)]
            draw_sticker(canvas, name, sx, sy, 260, t, t0, en, clock, sr, seed=i)


def scene_outro(canvas, t, cfg, clock, bgs):
    sc = cfg["outro"]
    st, en = sc["start"], sc["end"]
    p = (t - st) / 0.55
    paste(canvas, polaroid(), 960, 450, 1.05 * ease_out_back(p, 1.3) * (1 + 0.02 * clock.pulse(t)),
          -3 + 2 * math.sin(t))
    tx = text_img(sc["text"], F_BLACK, 110, RED)
    q = (t - st - 0.5) / 0.35
    if q > 0:
        paste(canvas, tx, 960, 900, ease_out_back(q))
    draw_sticker(canvas, "1faf6", 280, 250, 280, t, st + 0.25, en + 5, clock, -10)
    draw_sticker(canvas, "1f490", 1650, 260, 300, t, st + 0.75, en + 5, clock, 10)
    draw_sticker(canvas, "1f3eb", 250, 640, 260, t, st + 1.25, en + 5, clock, 8)
    draw_sticker(canvas, "2764-fe0f", 1680, 640, 240, t, st + 1.75, en + 5, clock, -8)


# ----------------------------------------------------------------- frame
def render_frame(t, cfg, clock, bgs):
    if cfg["chorus"][0]["start"] <= t < cfg["outro"]["start"]:
        canvas = bgs["light"].copy()
    else:
        canvas = bgs["full"].copy()
    if t < cfg["intro"]["end"]:
        scene_intro(canvas, t, cfg, clock, bgs)
    elif t < cfg["chorus"][0]["start"]:
        scene_verse(canvas, t, cfg, clock, bgs)
    elif t < cfg["outro"]["start"]:
        scene_chorus(canvas, t, cfg, clock, bgs)
    else:
        scene_outro(canvas, t, cfg, clock, bgs)

    # quick white flash + shake into the chorus (the "drop")
    tc = cfg["chorus"][0]["start"]
    if tc - 0.05 <= t < tc + 0.25:
        k = clamp(1 - (t - tc) / 0.25)
        flash = Image.new("RGBA", (W, H), (255, 255, 255, int(150 * k)))
        canvas.alpha_composite(flash)
        dx = int(14 * k * math.sin(t * 90))
        canvas = canvas.transform(canvas.size, Image.AFFINE, (1, 0, dx, 0, 1, 0), fillcolor=CREAM + (255,))

    # fade in/out
    a0, a1 = cfg["audio_start"], cfg["audio_end"]
    fade = min(clamp((t - a0) / 0.4), clamp((a1 - t) / 1.0))
    if fade < 1:
        black = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - fade))))
        canvas.alpha_composite(black)
    return canvas.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "ustazym.mp4"))
    ap.add_argument("--lyrics", default=os.path.join(HERE, "lyrics.json"))
    ap.add_argument("--preview", nargs="*", type=float, help="render only PNG frames at these seconds")
    args = ap.parse_args()

    cfg = json.load(open(args.lyrics, encoding="utf-8"))
    clock = Clock(cfg)
    bgs = {"full": make_background(strength=1.0), "light": make_background(strength=0.55)}

    if args.preview:
        for s in args.preview:
            p = os.path.join(os.path.dirname(args.out), f"preview_{s:05.2f}.png")
            render_frame(s, cfg, clock, bgs).save(p)
            print(p)
        return

    a0, a1 = cfg["audio_start"], cfg["audio_end"]
    n = int(round((a1 - a0) * FPS))
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-ss", str(a0), "-t", str(a1 - a0), "-i", args.audio,
           "-map", "0:v", "-map", "1:a",
           "-af", f"afade=t=out:st={a1 - a0 - 1.5}:d=1.5",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        t = a0 + i / FPS
        proc.stdin.write(render_frame(t, cfg, clock, bgs).tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{n}", file=sys.stderr, flush=True)
    proc.stdin.close()
    proc.wait()
    print(args.out)


if __name__ == "__main__":
    main()
