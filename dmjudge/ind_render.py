"""인디케이터 그림을 numpy로 직접 합성 (픽셀마다 진짜 반투명).

tkinter 투명색 방식은 '완전 투명 / 완전 불투명' 두 가지뿐이라, 흐려지는 선이 실제로는
검게 칠해진 불투명 선이 되어 뒤를 가렸다. 여기서는 선마다 알파를 줘서 겹쳐 그리므로
선이 많이 모인 곳일수록 진하고 밝아지고(= 경향성이 보임), 비어 있는 곳은 완전히 비친다.

결과는 premultiplied BGRA (Windows UpdateLayeredWindow 형식).
"""
from __future__ import annotations

import math
import os

import numpy as np

MARK = (63, 214, 200)      # 기본 선 (청록)
OUT = (255, 120, 90)       # 범위 밖
NEW = (255, 236, 120)      # 방금 친 노트
ZONE = (95, 211, 122)      # FAST/SLOW 범위 끝선
AXIS = (150, 156, 168)


class Canvas:
    """premultiplied RGBA float32, 'over' 합성.
    scale: 좌표는 기준 크기(w, h)로 받고 실제 픽셀은 scale배로 그린다 (OBS에서 크게/작게 쓸 때 선명하게)."""

    def __init__(self, w, h, scale=1.0):
        self.w, self.h = int(w), int(h)
        self.s = float(scale)
        self.pw, self.ph = max(1, int(round(self.w * self.s))), max(1, int(round(self.h * self.s)))
        self.px = np.zeros((self.ph, self.pw, 4), np.float32)

    def _over_px(self, ys, xs, rgb, a):
        """a: 스칼라 또는 (h, w) 배열 (0~1), 실제 픽셀 좌표"""
        reg = self.px[ys, xs]
        a = np.asarray(a, np.float32)
        if a.ndim == 2:
            a = a[..., None]
        col = np.array([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0], np.float32)
        reg *= (1.0 - a)
        reg += col * a

    def _over(self, ys, xs, rgb, a):
        """기준 좌표 slice. 배열 a는 기준 크기 → 실제 크기로 늘리거나 줄여서 합성"""
        if self.s == 1.0:
            return self._over_px(ys, xs, rgb, a)
        S = self.s
        y0 = max(0, int(round((ys.start or 0) * S))); y1 = min(self.ph, int(round((self.h if ys.stop is None else ys.stop) * S)))
        x0 = max(0, int(round((xs.start or 0) * S))); x1 = min(self.pw, int(round((self.w if xs.stop is None else xs.stop) * S)))
        if y1 <= y0 or x1 <= x0:
            return
        a = np.asarray(a, np.float32)
        if a.ndim == 2:
            import cv2
            a = cv2.resize(np.ascontiguousarray(a), (x1 - x0, y1 - y0),
                           interpolation=cv2.INTER_AREA if S < 1 else cv2.INTER_LINEAR)
        self._over_px(slice(y0, y1), slice(x0, x1), rgb, a)

    def rect(self, x0, x1, y0, y1, rgb, a):
        """가로는 소수점 위치 그대로 (가장자리 칸은 덮는 비율만큼) → 1px 미만 차이도 보임"""
        if a <= 0:
            return
        S = self.s
        x0, x1 = max(0.0, float(x0) * S), min(float(self.pw), float(x1) * S)
        y0, y1 = max(0, int(round(y0 * S))), min(self.ph, int(round(y1 * S)))
        if x1 <= x0 or y1 <= y0:
            return
        c0, c1 = int(math.floor(x0)), int(math.ceil(x1))
        cov = np.ones(c1 - c0, np.float32)
        if c1 - c0 == 1:
            cov[0] = x1 - x0
        else:
            cov[0] = 1.0 - (x0 - c0)
            cov[-1] = 1.0 - (c1 - x1)
        am = np.broadcast_to(cov[None, :] * a, (y1 - y0, c1 - c0))
        self._over_px(slice(y0, y1), slice(c0, c1), rgb, am)

    def vline(self, x, y0, y1, rgb, a, width=1.0):
        self.rect(x - width / 2.0, x + width / 2.0, y0, y1, rgb, a)

    def hline(self, x0, x1, y, rgb, a):
        self.rect(x0, x1, y - 0.5, y + 0.5, rgb, a)

    def _mask_px(self, m, x, y, rgb, a):
        x, y = int(round(x)), int(round(y))
        mh, mw = m.shape
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(self.pw, x + mw), min(self.ph, y + mh)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        sub = m[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        self._over_px(slice(dy0, dy1), slice(dx0, dx1), rgb, sub * a)

    def mask(self, m, x, y, rgb, a):
        """m: (h, w) 0~1 마스크를 왼쪽 위 (x, y)에 (기준 좌표)"""
        if self.s != 1.0:
            import cv2
            mh, mw = m.shape
            nw, nh = max(1, int(round(mw * self.s))), max(1, int(round(mh * self.s)))
            m = cv2.resize(np.ascontiguousarray(m, np.float32), (nw, nh),
                           interpolation=cv2.INTER_AREA if self.s < 1 else cv2.INTER_LINEAR)
        self._mask_px(m, x * self.s, y * self.s, rgb, a)

    def text(self, s, cx, cy, px, rgb, a, anchor="c", bold=True, shadow=0.55):
        S = self.s
        m = text_mask(s, max(5, int(round(px * S))), bold)    # 글자는 실제 크기로 새로 그려서 선명하게
        mh, mw = m.shape[0] / S, m.shape[1] / S
        x = cx - mw / 2.0 if anchor == "c" else (cx if anchor == "w" else cx - mw)
        y = cy - mh / 2.0
        if shadow > 0:                        # 밝은 배경에서도 읽히게 옅은 그림자
            self._mask_px(m, (x + 1) * S, (y + 1) * S, (0, 0, 0), a * shadow)
        self._mask_px(m, x * S, y * S, rgb, a)

    def tri(self, cx, ytop, half, height, rgb, a):
        """아래가 넓은 작은 삼각형 (평균 표시)"""
        h = max(1, int(round(height)))
        ys = np.arange(h, dtype=np.float32) + 0.5
        x0 = int(math.floor(cx - half - 1))
        xs = np.arange(x0, int(math.ceil(cx + half + 1)) + 1, dtype=np.float32) + 0.5
        hw = half * ys / h                                     # 줄마다 반폭
        cov = np.clip(hw[:, None] - np.abs(xs[None, :] - cx) + 0.5, 0, 1)
        self.mask(cov, x0, ytop, rgb, a)

    def bgra_premul(self):
        p = np.clip(self.px * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(p[..., [2, 1, 0, 3]])


# ------------------------------------------------------------------ 글자
_FONT_CACHE = {}
_TEXT_CACHE = {}
_FONT_FILES = {
    True: ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    False: ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def _font(px, bold):
    k = (px, bold)
    if k in _FONT_CACHE:
        return _FONT_CACHE[k]
    f = None
    try:
        from PIL import ImageFont
        for p in _FONT_FILES[bold]:
            if os.path.exists(p):
                f = ImageFont.truetype(p, px)
                break
        if f is None:
            try:
                f = ImageFont.load_default(px)
            except TypeError:
                f = ImageFont.load_default()
    except Exception:
        f = None
    _FONT_CACHE[k] = f
    return f


def text_mask(s, px, bold=True):
    k = (s, px, bold)
    m = _TEXT_CACHE.get(k)
    if m is not None:
        return m
    f = _font(px, bold)
    if f is not None:
        from PIL import Image, ImageDraw
        l, t, r, b = f.getbbox(s)
        img = Image.new("L", (r - l + 2, b - t + 2), 0)
        ImageDraw.Draw(img).text((1 - l, 1 - t), s, fill=255, font=f)
        m = np.asarray(img, np.float32) / 255.0
    else:                                         # PIL이 없으면 OpenCV 글꼴
        import cv2
        sc = px / 30.0
        th = 2 if bold else 1
        (tw, tht), bl = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        img = np.zeros((tht + bl + 4, tw + 4), np.uint8)
        cv2.putText(img, s, (2, tht + 2), cv2.FONT_HERSHEY_SIMPLEX, sc, 255, th, cv2.LINE_AA)
        m = img.astype(np.float32) / 255.0
    _TEXT_CACHE[k] = m
    return m


# ------------------------------------------------------------------ 인디케이터
def flash_text(kind, j=None):
    """범위 밖으로 쳤을 때 뜨는 글자: FAST 42.5ms / SLOW 47.0ms (영점에서 벗어난 크기)"""
    name = "FAST" if kind == "fast" else "SLOW"
    if j is None:
        return name
    return "%s %.1fms" % (name, abs(float(j)))


def flash_parts(kind, j, show_lab, show_ms):
    """범위 밖 글자를 설정대로: "FAST -52.3ms" / "FAST" / "-52.3ms" (숫자는 빠름 -, 느림 +)"""
    parts = []
    if show_lab:
        parts.append("FAST" if kind == "fast" else "SLOW")
    if show_ms and j is not None:
        parts.append(ms_text(j))
    return " ".join(parts)


TEXT_FADES = ("end", "mid", "linear")      # FAST/SLOW·ms 글자가 연해지는 방식


def text_alpha(f, mode="end"):
    """글자 진하기 (f = 경과 / "ms 글자 초"). end = 끝 30%만 흐려짐(예전 그대로),
    mid = 40% 지나면서부터 서서히, linear = 뜨자마자 일정하게 옅어짐"""
    if f >= 1.0:
        return 0.0
    hold = {"end": 0.7, "mid": 0.4, "linear": 0.0}.get(mode, 0.7)
    if f < hold:
        return 1.0
    return max(0.0, 1.0 - (f - hold) / (1.0 - hold))


FADE_MODES = ("weak", "mid", "strong")      # 인디케이터 선이 연해지는 정도


def line_alpha(age, span, mode="weak"):
    """오래된 선일수록 연하게. weak = 예전 그대로(55%→12%), mid = 70%→5%,
    strong = 처음 0.5초는 85%로 진하게 유지하다가 빠르게 3%까지."""
    r = max(0.0, min(1.0, age / max(1.0, span)))
    if mode == "strong":
        hold = min(500.0, span * 0.25)
        if age <= hold:
            return 0.85
        q = (age - hold) / max(1.0, span - hold)
        return 0.03 + 0.82 * (1.0 - min(1.0, q)) ** 2
    if mode == "mid":
        return 0.05 + 0.65 * (1.0 - r) ** 1.5
    return 0.12 + 0.43 * (1.0 - r)


def nice_step(R):
    for st in (2, 5, 10, 20, 25, 50, 100):
        if R / st <= 12:
            return st
    return 200


def ms_text(j):
    """모든 노트 ms 표시 글자 (빠름 -, 느림 +)"""
    return "%+.1fms" % float(j)


FAST_COL = (90, 200, 255)     # 빠름 (파랑)
SLOW_COL = (255, 110, 95)     # 느림 (빨강)


def fs_color(kind, opts):
    """FAST/SLOW 글자 색. opts["swap_colors"]면 반대로 (FAST 빨강 / SLOW 파랑 — 다른 게임에 익숙한 사람용)"""
    fast = (kind == "fast")
    if opts.get("swap_colors"):
        fast = not fast
    return FAST_COL if fast else SLOW_COL


def text_px(opts, base=14):
    """글자 크기(px): 기본 14px × 글자 크기 배율"""
    try:
        ts = float(opts.get("text_scale", 1.0))
    except (TypeError, ValueError):
        ts = 1.0
    return max(7, int(round(base * max(0.5, min(2.5, ts)))))


def text_y(H, bar, px=14):
    """글자 줄 높이: 막대가 있으면 막대 위(글자가 클수록 더 위로), 없으면 창 가운데"""
    return int(round(H * 0.62)) - 13 - 0.55 * px if bar else H / 2.0


def render(W, H, vis, R, fs, win, flashes, opts, moving=False, fl_ms=450.0, last=None, scale=1.0):
    """vis: [(오차 ms, 경과 ms)] 오래된 것부터, flashes: [(kind, 경과 ms[, 영점 기준 오차 ms[, 'L'/'R']])]
    last: (오차 ms, 경과 ms, 'fast'/'slow'/None) 가장 최근 노트와 유지 중인 FAST/SLOW (모든 노트 ms 표시용) 또는 None"""
    cv = Canvas(W, H, scale)
    bar = bool(opts.get("bar", True))
    span = float(opts.get("seconds", 3.0)) * 1000.0
    k = float(opts.get("strength", 1.0))           # 전체 진하기 배율
    if moving:
        cv.rect(0, W, 0, H, (32, 35, 43), 0.75)
        for x in range(2, W - 2, 7):
            cv.rect(x, min(W - 2, x + 4), 1, 3, (255, 209, 102), 1.0)
            cv.rect(x, min(W - 2, x + 4), H - 3, H - 1, (255, 209, 102), 1.0)
        for y in range(2, H - 2, 7):
            cv.rect(1, 3, y, min(H - 2, y + 4), (255, 209, 102), 1.0)
            cv.rect(W - 3, W - 1, y, min(H - 2, y + 4), (255, 209, 102), 1.0)
    elif opts.get("bg_alpha", 0) > 0:
        cv.rect(0, W, 0, H, (0, 0, 0), float(opts["bg_alpha"]))

    pad = 6
    x0, x1 = pad, W - pad

    def X(v):
        v = max(-R, min(R, v))
        return x0 + (v + R) / (2 * R) * (x1 - x0)

    mid = int(round(H * 0.62))
    n = len(vis)
    if bar:
        # 축: 아주 옅은 선 + 점선 눈금 (뒤가 다 비침)
        cv.hline(X(-R), X(R), mid, AXIS, 0.28 * k)
        st = nice_step(R)
        for v in range(-int(R) // st * st, int(R) + 1, st):
            cv.vline(X(v), mid - 2, mid + 3, AXIS, 0.45 * k)
        for v in (-R, R):                                   # 양끝
            cv.vline(X(v), mid - 6, mid + 7, (230, 230, 230), 0.6 * k, 1.5)
        # FAST/SLOW 범위: 채운 띠 대신 아주 옅은 바탕 + 끝선
        cv.rect(X(-fs), X(fs), mid - 8, mid + 9, ZONE, 0.07 * k)
        for v in (-fs, fs):
            cv.vline(X(v), mid - 8, mid + 9, ZONE, 0.75 * k, 1.5)
        if abs(fs - win) > 0.5 and win < R:
            for v in (-win, win):
                cv.vline(X(v), mid - 5, mid + 6, (170, 175, 185), 0.6 * k)
        cv.text("-%d" % round(R), x0 + 1, mid + 14, 10, (150, 156, 168), 0.7 * k, anchor="w", bold=False)
        cv.text("+%d" % round(R), x1 - 1, mid + 14, 10, (150, 156, 168), 0.7 * k, anchor="e", bold=False)

        # 분포 곡선: 최근 점들의 밀도를 막대 위에 옅은 언덕으로 (어디로 모이는지 한눈에)
        if n and opts.get("density", True):
            wts = np.array([max(0.0, 1.0 - a / span) for _, a in vis], np.float32)
            js = np.array([j for j, _ in vis], np.float32)
            tot = float(wts.sum())
            if tot > 0:
                xs = np.arange(W, dtype=np.float32) + 0.5
                v = (xs - x0) / max(1.0, (x1 - x0)) * 2 * R - R      # 칸 → ms
                sig = max(2.0, R * 0.065)
                d = (wts[None, :] * np.exp(-0.5 * ((v[:, None] - np.clip(js, -R, R)[None, :]) / sig) ** 2)).sum(1)
                dm = float(d.max())
                if dm > 0:
                    hmax = max(6.0, mid - 22.0)                  # 언덕 최대 높이 (글자 자리 아래까지)
                    amt = min(1.0, tot / 6.0)                    # 점이 적으면 옅게
                    hh = d / dm * hmax * (0.5 + 0.5 * amt)
                    yy = np.arange(H, dtype=np.float32)[:, None] + 0.5
                    top = mid - hh[None, :]
                    cov = np.clip(yy - top, 0, 1) * (yy < mid)    # 언덕 안쪽
                    edge = np.clip(1.0 - np.abs(yy - top), 0, 1) * (hh[None, :] > 0.6)
                    cv._over(slice(0, H), slice(0, W), MARK, cov * (0.2 * amt * k))
                    cv._over(slice(0, H), slice(0, W), MARK, edge * (0.55 * amt * k))

        # 선: 반투명으로 겹쳐 그림 → 많이 모인 곳일수록 진해짐
        # 같은 색끼리는 순서와 상관없이 1-Π(1-a) 로 한 번에 합성 (점이 많아도 빠르게)
        for col, sel in ((OUT, lambda j: abs(j) > fs), (MARK, lambda j: abs(j) <= fs)):
            lg = np.zeros(W, np.float64)
            for j, age in vis[:-1]:
                if not sel(j):
                    continue
                a = min(0.999, line_alpha(age, span, opts.get("fade", "weak")) * k)
                xa, xb = X(j) - 0.8, X(j) + 0.8
                c0, c1 = max(0, int(math.floor(xa))), min(W, int(math.ceil(xb)))
                for c in range(c0, c1):
                    cov = min(xb, c + 1) - max(xa, c)
                    if cov > 0:
                        lg[c] += math.log(1.0 - a * cov)
            am = (1.0 - np.exp(lg)).astype(np.float32)
            if am.max() > 0:
                cv._over(slice(mid - 7, mid + 8), slice(0, W), col, np.broadcast_to(am[None, :], (15, W)))
        if n:
            # 방금 친 노트: 어두운 테두리 + 굵고 긴 밝은 선 + 위쪽 작은 점 (다른 선들 사이에서도 눈에 띄게)
            xn = X(vis[-1][0])
            cv.vline(xn, mid - 14, mid + 15, (0, 0, 0), 0.55, 5.0)
            cv.vline(xn, mid - 13, mid + 14, NEW, 1.0, 3.0)
            cv.rect(int(round(xn - 2.5)), int(round(xn + 2.5)), mid - 17, mid - 12, NEW, 1.0)
        # 정중앙
        cv.vline(X(0), mid - 11, mid + 12, (255, 255, 255), min(1.0, 0.9 * k), 2.5)

    P = text_px(opts)                  # 글자 크기
    TY = text_y(H, bar, P)
    XC = X(0) if bar else W / 2.0      # 중앙선: 글자가 넘어가지 않게 왼쪽/오른쪽에 나눠 둔다
    M = 6                              # 중앙선과 글자 사이 여백

    def fit(samples, room):
        """samples 글자들이 모두 room(px) 안에 들어가는 가장 큰 크기 (최대 P)"""
        p = P
        while p > 7:
            w = max(text_mask(t, p).shape[1] for t in samples)
            if w <= room:
                break
            p = max(7, min(p - 1, int(p * room / w)))
        return p

    for fl in flashes:
        kind, age = fl[0], fl[1]
        jv = fl[2] if len(fl) > 2 else None
        f = age / fl_ms
        txt = fl[4] if len(fl) > 4 else flash_text(kind, jv)
        side = fl[3] if len(fl) > 3 else ("L" if kind == "fast" else "R")   # 왼손 → 왼쪽, 오른손 → 오른쪽
        col = fs_color(kind, opts)
        if side == "L":
            p = fit([txt], XC - M - 2)
            tw = text_mask(txt, p).shape[1]
            x = min(max(X(-R * 0.55), 2 + tw / 2), XC - M - tw / 2)
        else:
            p = fit([txt], W - 2 - (XC + M))
            tw = text_mask(txt, p).shape[1]
            x = max(min(X(R * 0.55), W - 2 - tw / 2), XC + M + tw / 2)
        a = text_alpha(f, opts.get("text_fade", "end"))
        cv.text(txt, x, TY, p, col, a)
    if last is not None:                          # 모든 노트 ms (다음 노트가 오면 바뀜)
        j, age, lab = last[0], last[1], (last[2] if len(last) > 2 else None)
        f = age / fl_ms
        a = text_alpha(f, opts.get("text_fade", "end"))
        if a > 0:
            out = abs(j) > fs
            col = fs_color("fast" if j < 0 else "slow", opts) if out else (255, 255, 255)
            txt = ms_text(j)
            # 크기는 가장 긴 경우 기준으로 정해서 숫자가 바뀌어도 크기가 흔들리지 않게
            p = min(fit(["-188.8ms"], W - 2 - (XC + M)), fit(["FAST", "SLOW"], XC - M - 2))
            if lab:                               # FAST/SLOW = 중앙선 왼쪽 (오른쪽 끝 고정)
                lc = fs_color(lab, opts)
                la = a
                if len(last) > 3 and last[3] is not None:     # 글자는 자기가 뜬 시각 기준으로 흐려짐
                    lf = last[3] / fl_ms
                    la = text_alpha(lf, opts.get("text_fade", "end"))
                cv.text(lab.upper(), XC - M, TY, p, lc, la, anchor="e")
            cv.text(txt, XC + M, TY, p, col, a, anchor="w")    # 숫자 = 중앙선 오른쪽 (왼쪽 끝 고정)
    if n and bar:
        avg = sum(j for j, _ in vis) / n
        cv.tri(X(avg), mid + 12, 4.5, 6, (200, 205, 214), 0.85 * k)
    if moving:
        cv.text("드래그 = 이동 · 휠 = 폭", W / 2, H - 9, 11, (255, 209, 102), 1.0)
    return cv
