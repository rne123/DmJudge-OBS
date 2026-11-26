"""디맥 결과 화면(JUDGEMENT DETAILS) 숫자 읽기.

전체 화면 한 장(BGR)을 받아, 결과 화면이면 [MAX 100%, 90%, 80%, ..., 10%, 1%, BREAK] 12개 숫자를 돌려준다.
- 글자 모양 표본: 사용자 1080p 결과 화면 6장에서 딴 숫자 97개 (줄마다 색이 달라도 '밝고 진한 색' 픽셀로 분리)
- 다른 해상도는 높이를 1080으로 맞추고 가운데 16:9 영역을 쓴다 (보더리스/전체 화면 기준)
- 결과 화면 판별: 정해진 12줄 위치에 숫자가 있고, 숫자 오른쪽 끝이 나란하고, 모든 글자가 표본과 충분히 닮았을 때만
"""
from __future__ import annotations

import base64
import zlib

import cv2
import numpy as np

_BLOB = "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"
_raw = zlib.decompress(base64.b64decode(_BLOB))
_N = 97
_DIG = np.frombuffer(_raw[:_N], np.uint8).astype(int)
_TMP = np.frombuffer(_raw[_N:], np.uint8).reshape(_N, 18, 12).astype(np.float32) / 255.0

# 1080p 기준 숫자 칸 (x 300~432, 줄 시작 y = 286 + 30*i, 글자 높이 약 19)
X0, X1, Y0, Y1 = 300, 432, 275, 645
ROW_Y = [286 + 30 * i for i in range(12)]
NAMES = ["MAX 100%", "MAX 90%", "MAX 80%", "MAX 70%", "MAX 60%", "MAX 50%", "MAX 40%",
         "MAX 30%", "MAX 20%", "MAX 10%", "MAX 1%", "BREAK"]
MAX_DIST = 60.0      # 표본과의 차이(픽셀 합, 216칸 중) — 이보다 크면 숫자가 아님


def to_1080(frame):
    """전체 화면을 높이 1080, 가운데 16:9로 맞춘다"""
    h, w = frame.shape[:2]
    gw = int(round(h * 16 / 9))
    x = max(0, (w - gw) // 2)
    g = frame[:, x:x + gw] if gw <= w else frame
    if g.shape[0] != 1080:
        g = cv2.resize(g, (1920, 1080), interpolation=cv2.INTER_AREA if g.shape[0] > 1080 else cv2.INTER_LINEAR)
    return g


def _mask(im):
    hsv = cv2.cvtColor(im[Y0:Y1, X0:X1], cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 150) & (hsv[..., 1] > 110)).astype(np.uint8)


def _classify(t):
    d = np.abs(_TMP - t).sum(axis=(1, 2))
    i = int(np.argmin(d))
    return int(_DIG[i]), float(d[i])


def _split_wide(r, gs):
    """저해상도를 키운 화면에서 붙어 버린 글자(예: '20')를 가장 얇은 세로줄에서 나눈다"""
    out = []
    for x, y, w, h in gs:
        k = int(round(w / 14.0))
        if w <= 17 or k < 2:
            out.append((x, y, w, h))
            continue
        cols = r[y:y + h, x:x + w].sum(0).astype(float)
        cuts = []
        for j in range(1, k):
            c = int(round(w * j / k))
            lo, hi = max(1, c - 4), min(w - 1, c + 5)
            cuts.append(lo + int(np.argmin(cols[lo:hi])))
        edges = [0] + cuts + [w]
        for a, b in zip(edges[:-1], edges[1:]):
            sub = r[y:y + h, x + a:x + b]
            ys = np.where(sub.any(1))[0]
            if len(ys):
                out.append((x + a, y + int(ys[0]), b - a, int(ys[-1] - ys[0] + 1)))
    return out


def read(frame, need_1080=True):
    """결과 화면이면 12개 숫자 리스트, 아니면 None"""
    im = to_1080(frame) if need_1080 else frame
    m = _mask(im)
    out, rights = [], []
    for y in ROW_Y:
        a, b = y - Y0 - 3, y - Y0 + 22
        r = m[a:b]
        if r.sum() < 20:
            return None
        n, lab, st, _ = cv2.connectedComponentsWithStats(r, 8)
        gs = sorted((st[i, 0], st[i, 1], st[i, 2], st[i, 3]) for i in range(1, n) if st[i, 4] >= 6)
        gs = _split_wide(r, gs)
        if not gs or len(gs) > 5:
            return None
        val = 0
        for x, yy, w, h in gs:
            if not (14 <= h <= 22) or w > 16:
                return None
            t = cv2.resize(r[yy:yy + h, x:x + w] * 255, (12, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            dgt, dist = _classify(t)
            if dist > MAX_DIST:
                return None
            val = val * 10 + dgt
        out.append(val)
        rights.append(gs[-1][0] + gs[-1][2])
    if max(rights) - min(rights) > 4:          # 숫자 오른쪽 끝이 나란해야 결과 화면
        return None
    if out[0] == 0:
        return None
    return out


# 미션 모드 결과 화면: 왼쪽에 JUDGEMENT DETAILS가 없고, 오른쪽 원 둘레에 MAX 100% / MAX 1~90% / BREAK 세 숫자만 있다.
# 1080p 기준 각 숫자 칸 (x0, y0, x1, y1)과 숫자 오른쪽 끝 x (숫자는 오른쪽 끝에 맞춰 왼쪽으로 늘어남)
MISSION_BOX = [(1060, 480, 1200, 522, 1185),     # MAX 100%  (노란 글자)
               (1700, 480, 1825, 522, 1810),     # MAX 1~90% (분홍 글자)
               (1560, 300, 1675, 340, 1662)]     # BREAK     (빨간 글자)


def _read_box(im, box):
    x0, y0, x1, y1, right = box
    hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 150) & (hsv[..., 1] > 110)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    gs = sorted((st[i, 0], st[i, 1], st[i, 2], st[i, 3]) for i in range(1, n)
                if st[i, 4] >= 20 and 20 <= st[i, 3] <= 28)          # 글자 높이 23~24 (밑줄·띠는 뺌)
    if not gs or len(gs) > 5:
        return None
    if abs(x0 + gs[-1][0] + gs[-1][2] - right) > 6:                # 오른쪽 끝이 제자리여야 결과 화면
        return None
    val = 0
    for x, y, w, h in gs:
        if w > 22:
            return None
        t = cv2.resize(m[y:y + h, x:x + w] * 255, (12, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        dgt, dist = _classify(t)
        if dist > 40.0:
            return None
        val = val * 10 + dgt
    return val


def read_mission(frame, need_1080=True):
    """미션 모드 결과 화면이면 [MAX 100%, MAX 1~90%, BREAK] 3개 숫자, 아니면 None
    (90%·80% … 단계별 개수는 화면에 없음 — 영점 보정에는 'MAX 100% 아닌 개수'만 있으면 됨)"""
    im = to_1080(frame) if need_1080 else frame
    out = []
    for box in MISSION_BOX:
        v = _read_box(im, box)
        if v is None:
            return None
        out.append(v)
    if out[0] == 0:
        return None
    return out


# ------------------------------------------------------------ 래더 결과 (라운드 점수만 나옴)
# 래더 대기 화면: 가운데 위부터 라운드마다 빨간 'ROUND n' 띠 → 곡 제목 → 왼쪽(내) 점수 / 오른쪽(상대) 점수.
# 점수는 가는 흰 글자(결과 화면 숫자와 글꼴이 달라 따로 표본: Lato 계열 글꼴 10개 + 실제 화면에서 딴 9개)
_LBLOB = "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"
_lraw = zlib.decompress(base64.b64decode(_LBLOB))
_LN = 19
_LDIG = np.frombuffer(_lraw[:_LN], np.uint8).astype(int)
_LTMP = np.frombuffer(_lraw[_LN:], np.uint8).reshape(_LN, 18, 12).astype(np.float32) / 255.0
L_BAR_X = (805, 1115)           # ROUND 띠 가로 범위 (1080p)
L_SCORE_X = (680, 872)          # 왼쪽 플레이어 점수 칸 (오른쪽 끝 약 857)
L_SCORE_RIGHT = 857
R_SCORE_X = (1048, 1240)        # 오른쪽 플레이어 점수 칸 (왼쪽 끝 약 1063)
R_SCORE_LEFT = 1063
L_SCORE_DY = (100, 170)         # ROUND 띠 위쪽에서 점수 줄까지 (첫 화면 기준 띠 126 → 숫자 244~276)
L_LABEL_DY = (82, 112)          # WIN / LOSE 글자 (점수 바로 위, 색 있는 글자)


def _classify_l(t):
    d = np.abs(_LTMP - t).sum(axis=(1, 2))
    best = {}
    for dg, dd in zip(_LDIG, d):
        best[int(dg)] = min(best.get(int(dg), 1e9), float(dd))
    r = sorted(best.items(), key=lambda kv: kv[1])
    return r[0][0], r[0][1], r[1][1] - r[0][1]        # 숫자, 차이, 2등과의 차이


def _round_bars(im):
    """빨간 ROUND 띠들의 위쪽 y (위→아래)"""
    x0, x1 = L_BAR_X
    hsv = cv2.cvtColor(im[:, x0:x1], cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)
    row = red.mean(axis=1) > 0.6
    bars, y = [], 0
    while y < len(row):
        if row[y]:
            y0 = y
            while y < len(row) and row[y]:
                y += 1
            if 14 <= y - y0 <= 40:
                bars.append(y0)
        y += 1
    return bars


def _read_lscore(im, by, side="L"):
    """side L = 왼쪽 플레이어(숫자 오른쪽 맞춤), R = 오른쪽 플레이어(숫자 왼쪽 맞춤)"""
    x0, x1 = L_SCORE_X if side == "L" else R_SCORE_X
    a, b = by + L_SCORE_DY[0], by + L_SCORE_DY[1]
    if b > im.shape[0]:
        return None
    hsv = cv2.cvtColor(im[a:b, x0:x1], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 190) & (hsv[..., 1] < 60)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    gs = sorted((st[i, 0], st[i, 1], st[i, 2], st[i, 3]) for i in range(1, n)
                if st[i, 4] >= 20 and 26 <= st[i, 3] <= 38)
    if not gs or len(gs) > 7:
        return None
    if side == "L" and abs(x0 + gs[-1][0] + gs[-1][2] - L_SCORE_RIGHT) > 8:      # 오른쪽 끝이 제자리
        return None
    if side == "R" and abs(x0 + gs[0][0] - R_SCORE_LEFT) > 8:                    # 왼쪽 끝이 제자리
        return None
    ys = [g[1] for g in gs]
    if max(ys) - min(ys) > 4:                                    # 한 줄
        return None
    val = 0
    for x, y, w, h in gs:
        if w > 26:
            return None
        t = cv2.resize(m[y:y + h, x:x + w] * 255, (12, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        dgt, dist, gap = _classify_l(t)
        if dist > 40.0 or gap < 8.0:                             # 확실하지 않으면 안 읽음
            return None
        val = val * 10 + dgt
    # 점수 위의 WIN / LOSE 글자(색 있는 글자)가 있어야 결과
    la, lb = by + L_LABEL_DY[0], by + L_LABEL_DY[1]
    lx0, lx1 = (790, L_SCORE_RIGHT + 4) if side == "L" else (R_SCORE_LEFT - 4, R_SCORE_LEFT + 75)
    h2 = cv2.cvtColor(im[la:lb, lx0:lx1], cv2.COLOR_BGR2HSV)
    if ((h2[..., 1] > 120) & (h2[..., 2] > 150)).mean() < 0.05:
        return None
    return val


def read_ladder(frame, need_1080=True):
    """래더 대기 화면에서 가장 아래(가장 최근) 라운드의 두 플레이어 점수.
    {"L": 왼쪽 점수 또는 None, "R": 오른쪽 점수 또는 None, "round", "bar_y"} 또는 None.
    내가 어느 쪽인지는 판마다 바뀌어서(내 카드가 왼쪽일 때도 오른쪽일 때도 있음) 여기서 정하지 않는다."""
    im = to_1080(frame) if need_1080 else frame
    bars = _round_bars(im)
    for k in range(len(bars) - 1, -1, -1):
        vl = _read_lscore(im, bars[k], "L")
        vr = _read_lscore(im, bars[k], "R")
        vl = vl if vl is not None and 0 < vl <= 1000000 else None
        vr = vr if vr is not None and 0 < vr <= 1000000 else None
        if vl is not None or vr is not None:
            return {"L": vl, "R": vr, "round": k + 1, "bar_y": bars[k]}
    return None


# ------------------------------------------------------------ 플레이 화면의 판정 종류 (래더·버서스)
# 래더·버서스 플레이 화면의 정보 패널: 위에 빨간 'ROUND n' 띠(폭 약 440), 원 아래에 'HARD JUDGEMENT' 빨간 띠 (하드일 때).
# 패널 위치는 기어 위치에 따라 다름 (기어 가운데 → 패널 왼쪽 x 80, 기어 오른쪽 → 패널 가운데 x 740, 기어 왼쪽 → 오른쪽)
JB_BAR_Y = (100, 160)                # ROUND 띠 세로 범위
JB_BADGE_DX = (125, 315)             # 패널 왼쪽 끝에서 판정 띠 가로 범위
JB_BADGE_Y = (670, 694)


def _hsv_frac(im, x0, y0, x1, y1):
    h = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    red = ((h[..., 0] <= 8) | (h[..., 0] >= 165)) & (h[..., 1] > 120) & (h[..., 2] > 120)
    wh = (h[..., 1] < 60) & (h[..., 2] > 200)
    return red, wh


def read_judge_badge(frame, need_1080=True):
    """래더·버서스 플레이 화면이면 'hard'(빨간 HARD JUDGEMENT 띠) / 'max'(자홍 MAX JUDGEMENT 띠) /
    'normal'(띠가 없음) / 'other'(모르는 색 띠), 아니면 None"""
    im = to_1080(frame) if need_1080 else frame
    y0, y1 = JB_BAR_Y
    red, wh = _hsv_frac(im, 0, y0, im.shape[1], y1)
    col = red[:10].mean(axis=0) > 0.5                    # 띠 맨 위 10줄(글자 없는 곳)로 띠의 가로 범위를 찾음
    runs, x = [], 0
    while x < len(col):                                  # 빨간 열이 이어진 구간 = ROUND 띠 후보
        if col[x]:
            xs = x
            while x < len(col) and col[x]:
                x += 1
            if 380 <= x - xs <= 500:
                runs.append((xs, x))
        x += 1
    found = None
    for xs, xe in runs:
        pw = wh[:, xs:xe].mean()
        if not (0.02 <= pw <= 0.2):                      # 띠 안에 흰 글자(ROUND n)
            continue
        bx0, bx1 = xs + JB_BADGE_DX[0], xs + JB_BADGE_DX[1]
        hb = cv2.cvtColor(im[JB_BADGE_Y[0]:JB_BADGE_Y[1], bx0:bx1], cv2.COLOR_BGR2HSV)
        sat = (hb[..., 1] > 120) & (hb[..., 2] > 120)
        w2 = ((hb[..., 1] < 60) & (hb[..., 2] > 200)).mean()
        if sat.mean() > 0.45 and 0.05 <= w2 <= 0.35:             # 색 띠 + 흰 글자
            hue = float(np.median(hb[..., 0][sat]))
            if hue <= 8 or hue >= 165:
                return "hard"                                     # HARD JUDGEMENT = 빨강
            if 138 <= hue <= 162:
                return "max"                                      # MAX JUDGEMENT = 자홍
        if sat.mean() < 0.08:
            return "normal"                                       # 노멀은 띠 자체가 없음
        found = "other"
    return found


def summary(vals):
    """(전체 노트, MAX 100% 아닌 개수(BREAK 제외), BREAK)
    vals = 일반 결과 화면 12개 [100%, 90% … 1%, BREAK] 또는 미션 결과 화면 3개 [100%, 1~90%, BREAK]"""
    if len(vals) == 3:
        return sum(vals), vals[1], vals[2]
    return sum(vals), sum(vals[1:11]), vals[11]
