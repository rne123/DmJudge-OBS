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


def summary(vals):
    """(전체 노트, MAX 100% 아닌 개수(BREAK 제외), BREAK)"""
    return sum(vals), sum(vals[1:11]), vals[11]
