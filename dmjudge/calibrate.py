"""플레이 영역 / 판정선 지정 도구 + 실시간 인식 미리보기 (OpenCV 창).
OpenCV 창은 한글이 깨질 수 있어 화면 안내는 영어로 표시한다.
"""
from __future__ import annotations

import time

import cv2
import numpy as np



def _closed(win):
    """창 오른쪽 위 X로 닫았는지 (OpenCV는 X를 눌러도 루프가 계속 그리면 창을 다시 만든다)"""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _destroy(win):
    """창을 확실히 닫는다 (destroyWindow 뒤에 이벤트를 몇 번 처리해야 실제로 사라짐)"""
    try:
        cv2.destroyWindow(win)
    except cv2.error:
        pass
    for _ in range(5):
        cv2.waitKey(1)

def _fit(img, max_w=1500, max_h=850):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return s


def auto_judge_line(img, x0, x1, top, bottom):
    """영역 아래쪽 30%에서 가로로 길게 밝고 채도 있는 띠(판정선)를 찾아 중앙 y를 반환"""
    y_from = int(top + (bottom - top) * 0.7)
    y_to = min(img.shape[0] - 1, int(bottom + (bottom - top) * 0.1))
    reg = img[y_from:y_to, int(x0):int(x1)]
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0]
    score = ((hsv[..., 2] > 200) & (hsv[..., 1] > 70) & (H >= 22) & (H <= 40)).mean(1)   # 밝은 노란 판정선
    on = score > 0.8          # 화면 폭 거의 전체가 노란 줄 (노트·그림자 제외)
    best, bl = None, 0
    y = 0
    while y < len(on):
        if on[y]:
            s = y
            while y < len(on) and on[y]:
                y += 1
            if y - s > bl:
                best, bl = (s, y - 1), y - s
        y += 1
    if best is None:
        return None
    return y_from + (best[0] + best[1]) / 2.0


def run_calibration(cfg, grab_fn):
    """반환: geometry dict 또는 None(취소)"""
    img = grab_fn()
    if img is None:
        return None
    s = _fit(img)
    view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    win = "dmjudge - drag playfield (lane 1 left edge ~ last lane right edge, top ~ judge line), ENTER=ok, C=cancel"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    r = cv2.selectROI(win, view, showCrosshair=False, fromCenter=False)
    _destroy(win)
    if r is None or r[2] < 20 or r[3] < 50:
        return None
    x0, top, w, h = [v / s for v in r]
    x1, bottom = x0 + w, top + h
    jy = auto_judge_line(img, x0, x1, top, bottom) or bottom
    occ = jy - (jy - top) * 0.17

    win2 = "dmjudge - W/S: judge line, E/D: tracking limit (Shift = x5), A: auto, ENTER: save, ESC: cancel"
    shown2 = False
    cv2.namedWindow(win2, cv2.WINDOW_AUTOSIZE)
    result = None
    while True:
        pad = 40
        ya, yb = int(max(0, top - pad)), int(min(img.shape[0], jy + 120))
        xa, xb = int(max(0, x0 - pad)), int(min(img.shape[1], x1 + pad))
        crop = img[ya:yb, xa:xb].copy()
        sc = min(1.0, 820 / crop.shape[0])
        disp = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        def Y(y): return int((y - ya) * sc)
        def X(x): return int((x - xa) * sc)
        cv2.rectangle(disp, (X(x0), Y(top)), (X(x1), Y(jy)), (0, 255, 0), 1)
        cv2.line(disp, (X(x0) - 15, Y(occ)), (X(x1) + 15, Y(occ)), (255, 0, 255), 1)
        cv2.line(disp, (X(x0) - 15, Y(jy)), (X(x1) + 15, Y(jy)), (0, 0, 255), 1)
        # 판정선 확대
        zy0, zy1 = int(jy - 25), int(jy + 25)
        z = img[max(0, zy0):zy1, int(x0):int(x1)].copy()
        if z.size:
            zf = max(2.0, min(6.0, 420.0 / max(1, z.shape[1]) * 2))
            # 가운데 부분만 크게 (판정선 두께를 보기 위함)
            zw = int(420 / zf)
            cx = z.shape[1] // 2
            z = z[:, max(0, cx - zw // 2): cx + zw // 2]
            z = cv2.resize(z, None, fx=zf, fy=zf, interpolation=cv2.INTER_NEAREST)
            cy = int((jy - max(0, zy0)) * zf)
            cv2.line(z, (0, cy), (z.shape[1], cy), (0, 0, 255), 1)
            zz = np.zeros((disp.shape[0], z.shape[1] + 10, 3), np.uint8)
            zz[:min(z.shape[0], zz.shape[0]), 10:] = z[:zz.shape[0]]
            disp = np.concatenate([disp, zz], 1)
        txt = "judge line y=%.1f  tracking limit y=%.1f  (red = judge line center, magenta = limit)" % (jy, occ)
        cv2.putText(disp, txt, (8, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if shown2 and _closed(win2):        # X로 닫음 = 취소
            break
        cv2.imshow(win2, disp)
        shown2 = True
        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        ch = k & 0xFF
        if ch == 27:
            break
        if ch in (13, 10):
            result = {"x0": float(x0), "x1": float(x1), "top": float(top),
                      "occl_y": float(occ), "judge_y": float(jy)}
            break
        step = 5 if chr(ch).isupper() else 1
        c = chr(ch).lower()
        if c == "w":
            jy -= step * 0.5
        elif c == "s":
            jy += step * 0.5
        elif c == "e":
            occ -= step
        elif c == "d":
            occ += step
        elif c == "a":
            jy = auto_judge_line(img, x0, x1, top, bottom) or jy
    _destroy(win2)
    return result


def run_preview(cfg, seconds=600):
    """실시간 인식 미리보기: 초록=레인 노트, 주황=트리거, 하늘=사이드, 빨강선=판정선 정중앙"""
    import dxcam
    from .capture import _region_and_geom, colors_of
    from .tracker import NoteTracker, LaneCounter

    region, geom = _region_and_geom(cfg)
    cam = dxcam.create(output_idx=int(cfg.get("monitor", 0)), output_color="BGR")
    mode = cfg.get("mode", "auto")
    lanes = 6 if mode == "auto" else int(cfg["modes"][mode]["lanes"])
    tr = NoteTracker(geom, colors_of(cfg), lanes=lanes)
    lc = LaneCounter(geom, colors_of(cfg))
    win = "dmjudge preview - ESC / Q / X to close"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    t_end = time.time() + seconds
    flashes = []
    k = 0
    while time.time() < t_end:
        f = cam.grab(region=region)
        if f is None:
            if cv2.waitKey(1) == 27 or (k > 0 and _closed(win)):
                break
            time.sleep(0.002)
            continue
        t = time.perf_counter() * 1000
        k += 1
        if k % 2 == 0:
            lc.feed(f)
        g = lc.guess(400)
        if mode == "auto" and g and g != tr.nl:
            tr = NoteTracker(geom, colors_of(cfg), lanes=g)
        blobs = tr.detect(f)
        for ch, ta, conf, lg in tr.update(f, t):
            flashes.append((t, ch, conf))
        d = f.copy()
        x0, x1 = int(geom["x0"]), int(geom["x1"])
        lw = (x1 - x0) / tr.nl
        for ch, tp, bt, ct, cb in blobs:
            if ch.startswith("L"):
                i = int(ch[1:]) - 1
                xa, xb, col = int(x0 + lw * i), int(x0 + lw * (i + 1)), (0, 255, 0)
            else:
                half = 0 if ch.endswith("L") else 1
                xa = x0 if half == 0 else (x0 + x1) // 2
                xb = (x0 + x1) // 2 if half == 0 else x1
                col = (0, 140, 255) if ch.startswith("T") else (255, 255, 0)
            cv2.rectangle(d, (xa + 2, tp), (xb - 2, bt), col, 1)
        cv2.line(d, (x0, int(geom["occl_y"])), (x1, int(geom["occl_y"])), (255, 0, 255), 1)
        cv2.line(d, (x0, int(round(geom["judge_y"]))), (x1, int(round(geom["judge_y"]))), (0, 0, 255), 1)
        flashes = [x for x in flashes if t - x[0] < 150]
        for ft, ch, conf in flashes:
            if ch.startswith("L"):
                i = int(ch[1:]) - 1
                xa, xb = int(x0 + lw * i), int(x0 + lw * (i + 1))
            else:
                xa, xb = (x0, (x0 + x1) // 2) if ch.endswith("L") else ((x0 + x1) // 2, x1)
            jy = int(round(geom["judge_y"]))
            cv2.rectangle(d, (xa + 3, jy - 4), (xb - 3, jy + 4), (0, 0, 255) if conf >= 1 else (128, 128, 128), -1)
        info = "lanes:%d  speed:%.3f px/ms  head:%.0fpx" % (tr.nl, tr.speed or 0, tr.head_h)
        cv2.putText(d, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if k > 1 and _closed(win):          # X로 닫음
            break
        cv2.imshow(win, d)
        if cv2.waitKey(1) in (27, ord("q")):
            break
    _destroy(win)
    try:
        del cam
    except Exception:
        pass
