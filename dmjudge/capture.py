"""화면 캡처 + 노트 추적 (별도 프로세스에서 실행).

dxcam(Desktop Duplication)으로 플레이 영역만 캡처한다.
주사율은 따로 설정하지 않는다: 새 프레임이 나올 때마다 시각을 찍으므로 60/144/240Hz 모두 자동.
메시지: ("arr", lanes, ch, t_ms, conf, is_long), ("lanes", n, t), ("seg", t),
        ("cap", fps, speed, head_h, proc_ms_p95, 건너뛴 프레임 수, 캡처 스레드 사용), ("error", msg),
        ("result", [MAX 100%, 90%, ..., 1%, BREAK], t) — 곡이 끝나고 결과 화면 숫자를 읽었을 때,
        ("songend", t) / ("songstart", t) — 기어의 판정선이 사라짐 / 다시 보임
"""
from __future__ import annotations

import math
import time


def capture_process_main(q, stop_evt, cfg, cmd_q=None):
    try:
        import ctypes
        # 보통보다 높음(게임을 방해하지 않을 정도)
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
    try:
        _run(q, stop_evt, cfg, cmd_q)
    except Exception as e:  # pragma: no cover
        import traceback
        q.put(("error", "캡처 오류: %r\n%s" % (e, traceback.format_exc())))


class Grabber:
    """캡처 전용 스레드: 새 프레임만 기다리다가 나오는 순간 시각을 찍는다.

    예전에는 분석이 끝난 뒤에 grab()을 불러서, 분석이 프레임 간격(144Hz = 6.9ms)만큼 걸리면
    이미 몇 ms 지난 프레임에 '지금' 시각이 찍혔다 → 노트 도달 시각이 늦게, 들쭉날쭉하게 잡힘.
    이제 분석 속도와 상관없이 시각은 '프레임이 처음 보인 순간' (직전 빈 확인과 이번 확인의 중간).
    """

    def __init__(self, grab_fn):
        import threading
        self.grab_fn = grab_fn
        self.cond = threading.Condition()
        self.frame = None
        self.t = 0.0
        self.seq = 0
        self.stop = False
        self.err = None
        self.full_fn = None        # 결과 화면 확인용 전체 화면 캡처 (요청이 있을 때만 한 장)
        self.full_req = False
        self.full = None
        self.th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.th.start()

    def _loop(self):
        try:
            import comtypes
            comtypes.CoInitializeEx(0)
        except Exception:
            pass
        prev_empty = None                       # 직전에 '새 프레임 없음'을 확인하기 시작한 시각
        while not self.stop:
            if self.full_req and self.full_fn is not None:
                try:
                    ff = self.full_fn()
                except Exception:
                    ff = None
                if ff is not None:
                    self.full = ff.copy()
                    self.full_req = False
                    continue
            t0 = time.perf_counter()
            try:
                f = self.grab_fn()
            except Exception as e:              # pragma: no cover
                self.err = e
                with self.cond:
                    self.cond.notify_all()
                return
            if f is None:
                prev_empty = t0
                time.sleep(0.0002)
                continue
            # 프레임은 prev_empty ~ t0 사이 어딘가에 나왔다 → 가운데로 추정 (간격이 넓으면 t0)
            if prev_empty is not None and t0 - prev_empty < 0.003:
                tf = (prev_empty + t0) / 2.0
            else:
                tf = t0
            prev_empty = None
            if not getattr(f, "flags", None) or not f.flags.owndata:
                f = f.copy()                    # 캡처 라이브러리 버퍼를 다음 grab이 덮어써도 안전하게
            with self.cond:
                self.frame, self.t = f, tf
                self.seq += 1
                self.cond.notify_all()

    def get(self, last_seq, timeout=0.05):
        """last_seq 다음 프레임 중 가장 최근 것: (seq, frame, t_sec, 건너뛴 수) 또는 None"""
        with self.cond:
            if self.seq == last_seq and self.err is None:
                self.cond.wait(timeout)
            if self.err is not None:
                raise self.err
            if self.seq == last_seq:
                return None
            return self.seq, self.frame, self.t, self.seq - last_seq - 1


def colors_of(cfg):
    c = dict(cfg["colors"])
    if cfg.get("dark_notes"):
        c["v_min"] = min(int(c.get("v_min", 140)), 90)
    c["ring_merge"] = bool(cfg.get("ring_notes", False))     # 동그란(고리) 노트 스킨: 위·아래 테두리 조각 합치기
    return c


BIG_EFFECT_FRAC = 0.30


def _region_and_geom(cfg):
    g = cfg["geometry"]
    pad = 2
    left = int(g["x0"]) - pad
    top = int(g["top"])
    right = int(g["x1"]) + pad
    bottom = int(g["judge_y"]) + 4
    occl = g["occl_y"]
    if cfg.get("big_effects"):
        # 판정 이펙트가 큰 스킨(판정선 위로 높이 솟는 캔 모양 등): 추적 한계선을 영역 높이 30% 지점까지 올림
        occl = min(occl, g["judge_y"] - (g["judge_y"] - g["top"]) * BIG_EFFECT_FRAC)
    geom = {
        "x0": g["x0"] - left, "x1": g["x1"] - left,
        "top": 0, "occl_y": occl - top, "judge_y": g["judge_y"] - top,
    }
    ind = indicator_rect(cfg)
    if ind is not None and ind[0] < g["x1"] and ind[0] + ind[2] > g["x0"]:
        geom["mask"] = (ind[1] - top - 2, ind[1] + ind[3] - top + 2)
    return (left, top, right, bottom), geom


def indicator_size(o):
    """(기본 높이, 실제 창 높이, 위로 늘린 만큼): 글자가 크면 창을 위로 늘리고 막대 줄은 그대로 둔다"""
    from .ind_render import text_px
    h0 = int(o["height"])
    P = text_px(o)
    h = max(h0, int(math.ceil((16 + 1.1 * P) / 0.62)) + 2)
    up = int(round((h - h0) * 0.62))
    return h0, h, up


def indicator_rect(cfg):
    """인디케이터 창의 화면 위치 (x, y, w, h). 꺼져 있으면 None.
    드래그로 옮긴 적이 있으면 그 위치(x, y), 없으면 플레이 영역 기준 기본 위치."""
    from .ind_state import opts_of
    from .edition import is_obs
    g = cfg.get("geometry")
    o = opts_of(cfg)
    if not g or not o["enabled"] or is_obs():      # OBS판: 게임 위에 띄우는 창이 없음
        return None
    h0, h, up = indicator_size(o)
    fw = g["x1"] - g["x0"]
    w = int(fw * max(0.2, min(1.5, float(o["width_frac"]))))
    if o.get("x") is not None and o.get("y") is not None:
        cx = float(o["x"])                     # 저장된 값 = 막대 가운데 x, 위쪽 y (기본 글자 크기 기준)
        return int(cx - w / 2), int(o["y"]) - up, w, h
    y = int(g["judge_y"] - (g["judge_y"] - g["top"]) * float(o["y_frac"]) - h0 / 2) - up
    x = int(g["x0"] + (fw - w) / 2)
    return x, y, w, h


class FrameDump:
    """진단용: 최근 1.2초 동안 분석기가 본 화면(플레이 영역, 3px 간격 세로줄)을 메모리에 두었다가
    요청이 오면 그 순간 앞뒤 프레임을 파일로 저장한다 (누르는 순간 예상과 확정값이 크게 다를 때)."""

    def __init__(self, geom, out_dir, keep_ms=1200.0, max_files=15):
        import collections
        self.top, self.occ = int(geom["top"]), int(geom["occl_y"])
        self.bot = int(round(geom["judge_y"])) + 4
        self.buf = collections.deque()
        self.keep = keep_ms
        self.pending = []
        self.out_dir = out_dir
        self.left = max_files
        self.last_req = -1e9
        self.geom = dict(geom)

    def feed(self, frame, t):
        self.buf.append((t, frame[self.top:self.bot, ::3].copy()))
        while self.buf and t - self.buf[0][0] > self.keep:
            self.buf.popleft()
        if self.pending and t > self.pending[0] + 300.0:
            tr = self.pending.pop(0)
            sel = [(tt, f) for tt, f in self.buf if tr - 900.0 <= tt <= tr + 300.0]
            if sel and self.left > 0:
                self.left -= 1
                self._save(tr, sel)

    def request(self, t):
        if self.left > 0 and t - self.last_req > 1500.0:
            self.last_req = t
            self.pending.append(t)

    def _save(self, tr, sel):
        import os
        import threading
        import numpy as np
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, "dump_%d.npz" % int(tr))
        ts = np.array([x[0] for x in sel])
        fr = np.stack([x[1] for x in sel])
        g = self.geom

        def w():
            try:
                np.savez_compressed(path, t=ts, frames=fr, t_req=tr, step=3, top=self.top,
                                    geom=np.array([g["x0"], g["x1"], g["top"], g["occl_y"], g["judge_y"]]))
            except Exception:
                pass
        threading.Thread(target=w, daemon=True).start()


def _run(q, stop_evt, cfg, cmd_q=None):
    import dxcam
    from .tracker import NoteTracker, LaneCounter, FastSlowText, JudgeLineWatch

    region, geom = _region_and_geom(cfg)
    cam = dxcam.create(output_idx=int(cfg.get("monitor", 0)), output_color="BGR")
    if cam is None:
        q.put(("error", "화면 캡처 장치를 만들 수 없어요 (모니터 번호 확인)"))
        return
    mode = cfg.get("mode", "auto")
    if mode == "auto":
        lane_set = [4, 5, 6]          # 판별 전까지 세 가지를 동시에 분석
    else:
        lane_set = [int(cfg["modes"][mode]["lanes"])]
    def make_all():
        return {n: NoteTracker(geom, colors_of(cfg), lanes=n) for n in lane_set}
    trackers = make_all()
    counter = LaneCounter(geom, colors_of(cfg)) if mode == "auto" else None
    bias = float(cfg.get("capture_bias_ms", 0.0))
    fstext = FastSlowText(geom, cfg.get("fs_text"))       # 게임 FAST/SLOW 글자 (보정 추천용)
    jwatch = JudgeLineWatch(geom)                          # 판정선이 사라지면 곡 끝
    dumper = None
    if cfg.get("frame_dump", True) and cmd_q is not None:
        try:
            from .config import app_dir
            import os
            dumper = FrameDump(geom, os.path.join(app_dir(), "dumps"))
        except Exception:
            dumper = None
    q.put(("info", "캡처 시작: 영역 %s" % (region,)))

    n, t_rep, k = 0, time.perf_counter(), 0
    proc = []
    last_arr = None
    guessed = None
    threaded = bool(cfg.get("capture_thread", True))
    grabber = None
    if threaded:
        import sys
        sys.setswitchinterval(0.0005)        # 분석 중에도 캡처 스레드가 바로 깨어나게
        grabber = Grabber(lambda: cam.grab(region=region))
        grabber.full_fn = lambda: cam.grab()
        grabber.start()
    seq, skipped = 0, 0
    last_fs_t = -1e9
    # 곡 끝 → 결과 화면 숫자 읽기: 노트(확실한 도달)가 100개 넘게 나온 뒤 1.5초 넘게 끊기면 0.4초마다 전체 화면 확인
    from .result_ocr import read as read_result
    rs = {"n": 0, "last": None, "next": 0.0, "prev": None}
    # [기록 저장]이 켜져 있으면 확정된 결과 화면 한 장을 메인으로 넘긴다 (곡이 끝난 뒤 한 번, 플레이 중 부하 없음)
    rec = {"on": bool(cfg.get("record_save", False))}

    def drain_cmds():
        if cmd_q is None:
            return
        try:
            while True:
                cmd = cmd_q.get_nowait()
                if cmd[0] == "dump" and dumper is not None:
                    dumper.request(cmd[1])
                elif cmd[0] == "recsave":
                    rec["on"] = bool(cmd[1])
        except Exception:
            pass

    def check_result():
        now = time.perf_counter() * 1000.0
        if rs["n"] < 100 or rs["last"] is None or now - rs["last"] < 1500.0:
            return
        if now - rs["last"] > 120000.0:          # 2분 넘게 결과 화면을 못 찾으면 포기
            rs["n"] = 0
            return
        full = None
        if grabber is not None:
            if grabber.full is not None:
                full, grabber.full = grabber.full, None
            elif now >= rs["next"] and not grabber.full_req:
                grabber.full_req = True
                rs["next"] = now + 400.0
        elif now >= rs["next"]:
            rs["next"] = now + 400.0
            full = cam.grab()
        if full is None:
            return
        try:
            vals = read_result(full)
        except Exception:
            vals = None
        if vals is None:
            rs["prev"] = None
            return
        if vals == rs["prev"]:                  # 숫자가 올라가는 애니메이션이 끝나 두 번 연속 같을 때 확정
            q.put(("result", vals, now, full if rec["on"] else None))
            rs["n"], rs["prev"] = 0, None
        else:
            rs["prev"] = vals

    while not stop_evt.is_set():
        check_result()
        if grabber is not None:
            got_f = grabber.get(seq)
            if got_f is None:
                continue
            seq, frame, tf, sk = got_f
            skipped += sk
            t0 = time.perf_counter()
            t = tf * 1000.0 + bias
        else:
            frame = cam.grab(region=region)
            if frame is None:
                time.sleep(0.0005)
                continue
            t0 = time.perf_counter()
            t = t0 * 1000.0 + bias
        got = False
        gate = jwatch.ended          # 곡이 끝나 판정선이 없는 동안: 화면에 잡히는 것은 노트로 치지 않음 (결과 화면 전환 그림 등)
        n_gated = 0                  # 그동안 버린 확실한 도달 수 (꾸준히 나오면 곡이 다시 시작된 것으로 봄)
        for lanes, tr in trackers.items():
            evs = tr.update(frame, t)
            if gate:
                n_gated = max(n_gated, sum(1 for e_ in evs if e_[2] >= 1))
                continue
            for (ch, ta, conf, is_long), nid in zip(evs, tr.last_ids):
                q.put(("arr", lanes, ch, ta, conf, bool(is_long), nid))
                if conf >= 1:
                    got = True
                    rs["n"] += 1
                    rs["last"] = time.perf_counter() * 1000.0
            if tr.vchanges:
                for tv, v0, v1 in tr.vchanges:
                    q.put(("vchg", lanes, tv, v0, v1))
                tr.vchanges = []
            pr = tr.predictions(t)
            if (pr or evs) and not gate:
                # 게임의 FAST/SLOW 글자가 떠 있는 동안(약 0.8초)은 그 글자가 지나가는 노트와 붙어 보여
                # 위치가 몇 ms 틀어질 수 있으므로 '믿을 만한 예상'에서 뺀다
                txt_ok = t - last_fs_t > 800.0
                q.put(("pred", lanes, [(x[0], x[1], x[2], bool(x[3]),
                                        bool(x[4]) and txt_ok if len(x) > 4 else False) for x in pr]))
        try:
            tr0_ = next(iter(trackers.values()))
            covered = jwatch.cover_from(getattr(tr0_, "last_blobs", ()), t)
            jev = jwatch.feed(frame, t, got, covered, n_gated)
        except Exception:
            jev = None
        if jev == "end":
            q.put(("songend", t))
            if rs["n"] >= 100 and rs["last"] is not None:      # 결과 화면 확인을 바로 시작
                rs["last"] = min(rs["last"], time.perf_counter() * 1000.0 - 1500.0)
        elif jev == "start":
            q.put(("songstart", t))
        for kind, te in fstext.feed(frame, t):
            q.put(("fstext", kind, te))
            last_fs_t = te
        if dumper is not None:
            dumper.feed(frame, t)
            drain_cmds()
        if got:
            last_arr = t
        # 레인 수 판별은 실제로 노트가 흘러가는 동안만 (결과 화면·선곡 화면의 그림을 노트로 착각해
        # 다음 곡 전체가 엉뚱한 모드로 분석되던 문제 방지)
        if counter is not None and last_arr is not None and t - last_arr < 3000:
            k += 1
            if k % 2 == 0:
                counter.feed(frame)
            if k % 30 == 0:
                g = counter.guess()
                q.put(("lc", len(counter.samples), counter.last_scores))
                if g is not None and g != guessed:
                    guessed = g
                    q.put(("lanes", g, t))
                    if g in trackers:            # 판별되면 나머지 분석은 끈다 (처리 시간 절약)
                        trackers = {g: trackers[g]}
        if counter is not None:
            # 노트가 4초 이상 없으면 곡이 끝난 것으로 보고 레인 판별을 새로 시작
            if last_arr is not None and t - last_arr > 4000:
                counter.reset()
                guessed = None
                last_arr = None
                trackers = make_all()
                q.put(("seg", t))
        n += 1
        now = time.perf_counter()
        proc.append((now - t0) * 1000.0)
        if now - t_rep >= 0.5:
            if dumper is None:
                drain_cmds()
            tr0 = trackers[guessed] if guessed in trackers else next(iter(trackers.values()))
            fps = (n + skipped) / (now - t_rep)         # 화면 프레임 수 (분석이 건너뛴 것 포함)
            p95 = sorted(proc)[int(len(proc) * 0.95)] if proc else 0.0
            q.put(("cap", fps, tr0.speed or 0.0, tr0.head_h, p95, skipped, threaded))
            n, t_rep, proc, skipped = 0, now, [], 0
    if grabber is not None:
        grabber.stop = True
        grabber.th.join(timeout=1)
    try:
        del cam
    except Exception:
        pass


def grab_fullscreen(monitor=0):
    """보정 도구용 전체 화면 1장."""
    import dxcam
    cam = dxcam.create(output_idx=int(monitor), output_color="BGR")
    img = None
    for _ in range(50):
        img = cam.grab()
        if img is not None:
            break
        time.sleep(0.02)
    del cam
    return img
