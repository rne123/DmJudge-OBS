"""인디케이터에 '무엇을 그릴지' 계산 (그리는 곳과 무관).

게임 위 오버레이(indicator.py)와 OBS 브라우저 소스(obs_server.py)가 같은 계산을 쓴다.
indicator.py가 없는 OBS판에서도 이 파일만으로 동작한다.
"""
from __future__ import annotations

import time

# zone_frac: 가운데 띠(FAST/SLOW 범위)가 막대 반쪽에서 차지하는 비율 (고정 → 모양이 항상 같음)
# render: "alpha"(반투명, 기본) / "classic"(예전 방식), strength: 전체 진하기, density: 분포 곡선
DEFAULTS = {"enabled": True, "y_frac": 0.45, "zone_frac": 0.416, "seconds": 3.0,
            "width_frac": 0.45, "height": 58, "x": None, "y": None,
            "render": "alpha", "strength": 1.0, "density": True, "bg_alpha": 0.0,
            "flash_sec": 1.5, "text_scale": 1.0, "swap_colors": False,
            "bar": True}   # 막대 / FAST·SLOW 글자 / 모든 노트 ms

FS_MODES = ("off", "fs", "win")            # FAST/SLOW 글자: 표기 안 함 / 패슬 범위 밖 / 100% 범위 밖
MS_MODES = ("off", "all", "fs", "win")     # ms 숫자: 표기 안 함 / 모든 노트 / 패슬 범위 밖만 / 100% 범위 밖만


def opts_of(cfg):
    d = dict(DEFAULTS)
    d.update(cfg.get("indicator", {}))
    return d


def bar_range(cfg, fs):
    """막대 양끝 ±ms = FAST/SLOW 범위 ÷ 띠 비율 (범위를 바꾸면 양끝이 자동으로 따라감)"""
    return float(fs) / max(0.1, min(0.9, float(opts_of(cfg)["zone_frac"])))


def modes_of(o):
    """(FAST/SLOW 글자 방식, ms 방식). 예전 설정(fs_text, ms_all, fs_label_win)도 그대로 읽어 옮김."""
    fm, mm = o.get("fs_mode"), o.get("ms_mode")
    if fm not in FS_MODES:
        fm = "off" if not o.get("fs_text", True) else ("win" if o.get("fs_label_win") else "fs")
    if mm not in MS_MODES:
        mm = "all" if o.get("ms_all") else ("fs" if o.get("fs_text", True) else "off")
    return fm, mm


class IndModel:
    """세션 기록에서 지금 그릴 내용(점, 글자)을 계산. 다시 그릴 필요가 없으면 None."""

    def __init__(self, track_latency=False):
        self._last_sig = None
        self._last_draw = 0.0
        self._flash_on = False
        self._ms_lab = None
        self._seen_t = None
        self.track_latency = track_latency     # 누른 순간 → 화면 표시 지연 기록 (한 곳에서만)

    def invalidate(self):
        self._last_sig = None

    def frame(self, cfg, ses, force=False, moving=False):
        o = opts_of(cfg)
        now = time.perf_counter() * 1000.0
        js = float(cfg.get("judge_setting_ms", 0)) + float(cfg.get("calib_ms", 0.0))
        win = float(cfg.get("window_ms", 42))
        span = float(o["seconds"]) * 1000.0
        # 최근 몇 초 안에 친 노트만 (판정 중심 시각 기준)
        vis = []
        newest_t = None
        FL = max(0.2, float(o.get("flash_sec", 1.5))) * 1000.0   # 글자 표시 시간 (사용자 설정)
        look = max(span, FL)
        last = None                                             # 가장 최근 노트 (모든 노트 ms 표시)
        fs = ses.fs_range()
        lab_new = None                                          # 가장 최근의 범위 밖 노트 (FAST/SLOW)
        fm, mm = modes_of(o)
        lab_lim = None if fm == "off" else (max(fs, win) if fm == "win" else fs)   # FAST/SLOW 글자가 뜨는 기준
        ms_lim = None if mm in ("off", "all") else (max(fs, win) if mm == "win" else fs)  # ms 숫자가 뜨는 기준
        hits = ses.recent_hits(now + js - look - 100.0) if hasattr(ses, "recent_hits") \
            else ses.display_hits()
        for h in reversed(hits):
            if h[4] < 1.0:
                continue
            age = now - (h[0] - js)
            if age > look:
                break
            if last is None:
                last = (h[3] + js, max(0.0, age))
            if lab_new is None and lab_lim is not None and abs(h[3] + js) > lab_lim:
                lab_new = ("fast" if h[3] + js < 0 else "slow", h[0])
            if age > span:
                continue
            if newest_t is None:
                newest_t = h[0]
            vis.append((h[3] + js, max(0.0, age)))
        vis.reverse()
        if mm != "all" or (last is not None and last[1] >= FL):
            last = None
        # 모든 노트 ms: FAST/SLOW는 다음 FAST/SLOW가 나올 때까지 유지 (노트가 멈추면 숫자와 같이 사라짐)
        if last is None:
            self._ms_lab = None
        else:
            if lab_new is not None and (self._ms_lab is None or lab_new[1] >= self._ms_lab[1]):
                self._ms_lab = lab_new
            # FAST/SLOW 글자는 다음 노트가 와도 남아 있되, 자기가 뜬 뒤 "ms 글자 초"가 지나면 사라짐
            lab, lab_age = None, None
            if self._ms_lab is not None:
                lab_age = now - (self._ms_lab[1] - js)
                if lab_age < FL:
                    lab = self._ms_lab[0]
                else:
                    lab_age = None
            last = (last[0], last[1], lab, lab_age)
        fl = [(f[0], now - f[1], f[2] if len(f) > 2 else None,
               f[3] if len(f) > 3 else ("L" if f[0] == "fast" else "R"))
              for f in ses.flashes if 0 <= now - f[1] < FL]
        # 모든 노트 ms가 아니면: 범위 밖으로 친 노트마다 손 쪽에 잠깐 (글자·숫자는 각자 기준을 넘을 때만)
        from .ind_render import flash_parts
        fl2 = []
        for x in fl:
            jv = x[2]
            show_lab = lab_lim is not None and (jv is None or abs(jv) > lab_lim)
            show_ms = ms_lim is not None and jv is not None and abs(jv) > ms_lim
            if show_lab or show_ms:
                fl2.append(x[:4] + (flash_parts(x[0], jv, show_lab, show_ms),))
        fl = fl2
        # 왼손에서 난 것은 왼쪽, 오른손은 오른쪽에. 같은 쪽에서는 가장 최근 것만 (겹치지 않게)
        side_last = {}
        for x in fl:
            if x[3] not in side_last or x[1] < side_last[x[3]][1]:
                side_last[x[3]] = x
        fl = list(side_last.values())
        if mm == "all":
            fl = []                     # 모든 노트 ms 모드에서는 FAST/SLOW를 가운데 숫자 옆에 붙여 보여줌
        # 새 노트가 찍혔는지만 비교 (경과 시간은 넣지 않음 → 흐려지는 효과는 33ms마다만 다시 그림)
        sig = (len(vis), newest_t, vis[-1][0] if vis else None, len(fl), js, win, fs, o["zone_frac"], o["seconds"],
               last[0] if last else None, last[2] if last else None, o.get("bar", True), fm, mm,
               o.get("fade", "weak"), o.get("text_fade", "end"),
               o.get("text_scale", 1.0), o.get("swap_colors", False))
        # 점이 있으면 흐려지는 효과 때문에 계속 다시 그린다 (없을 때는 한 번만)
        busy = bool(vis) or bool(fl) or self._flash_on or last is not None
        if not force and not moving and sig == self._last_sig and (not busy or now - self._last_draw < 33):
            return None
        self._flash_on = bool(fl) or last is not None
        if moving and not force and now - self._last_draw < 100:
            return None
        self._last_sig = sig
        self._last_draw = now
        if self.track_latency:
            lt = ses.last_instant_t
            if lt is not None and lt != self._seen_t:
                self._seen_t = lt
                ses.lat.append(now - lt)
                if len(ses.lat) > 200:
                    del ses.lat[:100]
        return {"vis": vis, "R": bar_range(cfg, fs), "fs": fs, "win": win, "fl": fl, "o": o,
                "FL": FL, "last": last, "span": span}
