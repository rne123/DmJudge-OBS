"""측정 세션: 키 입력·도달 이벤트를 받아 모드별로 매칭하고 통계를 만든다 (플랫폼 독립)."""
from __future__ import annotations

import csv
import time
from collections import defaultdict

import numpy as np

from .analysis import Matcher, summarize, histogram, suggest_zero, fit_zero
from .config import MODES, mode_channels

LANES_OF = {"4B": 4, "5B": 5, "6B": 6, "8B": 6}



GAME_MODES = ["프리스타일", "래더", "버서스 매치", "미션"]


def group_key(mode, gmode):
    """영점 그룹 이름: '6B · 프리스타일'"""
    return "%s · %s" % (mode or "?", gmode or "프리스타일")

def split_songs(hits, gap=6000.0):
    """짝지어진 기록(t, mode, ch, e, conf)을 시간순으로 놓고 gap ms 넘게 끊긴 곳에서 곡을 나눔 (신뢰도 무관)"""
    hs = sorted(hits, key=lambda h: h[0])
    groups, cur = [], []
    for h in hs:
        if cur and h[0] - cur[-1][0] > gap:
            groups.append(cur)
            cur = []
        cur.append(h)
    if cur:
        groups.append(cur)
    return groups


class Session:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rebuild()

    # ------------------------------------------------------------ 구성
    def update_keymap(self):
        """키 설정만 바뀐 경우: 기록은 유지하고 키→레인 표만 다시 만든다"""
        self.key_to_ch = {}
        for m in self.candidates:
            d = {}
            for ch, keys in self.cfg["modes"][m]["keymap"].items():
                for k in ([keys] if isinstance(keys, str) else keys):
                    d[k] = ch
            self.key_to_ch[m] = d
        self._game_keys = self._all_game_keys()

    def _all_game_keys(self):
        """모든 모드의 키 설정에 들어 있는 키 (게임에 쓰는 키). 이 밖의 키 입력은 기록하지 않는다."""
        ks = set()
        for mc in self.cfg.get("modes", {}).values():
            for keys in mc.get("keymap", {}).values():
                ks.update([keys] if isinstance(keys, str) else keys)
        return ks

    def rebuild(self):
        cfg = self.cfg
        fixed = cfg.get("mode", "auto")
        self.candidates = MODES if fixed == "auto" else [fixed]
        self.key_to_ch = {}
        for m in self.candidates:
            km = cfg["modes"][m]["keymap"]
            d = {}
            for ch, keys in km.items():
                if isinstance(keys, str):
                    keys = [keys]
                for k in keys:
                    d[k] = ch
            self.key_to_ch[m] = d
        self._game_keys = self._all_game_keys()
        self.matchers = {m: self._new_matcher() for m in self.candidates}
        self.past = []              # 지난 곡들의 확정 기록: (t, mode, ch, e, conf)
        self.past_missed = defaultdict(int)
        self.seg_start = 0.0
        self.seg_trigger = False
        self.lanes_guess = None
        self.vchg = []             # 변속 감지 [(t, lanes, 이전 속도, 새 속도)]
        self._suspect = set()      # 즉시값과 확정값이 크게 달라 믿지 않는 노트 id
        self.arr_log = []          # 그래프(밀도)용 노트 도달 [(t, lanes)]
        self.song_end = None       # 결과 화면으로 끝난 곡의 끝 시각 (그래프는 이 곡을 보여줌)
        self.song_t0 = -1e18       # 지금 곡의 시작 경계 (도달이 4초 넘게 끊겼다가 다시 나온 때 기준)
        self._last_arr = None
        self.chatter = defaultdict(int)
        self._last_up = {}
        self.paused = False
        self.flashes = []          # (kind, 시각, 영점 기준 오차 ms, 'L'/'R' 띄울 쪽)
        self._ev_idx = {}
        self.pred = {}             # lanes → {nid: (ch, 예상 도달 시각)}
        self.pred_ok = {}          # lanes → 예상이 믿을 만한(스크롤을 실제로 보고 있는 중) 노트 번호
        self._inst_flash = {}      # nid → 누르는 순간 띄운 글자 (확정값이 범위 안이면 지움)
        self._last_kid = {}        # (모드, 채널) → 마지막으로 누른 키 (5B 가운데 레인이 어느 손인지)
        self._inst_rel = {}        # nid → 누르는 순간 예상이 '믿을 만한' 것이었는지
        self.dump_requests = []    # 화면 저장 요청 (도달 시각) → 캡처 프로세스로
        self._hit_idx = {}
        self._long_nids = set()
        self.last_instant_t = None
        # 게임 FAST/SLOW 글자 ↔ 분석기 기록 비교 (어긋나면 보정 추천)
        self.fs_events = []        # (kind, 시각) 아직 연결 전
        self.text_links = {}       # (ch, 누른 시각) → "fast"/"slow"
        self.long_ch = {}          # 채널 → 지금 누르고 있는 게 롱노트인지
        self.long_ups = []         # 롱노트를 뗀 시각 (뗄 때 뜨는 LATE는 판단에서 제외)
        self.suggest = None        # (추천 보정값, 어긋난 FAST 수, 어긋난 SLOW 수)
        self._suggest_t = 0.0
        self._dismissed = 0
        self.lat = []              # 키 입력 → 인디케이터 표시까지 (ms)
        self.instant = {}          # nid → 키를 누른 순간 예상 도달 기준으로 계산한 기록 (인디케이터용)
        self._flashed = set()
        if not hasattr(self, "raw"):
            self.raw = []          # 진단용 원본 기록 (초기화해도 유지, [CSV 저장] 때 _raw.csv로)
        if not hasattr(self, "results"):
            self.results = load_results()   # 결과 화면으로 모은 곡들 (파일에 저장, 영점 추천용)
            self.zero = None
            self.zero_groups = {}
            self._zero_dismissed = 0
            self.update_zero()
        if not hasattr(self, "last_song"):
            self.last_song = None  # 마지막으로 끝난 곡의 화면 기준 오차 (게임 결과 보정용)
            self.calib_songs = []  # 게임 결과 보정에 모은 곡: (오차 배열, 판정 타이밍, 100% 아닌 개수)

    def reset(self):
        self.rebuild()
        self.log("reset")

    def log(self, *ev):
        """진단용 원본 기록 (최근 40만 개까지)"""
        self.raw.append((round(time.perf_counter() * 1000.0, 2),) + ev)
        if len(self.raw) > 400000:
            del self.raw[:100000]

    # ------------------------------------------------------------ 이벤트
    def on_key(self, kid, is_down, t):
        if kid not in self._game_keys:
            return      # 게임 키가 아니면 기록도 처리도 하지 않음 (채팅·비밀번호 등 다른 입력이 _raw.csv에 남지 않게)
        self.log("key", kid, int(bool(is_down)), round(t, 2))
        if self.paused:
            return
        if is_down:
            for m_, d_ in self.key_to_ch.items():
                ch_ = d_.get(kid)
                if ch_ is not None:
                    self._last_kid[(m_, ch_)] = kid
        if is_down:
            lu = self._last_up.get(kid)
            if lu is not None and t - lu < float(self.cfg.get("chatter_ms", 25)):
                self.chatter[kid] += 1
        else:
            self._last_up[kid] = t
        for m, mt in self.matchers.items():
            mt.on_key(kid, is_down, t, self.key_to_ch[m].get(kid))
        if is_down:
            self._instant(kid, t)
        else:
            m = self.active_mode()
            ch = self.key_to_ch[m].get(kid) if m else None
            if ch is not None and self.long_ch.pop(ch, False):
                self.long_ups.append(t)
                if len(self.long_ups) > 50:
                    self.long_ups = self.long_ups[-50:]

    def on_pred(self, lanes, items):
        self.pred[lanes] = {it[0]: (it[1], it[2]) for it in items}
        self.pred_ok[lanes] = {it[0] for it in items if len(it) > 4 and it[4]}
        for it in items:
            if len(it) > 3 and it[3]:
                self._long_nids.add(it[0])
        if len(self._long_nids) > 3000:
            self._long_nids = set(list(self._long_nids)[-1000:])

    def on_fstext(self, kind, t):
        self.log("text", kind, round(t, 2))
        self.fs_events.append((kind, t))

    def _link_fstext(self, now):
        """게임이 FAST/SLOW 글자를 띄운 순간 → 그 직전에 누른 노트와 연결 (롱노트를 뗄 때 뜬 글자는 제외)"""
        if not self.fs_events:
            return
        keep, hits = [], None
        c = self._center()
        for kind, te in self.fs_events:
            if now - te < 150:
                keep.append((kind, te))
                continue
            if any(te - 100 <= u <= te + 10 for u in self.long_ups):
                continue                            # 롱노트 뗄 때의 판정일 수 있음
            if hits is None:
                hits = self.display_hits()
            best, bj = None, None
            for h in reversed(hits):
                tp = h[0] + h[3]
                if tp < te - 90:
                    break
                if tp > te + 10 or h[4] < 1.0:
                    continue
                j = tp - (h[0] + c)
                if bj is None or (j > bj if kind == "slow" else j < bj):
                    best, bj = h, j
            if best is not None:
                self.text_links[(best[2], round(best[0] + best[3], 1))] = kind
        self.fs_events = keep

    def _check_mismatch(self, now):
        """게임은 SLOW(FAST)인데 분석기 기록은 범위 안인 노트가 있으면 보정을 추천"""
        if now - self._suggest_t < 1000:
            return
        self._suggest_t = now
        if not self.text_links:
            self.suggest = None
            return
        js = float(self.cfg.get("judge_setting_ms", 0))
        cb = float(self.cfg.get("calib_ms", 0.0))
        W = float(self.cfg.get("window_ms", 41.6))
        hs = [h for h in self.all_hits() if h[4] >= 1.0]
        lab = np.array([{"fast": -1, "slow": 1}.get(self.text_links.get((h[2], round(h[0] + h[3], 1))), 0)
                        for h in hs])
        j_raw = np.array([h[3] + js for h in hs])
        jn = j_raw + cb
        mf = int(((lab < 0) & (jn >= -W)).sum())       # 게임 FAST인데 분석기는 범위 안
        ms = int(((lab > 0) & (jn <= W)).sum())        # 게임 SLOW인데 분석기는 범위 안
        # 양쪽으로 비슷하게 어긋나면 영점 문제가 아니라 짝짓기 오차(롱노트 뗄 때 글자 등)이므로 추천 안 함
        if mf + ms < 2 or mf + ms <= self._dismissed or (min(mf, ms) > 0 and max(mf, ms) < 2 * min(mf, ms)):
            self.suggest = None
            return
        b = suggest_zero(j_raw, lab, W)
        if b is None or abs(b - cb) < 0.5:
            self.suggest = None
            return
        self.suggest = (b, mf, ms)

    def apply_suggestion(self):
        if self.suggest is not None:
            self.cfg["calib_ms"] = round(self.suggest[0], 1)
            cap = "thread" if self.cfg.get("capture_thread", True) else "old"
            self.cfg.setdefault("calib_by_capture", {})[cap] = self.cfg["calib_ms"]
        self.suggest = None

    def dismiss_suggestion(self):
        self._zero_dismissed = len(self.results)
        self.suggest = None

    # ------------------------------------------------------------ 결과 화면 → 곡 기록 → 영점 추천
    def on_result(self, vals, t):
        """곡이 끝나고 결과 화면 숫자를 읽었을 때: 그 곡 기록과 짝지어 저장. 반환: 저장한 기록 또는 None"""
        self.log("result", *vals)
        if getattr(self, "_last_result_t", -1e18) > t - 5000.0:
            return None
        m = self.active_mode()
        if m is None:
            return None
        g = self._last_song_hits(t)
        if g is None or len(g[2]) < 100:
            return None
        hits = g[2]
        self._last_result_t = t
        self.song_end = (g[0], g[1])
        js = float(self.cfg.get("judge_setting_ms", 0))
        rec = {"when": time.strftime("%m-%d %H:%M"), "mode": m, "game": [int(v) for v in vals],
               "non100": int(sum(vals[1:11])), "brk": int(vals[11]), "n": len(hits),
               "e0": [round(h[3] + js, 2) for h in hits], "js": js,
               "calib": float(self.cfg.get("calib_ms", 0.0)), "use": True,
               "judge": self.judge_kind(), "fs": self.fs_range(),
               "win": float(self.cfg.get("window_ms", 125.0 / 3.0)),
               "gmode": self.cfg.get("game_mode", "프리스타일"),
               "ch": [str(h[2]) for h in hits]}        # 노트별 레인 (키별 차이 확인용)
        self.results.append(rec)
        if len(self.results) > 120:
            self.results = self.results[-120:]
        save_results(self.results)
        self.update_zero()
        return rec

    def on_songend(self, t):
        """기어의 판정선이 사라짐 = 곡 끝 (그래프가 끝난 곡 전체를 보여주도록)"""
        self.log("songend", round(t, 2))
        g = self._last_song_hits()
        if g is not None and len(g[2]) >= 100:
            self.song_end = (g[0], g[1])

    def add_manual_result(self, non100):
        """결과 화면을 못 읽었을 때: 방금 끝난 곡에 '100% 아닌 개수'를 직접 넣음"""
        m = self.active_mode()
        g = self._last_song_hits()
        hits = g[2] if g else []
        if m is None or len(hits) < 100:
            return None
        js = float(self.cfg.get("judge_setting_ms", 0))
        rec = {"when": time.strftime("%m-%d %H:%M"), "mode": m, "game": None, "non100": int(non100), "brk": None,
               "n": len(hits), "e0": [round(h[3] + js, 2) for h in hits], "js": js,
               "calib": float(self.cfg.get("calib_ms", 0.0)), "use": True,
               "judge": self.judge_kind(), "fs": self.fs_range(),
               "win": float(self.cfg.get("window_ms", 125.0 / 3.0)),
               "gmode": self.cfg.get("game_mode", "프리스타일"),
               "ch": [str(h[2]) for h in hits]}        # 노트별 레인 (키별 차이 확인용)
        self.results.append(rec)
        save_results(self.results)
        self.update_zero()
        return rec

    def judge_kind(self):
        """지금 게임 판정 종류: 100% 범위 값으로 판단 (normal / hard / max / custom)"""
        p = self.cfg.get("fs_presets") or {}
        normal = 125.0 / 3.0
        fs = float(self.cfg.get("window_ms", normal))
        for k, v in (("normal", normal), ("hard", float(p.get("hard", round(normal / 2, 2)))),
                     ("max", float(p.get("max", round(normal / 4, 2))))):
            if abs(fs - v) < 0.01:
                return k
        return "custom"

    def update_zero(self):
        # 영점 보정은 노멀 판정 곡만으로 (하드·맥스는 판정 범위 자체가 달라서 섞으면 안 됨)
        use = [r for r in self.results if r.get("use", True) and r.get("judge", "normal") == "normal"][-30:]
        self.zero = fit_zero([(np.array(r["e0"]), r["non100"]) for r in use]) if use else None
        if self.zero is not None:
            self.zero["n"] = len(use)
        # 버튼 모드 × 게임 모드별로 따로 (루머 확인용: 그룹끼리 맞는 범위가 겹치는지 본다)
        groups = {}
        for r in self.results:
            if r.get("use", True) and r.get("judge", "normal") == "normal":
                groups.setdefault(group_key(r.get("mode"), r.get("gmode")), []).append(r)
        self.zero_groups = {}
        for k, rs in groups.items():
            rs = rs[-30:]
            z = fit_zero([(np.array(r["e0"]), r["non100"]) for r in rs])
            if z is not None:
                z["n"] = len(rs)
                self.zero_groups[k] = z

    def predicted_out(self, rec, calib=None):
        """이 곡 기록에서 영점 보정 calib일 때 분석기의 범위 밖 개수"""
        cb = float(self.cfg.get("calib_ms", 0.0)) if calib is None else calib
        # 그 곡을 친 판정 종류의 100% 범위 (노멀 = 41.67, 하드·맥스는 그때 범위)
        W = 125.0 / 3.0 if rec.get("judge", "normal") == "normal" else float(rec.get("win", rec.get("fs", 125.0 / 3.0)))
        e = np.array(rec["e0"]) + cb
        return int((np.abs(e) > W).sum())

    def _check_zero(self):
        """결과 화면 3곡 이상 모였고, 지금 영점 보정이 95% 범위 밖이면 추천 알림"""
        z = self.zero
        if z is None or z.get("n", 0) < 3 or len(self.results) <= self._zero_dismissed:
            self.suggest = None
            return
        cb = float(self.cfg.get("calib_ms", 0.0))
        if z["lo"] <= cb <= z["hi"]:
            self.suggest = None
            return
        self.suggest = (round(z["best"], 1), z["lo"], z["hi"], z["n"])

    # ------------------------------------------------------------ 그래프용
    SONG_GAP_MS = 6000.0
    SUSPECT_MS = 50.0

    def _last_song_hits(self, t=None):
        """마지막 곡의 (시작, 끝, 확실한 기록들): 실제로 키를 눌러 짝지어진 기록이 6초 넘게 끊긴 곳을 곡 경계로 본다.
        (곡이 끝나고 결과 화면으로 넘어갈 때 전환 그림이 노트처럼 잠깐 잡혀도 키를 안 누르니 곡으로 안 잡힘)
        경계는 신뢰도와 상관없이 짝지어진 기록 전체로 정한다 — 롱노트만 이어지는 구간(머리가 안 보여 신뢰도 0.3)이
        6초 넘으면 곡이 둘로 잘려 결과 기록에 곡 뒷부분만 들어가던 문제(v0.3.56) 때문."""
        g = split_songs([h for h in self.all_hits() if t is None or h[0] <= t], self.SONG_GAP_MS)
        if not g:
            return None
        g = g[-1]
        return g[0][0] - 1000.0, g[-1][0], [h for h in g if h[4] >= 1.0]

    def song_window(self, now=None):
        """그래프로 보여줄 곡의 (시작, 끝): 지금 치는 곡이 있으면 그것, 없으면 결과 화면으로 끝난 곡"""
        g = self._last_song_hits()
        if self.song_end is not None and (g is None or g[1] <= self.song_end[1] + 1.0 or len(g[2]) < 30):
            return self.song_end
        if g is None or len(g[2]) < 10:
            return None
        return g[0], g[1]

    def flow_points(self):
        """곡 흐름 그래프: [(곡 시작부터 초, 판정 기준 오차)]"""
        w = self.song_window()
        if w is None:
            return [], None
        c = self._center()
        pts = [((h[0] - w[0]) / 1000.0, h[3] - c) for h in self.all_hits() if h[4] >= 1.0 and w[0] <= h[0] <= w[1] + 1]
        return pts, (w[1] - w[0]) / 1000.0

    def density(self, bin_s=1.0):
        """밀도 그래프: 초당 노트 수 배열 (지금 모드의 레인 수 기준 도달)"""
        w = self.song_window()
        if w is None:
            return None
        m = self.active_mode()
        lanes = LANES_OF.get(m, 6) if m else 6
        ts = np.array([t for t, l in self.arr_log if l == lanes and w[0] - 500 <= t <= w[1] + 500])
        if len(ts) < 10:
            return None
        dur = max(1.0, (w[1] - w[0]) / 1000.0)
        nb = int(np.ceil(dur / bin_s)) + 1
        h, _ = np.histogram((ts - w[0]) / 1000.0, bins=nb, range=(0, nb * bin_s))
        return h / bin_s

    def flash_side(self, m, ch):
        """FAST/SLOW 글자를 띄울 쪽: 왼손 키 → 'L', 오른손 키 → 'R'"""
        g = self.cfg["modes"].get(m, {}).get("groups", {}).get(ch, "")
        if "왼" in g:
            return "L"
        if "오른" in g:
            return "R"
        if ch in ("SL", "TL"):
            return "L"
        if ch in ("SR", "TR"):
            return "R"
        keys = self.cfg["modes"].get(m, {}).get("keymap", {}).get(ch)
        if isinstance(keys, list) and len(keys) > 1:          # 5B 가운데: 실제로 누른 키로 판단
            kid = self._last_kid.get((m, ch))
            return "L" if kid is None or kid == keys[0] else "R"
        try:
            return "L" if int(ch[1:]) <= LANES_OF[m] / 2.0 else "R"
        except (ValueError, KeyError):
            return "L"

    def _final_nids(self, mt):
        s = {h[4] for h in mt.hits[-300:]}
        s.update(v[4] for v in mt.provisional.values())
        return s

    def _instant(self, kid, t):
        """키를 누른 순간: 그 레인에서 판정 중심이 가장 가까운 노트를 (예상 도달 시각으로) 찾아 바로 기록"""
        m = self.active_mode()
        if m is None:
            return
        ch = self.key_to_ch[m].get(kid)
        if ch is None:
            return
        mt = self.matchers[m]
        lanes = LANES_OF[m]
        c = self._center()
        W, WE = self.match_windows()
        taken = self._final_nids(mt) | set(self.instant.keys())
        cand = {}
        for nid, (pch, tp) in self.pred.get(lanes, {}).items():
            if pch == ch:
                cand[nid] = tp
        for a in mt.arrivals.get(ch, []):
            if a[4] is not None and a[1] >= 1.0:
                cand[a[4]] = a[0]                       # 실제 도달을 이미 알면 그 값
        best, bd = None, 1e9
        for nid, ta in cand.items():
            if nid in taken:
                continue
            dd = t - (ta + c)
            if dd > W or -dd > WE:
                continue
            d = abs(dd)
            if d < bd or (d == bd and best is not None and ta < cand[best]):
                best, bd = nid, d
        if best is None:
            return
        ta = cand[best]
        self.instant[best] = (ta, m, ch, t - ta, 1.0, t)
        self.last_instant_t = t
        self.long_ch[ch] = best in self._long_nids or any(
            a[4] == best and a[2] for a in mt.arrivals.get(ch, []))
        j = t - (ta + c)
        fs = self.fs_range()
        from_arrival = any(a[4] == best and a[1] >= 1.0 for a in mt.arrivals.get(ch, []))
        reliable = from_arrival or best in self.pred_ok.get(lanes, ())
        self.log("inst", ch, best, round(j, 1), int(from_arrival), int(reliable))
        self._inst_rel[best] = reliable
        if len(self._inst_rel) > 3000:
            self._inst_rel = dict(list(self._inst_rel.items())[-1000:])
        # FAST/SLOW 글자: 스크롤을 실제로 보고 있는 중의 예상(오차 1~2ms)이면 누르는 순간 바로 띄우고,
        # 짐작으로 이어가던 구간의 예상이면 노트 도달이 확인될 때까지 기다린다 (퍼펙트인데 FAST가 뜨는 일 방지).
        mode = self.cfg.get("indicator", {}).get("flash_mode", "auto")
        if (reliable or mode == "instant") and mode != "confirmed":
            if best not in self._flashed and abs(j) > fs:
                f = ("fast" if j < 0 else "slow", t, j, self.flash_side(m, ch))
                self.flashes.append(f)
                self._flashed.add(best)
                self._inst_flash[best] = f

    def on_arrival(self, lanes, ch, t, conf, is_long, nid=None):
        self.log("arr", lanes, ch, round(t, 2), round(float(conf), 2), int(bool(is_long)), nid)
        if conf > 0.0:
            self.arr_log.append((t, lanes))
            if len(self.arr_log) > 40000:
                del self.arr_log[:10000]
        if conf >= 1.0:
            if self._last_arr is not None and t - self._last_arr > 4000.0:
                self.song_t0 = self._last_arr + 1000.0     # 새 곡: 변속 횟수를 새로 셈
            if self._last_arr is None or t > self._last_arr:
                self._last_arr = t
        if self.paused:
            return
        if conf <= 0.0:
            # 3프레임 이상 일정하게 움직인 적 없는 것 = 노트가 아님 (FAST/SLOW 글자 등).
            # 짝짓기에 넣으면 진짜 노트의 입력을 가로채므로 버린다.
            return
        for m, mt in self.matchers.items():
            if LANES_OF[m] == lanes and ch in mode_channels(m):
                mt.on_arrival(ch, t, conf, is_long, nid)
        if lanes == 6 and ch in ("TL", "TR") and conf >= 1:
            self.seg_trigger = True

    def on_lanes(self, n):
        self.log("lanes", n)
        self.lanes_guess = n

    def on_vchg(self, lanes, t, v0, v1):
        """화면 노트 속도가 바뀜(변속) — 상태줄 표시·진단 기록용"""
        self.log("vchg", lanes, round(t, 2), round(float(v0), 4), round(float(v1), 4))
        self.vchg.append((t, lanes, v0, v1))
        if len(self.vchg) > 500:
            self.vchg = self.vchg[-300:]

    def speed_changes(self):
        """지금 곡에서 감지된 변속 횟수와 속도 범위 (지금 모드의 레인 수 기준)"""
        m = self.active_mode()
        lanes = LANES_OF.get(m, 6) if m else 6
        ev = [e for e in self.vchg if e[1] == lanes and e[0] >= self.song_t0]
        if not ev:
            return 0, None
        vs = [e[2] for e in ev] + [e[3] for e in ev]
        return len(ev), (min(vs), max(vs))

    def on_segment(self, t):
        """곡이 바뀐 것으로 판단될 때: 현재 곡 기록을 확정하고 판별을 새로 시작"""
        self.log("seg", round(t, 2))
        now = time.perf_counter() * 1000.0
        for mt in self.matchers.values():
            mt.center = self._center()
            mt.resolve(now + 1e6)
        m = self.active_mode()
        if m is not None:
            seg = self._seg_hits(m)
            e = np.array([h[3] for h in seg if h[4] >= 1.0], dtype=float)
            if len(e) >= 100:
                self.last_song = e
            self.past += seg
            for ch, c in self.matchers[m].missed.items():
                self.past_missed[ch] += c
        self.matchers = {mm: self._new_matcher() for mm in self.candidates}
        self.seg_start = t
        self.seg_trigger = False
        self.lanes_guess = None
        self._ev_idx = {}
        self._hit_idx = {}
        self._inst_flash = {}
        self.instant = {}
        self._flashed = set()

    def _new_matcher(self):
        return Matcher(float(self.cfg.get("match_window_ms", 150)), float(self.cfg.get("chatter_ms", 25)),
                       float(self.cfg.get("match_early_ms", 190)))

    def _center(self):
        return -(float(self.cfg.get("judge_setting_ms", 0)) + float(self.cfg.get("calib_ms", 0.0)))

    def match_windows(self):
        """키 입력과 노트를 짝지을 최대 거리 (늦은 쪽, 이른 쪽).
        게임이 판정해 주는 범위 끝 ≈ MAX 10% 끝 = 100% 범위 + 9단계(한 단계 = 0.4 x 100% 범위) = 4.6 x 100% 범위.
        노멀(41.67)이면 191.7 → 실측 값(늦은 쪽 150, 이른 쪽 190) 그대로, 하드(20.83)면 95.8까지만 짝지음.
        (하드에서 게임이 무시하는 헛친 키를 노트에 짝지어 'FAST 160ms'처럼 크게 튄 글자가 뜨던 문제)"""
        lim = 4.6 * float(self.cfg.get("window_ms", 125.0 / 3.0))
        return (min(float(self.cfg.get("match_window_ms", 150)), lim),
                min(float(self.cfg.get("match_early_ms", 190)), lim))

    def tick(self):
        now = time.perf_counter() * 1000.0
        c = self._center()
        fw = float(self.cfg.get("window_ms", 42))
        fs = self.fs_range()
        mw, mwe = self.match_windows()
        for mt in self.matchers.values():
            mt.center = c
            mt.fast_win = fw
            mt.fs = fs
            mt.W, mt.WE = mw, mwe
            mt.resolve(now)
        # 인디케이터 반짝임: 지금 모드의 FAST/SLOW 이벤트만
        m = self.active_mode()
        if m is not None:
            mt = self.matchers[m]
            # 누르는 순간 예상(즉시값)과 확정값이 50ms 넘게 다르면 둘 중 하나가 크게 틀린 것 → 그 노트는 믿지 않음
            # (FAST/SLOW 표시 안 함, 신뢰도 0.3으로 내려 통계·인디케이터에서 뺌). 정상이면 차이는 수 ms.
            for i in range(self._hit_idx.get(m, 0), len(mt.hits)):
                h = mt.hits[i]
                nid = h[4] if len(h) > 4 else None
                iv = self.instant.get(nid) if nid is not None else None
                if (iv is not None and iv[1] == m and h[3] >= 1.0 and abs(h[0] + h[2] - iv[5]) < 3.0
                        and abs(h[2] - iv[3]) > self.SUSPECT_MS):         # 같은 키 입력인데 노트 도달 시각만 크게 다름
                    mt.hits[i] = (h[0], h[1], h[2], 0.3) + tuple(h[4:])
                    self._suspect.add(nid)
                    self.log("suspect", h[1], nid, round(iv[3] - c, 1), round(h[2] - c, 1), round(h[0], 2))
            i0 = self._ev_idx.get(m, 0)
            for ev in mt.events[i0:]:
                nid = ev[4] if len(ev) > 4 else None
                if nid is not None and (nid in self._flashed or nid in self._suspect):
                    continue
                if nid is not None:
                    self._flashed.add(nid)
                self.flashes.append((ev[0], ev[2], ev[3], self.flash_side(m, ev[1])))
                self.log("flash", ev[0], ev[1], nid, round(ev[3], 1))
            self._ev_idx[m] = len(mt.events)
            # 누르는 순간 '믿을 만한' 예상과 확정값이 10ms 넘게 다르면: 원인 진단용 화면 저장 요청
            for h in mt.hits[self._hit_idx.get(m, 0):]:
                nid = h[4] if len(h) > 4 else None
                iv = self.instant.get(nid) if nid is not None else None
                if iv is not None and self._inst_rel.get(nid) and abs(h[2] - iv[3]) > 10.0:
                    self.log("mismatch", h[1], nid, round(iv[3] - c - 0, 1), round(h[2] - c, 1), round(h[0], 2))
                    self.dump_requests.append(h[0])
            # 누르는 순간 띄운 글자가 확정값으로 보니 범위 안이면 바로 지운다
            if self._inst_flash:
                h0 = self._hit_idx.get(m, 0)
                js_c = -self._center()
                for h in mt.hits[h0:]:
                    f = self._inst_flash.pop(h[4], None) if len(h) > 4 else None
                    if f is not None and (abs(h[2] + js_c) <= fs or h[4] in self._suspect):
                        self.flashes = [x for x in self.flashes if x is not f]
                        self.log("unflash", h[1], h[4], round(h[2] + js_c, 1))
                if len(self._inst_flash) > 200:
                    self._inst_flash = dict(list(self._inst_flash.items())[-50:])
            self._hit_idx[m] = len(mt.hits)
            if len(self.flashes) > 50:
                self.flashes = self.flashes[-50:]
        self._link_fstext(now)
        self._check_zero()
        # 오래된 즉시 기록 정리
        if self.instant:
            self.instant = {k: v for k, v in self.instant.items() if now - v[5] < 6000}
        if len(self._flashed) > 2000:
            self._flashed = set(list(self._flashed)[-500:])
        if len(self._suspect) > 2000:
            self._suspect = set(list(self._suspect)[-500:])

    def fs_range(self):
        v = self.cfg.get("fs_ms")
        return float(v) if v not in (None, "") else float(self.cfg.get("window_ms", 42))

    # ------------------------------------------------------------ 상태
    def active_mode(self):
        fixed = self.cfg.get("mode", "auto")
        if fixed != "auto":
            return fixed
        n = self.lanes_guess
        if n is not None:
            if n == 6:
                return "8B" if self.seg_trigger else "6B"
            return {4: "4B", 5: "5B"}.get(n)
        # 화면으로 판별이 안 되면: 실제 친 키와 노트가 가장 많이 짝지어진 모드
        return self.best_by_hits()

    def best_by_hits(self, min_hits=15):
        best, bn = None, 0
        for m, mt in self.matchers.items():
            k = sum(1 for h in mt.hits if h[3] >= 1 and h[0] >= self.seg_start)
            if k > bn:
                best, bn = m, k
        return best if bn >= min_hits else None

    def _seg_hits(self, m):
        return [(h[0], m, h[1], h[2], h[3]) for h in self.matchers[m].hits if h[0] >= self.seg_start]

    def all_hits(self):
        m = self.active_mode()
        cur = self._seg_hits(m) if m is not None else []
        return self.past + cur

    def display_hits(self):
        """인디케이터용: 확정 기록 + 아직 확정 대기 중인 늦은 입력"""
        m = self.active_mode()
        if m is None:
            return self.past
        mt = self.matchers[m]
        prov = [(h[0], m, h[1], h[2], h[3]) for h in mt.provisional.values()]
        fin = self._final_nids(mt)
        inst = [v[:5] for k, v in self.instant.items() if k not in fin and v[1] == m]
        extra = prov + inst
        out = self.all_hits() + extra
        if extra:
            out.sort(key=lambda h: h[0])
        return out

    def recent_hits(self, since, tail=400):
        """인디케이터용 (매 ms 불리므로 가볍게): since 이후 기록만. 형식은 display_hits와 같음"""
        m = self.active_mode()
        out = [h for h in self.past[-tail:] if h[0] >= since]
        if m is None:
            return out
        mt = self.matchers[m]
        lo = max(since, self.seg_start)
        out += [(h[0], m, h[1], h[2], h[3]) for h in mt.hits[-tail:] if h[0] >= lo]
        out += [(h[0], m, h[1], h[2], h[3]) for h in mt.provisional.values() if h[0] >= since]
        if self.instant:
            fin = self._final_nids(mt)
            out += [v[:5] for k, v in self.instant.items() if k not in fin and v[1] == m and v[0] >= since]
        out.sort(key=lambda h: h[0])
        return out

    def missed(self):
        out = defaultdict(int, self.past_missed)
        m = self.active_mode()
        if m is not None:
            for ch, c in self.matchers[m].missed.items():
                out[ch] += c
        return out

    # ------------------------------------------------------------ 통계
    def stats(self):
        cfg = self.cfg
        hits = [h for h in self.all_hits() if h[4] >= 1.0]
        N = int(cfg.get("recent", 0) or 0)
        if N > 0:
            hits = hits[-N:]
        js = float(cfg.get("judge_setting_ms", 0))
        win = float(cfg.get("window_ms", 42))
        b = float(cfg.get("calib_ms", 0.0))
        fs = self.fs_range()
        e = np.array([h[3] for h in hits], dtype=float)
        out = {"overall": summarize(e, js, win, b, fs), "groups": [], "channels": [],
               "hist": histogram(e + js + b) if len(e) else None, "mode": self.active_mode()}
        # 그룹별/채널별
        by_g, by_c = defaultdict(list), defaultdict(list)
        for t, m, ch, ee, c in hits:
            g = cfg["modes"][m]["groups"].get(ch, "기타")
            by_g[g].append(ee)
            by_c[(m, ch)].append(ee)
        for g, v in by_g.items():
            out["groups"].append((g, summarize(np.array(v), js, win, b, fs, rec=False)))
        miss = self.missed()
        for (m, ch), v in sorted(by_c.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            keys = cfg["modes"][m]["keymap"].get(ch, [])
            out["channels"].append((m, ch, keys, summarize(np.array(v), js, win, b, fs, rec=False), miss.get(ch, 0)))
        out["chatter"] = dict(self.chatter)
        return out

    def song_e(self):
        """게임 결과 보정용 '방금 친 곡': 진행 중인 곡 기록이 충분하면 그것, 아니면 마지막으로 끝난 곡"""
        m = self.active_mode()
        if m is not None:
            e = np.array([h[3] for h in self._seg_hits(m) if h[4] >= 1.0], dtype=float)
            if len(e) >= 100:
                return e
        return self.last_song

    def raw_e(self):
        return np.array([h[3] for h in self.all_hits() if h[4] >= 1.0], dtype=float)

    def export_raw(self, path):
        """진단용: 키 입력 / 노트 도달 / 게임 글자 원본 + 설정 요약"""
        import json
        keys = {k: v for k, v in self.cfg.items() if k not in ("colors",)}
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["#cfg", json.dumps(keys, ensure_ascii=False)])
            w.writerow(["#colors", json.dumps(self.cfg.get("colors", {}), ensure_ascii=False)])
            for ev in self.raw:
                w.writerow(ev)

    def export_csv(self, path):
        js = float(self.cfg.get("judge_setting_ms", 0)) + float(self.cfg.get("calib_ms", 0.0))
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["시각(ms)", "모드", "채널", "그룹", "화면기준 오차(ms)", "판정기준 오차(ms)", "신뢰도"])
            for t, m, ch, e, c in self.all_hits():
                g = self.cfg["modes"][m]["groups"].get(ch, "")
                w.writerow([round(t, 2), m, ch, g, round(e, 2), round(e + js, 2), c])


def _results_path():
    import os
    from .config import app_dir
    return os.path.join(app_dir(), "results.json")


def load_results():
    import json
    try:
        with open(_results_path(), encoding="utf-8") as f:
            d = json.load(f)
        return [r for r in d if isinstance(r, dict) and "e0" in r and "non100" in r]
    except Exception:
        return []


def save_results(rs):
    import json
    try:
        with open(_results_path(), "w", encoding="utf-8") as f:
            json.dump(rs, f, ensure_ascii=False)
    except Exception:
        pass


# ------------------------------------------------------------ 저장한 기록 다시 보기
class SongView:
    """불러온 기록의 곡 하나 — 그래프 그리기에 필요한 것만 (Session과 같은 이름의 메서드)"""

    def __init__(self, song, cfg):
        self.s = song
        self.cfg = cfg
        self.song_end = (song["start"], song["end"])

    def song_window(self, now=None):
        return self.song_end

    def flow_points(self):
        c = -(self.s["js"] + self.s["calib"])
        t0 = self.s["start"]
        pts = [((h[0] - t0) / 1000.0, h[3] - c) for h in self.s["hits"]]
        return pts, (self.s["end"] - t0) / 1000.0

    def density(self, bin_s=1.0):
        t0, t1 = self.s["start"], self.s["end"]
        ts = np.array([t for t in self.s["arr"] if t0 - 500 <= t <= t1 + 500])
        if len(ts) < 10:
            return None
        dur = max(1.0, (t1 - t0) / 1000.0)
        nb = int(np.ceil(dur / bin_s)) + 1
        h, _ = np.histogram((ts - t0) / 1000.0, bins=nb, range=(0, nb * bin_s))
        return h / bin_s

    def judge_errors(self):
        c = -(self.s["js"] + self.s["calib"])
        return np.array([h[3] - c for h in self.s["hits"]])


def replay_raw(path):
    """[CSV 저장] 때 만든 _raw.csv를 다시 흘려 넣어 곡별 기록을 만든다.
    반환: (설정, [곡 dict: start, end, hits, arr, result, mode, js, calib, n])"""
    import json
    from . import config as C
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    if not rows or rows[0][0] != "#cfg":
        raise ValueError("분석기 진단 기록(_raw.csv)이 아니에요")
    cfg = C._merge(C.DEFAULTS, json.loads(rows[0][1]))
    clock = [0.0]
    real = time.perf_counter
    songs = []
    res_rows = []
    try:
        time.perf_counter = lambda: clock[0] / 1000.0
        ses = Session.__new__(Session)
        ses.results = []
        ses.zero = None
        ses._zero_dismissed = 0
        Session.__init__(ses, cfg)
        arr_all = []

        def harvest():
            for _ in range(3):
                clock[0] += 500.0
                ses.tick()
            m = ses.active_mode()
            for g in split_songs(ses.all_hits(), Session.SONG_GAP_MS):
                g = [h for h in g if h[4] >= 1.0]
                if len(g) < 30:
                    continue
                songs.append({"start": g[0][0] - 1000.0, "end": g[-1][0], "hits": g, "mode": g[0][1] or m,
                              "arr": [t for t in arr_all if g[0][0] - 2000 <= t <= g[-1][0] + 1000],
                              "js": float(cfg.get("judge_setting_ms", 0)), "calib": float(cfg.get("calib_ms", 0.0)),
                              "result": None, "n": len(g)})

        n = 0
        for r in rows[2:]:
            if len(r) < 2:
                continue
            try:
                t = float(r[0])
            except ValueError:
                continue
            clock[0] = t
            k = r[1]
            if k == "reset":
                harvest()
                ses.reset()
            elif k == "arr":
                conf = float(r[5])
                if conf > 0:
                    arr_all.append(float(r[4]))
                ses.on_arrival(int(r[2]), r[3], float(r[4]), conf, r[6] == "1", r[7] if len(r) > 7 else None)
            elif k == "key":
                ses.on_key(r[2], r[3] == "1", float(r[4]))
            elif k == "result":
                res_rows.append((t, [int(x) for x in r[2:14]]))
            n += 1
            if n % 50 == 0:
                ses.tick()
        harvest()
    finally:
        time.perf_counter = real
    # 결과 화면 숫자는 그 직전에 끝난 곡에 붙인다
    for t, vals in res_rows:
        cand = [sg for sg in songs if sg["end"] <= t and t - sg["end"] < 120000.0 and sg["result"] is None]
        if cand:
            max(cand, key=lambda sg: sg["end"])["result"] = vals
    songs.sort(key=lambda sg: sg["start"])
    return cfg, songs
