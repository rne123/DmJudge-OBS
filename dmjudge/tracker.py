"""노트 추적 엔진 (플랫폼 독립, numpy/opencv만 사용).

원리
- 화면의 모든 노트는 같은 순간 같은 속도로 스크롤된다.
  → 프레임마다 보이는 노트 전체로 '스크롤 위치 S(t)'를 추정한다.
- 각 노트는 고정된 채보 위치 p 를 가진다 (화면 y = p + S(t)).
- 노트 중앙이 판정선 정중앙(judge_y)에 닿는 순간 = S(t) 가 judge_y - p 를 지나는 순간.
  판정선 근처가 이펙트로 가려져도, 다른 노트로 S(t)를 계속 알 수 있으므로 계산 가능.
- 변속/정지도 S(t) 자체가 따라가므로 그대로 대응된다.

좌표는 모두 '캡처 영역 기준 픽셀', 시간은 ms.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

_NID = itertools.count(1)

# 채널 이름: 일반 레인 L1~L6, 트리거 TL/TR, 사이드 트랙 SL/SR
LANES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]


class Note:
    __slots__ = ("ch", "p_bot", "h", "n", "first_t", "last_t", "done", "long", "obs", "anchor", "nid")

    def __init__(self, ch, p_bot, h, t, is_long, anchor=False):
        self.ch = ch
        self.p_bot = p_bot      # 아랫면의 채보 위치 (y - S)
        self.h = h              # 머리 두께(px)
        self.n = 1
        self.first_t = t
        self.last_t = t
        self.done = False
        self.long = is_long
        self.obs = None
        self.anchor = anchor    # 롱노트 윗면(스크롤 기준점 전용, 판정 이벤트 없음)
        self.nid = next(_NID)   # 노트 고유 번호 (예측 ↔ 실제 도달 연결용)

    @property
    def p_center(self):
        return self.p_bot - self.h / 2.0


class NoteTracker:
    def __init__(self, geom: dict, colors: dict, lanes: int = 6):
        """geom: x0,x1 (플레이 영역 좌우), top (추적 시작 y), occl_y (추적 끝 y,
        이 아래는 이펙트로 가려지는 구간), judge_y (판정선 정중앙). 모두 캡처 영역 기준.
        """
        self.g = dict(geom)
        self.c = dict(colors)
        self.nl = lanes
        x0, x1 = self.g["x0"], self.g["x1"]
        self.lw = (x1 - x0) / lanes
        step = max(2, int(round(self.lw / 18)))           # 레인당 약 18개 샘플
        self.xs = np.arange(int(x0) + 1, int(x1) - 1, step)
        rel = (self.xs - x0) / self.lw
        self.lane_of = np.clip(rel.astype(int), 0, lanes - 1)
        frac = rel - self.lane_of
        self.inner = (frac > 0.15) & (frac < 0.85)          # 레인 안쪽 샘플
        # 반폭 노트(트리거/사이드)는 레인 수와 무관하게 화면 절반 기준 (5B 대응)
        self.half_of = (self.xs >= (x0 + x1) / 2.0).astype(int)
        self.half_n = np.array([(self.half_of == h).sum() for h in (0, 1)])
        # 처리 속도용 미리 계산: 레인별 안쪽 샘플 열(연속) / 반쪽 화면별 레인 열
        self._inner_cols = np.where(self.inner)[0]
        il = self.lane_of[self._inner_cols]
        self._lane_ids = [l for l in range(lanes) if (il == l).any()]
        self._lane_starts = np.array([np.where(il == l)[0][0] for l in self._lane_ids])
        self._lane_k = np.array([(il == l).sum() for l in self._lane_ids])
        self._lane_cols = {l: np.where(self.lane_of == l)[0] for l in range(lanes)}
        self._half_cols = []
        for h in (0, 1):
            m = self.half_of == h
            cols = np.where(m)[0]
            lane_cols = [np.where(m & (self.lane_of == l))[0] for l in range(lanes)]
            self._half_cols.append((cols, [lc for lc in lane_cols if len(lc)]))
        # 인디케이터 등으로 가려지는 줄 (캡처 영역 기준 y 범위) → 추적 구간에서 '가려짐'으로 처리
        self.band = None
        if self.g.get("mask"):
            a, b = self.g["mask"]
            ra, rb = int(a - self.g["top"]), int(b - self.g["top"]) + 1
            rows = int(self.g["occl_y"]) - int(self.g["top"])
            ra, rb = max(0, ra), min(rows, rb)
            if rb > ra:
                self.band = (ra, rb)

        self.S = 0.0
        self.v = None          # px/ms
        self.hist = []         # (t, S, observed)
        self.last_obs_t = None
        self.notes: list[Note] = []
        self.head_h = self.lw * 0.33   # 탭 노트 두께 추정(초기값, 이후 실측)
        self._tap_h = []
        self.prev_t = None
        self._boot_prev = None
        self._lost = 0
        self.debug = False
        self.pending = []
        self.last_ids = []
        self._sb = []          # 직전 프레임 덩어리 [(ch, top, bot, 제자리 연속 횟수)]
        self.n_static = 0      # 이번 프레임에서 '화면 고정 그림'으로 뺀 덩어리 수
        self.static_on = True
        self._zones = {}       # 채널 → [[top, bot, 마지막 확인 t]] 고정 그림이 있는 줄(가림 구역)
        self._zclip = set()    # 이번 프레임에서 가림 구역에 걸려 잘린 덩어리 (새 노트로 등록 안 함)
        self._vref = None      # 변속 감지: 지금 기준 속도 (px/ms)
        self._vanchor = None   # 마지막 변속 이후 기준 속도
        self._vc = []          # 기준과 다른 속도가 연속으로 나온 값들
        self._vdet_t = None
        self.vchanges = []     # [(t, 이전 속도, 새 속도)] — 꺼내 간 쪽이 비움

    # ------------------------------------------------------------ 검출
    def detect(self, frame):
        """frame: BGR 이미지(캡처 영역 전체). 반환: [(ch, top, bot, clipped_top, clipped_bot)]"""
        top, occ = int(self.g["top"]), int(self.g["occl_y"])
        strip = frame[top:occ].take(self.xs, axis=1)          # rows, ns, 3 (take가 팬시 인덱싱보다 빠름)
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        c = self.c
        vmin = int(min(255, max(0, c["v_min"])))
        smin = int(min(255, max(0, c["sat_min"])))

        def rng(h0, h1, s0):
            lo = np.array([min(255, max(0, int(h0))), s0, vmin], np.uint8)
            hi = np.array([min(255, max(0, int(h1))), 255, 255], np.uint8)
            if int(h1) < 0 or int(h0) > 255 or int(h0) > int(h1):
                return np.zeros(hsv.shape[:2], bool)
            m = cv2.inRange(hsv, lo, hi)                       # 0/255
            np.bitwise_and(m, 1, out=m)                        # 0/1 로 만든 뒤 bool로 보기 (복사 없음)
            return m.view(bool)

        bright = rng(0, 255, 0)
        trig = rng(c["trig_h"][0], c["trig_h"][1], smin)
        side = rng(c["side_h"][0], c["side_h"][1], smin)
        if self.band is not None:
            bright[self.band[0]:self.band[1]] = False
            trig[self.band[0]:self.band[1]] = False
            side[self.band[0]:self.band[1]] = False

        out = []
        rows = occ - top
        # 반폭 노트(트리거/사이드): 반쪽 화면에서 그 색이 일정 비율 이상이고, 레인 2개 이상에 걸쳐 있어야 함
        # (하늘색·주황색 노트 스킨의 일반 노트 하나를 사이드/트리거로 착각하지 않도록)
        excl = None
        for h, (tl, sl) in enumerate((("TL", "SL"), ("TR", "SR"))):
            cols, lane_cols = self._half_cols[h]
            n = self.half_n[h]
            for mask, name in ((trig, tl), (side, sl)):
                sub = mask[:, cols]
                cnt = sub.sum(1)
                if not cnt.any():                   # 이 색이 반쪽에 하나도 없으면 건너뜀 (대부분의 프레임)
                    continue
                frac = cnt / n
                nl_on = np.zeros(rows, dtype=np.int16)
                for lc in lane_cols:
                    nl_on += mask[:, lc].any(1)
                ok = (frac >= 0.55) | ((frac >= c["wide_frac"]) & (nl_on >= 2))
                out += self._runs(ok.copy(), name, rows, top)
                if ok.any():
                    if excl is None:
                        excl = np.zeros_like(bright)
                    excl[:, cols] |= sub & ok[:, None]
        # 일반 레인: 색과 상관없이 밝은 노트 (반폭 노트로 판정된 줄의 그 색 픽셀만 제외)
        note = bright if excl is None else (bright & ~excl)
        sums = np.add.reduceat(note[:, self._inner_cols].astype(np.int16), self._lane_starts, axis=1)
        for i, l in enumerate(self._lane_ids):
            on = sums[:, i] >= max(1, int(np.ceil(self._lane_k[i] * c["lane_frac"])))
            ring = None
            if on.any() and c.get("ring_merge"):
                # 고리(테두리) 모양 동그란 노트: 가운데 줄은 양옆 테두리만 밝아서 노트로 안 잡힘 →
                # 그 줄에서 왼쪽 끝·오른쪽 끝 밝은 점이 레인 폭의 절반 넘게 벌어져 있으면 '고리 옆면'으로 표시
                cols = self._lane_cols[l]
                sub = note[:, cols]
                anyb = sub.any(1)
                if anyb.any():
                    left = np.argmax(sub, axis=1)
                    right = sub.shape[1] - 1 - np.argmax(sub[:, ::-1], axis=1)
                    ring = anyb & ((self.xs[cols][right] - self.xs[cols][left]) >= 0.5 * self.lw)
            out += self._runs(on, LANES[l], rows, top, ring)
        return out

    def _runs(self, mask, ch, rows, top, ring=None):
        res = []
        if not mask.any():
            return res
        # 1) 노트 내부 무늬로 생긴 짧은 틈 메우기
        g = max(3, int(self.lw * 0.08))
        d0 = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        st, en = np.where(d0 == 1)[0], np.where(d0 == -1)[0]
        for a, b in zip(en[:-1], st[1:]):
            if b - a <= g and not (self.band is not None and a <= self.band[0] < b + 1 and self.band[1] > a):
                mask[a:b] = True
        # 2) 속 빈 테두리 노트: 얇은 위 테두리와 얇은 아래 테두리 사이를 메움 (둘 다 얇을 때만)
        thin = max(2, int(self.lw * 0.15))
        g2 = max(4, int(self.lw * float(self.c.get("gap_frac", 0.35))))
        d0 = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        st, en = np.where(d0 == 1)[0], np.where(d0 == -1)[0]
        for i in range(len(st) - 1):
            a, b = en[i], st[i + 1]
            if b - a <= g2 and en[i] - st[i] <= thin and en[i + 1] - st[i + 1] <= thin and \
                    not (self.band is not None and a <= self.band[0] < b + 1 and self.band[1] > a):
                mask[a:b] = True
        # 3) 고리 모양 동그란 노트: 위 테두리 / (가운데 무늬) / 아래 테두리 조각 사이가 전부 '고리 옆면' 줄이고,
        #    합친 높이가 레인 폭 정도(동그라미 지름, 0.7~1.35배)이면 한 노트로 합친다
        if ring is not None:
            d0 = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
            st, en = np.where(d0 == 1)[0], np.where(d0 == -1)[0]
            bd = self.band
            i = 0
            while i < len(st) - 1:
                j = i
                while j + 1 < len(st):
                    a, b = en[j], st[j + 1]
                    if b - a > 1.0 * self.lw or en[j + 1] - st[i] > 1.35 * self.lw:
                        break
                    if bd is not None and a <= bd[0] < b + 1 and bd[1] > a:
                        break
                    if ring[a:b].mean() < 0.8:
                        break
                    j += 1
                n_ = len(mask)
                top0 = 0 if bd is None or st[i] < bd[1] else bd[1]
                bot0 = n_ if bd is None or en[j] > bd[0] else bd[0]
                # 위/아래가 화면(또는 가림 구간)에서 잘려 보이는 중이면 짧아도 합침
                # (잘린 쪽 끝까지 고리 옆면만 이어지는 경우 포함)
                clipped = (st[i] == top0 or (st[i] - top0 < self.lw and ring[top0:st[i]].mean() >= 0.8) or
                           en[j] == bot0 or (bot0 - en[j] < self.lw and ring[en[j]:bot0].mean() >= 0.8))
                if j > i and (en[j] - st[i] >= 0.7 * self.lw or clipped):
                    mask[st[i]:en[j]] = True
                i = j + 1
        d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        min_h = max(3, int(self.lw * 0.18))   # 노트 두께의 절반 정도 미만은 잡음
        bd = self.band
        for s, e in zip(starts, ends):
            ct = s == 0 or (bd is not None and s == bd[1])        # 위쪽이 가려짐
            cb = e == rows - 1 or (bd is not None and e == bd[0] - 1)   # 아래쪽이 가려짐
            if e - s + 1 < min_h and not ct and not cb:
                continue
            res.append((ch, s + top, e + top, ct, cb))
        return res

    # ------------------------------------------------------------ 갱신
    def update(self, frame, t):
        """한 프레임 처리. 반환: 판정선 도달 이벤트 [(ch, t_arrival, conf, is_long)]"""
        self.last_ids = []
        blobs = self._drop_static(self.detect(frame), t)
        self.last_blobs = blobs
        if self.prev_t is None:
            self.prev_t = t
            self._register_new(blobs, t, force=True)
            self.hist.append((t, self.S, False))
            return []
        dt = t - self.prev_t
        if dt <= 0:
            return []
        if self._frozen(blobs, t, dt):
            return []

        if self.v is None or self._lost >= 2:
            self._bootstrap(blobs, t)
        self._boot_prev = (t, [(b[0], b[2]) for b in blobs if not b[4]])
        v = self.v or 0.0
        S_pred = self.S + v * dt

        # 관측 노트와 기존 노트 매칭 → S 보정
        # 변속으로 속도가 급변해도 맞도록, 넓은 범위의 후보 중 '가장 많은 노트가 동의하는 보정량'을 고른다.
        hh = self.head_h
        wide = 3.0 * hh + 0.8 * v * dt + 6.0
        cand = [nt for nt in self.notes if not nt.done]
        bych = {}
        for nt in cand:
            bych.setdefault(nt.ch, []).append(nt)
        obsl = []          # (key, y, blob_idx)
        for bi, (ch, tp, bt, ct, cb) in enumerate(blobs):
            if not cb:
                obsl.append((ch, bt, bi))
            if not ct and (bt - tp + 1) > self.long_thr():
                obsl.append((ch + "^", tp, bi))      # 롱노트 윗면
        allr = []          # (residual, obs_idx, note)
        for oi, (key, y, bi) in enumerate(obsl):
            for nt in bych.get(key, ()):
                r = (y - S_pred) - nt.p_bot
                if abs(r) < wide:
                    allr.append((r, oi, nt))
        resid, pairs = [], []
        if allr:
            rs = np.array([a[0] for a in allr])
            band = max(3.0, 0.35 * hh)
            score = [(np.abs(rs - r0) <= band).sum() - 0.001 * abs(r0) for r0 in rs]
            r_star = rs[int(np.argmax(score))]
            used_b, used_n = set(), set()
            for r, bi, nt in sorted(allr, key=lambda a: abs(a[0] - r_star)):
                if abs(r - r_star) > band or bi in used_b or id(nt) in used_n:
                    continue
                used_b.add(bi); used_n.add(id(nt))
                resid.append(r); pairs.append((bi, nt))
        observed = len(resid) > 0
        corr = float(np.median(resid)) if observed else 0.0
        movable = [b for b in blobs if not b[4]]
        self._lost = 0 if (observed or not movable) else self._lost + 1
        S_now = S_pred + corr
        if not observed and self.v:
            # 관측이 끊긴 프레임: 마지막 몇 프레임 속도(self.v)로만 외삽하면, 그 몇 프레임 중 하나가
            # 늦게 찍혔을 때 속도가 크게 틀려 판정선까지 가는 동안 수십 ms가 밀린다 → 최근 관측 여러 개의 직선으로 외삽
            # (BPM이 바뀐 마디가 그 안에 있으면 바뀐 뒤 구간만)
            bf = self.blind_fit(t)
            if bf is not None and bf[0] > self.S:
                S_now = bf[0]
        if observed:
            self.last_obs_t = t

        # 매칭된 노트 위치 갱신(평균)
        matched_obs = {oi for oi, _ in pairs}
        for oi, nt in pairs:
            key, y, bi = obsl[oi]
            ch, tp, bt, ct, cb = blobs[bi]
            if nt.anchor:
                nt.p_bot = (nt.p_bot * nt.n + (y - S_now)) / (nt.n + 1)
                nt.n += 1
                nt.last_t = t
                continue
            pb = bt - S_now
            nt.p_bot = (nt.p_bot * nt.n + pb) / (nt.n + 1)
            nt.n += 1
            nt.last_t = t
            if self.debug:
                nt.obs = (nt.obs or []) + [(t, bt, tp, ct)]
            hgt = bt - tp + 1
            if hgt > self.long_thr():
                if not nt.long:
                    nt.long = True
                    nt.h = self.head_h
            elif not ct and not nt.long:
                if ch.startswith("L") and 0.15 * self.lw <= hgt <= 1.2 * self.lw:
                    self._tap_h.append(hgt)
                    if len(self._tap_h) > 400:
                        self._tap_h = self._tap_h[-400:]
                    self.head_h = float(np.median(self._tap_h))
                    nt.h = hgt if nt.n < 3 else (nt.h * 0.7 + hgt * 0.3)
        matched_idx = {obsl[oi][2] for oi in matched_obs if not obsl[oi][0].endswith("^")}
        new_tops = [obsl[oi] for oi in range(len(obsl))
                    if obsl[oi][0].endswith("^") and oi not in matched_obs]
        S_prev = self.S
        self.S = S_now
        self.hist.append((t, S_now, observed))
        if len(self.hist) > 64:
            self.hist = self.hist[-64:]
        self._update_v()

        if observed:
            self._detect_speed_change(t)
        self._register_new([b for i, b in enumerate(blobs) if i not in matched_idx], t)
        for key, y, bi in new_tops:
            self.notes.append(Note(key, y - self.S, 0, t, True, anchor=True))

        # 도달 판정: S가 (judge_y - p_center)를 지나갔는가
        ev = []
        J = self.g["judge_y"]
        gap = (t - self.last_obs_t) if self.last_obs_t is not None else 1e9
        # 관측이 끊긴 구간(추정 스크롤)에서 계산된 도달은 보류했다가, 다음 관측으로 검증
        if self.pending:
            if observed:
                ok = abs(corr) <= max(3.0, 0.35 * self.head_h)
                ev += [(e[0], e[1], (1.0 if ok else 0.0)) + e[3:] for e in self.pending]
                self.pending = []
            elif t - self.pending[0][1] > self.c["max_blind_ms"]:
                ev += [(e[0], e[1], min(e[2], 0.3)) + e[3:] for e in self.pending]
                self.pending = []
        fit = None
        for nt in self.notes:
            if nt.done or nt.anchor:
                continue
            target = J - nt.p_center
            if S_prev < target <= S_now:
                frac = (target - S_prev) / (S_now - S_prev)
                t_arr = self.prev_t + frac * dt
                # 두 프레임 사이 보간은 한 프레임이 늦게 찍히면 몇 ms 틀어진다 → 최근 스크롤 직선으로 다시 계산
                if fit is None:
                    fit = self.scroll_fit(t) or (0.0, 0.0, False)
                if fit[2]:
                    tf = t + (target - fit[0]) / fit[1]
                    if abs(tf - t_arr) < 15.0:
                        t_arr = tf
                conf = 0.0 if nt.n < 3 else 1.0
                e = (nt.ch, t_arr, conf, nt.long, nt.nid, nt)
                if observed or conf == 0.0:
                    ev.append(e)
                else:
                    self.pending.append(e)
                nt.done = True
        # 오래된 노트 정리
        self.notes = [nt for nt in self.notes
                      if not nt.done and t - nt.last_t < self.c["note_ttl_ms"]
                      and not (nt.anchor and nt.p_bot + S_now > self.g["occl_y"] + 5)]
        self.prev_t = t
        self.last_ids = [e[4] for e in ev]
        return [(e[0], e[1], e[2], e[3]) + ((e[5],) if self.debug else ()) for e in ev]

    FREEZE_MAX_MS = 120.0

    def _frozen(self, blobs, t, dt):
        """게임 화면이 잠깐 멈춘 프레임(노트가 전부 그대로)인지. 맞으면 이 프레임은 건너뛴다.
        실측(ENERGY SYNERGY MATRIX): 노트가 40ms쯤 멈췄다가 멈춘 만큼 한 번에 따라잡음 — 게임 시간은 그대로 흐름.
        건너뛰지 않으면 멈춘 동안 속도를 0 가까이로 낮췄다가 따라잡는 움직임을 놓쳐서, 그 전에 보인 노트들의
        도달이 수십 ms 늦게 계산된다. 일시정지처럼 길게 멈추면(FREEZE_MAX_MS 넘음) 예전처럼 처리한다."""
        prev = getattr(self, "_frz_prev", None)
        cur = sorted((b[0], int(b[2])) for b in blobs if not b[4] and not b[3])
        self._frz_prev = cur
        v = self.v or 0.0
        if prev is None or len(cur) < 2 or v < 0.4 or v * dt < 4.0 or cur != prev:
            if getattr(self, "_frz_t0", None) is not None:
                self._frz_t0 = None
            return False
        if getattr(self, "_frz_t0", None) is None:
            self._frz_t0 = self.prev_t
        if t - self._frz_t0 > self.FREEZE_MAX_MS:
            # 길게 멈춤(일시정지 등): 건너뛰기를 그만두고, 멈춘 마지막 프레임부터 이어서 예전처럼
            self.prev_t = max(self.prev_t, getattr(self, "_frz_last", self.prev_t))
            return False
        self._frz_last = t
        return True

    def _drop_static(self, blobs, t):
        """화면에 고정된 그림(SIDE TRACK 알람 상자, FAST/SLOW 글자 등)을 '가림 구역'으로 처리한다.
        - 다른 노트들은 분명히 내려오는데(2개 이상) 혼자 3프레임 연속 제자리인 덩어리 → 그 채널의 그 줄을 가림 구역으로 기억
          (제자리: 보이는 아랫면 ±1.5px·윗면 ±6px, 아랫면이 가려진 덩어리는 윗면 ±1.5px. 위아래 다 가려진 것은 판단 안 함)
        - 구역 안에 들어간 덩어리는 버리고, 구역에 걸친 덩어리는 걸친 쪽을 '가려짐'으로 바꾼다
          (노트가 알람 글자 뒤를 지나며 한 덩어리로 붙어도 아랫면이 멈춘 것처럼 보이지 않게)
        - 구역은 고정 그림이 다시 확인되지 않으면 0.5초 뒤 사라진다
        - 모든 덩어리가 멈춘 프레임(진짜 정지, 캡처가 거의 같은 화면을 준 경우)에는 새 구역을 만들지 않는다."""
        self._zclip = set()
        if not self.static_on:
            self.n_static = 0
            return blobs
        prev = self._sb
        H = self.g["occl_y"] - self.g["top"]
        top0, occ = self.g["top"], self.g["occl_y"]
        cur, moving = [], 0
        for b in blobs:
            ch, tp, bt, ct, cb = b
            k, mv = 0, False
            if not (ct and cb):
                for pch, ptp, pbt, pct, pcb, pk in prev:
                    if pch != ch or pcb != cb or pct != ct:
                        continue
                    if cb:
                        same = abs(tp - ptp) <= 1.5
                    else:
                        same = abs(bt - pbt) <= 1.5 and (ct or abs(tp - ptp) <= 6)
                    if same:
                        k = max(k, pk + 1)
                    elif not cb and 2.0 < bt - pbt < 0.25 * H:
                        mv = True
            if mv and k == 0:
                moving += 1
            cur.append((ch, tp, bt, ct, cb, k))
        self._sb = cur
        zs = self._zones
        # 디맥엔 정지가 없다(일시정지는 창이 노트를 가림). 스크롤이 움직이는 중(최근 속도가 있음)이거나
        # 다른 덩어리가 내려오고 있으면, 3프레임 연속 제자리인 것은 화면 고정 그림이다.
        # (노트가 하나뿐일 때 SIDE TRACK 알람이 스크롤을 붙잡던 문제)
        scrolling = self.v is not None and self.v > 0.3 * (self._vref or self.v) and self.v > 0.05
        if moving >= 1 or scrolling:
            zmax = 0.3 * H
            for ch, tp, bt, ct, cb, k in cur:
                # 구역은 위아래가 다 보이는 짧은 덩어리로만 만든다 (위가 잘린 롱노트가 고정 그림과 붙은 덩어리 등으로
                # 레인 전체가 구역이 되지 않게)
                if k < 2 or ct or cb or bt - tp > zmax:
                    continue
                for z in zs.setdefault(ch, []):
                    if tp <= z[1] + 2 and bt >= z[0] - 2:
                        if max(z[1], bt) - min(z[0], tp) <= zmax:
                            z[0], z[1] = min(z[0], tp), max(z[1], bt)
                        z[2] = t
                        break
                else:
                    zs[ch].append([tp, bt, t])
        for ch in list(zs):
            zs[ch] = [z for z in zs[ch] if t - z[2] <= 500.0]
            if not zs[ch]:
                del zs[ch]
        if not zs:
            self.n_static = 0
            return blobs
        # 알람 상자는 불투명하고 글자가 굴러가는 애니메이션이 있다 → 글자 줄 단위가 아니라 상자 전체를 가림 구역으로:
        # 같은 레인에서 가까운 구역끼리 합치고(틈 ≤ 레인 폭 30%) 위아래로 여유(레인 폭 12%, 최소 4px)를 둔다.
        # (롱노트 아래 끝이 상자 윗변에 걸려 멈춘 것처럼 보이거나, 굴러가는 글자 조각이 노트와 엉키던 문제)
        pad, gap = max(4.0, 0.12 * self.lw), 0.3 * self.lw
        zuse = {}
        for ch, lst in zs.items():
            m = []
            for z0, z1, zt in sorted(lst):
                if m and z0 - m[-1][1] <= gap:
                    m[-1][1] = max(m[-1][1], z1)
                else:
                    m.append([z0, z1])
            zuse[ch] = [(a - pad, b + pad, 0) for a, b in m]
        out, n = [], 0
        for ch, tp, bt, ct, cb in blobs:
            drop, clip = False, False
            for z0, z1, zt in zuse.get(ch, ()):
                if tp >= z0 - 2 and bt <= z1 + 2:          # 구역 안 → 고정 그림 자체
                    drop = True
                    break
                if z0 - 2 <= bt <= z1 + 2 and tp < z0 - 2:  # 아랫면이 구역에 걸림 → 아래쪽이 가려진 노트
                    bt, cb, clip = z0 - 3, True, True
                elif z0 - 2 <= tp <= z1 + 2 and bt > z1 + 2:  # 윗면이 구역에 걸림 → 위쪽이 가려진 노트
                    tp, ct, clip = z1 + 3, True, True
                # 노트가 구역을 완전히 덮고 위아래로 삐져나옴(롱노트 몸통) → 그대로 둠
            if drop:
                n += 1
            else:
                out.append((ch, tp, bt, ct, cb))
                if clip:
                    self._zclip.add(out[-1])
        self.n_static = n
        return out

    def _detect_speed_change(self, t):
        """노트 속도 변화(변속) 감지 — 상태줄 표시용.
        최근 250ms 스크롤 직선의 속도가 기준과 6% 넘게 다른 상태가 200ms 넘게 이어지고 그동안 서로 비슷하면 한 번.
        서서히 바뀌는 경우: 기준이 마지막 변속 때보다 6% 넘게 움직이면 한 번.
        캡처 시각이 한두 프레임 튀어 잠깐 빨라졌다 돌아오는 것은 세지 않는다.
        1.5초 넘게 노트를 못 봤으면(곡 사이) 기준을 새로 잡는다."""
        f = self.scroll_fit(t, span=250.0, nmax=40, min_n=8)
        if f is None or not f[2]:
            return
        v = f[1]
        if self._vdet_t is None or t - self._vdet_t > 1500.0:
            self._vref = self._vanchor = v
            self._vc = []
        self._vdet_t = t
        if abs(v / self._vref - 1.0) > 0.06:
            self._vc.append((t, v))
            vs = [x[1] for x in self._vc]
            if max(vs) - min(vs) > 0.05 * v:
                self._vc = self._vc[-1:]          # 아직 자리 잡는 중: 새로 시작
            elif t - self._vc[0][0] >= 200.0:
                nv = float(np.median(vs))
                self.vchanges.append((t, self._vref, nv))
                self._vref = self._vanchor = nv
                self._vc = []
            return
        self._vc = []
        self._vref = 0.97 * self._vref + 0.03 * v
        if abs(self._vref / self._vanchor - 1.0) > 0.06:
            self.vchanges.append((t, self._vanchor, self._vref))
            self._vanchor = self._vref

    def blind_fit(self, t, span=350.0, nmax=64):
        """관측이 끊겼을 때 쓸 스크롤 직선 (S, v). 최근 span ms의 관측 S에 직선을 맞추되,
        디맥은 BPM이 바뀌는 마디에서 속도가 계단처럼 바뀌므로 두 직선으로 나눠 맞춰 보고
        기울기가 12% 넘게 다르고 확실히 더 잘 맞으면 바뀐 뒤 구간만 쓴다. 점이 부족하면 None."""
        obs = [(a, b) for a, b, o in self.hist[-nmax:] if o and t - a <= span]
        n = len(obs)
        if n < 8:
            return None
        ts = np.array([o[0] for o in obs]) - t
        ss = np.array([o[1] for o in obs])

        def fit(x, y):
            A = np.vstack([x, np.ones(len(x))]).T
            (k, c), *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - (k * x + c)
            return k, c, float((r * r).sum())

        k1, c1, e1 = fit(ts, ss)
        best = None
        for j in range(4, n - 7):
            if ts[-1] - ts[j] < 60.0:          # 바뀐 뒤 구간은 점 8개·60ms 이상 (몇 프레임 튄 것을 변속으로 보지 않게)
                break
            ka, ca, ea = fit(ts[:j], ss[:j])
            kb, cb, eb = fit(ts[j:], ss[j:])
            if best is None or ea + eb < best[0]:
                best = (ea + eb, j, ka, kb, cb)
        k, c = k1, c1
        if best is not None:
            e2, j, ka, kb, cb = best
            if kb > 0.05 and abs(kb - ka) > 0.12 * max(ka, kb) and e2 < 0.5 * e1:
                k, c = kb, cb
        if k <= 0.05:
            return None
        return float(c), float(k)

    def scroll_fit(self, t, span=100.0, nmax=14, min_n=5):
        """최근 관측된 스크롤 위치들에 직선을 맞춰 지금(t) 위치와 속도를 구한다.
        한두 프레임이 늦게 찍히거나(게임/캡처 끊김) 위치가 몇 px 튀어도 예상이 흔들리지 않게,
        크게 벗어난 점은 빼고 다시 맞춘다. 반환 (S, v, ok) / 점이 부족하면 None.
        ok: 남은 점들이 직선에 3px 안으로 붙어 있음 (속도가 일정 = 예상을 믿어도 됨)."""
        obs = [(a, b) for a, b, o in self.hist[-nmax:] if o and t - a <= span]
        if len(obs) < min_n:
            return None
        ts = np.array([o[0] for o in obs]) - t
        ss = np.array([o[1] for o in obs])
        keep = np.ones(len(ts), bool)
        drop_max = max(2, len(ts) // 4)
        while True:
            A = np.vstack([ts[keep], np.ones(keep.sum())]).T
            k, c0 = np.linalg.lstsq(A, ss[keep], rcond=None)[0]
            r = ss - (k * ts + c0)
            ra = np.where(keep, np.abs(r), -1.0)
            w = int(np.argmax(ra))
            if ra[w] <= 3.0:
                break
            if (~keep).sum() >= drop_max or keep.sum() <= 5:
                return float(c0), float(k), False
            keep[w] = False                       # 가장 크게 벗어난 점(늦게 찍힌 프레임 등)을 빼고 다시
        ok = k > 0.05
        return float(c0), float(k), bool(ok)

    def predictions(self, t, horizon=400.0):
        """아직 판정선에 닿지 않은 노트의 '예상' 도달 시각 [(nid, ch, t_pred, is_long, reliable)].
        키를 누르는 순간 바로 오차를 계산하기 위해 쓴다 (실제 도달 시 정확한 값으로 교체).
        reliable: 지금 화면의 노트들로 스크롤을 실제로 보고 있고, 최근 스크롤이 일정한 속도로 움직임."""
        v = self.v or 0.0
        if v <= 0.05:
            return []
        J = self.g["judge_y"]
        S = self.S
        rel = (self.last_obs_t is not None and t - self.last_obs_t <= 40.0
               and not self.pending and len(self.hist) >= 3 and all(o for _, _, o in self.hist[-3:]))
        f = self.scroll_fit(t)
        if f is not None:                      # 점이 부족하면(저fps) 예전 방식 그대로
            if f[2]:
                S, v = f[0], f[1]
            else:
                rel = False
        out = []
        for nt in self.notes:
            if nt.done or nt.anchor or nt.n < 3:
                continue
            dt = (J - nt.p_center - S) / v
            if -30.0 < dt < horizon:
                out.append((nt.nid, nt.ch, t + dt, nt.long, rel and nt.n >= 5))
        return out

    def long_thr(self):
        """이 높이보다 크면 롱노트. 노트 두께를 충분히 재기 전에는 넉넉하게 (동그란·큰 노트를 롱노트로 오해하지 않게)"""
        if len(self._tap_h) >= 15:
            return 1.8 * self.head_h
        return 1.3 * self.lw

    def _register_new(self, blobs, t, force=False):
        hh = self.head_h
        lim = self.g["top"] + (self.g["occl_y"] - self.g["top"]) * 0.6
        for b in blobs:
            ch, tp, bt, ct, cb = b
            if cb:
                continue
            if not force and b in self._zclip:
                continue                 # 고정 그림(가림 구역)에 걸려 잘린 조각: 기존 노트 짝짓기에만 씀
            if not force and bt > lim and not ct:
                continue                 # 갑자기 아래쪽에 나타난 것은 잡음으로 간주
            hgt = bt - tp + 1
            is_long = hgt > self.long_thr()
            h = hh if is_long else hgt
            self.notes.append(Note(ch, bt - self.S, h, t, is_long))

    def _update_v(self):
        obs = [(t, s) for t, s, o in self.hist[-5:] if o]
        if len(obs) >= 3:
            ts = np.array([o[0] for o in obs])
            ss = np.array([o[1] for o in obs])
            if ts[-1] - ts[0] > 1e-3:
                v = float(np.polyfit(ts - ts[-1], ss, 1)[0])
                if v >= -0.05:
                    self.v = max(0.0, v)

    def _bootstrap(self, blobs, t):
        """속도를 모르거나 놓쳤을 때: 직전 프레임 대비 같은 채널 아랫면 이동량으로 추정"""
        cur = [(ch, bt) for ch, tp, bt, ct, cb in blobs if not cb]
        if self._boot_prev is not None and 0 < t - self._boot_prev[0] < 100:
            pt, prev = self._boot_prev
            dt = t - pt
            ds = []
            for ch, bt in cur:
                for ch2, bt2 in prev:
                    if ch == ch2 and 0 < bt - bt2 < 0.25 * (self.g["occl_y"] - self.g["top"]):
                        ds.append((bt - bt2) / dt)
            if len(ds) >= 2:
                ds = np.array(ds)
                hist, edges = np.histogram(ds, bins=np.arange(0, ds.max() + 0.1, 0.05))
                k = hist.argmax()
                sel = ds[(ds >= edges[k] - 0.05) & (ds <= edges[k + 1] + 0.05)]
                if len(sel) >= 2:
                    self.v = float(np.median(sel))
                    self._lost = 0

    @property
    def speed(self):
        return self.v


DEFAULT_COLORS = {
    "v_min": 140,          # 밝기 기준
    "sat_min": 120,
    "trig_h": [0, 13],     # 트리거(붉은 주황) 색상 범위 (OpenCV H: 0~180)
    "side_h": [80, 97],    # 사이드 트랙(하늘색)
    "wide_frac": 0.2,      # 반폭에서 이 비율 이상이면 트리거/사이드로 인정 (가려져도 OK)
    "lane_frac": 0.45,     # 레인 안쪽 샘플 중 이 비율 이상 밝으면 노트 (동그라미·아이콘 모양도 잡히게)
    "gap_frac": 0.35,      # 속 빈 테두리 노트의 위·아래 테두리 사이를 메울 최대 길이 (레인 폭 비율)
    "max_blind_ms": 120,   # 스크롤 관측이 이만큼 끊기면 신뢰도 낮음
    "note_ttl_ms": 3000,
}


class LaneCounter:
    """화면에 보이는 일반 노트의 가로 폭으로 레인 수(4/5/6)를 자동 판별."""

    def __init__(self, geom: dict, colors: dict):
        self.g, self.c = geom, colors
        self.W = geom["x1"] - geom["x0"]
        self.samples = []
        self.last_scores = {}

    def feed(self, frame):
        top, occ = int(self.g["top"]), int(self.g["occl_y"])
        x0, x1 = int(self.g["x0"]), int(self.g["x1"])
        rows = frame[top:occ:3, x0:x1]
        hsv = cv2.cvtColor(np.ascontiguousarray(rows), cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        c = self.c
        colored = S >= c["sat_min"]
        wide = colored & (((H >= c["trig_h"][0]) & (H <= c["trig_h"][1])) |
                          ((H >= c["side_h"][0]) & (H <= c["side_h"][1])))
        m = (V >= c["v_min"]) & ~wide
        if self.g.get("mask"):
            a, b = self.g["mask"]
            ra, rb = int((a - top) // 3), int((b - top) // 3) + 1
            m[max(0, ra):max(0, rb)] = False
        lo, hi = self.W / 7.5, self.W / 3.3
        # 모든 줄의 연속 구간을 한 번에 (줄 순서·왼쪽부터 순서는 예전과 같음)
        pad = np.zeros((m.shape[0], m.shape[1] + 2), np.int8)
        pad[:, 1:-1] = m
        d = np.diff(pad, axis=1)
        st = np.nonzero(d == 1)[1]
        en = np.nonzero(d == -1)[1]
        if len(st):
            ln = en - st
            ok = (ln >= lo) & (ln <= hi)
            self.samples.extend(zip(ln[ok].tolist(), (st[ok] + ln[ok] / 2.0).tolist()))
        if len(self.samples) > 4000:
            self.samples = self.samples[-4000:]

    def reset(self):
        self.samples = []

    def guess(self, min_samples=800):
        if len(self.samples) < min_samples:
            return None
        arr = np.array(self.samples)
        scores = {}
        for n in (4, 5, 6):
            lw = self.W / n
            # 노트 중심이 레인 중심 근처에 있고, 폭이 레인 폭과 맞는 비율
            centers = (np.arange(n) + 0.5) * lw
            dc = np.min(np.abs(arr[:, 1][:, None] - centers[None, :]), axis=1)
            ok = (dc < 0.12 * lw) & (arr[:, 0] > 0.75 * lw) & (arr[:, 0] < 1.05 * lw)
            scores[n] = ok.mean()
        self.last_scores = {k: round(float(v), 2) for k, v in scores.items()}
        order = sorted(scores, key=scores.get, reverse=True)
        b, s2 = order[0], scores[order[1]]
        if scores[b] > 0.35 and scores[b] - s2 > 0.15:
            return b
        return None



class FastSlowText:
    """게임이 띄우는 FAST(왼쪽)/SLOW(오른쪽) 노란 글자가 '새로 나타나는 순간'을 잡는다.
    (보정이 필요한지 알려주는 용도로만 쓴다. 영점을 자동으로 바꾸지 않음)
    글자는 나타날 때 가장 밝고 0.6초쯤에 걸쳐 어두워지므로, 밝기가 갑자기 올라가는 순간을 새 글자로 본다.
    노트(가로로 길게 이어지고 움직임)와 글자(끊겨 있고 제자리)를 구분한다."""

    DEFAULT = {"y_frac": 0.555, "band_frac": 0.03,
               "fast_x": [0.19, 0.41], "slow_x": [0.71, 0.88]}

    def __init__(self, geom: dict, opts: dict | None = None):
        o = dict(self.DEFAULT)
        o.update(opts or {})
        x0, x1, top, J = geom["x0"], geom["x1"], geom["top"], geom["judge_y"]
        H, W = J - top, x1 - x0
        yc = J - H * o["y_frac"]
        b = max(6, H * o["band_frac"])
        self.ya, self.yb = int(max(0, yc - b)), int(yc + b)
        self.boxes = {
            "fast": (int(x0 + W * o["fast_x"][0]), int(x0 + W * o["fast_x"][1])),
            "slow": (int(x0 + W * o["slow_x"][0]), int(x0 + W * o["slow_x"][1])),
        }
        self.prev = {"fast": 0.0, "slow": 0.0}
        self.cand = {"fast": None, "slow": None}
        self.max_run = max(8, int(W * 0.09))

    def feed(self, frame, t):
        band = frame[self.ya:self.yb]
        if band.size == 0:
            return []
        out = []
        for kind, (xa, xb) in self.boxes.items():
            hsv = cv2.cvtColor(np.ascontiguousarray(band[:, xa:xb]), cv2.COLOR_BGR2HSV)
            Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            m = (Hh >= 15) & (Hh <= 38) & (S > 110) & (V > 60)
            fill = float(m.mean())
            longest = 0
            if fill > 0.03:
                for r in m:
                    if r.any():
                        d = np.diff(np.concatenate([[0], r.astype(np.int8), [0]]))
                        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
                        longest = max(longest, int((en - st).max()))
            note_like = fill > 0.33 or longest > self.max_run
            v = float(V[m].mean()) if (fill > 0.03 and not note_like) else 0.0
            cand = self.cand.get(kind)
            if cand is not None:
                ct, cm = cand
                inter = (m & cm).sum()
                union = (m | cm).sum()
                if not note_like and union > 0 and inter / union > 0.6 and v > 130:
                    out.append((kind, ct))
                self.cand[kind] = None
            if not note_like and v > 150 and v - self.prev[kind] > 40:
                self.cand[kind] = (t, m.copy())
            if not note_like:
                self.prev[kind] = v
        return out


class JudgeLineWatch:
    """기어의 판정선이 화면에서 사라지면 곡이 끝난 것으로 본다 (결과 화면·선택 화면에는 판정선이 없음).

    - 판정선 모양: 곡이 시작될 때(첫 확실한 도달 직전 0.5~2초, 기어는 보이고 이펙트는 거의 없을 때) 판정선 줄을 기억한다.
      기어 스킨마다 색이 달라도 되게 색을 고정하지 않는다.
    - 판정 이펙트·롱노트가 판정선을 자주 가리므로, '1.2초 넘게 한 번도 안 보이고, 그동안 판정선 쪽으로 이어지는 노트(아래가
      가려진 덩어리)도 없을 때'만 곡 끝.
    - 안전장치: 그 곡에서 판정선이 충분히 자주 보였을 때만 곡 끝을 낸다 (모양을 잘못 기억했으면 아무것도 안 함).
    feed() 반환: "end"(곡 끝) / "start"(판정선이 다시 보임) / None
    """

    def __init__(self, geom: dict, ncol: int = 64):
        x0, x1 = geom["x0"], geom["x1"]
        self.xs = np.linspace(x0 + 2, x1 - 2, ncol).astype(int)
        self.y = int(round(geom["judge_y"]))
        self.buf = []            # (t, band) 최근 2.5초
        self.tmpl = None
        self.last_seen = None
        self.seen_n = 0
        self.samp_n = 0
        self.playing = False
        self.ended = False
        self.last_cover = None
        self.first_arr = None
        self.last_arr = None
        self._tops = []          # 최근 몇 프레임의 '아래가 가려진 덩어리' 윗끝 {레인: [y]}
        self._gated = []         # '곡 끝' 상태에서 버려진 확실한 도달 시각 (다시 시작 감지용)

    def _band(self, frame):
        y0 = max(0, self.y - 2)
        return frame[y0:self.y + 3].take(self.xs, axis=1).astype(np.float32)

    def visible(self, band):
        if self.tmpl is None or band.shape != self.tmpl.shape:
            return False
        d = np.abs(band - self.tmpl).mean(axis=(0, 2))
        return float((d < 40).mean()) > 0.45

    def cover_from(self, blobs, t):
        """판정선 쪽으로 이어지는 '진짜 노트'가 있는지.
        아래가 가려진 덩어리 중 윗끝이 내려오고 있는 것(롱노트 몸통)만 센다.
        위아래가 다 잘린 덩어리(레인을 꽉 채운 롱노트)는 마지막 도달 4초 안에서만 센다.
        (곡 끝 효과·결과 화면 그림처럼 제자리에 있는 덩어리 때문에 곡 끝을 못 잡던 문제)"""
        cur, res = {}, False
        for b in blobs:
            ch, tp, bt, ct, cb = b[:5]
            if not cb:
                continue
            if ct:
                if self.last_arr is not None and t - self.last_arr < 4000.0:
                    res = True
                continue
            cur.setdefault(ch, []).append(tp)
            for old in self._tops:
                if any(1.5 < tp - y < 120 for y in old.get(ch, ())):
                    res = True
                    break
        self._tops.append(cur)
        if len(self._tops) > 6:
            self._tops.pop(0)
        return res

    REOPEN_N = 10          # '곡 끝' 상태에서 확실한 도달이 이만큼
    REOPEN_SPAN = 1500.0   # 이 시간(ms) 넘게 이어지면 곡이 다시 시작된 것으로 봄
    REOPEN_WIN = 3000.0

    def _regrab(self, t, vis):
        """곡 중 판정선이 잘 안 보이면(노트·이펙트가 많은 순간에 모양을 기억한 경우) 2.5초마다
        최근 2.5초 화면의 중앙값으로 모양을 다시 만들어 보고, 그게 최근 화면과 더 잘 맞을 때만 바꾼다.
        (곡 끝을 못 잡던 문제: 안전장치 '판정선이 15% 넘게 보여야 곡 끝'에 막힘)"""
        self._w_n = getattr(self, "_w_n", 0) + 1
        self._w_seen = getattr(self, "_w_seen", 0) + vis
        if t - getattr(self, "_w_t", -1e18) < 2500.0:
            return
        r = self._w_seen / max(1, self._w_n)
        self._w_t, self._w_n, self._w_seen = t, 0, 0
        if r >= 0.3 or len(self.buf) < 60:
            return
        bands = np.stack([b for _, b in self.buf])
        cand = np.median(bands, axis=0)

        def score(tm):
            d = np.abs(bands - tm[None]).mean(axis=(1, 3))
            return int(((d < 40).mean(axis=1) > 0.45).sum())
        if self.tmpl is None or score(cand) > score(self.tmpl) * 1.5 + 5:
            self.tmpl = cand
            self.seen_n = self.samp_n = 0

    def feed(self, frame, t, arrived: bool, covered: bool, gated: int = 0):
        """arrived: 이번 프레임에 확실한 도달이 있었음 / covered: 판정선 쪽으로 이어지는 노트가 보임
        gated: '곡 끝' 상태라서 버려진 이번 프레임의 확실한 도달 수"""
        try:
            band = self._band(frame)
        except Exception:
            return None
        self.buf.append((t, band))
        while self.buf and t - self.buf[0][0] > 2500.0:
            self.buf.pop(0)
        if covered:
            self.last_cover = t
        if arrived:
            self.last_arr = t
        ev = None
        if arrived and not self.playing:
            # 곡 시작: 첫 도달 0.5~2초 전 판정선 줄의 중앙값을 판정선 모양으로
            old = [b for tt, b in self.buf if 500.0 <= t - tt <= 2000.0]
            if len(old) >= 5:
                self.tmpl = np.median(np.stack(old), axis=0)
            self.playing, self.ended = True, False
            self.seen_n = self.samp_n = 0
            self.last_seen = t
        vis = self.visible(band)
        if vis:
            self.last_seen = t
            if self.ended:
                self.ended = False
                self._gated = []
                ev = "start"
        elif self.ended and gated:
            # 기억한 판정선 모양이 안 보여도(다른 버튼 수·기어, 선곡 화면 그림을 잘못 기억한 경우) 확실한 도달이
            # 1.5초 넘게 꾸준히 나오면 곡이 시작된 것: 다시 받고, 판정선 모양은 다음 도달 때 이 곡 기준으로 새로 기억
            self._gated.append(t)
            while self._gated and t - self._gated[0] > self.REOPEN_WIN:
                self._gated.pop(0)
            if len(self._gated) >= self.REOPEN_N and t - self._gated[0] >= self.REOPEN_SPAN:
                self.ended = False
                self.playing = False
                self._gated = []
                ev = "start"
        if self.playing:
            self.samp_n += 1
            self.seen_n += vis
            self._regrab(t, vis)
            if arrived:
                self.last_seen = max(self.last_seen or t, t - 300.0)
            quiet = t - (self.last_seen or t)
            cov = self.last_cover is not None and t - self.last_cover < 1200.0
            if quiet > 1200.0 and not cov and self.samp_n > 200 and self.seen_n > 0.15 * self.samp_n:
                self.playing, self.ended = False, True
                ev = "end"
        return ev
