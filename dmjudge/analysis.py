"""키 입력 ↔ 노트 도달 매칭, 통계, 추천 판정값 계산 (플랫폼 독립)."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class Matcher:
    """채널별로 '눌린 시각'과 '노트 중앙이 판정선 정중앙에 닿은 시각'을 짝짓는다.

    e = t_press - t_arrival  (ms, 화면 기준. 음수 = 빨리 침)
    도달 이벤트는 실시간으로 들어오므로, t_arrival + window 가 지난 뒤에 확정한다.
    """

    def __init__(self, window_ms=100.0, chatter_ms=25.0, early_ms=None):
        self.W = window_ms                  # 늦게 친 쪽 최대 거리
        self.WE = window_ms if early_ms is None else early_ms   # 일찍 친 쪽 최대 거리 (게임은 빠른 쪽을 더 넓게 받음)
        # 짝짓기 중심: 화면 도달 시각 + center (= 실제 판정 중심).
        # 판정 타이밍을 +N 으로 올리면 보이지 않는 판정선이 N ms 위에 있으므로 center = -(N + 보정값)
        self.center = 0.0
        self.fast_win = 42.0
        self.fs = 42.0                      # FAST/SLOW 범위 (판정 기준 ±ms)
        self.events = []                    # ("fast"/"slow", ch, 발생 시각, 오차)
        self.provisional = {}               # 확정 대기 중인 늦은 입력 (인디케이터 즉시 표시용)
        self.chatter_ms = chatter_ms
        self.presses = defaultdict(list)    # ch -> [t, used, chatter]
        self.arrivals = defaultdict(list)   # ch -> [t, conf, long]
        self.hits = []                      # (t_arr, ch, e, conf)
        self.missed = defaultdict(int)
        self.phantom = 0                    # 가짜 도달(전환 화면 등)로 걸러낸 수
        self._ph_t = []                     # 가짜 묶음이 나온 시각
        self.ghost = defaultdict(int)
        self.chatter = defaultdict(int)     # key_id -> 횟수
        self._last_up = {}
        self._down = {}
        self._any_press = deque(maxlen=64)  # 레인 상관없이 최근 누른 시각 (아무것도 안 누른 구간의 '놓침'은 세지 않음)

    def on_key(self, key_id, is_down, t, ch):
        if is_down:
            ch_ = ch
            lu = self._last_up.get(key_id)
            chat = lu is not None and (t - lu) < self.chatter_ms
            if chat:
                self.chatter[key_id] += 1
            if ch_ is not None:
                self.presses[ch_].append([t, False, chat])
                self._any_press.append(t)
        else:
            self._last_up[key_id] = t

    PHANTOM_MS = 25.0      # 같은 레인에 이보다 가깝게 붙은 도달은 노트가 아님 (실제 채보 최소 간격은 40ms 이상)
    PHANTOM_SPAN = 300.0

    def on_arrival(self, ch, t, conf, is_long, nid=None):
        arr = self.arrivals[ch]
        # 곡 시작·전환 화면의 번쩍임/와이프는 같은 레인에 몇 ms 간격으로 '도달'을 줄줄이 만든다
        # → 서로 붙어 있는 것끼리 전부 가짜로 표시 (짝짓기·놓침 집계에서 빠짐).
        # 그런 묶음이 나온 앞뒤 PHANTOM_SPAN ms 안의 다른 레인 도달도 같은 화면 효과로 보고 뺀다.
        fake = any(abs(pt - t) < self.PHANTOM_SPAN for pt in self._ph_t[-8:])
        for a in arr[-6:]:
            if abs(a[0] - t) < self.PHANTOM_MS:
                a[1] = -1.0
                fake = True
                self._ph_t.append(t)
        if fake:
            sp = self.PHANTOM_SPAN
            for lst in self.arrivals.values():
                for a in lst[-12:]:
                    if a[1] >= 0 and abs(a[0] - t) < sp:
                        a[1] = -1.0
                        self.phantom += 1
            self.phantom += 1
            conf = -1.0
        arr.append([t, conf, is_long, False, nid])   # [3] = SLOW 조기 확정 여부, [4] = 노트 번호

    def resolve(self, now):
        """now 시각까지 확정할 수 있는 노트를 확정한다.
        - 입력이 있으면: 앞으로 올 어떤 입력도 이보다 가까울 수 없게 된 순간 바로 확정
          (빠르게 친 노트, 범위 안 노트는 즉시 / 범위 밖 늦은 입력은 창이 끝날 때)
        - 판정 중심 + FAST/SLOW 범위가 지났는데 아직 입력이 없으면: 그 순간 SLOW 확정 이벤트
        """
        W = self.W
        WE = self.WE
        fs = self.fs
        c = self.center
        for ch in list(self.arrivals.keys()):
            arr = self.arrivals[ch]
            prs = self.presses.get(ch, [])
            done = 0
            for i, a in enumerate(arr):
                ta, conf = a[0], a[1]
                if conf < 0:                     # 가짜 도달: 버림
                    self.provisional.pop(id(a), None)
                    done = i + 1
                    continue
                tc = ta + c                      # 이 노트의 판정 중심 시각
                best, bd = None, W + 1
                for p in prs:
                    if p[1] or p[0] > now or p[0] - tc > W or tc - p[0] > WE:
                        continue
                    d = abs(p[0] - tc)
                    # 다음 노트에 더 가까운 입력은 이 노트 것이 아님
                    if any(b[1] >= 0 and abs(p[0] - (b[0] + c)) < d for b in arr[i + 1:i + 4]):
                        continue
                    # 게임처럼 '창 안에서 먼저 누른 입력'이 이 노트의 판정 (더 가까운 나중 입력이 아니라)
                    if best is None or p[0] < best[0]:
                        best, bd = p, d
                if best is not None:
                    dd = best[0] - tc
                    sure = True                  # 먼저 누른 입력이 정해졌으면 더 기다릴 것 없음
                    if sure or now >= tc + W:
                        best[1] = True
                        self.provisional.pop(id(a), None)
                        self.hits.append((ta, ch, best[0] - ta, conf, a[4]))
                        if conf >= 1.0:
                            if dd < -fs:
                                self.events.append(("fast", ch, now, dd, a[4]))
                            elif dd > fs and not a[3]:
                                self.events.append(("slow", ch, now, dd, a[4]))
                        done = i + 1
                        continue
                    # 늦게 친 입력: 통계 확정은 창 끝까지 기다리지만, 인디케이터에는 지금 바로 찍는다
                    self.provisional[id(a)] = (ta, ch, best[0] - ta, conf, a[4])
                if now >= tc + W:
                    self.provisional.pop(id(a), None)
                    # 앞뒤로 키를 하나도 안 누르던 구간(곡 시작 전 화면, 데모 재생 등)은 놓침으로 안 셈
                    if conf >= 1.0 and any(abs(p - ta) < 1500.0 for p in self._any_press):
                        self.missed[ch] += 1
                    done = i + 1
                    continue
                break
            if done:
                del arr[:done]
            # 오래된 입력 정리
            keep = []
            for p in prs:
                if now - p[0] > 3 * max(W, WE):
                    if not p[1]:
                        self.ghost[ch] += 1
                    continue
                keep.append(p)
            if ch in self.presses:
                self.presses[ch] = keep

    def reset(self):
        c, fw, fs = self.center, self.fast_win, self.fs
        self.__init__(self.W, self.chatter_ms, self.WE)
        self.center, self.fast_win, self.fs = c, fw, fs


# ------------------------------------------------------------ 점수 (게임 판정 단계)
# 100% 범위 바깥으로 100% 범위의 0.4배(노멀 16.67ms = 1/60초)마다 한 단계씩 90% → 80% … 10% → 1%, 그 밖은 BREAK.
GRADE_PCT = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1)      # 레벨 0~10, 레벨 11 = BREAK
GRADE_PTS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0)   # 1%는 10점 미만이라 0점으로 셈
MAX_SCORE = 1000000
BREAK_LV = len(GRADE_PCT)


def grade_levels(j, win: float) -> np.ndarray:
    """판정 기준 오차(ms) → 판정 레벨 (0 = 100%, 1 = 90% … 10 = 1%, 11 = BREAK)"""
    a = np.abs(np.asarray(j, dtype=float))
    step = float(win) * 0.4
    lv = np.ceil((a - float(win)) / step)
    lv = np.where(a <= float(win), 0, lv)
    return np.minimum(lv, BREAK_LV).astype(int)


def judge_score(j, win: float, n_break: int = 0):
    """만점 1,000,000점을 노트 수로 나눠 노트 하나의 점수, 판정 %만큼 받음 (1%·BREAK = 0점).
    반환 dict(score 점수, rate 퍼센트, counts 레벨별 개수, n 노트 수) 또는 노트가 없으면 None"""
    lv = grade_levels(j, win)
    counts = np.bincount(lv, minlength=BREAK_LV + 1)
    counts[BREAK_LV] += int(n_break)
    n = int(counts.sum())
    if n == 0:
        return None
    rate = float((counts * np.array(GRADE_PTS)).sum()) / n
    return {"score": int(round(MAX_SCORE * rate)), "rate": 100.0 * rate, "counts": counts.tolist(), "n": n}


def score_from_counts(counts):
    """판정 개수 [100%, 90%, 80% … 10%, 1%, BREAK] (12개, 게임 결과 화면과 같은 순서) → judge_score와 같은 dict"""
    c = np.asarray(list(counts)[:BREAK_LV + 1], dtype=float)
    if len(c) < BREAK_LV + 1 or c.sum() <= 0:
        return None
    n = int(c.sum())
    rate = float((c * np.array(GRADE_PTS)).sum()) / n
    return {"score": int(round(MAX_SCORE * rate)), "rate": 100.0 * rate, "counts": [int(x) for x in c], "n": n}


def summarize(e: np.ndarray, judge_setting: float, win: float, bias: float = 0.0, fs: float = 0.0,
              rec: bool = True):
    """e: 화면 기준 오차(ms) 배열. 반환: 통계 dict"""
    n = len(e)
    if n == 0:
        return {"n": 0}
    j = e + judge_setting + bias          # 판정 기준 오차
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med)) * 1.4826)
    res = {
        "n": n,
        "mean_e": float(e.mean()),
        "median_e": med,
        "std": float(e.std()),
        "robust_std": mad,
        "mean_j": float(j.mean()),
        "median_j": float(np.median(j)),
        "fast": int((j < -fs).sum()),        # FAST/SLOW 기준(±fs ms) 밖
        "slow": int((j > fs).sum()),
        "in_now": float((np.abs(j) <= win).mean()),
        "rate_now": float(judge_score(j, win)["rate"]),     # 예상 정확도(점수 %): 판정 단계별 점수로
        "early_now": int((j < -win).sum()),
        "late_now": int((j > win).sum()),
    }
    if rec:
        # 계산량 제한: 최근 3000개로 추천값 계산 (화면 반응이 끊기지 않게)
        res.update(recommend(e[-3000:], win, bias))
    return res


def recommend(e: np.ndarray, win: float, bias: float = 0.0, lo=-120, hi=120):
    """예상 정확도(점수 %)가 가장 높아지는 판정 타이밍 값(정수 ms)."""
    if len(e) == 0:
        return {}
    ss = np.arange(lo, hi + 1)
    je = e[None, :] + ss[:, None] + bias
    a = np.abs(je)
    cnt = (a <= win).sum(1)
    lv = np.where(a <= win, 0, np.minimum(np.ceil((a - win) / (0.4 * win)), BREAK_LV)).astype(int)
    rate = np.asarray(GRADE_PTS)[lv].mean(1)
    mx = rate.max()
    plateau = ss[rate >= mx - 1e-12]
    # 최대 구간이 여러 개면 중앙값 기준점에 가장 가까운 구간의 가운데
    center = -(np.median(e) + bias)
    groups = np.split(plateau, np.where(np.diff(plateau) > 1)[0] + 1)
    g = min(groups, key=lambda gg: abs((gg[0] + gg[-1]) / 2 - center))
    s_best = int(round((g[0] + g[-1]) / 2))
    margin_lo = win - (-(e + s_best + bias).min())   # 가장 빠른 노트까지 남은 여유
    margin_hi = win - (e + s_best + bias).max()
    return {
        "best_setting": s_best,
        "in_best": float(cnt[s_best - lo] / len(e)),
        "rate_best": float(rate[s_best - lo] * 100.0),
        "margin_early": float(margin_lo),
        "margin_late": float(margin_hi),
    }


def histogram(values, lo=-100, hi=100, step=4):
    edges = np.arange(lo, hi + step, step)
    h, _ = np.histogram(np.clip(values, lo, hi - 1e-6), bins=edges)
    return edges, h


def calibrate_from_game(e: np.ndarray, judge_setting: float, win: float, n_fast: int, n_slow: int):
    """게임 결과(100% 밖 FAST 개수, SLOW 개수)와 가장 잘 맞는 보정값(ms)을 찾는다.
    반환: (보정값, 그때의 FAST 밖 개수, SLOW 밖 개수)"""
    if len(e) == 0:
        return None
    bs = np.arange(-80, 80.01, 0.5)
    j = e[None, :] + judge_setting + bs[:, None]
    ce = (j < -win).sum(1)
    cl = (j > win).sum(1)
    cost = np.abs(ce - n_fast) + np.abs(cl - n_slow)
    best = bs[cost == cost.min()]
    groups = np.split(best, np.where(np.diff(best) > 0.51)[0] + 1)
    g = max(groups, key=len)
    b = float((g[0] + g[-1]) / 2)
    k = int(np.argmin(np.abs(bs - b)))
    return b, int(ce[k]), int(cl[k])


def calibrate_from_total(e: np.ndarray, judge_setting: float, win: float, n_out: int, lo=-40.0, hi=40.0):
    """게임 결과 화면의 'MAX 100%가 아닌 노트 수'(MAX 90% 이하 합계, BREAK 제외)와 맞는 보정값.
    디맥 결과 화면은 FAST/SLOW를 나눠 보여주지 않으므로 합계만으로 찾는다.
    개수가 맞는 구간이 여러 개면 0에 가장 가까운 구간의 가운데. 반환: (보정값, FAST 밖, SLOW 밖, 구간 폭)"""
    if len(e) == 0:
        return None
    bs = np.arange(lo, hi + 0.01, 0.5)
    j = e[None, :] + judge_setting + bs[:, None]
    ce = (j < -win).sum(1)
    cl = (j > win).sum(1)
    cost = np.abs(ce + cl - n_out)
    best = bs[cost == cost.min()]
    groups = np.split(best, np.where(np.diff(best) > 0.51)[0] + 1)
    g = min(groups, key=lambda g: (abs((g[0] + g[-1]) / 2), -len(g)))
    b = float((g[0] + g[-1]) / 2)
    k = int(np.argmin(np.abs(bs - b)))
    return b, int(ce[k]), int(cl[k]), float(g[-1] - g[0])


def calibrate_joint(songs, win: float, lo=-40.0, hi=40.0):
    """여러 곡의 (화면 기준 오차 배열, 판정 타이밍, 게임의 100% 아닌 개수)로 한 번에 보정값 찾기.
    곡마다 '분석기 기준 범위 밖 개수'와 게임 숫자의 차이 합이 가장 작은 값.
    반환: (보정값, 맞는 구간 시작, 끝, 차이 합) 또는 None"""
    songs = [s for s in songs if len(s[0])]
    if not songs:
        return None
    bs = np.arange(lo, hi + 0.01, 0.5)
    cost = np.zeros(len(bs))
    for e, js, n in songs:
        j = np.asarray(e)[None, :] + float(js) + bs[:, None]
        cost += np.abs(((j < -win) | (j > win)).sum(1) - int(n))
    best = bs[cost == cost.min()]
    groups = np.split(best, np.where(np.diff(best) > 0.51)[0] + 1)
    g = min(groups, key=lambda g: (abs((g[0] + g[-1]) / 2), -len(g)))
    return float((g[0] + g[-1]) / 2), float(g[0]), float(g[-1]), int(cost.min())


def suggest_zero(j_raw: np.ndarray, labels: np.ndarray, win: float, w_label: float = 3.0, lo=-40.0, hi=40.0):
    """게임 FAST/SLOW 글자와 가장 잘 맞는 영점 보정값 (추천용, 자동 적용하지 않음).
    j_raw: 보정 전 판정 기준 오차, labels: -1(게임 FAST) / +1(게임 SLOW) / 0(글자 없음)"""
    if len(j_raw) == 0 or not (labels != 0).any():
        return None
    bs = np.arange(lo, hi + 0.01, 0.5)
    J = j_raw[None, :] + bs[:, None]
    L = labels[None, :]
    agree = np.where(L > 0, (J > win) * w_label, np.where(L < 0, (J < -win) * w_label, (np.abs(J) <= win) * 1.0))
    sc = agree.sum(1)
    best = bs[sc >= sc.max() - 1e-9]
    groups = np.split(best, np.where(np.diff(best) > 0.51)[0] + 1)
    g = max(groups, key=len)
    return float((g[0] + g[-1]) / 2)


# 영점 거울 해를 게임 FAST/SLOW 글자로 가를 때: 실시간 글자 짝짓기는 틀린 노트에 붙는 일이 많아서(실측 절반 가까이) 보수적으로
LAB_SIGMA = 5.0
LAB_NOISE = 0.3
LAB_MIN = 8
LAB_MARGIN = 4.0
# 곡마다 추가 흔들림(추적 오류 곡·결과 숫자와 안 맞는 노트): 분산에 더함. 0이면 예전과 같음.
# 사용자 22곡 실측: 예전(0, 0)은 3~5곡일 때 "95% 범위"에 실제 값이 66~68%만 들어갔음 → (0.5, 0.2)로 95~98%
OVER_A = 0.5
OVER_B = 0.2


def _score_ll(e, ex, win, ks, sigma, phi):
    """점수만 아는 곡(래더)의 영점 K별 로그 우도.
    잃은 점수 D = (1 - 점수/1,000,000) x 노트 수 = 노트마다 (100% - 판정%)의 합 (1%·BREAK는 1).
    분석기 오차 + K로 노트마다 판정 단계를 정해 잃은 점수의 기댓값·분산을 구한다(단계 경계 근처는 측정 흔들림만큼 나눔).
    분석기가 못 본 BREAK 개수 b는 모르므로, 지금까지 결과 화면에서 본 BREAK·1% 비율(lam)로 b를 평균낸다."""
    from math import lgamma, log
    N = max(float(ex.get("N") or len(e)), float(len(e)))
    D = (1.0 - float(ex["score"]) / MAX_SCORE) * N
    lam = max(1e-4, float(ex.get("lam") or 0.003))
    h = float(len(e))
    loss = 1.0 - np.array(GRADE_PTS)
    step = 0.4 * win
    bs = np.arange(0, 41)
    lpb = np.array([-lam * N + b * log(lam * N) - lgamma(b + 1) for b in bs])
    kc = np.arange(ks[0], ks[-1] + 0.25, 0.25)              # 계산은 0.25ms 간격으로 하고 사이는 이어 붙임 (가볍게)
    ll = np.empty(len(kc))
    mus = np.empty(len(kc))
    for i, k in enumerate(kc):
        a = np.abs(e + k)
        # 가장 가까운 단계 경계: win + step*j (j = 0..10)
        j = np.clip(np.round((a - win) / step), 0, 10)
        edge = win + step * j
        p_out = 1.0 - phi((edge - a) / sigma)               # 경계 바깥(한 단계 아래)일 확률
        lo_lv = j.astype(int)                                # 경계 안쪽 단계 = j (0 = 100%)
        hi_lv = np.minimum(lo_lv + 1, BREAK_LV)
        L1, L2 = loss[lo_lv], loss[hi_lv]
        m_ = L1 + p_out * (L2 - L1)
        v_ = p_out * (1 - p_out) * (L2 - L1) ** 2
        T, vT = float(m_.sum()), float(v_.sum())
        mb = T * (N - bs) / h + bs                           # 못 본 BREAK b개: 나머지는 본 노트처럼
        var = vT + 0.05 + OVER_A * 0.04 + (OVER_B * mb) ** 2
        q = lpb - 0.5 * (D - mb) ** 2 / var - 0.5 * np.log(var)
        mx = q.max()
        ll[i] = mx + np.log(np.exp(q - mx).sum())
        mus[i] = T * N / h
    return np.interp(ks, kc, ll), np.interp(ks, kc, mus)


def fit_zero(songs, win: float = 125.0 / 3.0, sigma: float = 1.5, lo=-30.0, hi=40.0, step=0.05, brk=1e9):
    """결과 화면 기반 영점 보정 추천.
    songs: [(e0 배열, 게임의 MAX 100% 아닌 개수(BREAK 제외))] — e0 = 화면 기준 오차 + 판정 타이밍 (보정 전).
    게임 오차 = e0 + 보정값 이라고 보고, 노트마다 분석기 측정 흔들림(sigma)을 반영해
    '보정값 K일 때 범위 밖 개수'의 기댓값·분산을 구한 뒤, 모든 곡의 개수가 동시에 나올 확률이 가장 높은 K를 찾는다.
    (brk: 이보다 큰 오차는 BREAK로 볼 때 쓰는 값. 판정 등급이 1/60초씩 넓어져 MAX 1%가 분석기 짝짓기 범위(-190~+150)보다 넓으므로 기본은 끔)  
    반환: dict(best, lo, hi(95%), per=[(게임 개수, K=best일 때 예상 개수)]) 또는 None"""
    from math import erf, sqrt
    # songs 항목: (e0, 게임 개수[, 노트별 게임 글자 -1 FAST / +1 SLOW / 0 못 봄[, 그 곡의 100% 범위]])
    # 영점(기기 지연)은 판정 종류와 상관없으므로, 하드·맥스 곡도 그 곡의 100% 범위로 계산해 같이 쓸 수 있다.
    # 5번째 값 = 래더처럼 점수만 아는 곡: {"score": 게임 점수, "N": 노트 수, "lam": BREAK·1% 비율} (게임 개수는 None)
    raw = []
    wins = []
    extras = []
    for s_ in songs:
        e, n = s_[0], s_[1]
        lab = s_[2] if len(s_) > 2 else None
        if len(e) < 50:
            continue
        e = np.asarray(e, float)
        lab = np.asarray(lab, float) if lab is not None and len(lab) == len(e) else np.zeros(len(e))
        raw.append((e, -1 if n is None else int(n), lab))
        wins.append(float(s_[3]) if len(s_) > 3 and s_[3] else float(win))
        extras.append(s_[4] if len(s_) > 4 else None)
    songs = [(e, n) for e, n, _ in raw]
    if not songs:
        return None
    ks = np.arange(lo, hi + step / 2, step)
    _erf = np.vectorize(erf, otypes=[float])     # 경계 근처 노트가 0개일 때도 동작 (v0.3.61 시작 시 오류)

    def phi(x):
        return 0.5 * (1.0 + _erf(x / sqrt(2.0)))

    LL = np.zeros(len(ks))
    curves = []
    for (e, n), win, ex in zip(songs, wins, extras):
        e = e[np.abs(e) < 300]
        if ex is not None and ex.get("score") is not None:
            ll_s, mu_s = _score_ll(e, ex, win, ks, sigma, phi)
            LL += ll_s
            curves.append(mu_s)
            continue
        mu = np.empty(len(ks))
        var = np.empty(len(ks))
        # 범위 근처 노트만 계산 (멀리 있는 노트는 확률 0 또는 1)
        for i, k in enumerate(ks):
            d = e + k
            near = np.abs(np.abs(d) - win) < 6 * sigma
            far_out = (~near) & (np.abs(d) > win) & (np.abs(d) <= brk)
            P = 1.0 - (phi((win - d[near]) / sigma) - phi((-win - d[near]) / sigma))
            mu[i] = far_out.sum() + P.sum()
            var[i] = (P * (1 - P)).sum() + 0.25 + OVER_A + (OVER_B * mu[i]) ** 2
        LL += -0.5 * (n - mu) ** 2 / var - 0.5 * np.log(var)
        curves.append(mu)
    i = int(np.argmax(LL))
    ok = ks[LL >= LL[i] - 1.92]
    res = {"best": float(ks[i]), "lo": float(ok.min()), "hi": float(ok.max())}
    # ---- 거울 해 검사: '100% 아닌 개수'만 보면 FAST 쪽으로 밀린 값과 SLOW 쪽으로 밀린 값이 똑같이 맞을 수 있다
    #      (예: 실제는 FAST 1·SLOW 4인데 FAST 4·SLOW 1로 개수만 맞는 값). 곡이 적을 때 생김.
    #      거의 같은 높이의 봉우리가 둘 이상이면, 게임 FAST/SLOW 글자(방향)로 고르고, 글자가 모자라면 '아직 모름'.
    AMB = 3.0                                    # 우도 차이 3 = 약 20배 안쪽이면 '비슷하게 맞음'
    good = LL >= LL[i] - AMB
    peaks = []                                   # 이어진 구간마다 가장 높은 점
    j = 0
    while j < len(ks):
        if good[j]:
            k0 = j
            while j + 1 < len(ks) and good[j + 1]:
                j += 1
            seg = np.arange(k0, j + 1)
            peaks.append(int(seg[np.argmax(LL[seg])]))
        j += 1
    if len(peaks) > 1:
        E = np.concatenate([e for e, _, _ in raw])
        L = np.concatenate([lab for _, _, lab in raw])
        Wn = np.concatenate([np.full(len(e), w_) for (e, _, _), w_ in zip(raw, wins)])
        m_ = L != 0
        sc = []
        for pk in peaks:
            d = E[m_] + ks[pk]
            s3 = LAB_SIGMA
            pf = phi((-Wn[m_] - d) / s3)                  # 게임 오차가 FAST 쪽 범위 밖일 확률
            ps = 1.0 - phi((Wn[m_] - d) / s3)
            p = np.where(L[m_] < 0, pf, ps)
            sc.append(float(np.log(LAB_NOISE / 2 + (1 - LAB_NOISE) * p).sum()))   # 글자 짝짓기 실수 몫
        order = np.argsort(sc)[::-1]
        if m_.sum() >= LAB_MIN and sc[order[0]] - sc[order[1]] >= LAB_MARGIN:
            pk = peaks[int(order[0])]
            # 고른 봉우리 주변만으로 95% 범위
            a_ = pk
            while a_ > 0 and LL[a_ - 1] >= LL[pk] - 1.92:
                a_ -= 1
            b_ = pk
            while b_ + 1 < len(ks) and LL[b_ + 1] >= LL[pk] - 1.92:
                b_ += 1
            i = pk
            res = {"best": float(ks[pk]), "lo": float(ks[a_]), "hi": float(ks[b_]), "by_text": True,
                   "alt": [float(ks[q]) for q in peaks if q != pk]}
        else:
            res["ambiguous"] = [float(ks[q]) for q in sorted(peaks, key=lambda q: -LL[q])]
            res["lo"] = min(res["lo"], float(ks[min(peaks)]))
            res["hi"] = max(res["hi"], float(ks[max(peaks)]))
    res["per"] = [(n, float(c[i])) for (e, n), c in zip(songs, curves)]
    return res
