"""오버레이 창 (tkinter)."""
from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import config as C
from .keys_win import key_name

FONT = "Malgun Gothic"
BG, FG, DIM = "#14161c", "#e8e8ea", "#8a8f99"
EARLY, LATE, OKC = "#38b6ff", "#ff5a5a", "#ffd166"


def cv_w(cv):
    """캔버스 실제 폭 (아직 안 그려졌으면 300)"""
    try:
        return max(int(cv.winfo_width()), 300)
    except Exception:
        return 420


def fmt_ms(v, plus=True):
    if v is None:
        return "-"
    return ("%+.1f" if plus else "%.1f") % v


def numeric_only(widget):
    """숫자 칸(Spinbox)에 글자가 못 들어가게: 숫자, 소수점(소수 칸만), 마이너스(음수 가능한 칸만)만 허용"""
    import re as _re
    try:
        kids = widget.winfo_children()
    except Exception:
        return
    for w in kids:
        numeric_only(w)
        if w.winfo_class() != "Spinbox":
            continue
        try:
            inc = float(w.cget("increment"))
            fmt = str(w.cget("format") or "")
            lo = float(w.cget("from"))
        except Exception:
            continue
        dec = (inc % 1 != 0) or (".0f" not in fmt and "f" in fmt)
        neg = lo < 0
        pat = _re.compile("^%s\\d*%s$" % ("-?" if neg else "", "(\\.\\d*)?" if dec else ""))
        vc = w.register(lambda P, pat=pat: bool(pat.match(P)))
        try:
            w.configure(validate="key", validatecommand=(vc, "%P"))
        except Exception:
            pass


class WrapFrame(tk.Frame):
    """창 폭에 맞춰 안의 위젯들을 자동으로 줄바꿈해서 놓는 줄 (가로로 끝없이 길어지지 않게)"""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.items = []
        self._lastw = 0
        self.bind("<Configure>", self._relayout)

    def add(self, w, padl=0, padr=4):
        self.items.append((w, padl, padr))
        self._relayout()
        if len(self.items) == 1:
            try:
                self.after_idle(self._relayout)      # 위젯 크기가 계산된 뒤 한 번 더
            except Exception:
                pass
        return w

    def _relayout(self, ev=None):
        W = int(ev.width if ev is not None else self.winfo_width())
        if W <= 1:
            try:
                W = max(300, int(self.master.winfo_width()) - 16)
            except Exception:
                W = 640
            if W <= 300:
                W = 640
        if ev is not None and W == self._lastw:
            return
        if ev is not None:
            self._lastw = W
        x = y = rowh = 0
        for w, pl, pr in self.items:
            need = int(w.winfo_reqwidth()) + pl + pr
            if x > 0 and x + need > W:
                x, y, rowh = 0, y + rowh, 0
            w.place(x=x + pl, y=y)
            x += need
            rowh = max(rowh, int(w.winfo_reqheight()) + 2)
        h = max(1, y + rowh)
        try:
            cur = int(self.cget("height") or 0)
        except Exception:
            cur = -1
        if cur != h:
            self.configure(height=h)


GRAPH_NAMES = ["오차 분포", "곡 흐름 (빠름/느림 쏠림)", "밀도 (초당 노트)", "판정 단계 (FAST/SLOW 기준)",
               "곡 흐름 + 밀도 (2개 같이 보기, 표 숨김)", "오차 분포 + 판정 단계 (2개 같이 보기, 표 숨김)"]


class GraphDraw:
    _title_pad = 16          # 그래프 캔버스 위쪽 여백 (캔버스 안에 제목을 2줄로 쓸 때는 넓힘)

    def _fs(self, win):
        """FAST/SLOW 범위 (설정값). 곡 흐름·판정 단계는 이 값을 기준으로 그림"""
        try:
            v = self.ses.cfg.get("fs_ms") if hasattr(self.ses, "cfg") else None
            if v in (None, ""):
                v = self.cfg.get("fs_ms")
            return float(v) if v not in (None, "") else float(win)
        except Exception:
            return float(win)

    def _tpad(self):
        return self._title_pad

    def _draw_choice(self, g, cv1, cv2, win, hist, titles):
        """g = 고른 그래프 이름. 2개 같이 보기면 cv1 위·cv2 아래에 하나씩.
        titles=True면 캔버스 안에 제목·요약을 쓰고 "" 반환, 아니면 요약 글자를 반환(그래프 1개일 때)."""
        i = GRAPH_NAMES.index(g) if g in GRAPH_NAMES else 0
        if i >= 4:
            self._title_pad = 30
            if i == 4:
                t1 = self._draw_flow(win, cv1)
                dur = self.ses.flow_points()[1]           # 밀도 가로축을 곡 흐름과 같은 시작~끝으로
                t2 = self._draw_density(cv2, dur=dur)
                names = ("곡 흐름", "밀도")
            else:
                t1 = self._draw_hist(hist, win, cv1)
                t2 = self._draw_grades(win, cv2)
                names = ("오차 분포", "판정 단계")
            for cv, nm, t in ((cv1, names[0], t1), (cv2, names[1], t2)):
                self._title(cv, nm + ("  · " + t if t else ""))
            return ""
        self._title_pad = 30 if titles else 16
        if i == 1:
            t = self._draw_flow(win, cv1)
        elif i == 2:
            t = self._draw_density(cv1)
        elif i == 3:
            t = self._draw_grades(win, cv1)
        else:
            t = self._draw_hist(hist, win, cv1)
        if titles:
            if t:
                self._title(cv1, t)
            return ""
        return t

    def _title(self, cv, text):
        """캔버스 위쪽에 제목·요약 (폭에 맞춰 최대 2줄)"""
        cv.create_text(cv_w(cv) // 2, 2, text=text, fill=FG, font=(FONT, 8, "bold"), width=cv_w(cv) - 12,
                       anchor="n", justify="center")

    """그래프 그리기 (메인 창과 '기록 불러오기' 창이 같이 씀). self.ses에 flow_points/density/song_end/song_window 필요"""

    def _song_label(self, dur):
        if getattr(self, "label_prefix", None):
            head = self.label_prefix
        else:
            done = self.ses.song_end is not None and self.ses.song_window() == self.ses.song_end
            head = "방금 끝난 곡" if done else "지금 곡"
        return head + " · %d:%02d" % (int(dur) // 60, int(dur) % 60)

    def _draw_flow(self, win, cv):
        win = self._fs(win)
        """곡 흐름: 세로축 = 인디케이터 범위(±FAST/SLOW 범위 x2.4).
        위 파랑 색칠 = 범위 안에서 빠르게 친 노트끼리 평균, 아래 빨강 색칠 = 범위 안에서 느리게 친 노트끼리 평균.
        노란 선 = 색칠을 따라가다 범위 밖으로 튄 노트가 있는 곳에서 그 노트 높이까지 언덕처럼 솟음(위아래 같음).
        점 = FAST/SLOW 범위 밖으로 친 노트의 실제 위치, 세로축 밖이면 끝에 삼각형."""
        cv.delete("all")
        W = cv_w(cv)
        H = int(cv["height"])
        pts, dur = self.ses.flow_points()
        if len(pts) < 10:
            return "곡을 치면 시간에 따른 쏠림이 보여요"
        import numpy as np
        ts = np.array([p[0] for p in pts])
        es = np.array([p[1] for p in pts])
        dur = max(dur or ts.max(), 1.0)
        Rg = float(win) * 2.4                                   # 인디케이터 양끝
        L, R, T, B = 36, W - 8, self._tpad(), H - 16
        def X(t): return L + t / dur * (R - L)
        def Y(v): return T + (Rg + max(-Rg, min(Rg, v))) / (2 * Rg) * (B - T)   # 빠름(-) 위, 느림(+) 아래
        y0 = Y(0)
        cv.create_rectangle(L, Y(win), R, Y(-win), fill="#171b22", outline="")   # FAST/SLOW 안쪽
        for v in (-Rg, -win, win, Rg):
            cv.create_line(L, Y(v), R, Y(v), fill="#3a4050" if abs(v) == win else "#20242d",
                           dash=(3, 3) if abs(v) == win else ())
            cv.create_text(L - 4, Y(v), text="%+d" % round(v), fill=DIM, font=(FONT, 7), anchor="e")
        cv.create_text(L - 4, y0, text="0", fill=DIM, font=(FONT, 7), anchor="e")
        nb = int(max(20, min(120, (R - L) // 5)))               # 시간 구간 수
        edges = np.linspace(0, dur, nb + 1)
        cen = (edges[:-1] + edges[1:]) / 2.0
        half = 1.5 * dur / nb                                   # 각 구간 앞뒤로 조금씩 더 봄(부드럽게)
        def side_mean(mask):
            tt, ee = ts[mask], es[mask]
            out = np.full(nb, np.nan)
            for i, c in enumerate(cen):
                sel = ee[np.abs(tt - c) <= half]
                if len(sel):
                    out[i] = float(sel.mean())
            return out
        inr = np.abs(es) <= win
        slow_in, fast_in = side_mean((es > 0) & inr), side_mean((es < 0) & inr)    # 색칠 = 범위 안 노트끼리 평균
        for m, col in ((slow_in, "#7a2a2a"), (fast_in, "#1f4a6e")):
            for i in range(nb):
                if not np.isnan(m[i]):
                    cv.create_rectangle(X(edges[i]), min(y0, Y(m[i])), X(edges[i + 1]) + 0.5, max(y0, Y(m[i])),
                                        fill=col, outline="")
        cv.create_line(L, y0, R, y0, fill="#ffffff")
        # 선 = 색칠(범위 안 평균)을 따라가다가 튄 노트가 있는 곳에서 그 노트 높이까지 언덕처럼 솟음 (아래쪽도 똑같이)
        gx = np.linspace(0, dur, int(max(100, R - L)))
        sig = max(0.6, dur * 0.012)                              # 언덕 폭 (초)
        def base_on(m):
            ok = ~np.isnan(m)
            if not ok.any():
                return None
            return np.interp(gx, cen[ok], m[ok])
        for m, sgn in ((slow_in, 1), (fast_in, -1)):
            base = base_on(m)
            if base is None:
                base = np.zeros_like(gx)
            line = base.copy()
            for t, e in zip(ts, es):
                if sgn * e <= win:
                    continue
                g = np.exp(-((gx - t) / sig) ** 2)
                hill = base + (e - base) * g
                line = np.maximum(line, hill) if sgn > 0 else np.minimum(line, hill)
            pts_ = []
            for x_, v in zip(gx, line):
                pts_ += [X(x_), Y(v)]
            cv.create_line(*pts_, fill=OKC, width=2, smooth=True)
        out_s = out_f = 0
        for t, e in zip(ts, es):                                # 범위 밖 노트: 실제 위치에 점
            if abs(e) <= win:
                continue
            col = LATE if e > 0 else EARLY
            if e > 0:
                out_s += 1
            else:
                out_f += 1
            x = X(t)
            if abs(e) > Rg:                                     # 세로축 밖: 끝에 삼각형
                y = B if e > 0 else T
                d = -5 if e > 0 else 5
                cv.create_polygon(x - 3, y + d, x + 3, y + d, x, y, fill=col, outline="")
            else:
                y = Y(e)
                cv.create_oval(x - 2, y - 2, x + 2, y + 2, fill=col, outline="")
        cv.create_text(L + 4, T + 7, text="빠름", fill=EARLY, font=(FONT, 8, "bold"), anchor="w")
        cv.create_text(L + 4, B - 7, text="느림", fill=LATE, font=(FONT, 8, "bold"), anchor="w")
        for i in range(0, int(dur) + 1, 30 if dur > 90 else 15):
            cv.create_text(X(i), H - 8, text="%d:%02d" % (i // 60, i % 60), fill=DIM, font=(FONT, 7))
        sl, fa = es[es > 0], es[es < 0]
        def avg(a): return "%+.1f" % float(a.mean()) if len(a) else "-"
        def far(a, f): return "%+.0f" % float(f(a)) if len(a) else "-"
        return "%s · 빠름 %s (튄 노트 포함 %s, 최대 %s) · 느림 %s (포함 %s, 최대 %s) · 범위 밖 빠름 %d / 느림 %d" % (
            self._song_label(dur), avg(fa[fa >= -win]), avg(fa), far(fa, np.min),
            avg(sl[sl <= win]), avg(sl), far(sl, np.max), out_f, out_s)

    def _draw_density(self, cv, dur=None):
        """밀도: 초당 지나간 노트 수(막대)와 평균·최고"""
        cv.delete("all")
        W = cv_w(cv)
        H = int(cv["height"])
        d = self.ses.density()
        if d is None:
            return "곡을 치면 초당 노트 수가 보여요"
        import numpy as np
        L, R, T, B = 36, W - 8, self._tpad(), H - 16          # 왼쪽 여백 = 곡 흐름과 같게
        mx = max(4.0, float(d.max()))
        top = float(np.ceil(mx / 5.0) * 5)
        n = len(d)
        span = float(dur) if dur else float(n)                 # 가로축 길이(초), 곡 흐름과 맞출 때는 그 길이
        def X(i): return L + min(i, span) / span * (R - L)
        def Y(v): return B - v / top * (B - T)
        for v in np.linspace(0, top, 3):
            cv.create_line(L, Y(v), R, Y(v), fill="#262a35")
            cv.create_text(L - 4, Y(v), text="%d" % v, fill=DIM, font=(FONT, 7), anchor="e")
        avg = float(d[d > 0].mean()) if (d > 0).any() else 0.0
        for i, v in enumerate(d):
            if v <= 0:
                continue
            col = LATE if v >= 0.8 * mx else OKC if v >= avg else "#4a6fa5"
            cv.create_rectangle(X(i) + 0.5, Y(v), X(i + 1) - 0.5, B, fill=col, outline="")
        cv.create_line(L, Y(avg), R, Y(avg), fill="#ffffff", dash=(3, 3))
        dur = span
        for i in range(0, int(dur) + 1, 30 if dur > 90 else 15):
            cv.create_text(X(i), H - 8, text="%d:%02d" % (i // 60, i % 60), fill=DIM, font=(FONT, 7))
        pk = int(np.argmax(d))
        return "%s · 평균 %.1f · 최고 %.0f 노트/초 (%d:%02d)" % (self._song_label(dur), avg, mx, pk // 60, pk % 60)

    GRADE_NAMES = ["90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "1%"]

    def _draw_grades(self, win, cv):
        win = self._fs(win)
        """판정 단계: FAST/SLOW 범위(설정값)를 가운데(100%)로, 바깥으로 범위의 0.4배(기본 16.67ms = 1/60초)마다 한 단계씩
        (게임의 MAX 90%, 80% … 와 같은 간격). 위 = 빠름, 아래 = 느림. 막대 길이 = 개수(제곱근 눈금)."""
        cv.delete("all")
        W = cv_w(cv)
        H = int(cv["height"])
        pts, dur = self.ses.flow_points()
        if len(pts) < 10:
            return "곡을 치면 판정 단계별 개수가 보여요"
        import numpy as np
        es = np.array([p[1] for p in pts])
        step = float(win) * 0.4      # 기본(41.67)이면 1/60초(16.67ms) = 게임과 같음, 범위를 줄이면 같은 비율로 줄어듦
        nlv = len(self.GRADE_NAMES)
        def level(e):
            a = abs(e)
            if a <= win:
                return 0
            return int(min(nlv, np.ceil((a - win) / step)))
        lv = np.array([level(e) for e in es])
        fast = [int(((lv == k) & (es < 0)).sum()) for k in range(1, nlv + 1)]
        slow = [int(((lv == k) & (es > 0)).sum()) for k in range(1, nlv + 1)]
        center = int((lv == 0).sum())
        far = max([k + 1 for k in range(nlv) if fast[k] or slow[k]] + [3])
        rows = []
        for k in range(far - 1, -1, -1):
            rows.append(("FAST " + self.GRADE_NAMES[k], fast[k], EARLY))
        rows.append(("100%% (±%.2f)" % win, center, OKC))
        for k in range(far):
            rows.append(("SLOW " + self.GRADE_NAMES[k], slow[k], LATE))
        total = max(1, len(es))
        L, R, T, B = 92, W - 70, self._tpad(), H - 4
        rh = (B - T) / len(rows)
        mx = max(r[1] for r in rows) or 1
        def bw(n): return (R - L) * (n / mx) ** 0.5
        for i, (name, n, col) in enumerate(rows):
            y = T + (i + 0.5) * rh
            th = max(2.0, min(12.0, rh * 0.7))
            cv.create_text(L - 6, y, text=name, fill=col if n else DIM, font=(FONT, 7, "bold" if col == OKC else ""),
                           anchor="e")
            if n:
                cv.create_rectangle(L, y - th / 2, L + max(2.0, bw(n)), y + th / 2, fill=col, outline="")
            cv.create_text(L + max(2.0, bw(n)) + 4, y, text="%d (%.1f%%)" % (n, 100.0 * n / total) if n else "0",
                           fill=FG if n else DIM, font=(FONT, 7), anchor="w")
        cv.create_line(L, T, L, B, fill="#3a4050")
        return "%s · FAST %d · 100%% %d · SLOW %d · 한 단계 %.2fms, 막대 길이는 제곱근 눈금" % (
            self._song_label(dur), sum(fast), center, sum(slow), step)

    def _draw_hist(self, hist, win, cv=None):
        cv = cv or self.cv
        cv.delete("all")
        W = cv_w(cv)
        H = int(cv["height"])
        lo, hi = -100, 100
        def X(v): return 10 + (v - lo) / (hi - lo) * (W - 20)
        cv.create_rectangle(X(-win), 0, X(win), H - 16, fill="#1a2a1f", outline="")
        if hist is not None:
            edges, h = hist
            mx = max(1, h.max())
            for i, c in enumerate(h):
                a, b = edges[i], edges[i + 1]
                y = (H - 18) * (1 - c / mx)
                col = EARLY if b <= 0 else LATE if a >= 0 else "#cccccc"
                if abs(a) >= win or abs(b) > win:
                    col = "#555b66"
                cv.create_rectangle(X(a) + 1, y + 2, X(b) - 1, H - 16, fill=col, outline="")
        for v, col in ((0, "#ffffff"), (-win, "#5fd37a"), (win, "#5fd37a")):
            cv.create_line(X(v), 0, X(v), H - 16, fill=col)
        for v in (-100, -50, 0, 50, 100):
            cv.create_text(X(v), H - 8, text=str(v), fill=DIM, font=(FONT, 8))
        cv.create_text(14, 8, text="빠름", fill=EARLY, font=(FONT, 8), anchor="w")
        cv.create_text(W - 14, 8, text="느림", fill=LATE, font=(FONT, 8), anchor="e")
        return ""


class App(GraphDraw):
    def __init__(self, root, cfg, session, ctrl):
        """ctrl: start_capture(), stop_capture(), calibrate(), preview(), save_cfg()"""
        self.root, self.cfg, self.ses, self.ctrl = root, cfg, session, ctrl
        self.bind_req = None
        self.indicator = None          # 게임 위 오버레이 (일반판만)
        self.obs = None                # OBS 브라우저 소스
        from .edition import is_obs
        self.obs_edition = is_obs()
        self._restart_job = None
        self.status = {"fps": 0.0, "speed": 0.0, "proc": 0.0, "skip": 0.0, "thr": False,
                       "keys": "대기", "msg": "", "lc": None}
        root.title("디맥 판정 분석기" + (" (OBS판)" if self.obs_edition else ""))
        root.configure(bg=BG)
        if not self.obs_edition:           # OBS판: 분석기 창도 게임 위에 강제로 올리지 않음 (보통 창)
            root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", float(cfg["ui"].get("alpha", 0.92)))
        except Exception:
            pass
        # 창 크기를 고정해서 글자 길이가 바뀔 때마다 창이 늘었다 줄었다 하지 않게 한다
        # 저장된 크기가 화면보다 크면 화면 안에 들어오게 줄임 (아래 버튼이 화면 밖으로 안 나가게)
        geo = cfg["ui"].get("geometry") or "680x860"
        try:
            import re as _re
            m = _re.match(r"(\d+)x(\d+)([+-]-?\d+)?([+-]-?\d+)?", geo)
            sw, sh = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
            w, h = int(m.group(1)), int(m.group(2))
            w, h = min(w, max(560, sw - 20)), min(h, max(500, sh - 90))    # 작업 표시줄·제목 표시줄 몫
            x = int(m.group(3).lstrip("+")) if m.group(3) else None
            y = int(m.group(4).lstrip("+")) if m.group(4) else None
            if x is not None and y is not None:
                x = max(0, min(x, sw - w))
                y = max(0, min(y, sh - h - 60))
                geo = "%dx%d+%d+%d" % (w, h, x, y)
            else:
                geo = "%dx%d" % (w, h)
            root.geometry(geo)
        except Exception:
            root.geometry("680x860")
        root.minsize(560, 500)
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("Treeview", background="#1d2029", fieldbackground="#1d2029", foreground=FG,
                     font=(FONT, 9), rowheight=20, borderwidth=0)
        st.configure("Treeview.Heading", background="#262a35", foreground=FG, font=(FONT, 9, "bold"))
        st.configure("TCombobox", fieldbackground="#262a35", background="#262a35", foreground="#000")
        # 탭: 어두운 바탕에, 고른 탭만 밝게
        st.configure("Dm.TNotebook", background=BG, borderwidth=0, tabmargins=(0, 2, 0, 0))
        st.configure("Dm.TNotebook.Tab", background="#20232c", foreground=DIM, font=(FONT, 9, "bold"),
                     padding=(10, 4), borderwidth=0)
        st.map("Dm.TNotebook.Tab", background=[("selected", "#343a48"), ("active", "#2a2e3a")],
               foreground=[("selected", FG), ("active", FG)], padding=[("selected", (10, 5))])
        self._build()
        numeric_only(root)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------ 화면 구성
    def _lbl(self, parent, text="", size=10, bold=False, fg=FG, bg=BG, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, size, "bold" if bold else "normal"), **kw)

    def _btn(self, parent, text, cmd, bold=False):
        return tk.Button(parent, text=text, command=cmd, bg="#262a35", fg=FG, activebackground="#343a48",
                         activeforeground=FG, relief="flat", font=(FONT, 9, "bold" if bold else "normal"), padx=6)

    def _hint(self, parent, text):
        """탭 안 작은 설명 줄 (창 폭에 맞춰 줄바꿈)"""
        lb = self._lbl(parent, text, 9, fg=DIM, anchor="w", justify="left", width=1)
        lb.pack(fill="x", padx=8, pady=(4, 2))
        self._wrap_lbls.append(lb)
        return lb

    def _build(self):
        r = self.root
        self._wrap_lbls = []
        ck = dict(bg=BG, fg=FG, selectcolor="#262a35", activebackground=BG, activeforeground=FG, font=(FONT, 9))
        # 맨 아래 고정 버튼 (_build 처음에 자리를 잡아 둬서 창이 짧아도 항상 보임)
        self._btnbar = WrapFrame(r, bg=BG)
        self._btnbar.pack(side="bottom", fill="x", padx=8, pady=(2, 8))
        top = tk.Frame(r, bg=BG)
        top.pack(fill="x", padx=8, pady=(6, 2))
        # 상태 줄: 높이를 3줄로 고정 (글자가 길어지거나 알림이 붙어도 아래 화면이 위아래로 들썩이지 않게)
        self.lbl_status = self._lbl(top, "", 9, fg=DIM, anchor="nw", justify="left", width=1, height=3)
        self.lbl_status.pack(fill="x")
        self.root.bind("<Configure>", self._on_resize)

        # 모드 선택 버튼 (자주 바꾸니 탭 밖, 항상 보임)
        fm = tk.Frame(r, bg=BG)
        fm.pack(fill="x", padx=8, pady=(2, 2))
        self._lbl(fm, "모드", 10, True).pack(side="left", padx=(0, 6))
        self.mode_btns = {}
        for val, txt in (("4B", "4B"), ("5B", "5B"), ("6B", "6B"), ("8B", "8B"), ("auto", "자동")):
            b = tk.Button(fm, text=txt, width=5, command=lambda v=val: self.on_mode(v), relief="flat",
                          font=(FONT, 10, "bold"), bg="#262a35", fg=FG, activebackground="#343a48",
                          activeforeground=FG)
            b.pack(side="left", padx=2)
            self.mode_btns[val] = b
        self._paint_mode()

        # 요약 (판정선 추천 결과 + 영점 추천 알림: 어느 탭에서든 항상 보임)
        box = tk.Frame(r, bg="#1b1e26", highlightthickness=1, highlightbackground="#2b303c")
        box.pack(fill="x", padx=8, pady=6)
        self.lbl_line = self._lbl(box, "", 15, True, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_line.pack(fill="x", padx=8, pady=(6, 0))
        self.lbl_spread = self._lbl(box, "", 11, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_spread.pack(fill="x", padx=8)
        self.lbl_fs = self._lbl(box, "", 13, True, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_fs.pack(fill="x", padx=8)
        self.lbl_in = self._lbl(box, "", 11, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_in.pack(fill="x", padx=8)
        self.lbl_rec = self._lbl(box, "", 13, True, fg=OKC, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_rec.pack(fill="x", padx=8, pady=(4, 0))
        self.lbl_margin = self._lbl(box, "", 9, fg=DIM, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_margin.pack(fill="x", padx=8)
        self.lbl_sync = self._lbl(box, "", 9, fg=DIM, bg="#1b1e26", anchor="w", width=1, justify="left")
        self.lbl_sync.pack(fill="x", padx=8, pady=(0, 6))
        # 영점 보정 추천 알림 (결과 화면으로 모은 곡 기준, 지금 보정값이 맞는 범위 밖일 때만 보임)
        self.banner = tk.Frame(box, bg="#3a2f12")
        self.lbl_banner = tk.Label(self.banner, text="", bg="#3a2f12", fg=OKC, font=(FONT, 10, "bold"),
                                   anchor="w", justify="left", width=1)
        self.lbl_banner.pack(side="left", fill="x", expand=True, padx=6, pady=4)
        for txt, cmd in (("무시", self.on_dismiss), ("보정 적용", self.on_apply_suggest)):
            tk.Button(self.banner, text=txt, command=cmd, bg="#5a4a1a", fg=FG, relief="flat",
                      font=(FONT, 9), padx=6).pack(side="right", padx=3, pady=3)
        self._banner_on = False

        # ---- 탭: 기능별로 나눠서 한 번에 하나만 보이게
        self.nb = ttk.Notebook(r, style="Dm.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 2))
        self.TAB_NAMES = ("그래프", "판정·영점", "인디케이터", "오답노트·기록", "설정")
        tabs = []
        for name in self.TAB_NAMES:
            fr = tk.Frame(self.nb, bg=BG)
            self.nb.add(fr, text=" %s " % name)
            tabs.append(fr)
        t_graph, t_judge, t_ind, t_rec, t_set = tabs

        # ===== 탭 1: 그래프
        gb = WrapFrame(t_graph, bg=BG)
        gb.pack(side="bottom", fill="x", padx=6, pady=(0, 4))
        gb.add(self._btn(gb, "기록 불러오기", self.on_load), 2, 2)
        gb.add(self._btn(gb, "CSV 저장", self.on_csv), 2, 2)
        gsel = tk.Frame(t_graph, bg=BG)
        gsel.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(gsel, text="그래프", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")
        self.graph_names = GRAPH_NAMES
        gsv = self.cfg.get("graph", self.graph_names[0])
        if str(gsv).startswith("전체 보기"):
            gsv = self.graph_names[4]
        self.graph_var = tk.StringVar(value=gsv)
        if self.graph_var.get() not in self.graph_names:
            self.graph_var.set(self.graph_names[0])
        self.cb_graph = ttk.Combobox(gsel, textvariable=self.graph_var, values=self.graph_names,
                                     state="readonly", width=24, font=(FONT, 9))
        self.cb_graph.pack(side="left", padx=6)
        self.cb_graph.bind("<<ComboboxSelected>>", lambda e: self.on_graph())
        # 그래프 설명은 콤보 아래 줄에, 높이 2줄 고정 (옆으로 길어지거나 위아래로 들썩이지 않게)
        self.lbl_graph = tk.Label(t_graph, text="", bg=BG, fg=DIM, font=(FONT, 9), anchor="nw", justify="left",
                                  width=1, height=2)
        self.lbl_graph.pack(fill="x", padx=8)
        self.cv = tk.Canvas(t_graph, width=420, height=130, bg="#101217", highlightthickness=0)
        self.cv.pack(padx=8, pady=2, fill="x")
        # '전체 보기'용 그래프 2개 (평소엔 숨김)
        self.cv2 = tk.Canvas(t_graph, width=420, height=130, bg="#101217", highlightthickness=0)
        self._all_view = False
        cols = ("n", "mean", "std", "fast", "slow", "in", "etc")
        self.tree = ttk.Treeview(t_graph, columns=cols, height=5)     # 창이 짧으면 표가 먼저 줄어듦
        self.tree.heading("#0", text="구분")
        for c, t, w in zip(cols, ("개수", "평균", "흔들림", "FAST", "SLOW", "100%", "놓침/채터"),
                           (42, 52, 52, 44, 44, 52, 62)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="e")
        self.tree.column("#0", width=120)
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)

        # ===== 탭 2: 판정·영점
        self._hint(t_judge, "게임 설정과 똑같이 맞춰 주세요. 게임 모드마다 판정·출력 타이밍과 영점을 따로 기억해요(고르면 그 값으로 바뀜). "
                            "영점 보정은 결과 화면이 3곡 이상 모이면 추천해요.")
        fg_ = tk.Frame(t_judge, bg=BG)
        fg_.pack(fill="x", padx=8, pady=2)
        from .session import GAME_MODES
        self._lbl(fg_, "게임 모드", 9).pack(side="left", padx=(0, 2))
        self.var_gmode = tk.StringVar(value=self.cfg.get("game_mode", GAME_MODES[0]))
        cgm = ttk.Combobox(fg_, textvariable=self.var_gmode, values=GAME_MODES, state="readonly", width=10,
                           font=(FONT, 9))
        cgm.pack(side="left")
        cgm.bind("<<ComboboxSelected>>", lambda e: self.on_gmode())
        f = WrapFrame(t_judge, bg=BG)
        f.pack(fill="x", padx=8, pady=2)
        self.var_judge = tk.IntVar(value=int(self.cfg.get("judge_setting_ms", 0)))
        self.var_out = tk.IntVar(value=int(self.cfg.get("output_setting_ms", 0)))
        self.var_win = tk.DoubleVar(value=float(self.cfg.get("window_ms", 41.6)))
        self.var_fs = tk.DoubleVar(value=float(self.cfg.get("fs_ms", 41.6) or 41.6))
        for i, (txt, var, lo, hi) in enumerate([("판정 타이밍", self.var_judge, -150, 150),
                                                 ("출력 타이밍", self.var_out, -150, 150),
                                                 ("100% 범위 ±", self.var_win, 5, 100),
                                                 ("FAST/SLOW 범위 ±", self.var_fs, 1, 100)]):
            pf = tk.Frame(f, bg=BG)
            self._lbl(pf, txt, 9).pack(side="left", padx=(0, 2))
            isf = isinstance(var, tk.DoubleVar)
            sp = tk.Spinbox(pf, from_=lo, to=hi, width=6 if isf else 5, textvariable=var, font=(FONT, 10),
                            increment=0.01 if isf else 1, format="%.2f" if isf else "%.0f",
                            command=self.on_settings, bg="#262a35", fg=FG, buttonbackground="#262a35",
                            insertbackground=FG, relief="flat")
            sp.pack(side="left")
            if var is self.var_fs or var is self.var_win:
                # 판정 종류: 고르면 그 범위가 그 값으로, 숫자를 직접 바꾸면 '사용자 설정' (100%·FAST/SLOW 따로)
                pv_ = tk.StringVar(value=self._fs_preset_name(float(var.get())))
                cbp = ttk.Combobox(pf, textvariable=pv_, state="readonly", width=9, font=(FONT, 9),
                                   values=[n for n, _ in self._fs_presets()] + ["사용자 설정"])
                cbp.pack(side="left", padx=(4, 0))
                if var is self.var_fs:
                    self.var_fsp = pv_
                    cbp.bind("<<ComboboxSelected>>", lambda e: self.on_fs_preset(self.var_fsp, self.var_fs))
                else:
                    self.var_winp = pv_
                    cbp.bind("<<ComboboxSelected>>", lambda e: self.on_fs_preset(self.var_winp, self.var_win))
            f.add(pf, 0, 10)
            sp.bind("<Return>", lambda e: self.on_settings())
            sp.bind("<FocusOut>", lambda e: self.on_settings())
        # 영점 보정 (직접 조절): + 로 올리면 기록이 늦은 쪽으로, - 로 내리면 빠른 쪽으로 옮겨짐
        self.var_calib = tk.DoubleVar(value=float(self.cfg.get("calib_ms", 0.0)))
        pf = tk.Frame(f, bg=BG)
        self._lbl(pf, "영점 보정", 9).pack(side="left", padx=(0, 2))
        self.sp_calib = tk.Spinbox(pf, from_=-40, to=40, width=5, textvariable=self.var_calib, font=(FONT, 10),
                                   increment=0.5, format="%.1f", command=self.on_calib_manual,
                                   bg="#262a35", fg=FG, buttonbackground="#262a35", insertbackground=FG,
                                   relief="flat")
        self.sp_calib.pack(side="left")
        f.add(pf, 0, 10)
        self.sp_calib.bind("<Return>", lambda e: self.on_calib_manual())
        self.sp_calib.bind("<FocusOut>", lambda e: self.on_calib_manual())
        # 버튼별 영점 따로 쓰기: 체크한 버튼은 그 버튼 전용 영점, 나머지는 공통 영점
        fb = WrapFrame(t_judge, bg=BG)
        fb.pack(fill="x", padx=8, pady=2)
        fb.add(self._lbl(fb, "버튼별 영점 따로:", 9), 0, 4)
        self.var_split = {}
        sp_ = self.cfg.get("calib_split") or {}
        for m in ("4B", "5B", "6B", "8B"):
            v = tk.IntVar(value=1 if sp_.get(m) else 0)
            self.var_split[m] = v
            fb.add(tk.Checkbutton(fb, text=m, variable=v, command=lambda m=m: self.on_calib_split(m), **ck), 0, 2)
        self.lbl_calib_src = self._lbl(t_judge, "", 9, fg=DIM, anchor="w", justify="left", width=1)
        self.lbl_calib_src.pack(fill="x", padx=8)
        self._wrap_lbls.append(self.lbl_calib_src)
        f2 = WrapFrame(t_judge, bg=BG)
        f2.pack(fill="x", padx=8, pady=2)
        self.var_measure = tk.StringVar(value=self.cfg.get("measure", "eye"))
        for txt, val in (("눈판정 측정(소리 끔)", "eye"), ("귀판정 측정", "ear")):
            f2.add(tk.Radiobutton(f2, text=txt, value=val, variable=self.var_measure, command=self.on_settings, **ck))
        fj = WrapFrame(t_judge, bg=BG)
        fj.pack(fill="x", padx=8, pady=(6, 2))
        fj.add(self._btn(fj, "영점 보정 (결과 화면)", self.on_game_calib, bold=True), 0, 4)

        # ===== 탭 3: 인디케이터
        io = self.cfg.get("indicator", {})
        f3 = WrapFrame(t_ind, bg=BG)
        f3.pack(fill="x", padx=8, pady=(6, 2))
        self.var_ind = tk.IntVar(value=1 if io.get("enabled", True) else 0)
        if not self.obs_edition:
            f3.add(tk.Checkbutton(f3, text="기어 위 인디케이터", variable=self.var_ind, command=self.on_indicator, **ck))
        self.var_iy = tk.IntVar(value=int(round(float(io.get("y_frac", 0.45)) * 100)))
        self.var_in = tk.DoubleVar(value=float(io.get("seconds", 3.0)))
        self.var_fl = tk.DoubleVar(value=float(io.get("flash_sec", 1.5)))
        self.var_its = tk.IntVar(value=int(round(float(io.get("text_scale", 1.0)) * 100)))
        self.var_iw = tk.IntVar(value=int(round(float(io.get("width_frac", 0.7)) * 100)))
        self.btn_move = self._btn(f3, "인디케이터 이동", self.on_ind_move)
        if not self.obs_edition:
            f3.add(self.btn_move, 6)
            f3.add(self._btn(f3, "위치 초기화", lambda: self.indicator and self.indicator.reset_position()), 2)
            f3.add(self._btn(f3, "폭 초기화", lambda: self.indicator and self.indicator.reset_width()), 2)
        else:
            # OBS 브라우저 소스 (방송·녹화 화면에만 보임)
            from .obs_server import obs_opts, obs_scale
            oo = obs_opts(self.cfg)
            self.var_obs = tk.IntVar(value=1 if oo["enabled"] else 0)
            f3.add(tk.Checkbutton(f3, text="OBS 인디케이터 (브라우저 소스)", variable=self.var_obs,
                                  command=self.on_obs, **ck))
            self.var_obss = tk.IntVar(value=int(round(obs_scale(self.cfg) * 100)))
            pf = tk.Frame(f3, bg=BG)
            self._lbl(pf, "크기 %", 9).pack(side="left")
            sp = tk.Spinbox(pf, from_=50, to=400, increment=10, width=4, textvariable=self.var_obss,
                            command=self.on_obs, bg="#262a35", fg=FG, buttonbackground="#262a35",
                            relief="flat", font=(FONT, 9))
            sp.pack(side="left", padx=2)
            sp.bind("<Return>", lambda e: self.on_obs())
            f3.add(pf, 6)
            f3.add(self._btn(f3, "주소 복사", self.on_obs_copy), 2)
            f3.add(self._btn(f3, "브라우저로 보기", self.on_obs_open), 2)
        f3s = WrapFrame(t_ind, bg=BG)
        f3s.pack(fill="x", padx=8, pady=2)
        for txt, var, lo, hi, inc in (("표시 초", self.var_in, 0.5, 10, 0.5), ("ms 글자 초", self.var_fl, 0.3, 5, 0.1),
                                      ("글자 크기 %", self.var_its, 50, 250, 10)):
            pf = tk.Frame(f3s, bg=BG)
            self._lbl(pf, txt, 9).pack(side="left")
            sp = tk.Spinbox(pf, from_=lo, to=hi, increment=inc, width=4, textvariable=var,
                            command=self.on_indicator, bg="#262a35", fg=FG, buttonbackground="#262a35",
                            relief="flat", font=(FONT, 9))
            sp.pack(side="left", padx=2)
            sp.bind("<Return>", lambda e: self.on_indicator())
            f3s.add(pf, 0, 10)
        # 인디케이터 표시 항목 (각각 켜고 끄기)
        f3b = WrapFrame(t_ind, bg=BG)
        f3b.pack(fill="x", padx=8, pady=(0, 2))
        from .ind_state import modes_of
        fm, mm = modes_of(io)
        self.var_ibar = tk.IntVar(value=1 if io.get("bar", True) else 0)
        f3b.add(tk.Checkbutton(f3b, text="인디케이터 막대 표시", variable=self.var_ibar, command=self.on_indicator, **ck))
        self.var_iswap = tk.IntVar(value=1 if io.get("swap_colors", False) else 0)
        f3b.add(tk.Checkbutton(f3b, text="FAST/SLOW 색 반대로", variable=self.var_iswap, command=self.on_indicator, **ck))
        self.FS_LABELS = {"off": "표기 안 함", "fs": "FAST/SLOW 범위를 벗어났을 때", "win": "100% 범위를 벗어났을 때"}
        self.MS_LABELS = {"off": "표기 안 함", "all": "모든 노트 (FAST/SLOW 범위 밖은 색으로)",
                          "fs": "FAST/SLOW 범위를 벗어났을 때만", "win": "100% 범위를 벗어났을 때만"}
        self.FADE_LABELS = {"weak": "약하게", "mid": "보통", "strong": "강하게 (최근 것만 진하게)"}
        self.var_ifm = tk.StringVar(value=self.FS_LABELS[fm])
        self.var_imm = tk.StringVar(value=self.MS_LABELS[mm])
        self.var_ifade = tk.StringVar(value=self.FADE_LABELS.get(io.get("fade", "weak"), "약하게"))
        self.TFADE_LABELS = {"end": "끝에서만 (마지막 30%)", "mid": "보통 (40%부터 서서히)", "linear": "뜨자마자 서서히"}
        self.var_itfade = tk.StringVar(value=self.TFADE_LABELS.get(io.get("text_fade", "end"), "끝에서만 (마지막 30%)"))
        # 표시 방식 콤보는 한 줄에 하나씩 (이름 칸 폭을 맞춰서 세로로 가지런히)
        for txt, var, labels, w in (("FAST/SLOW 표시", self.var_ifm, self.FS_LABELS, 30),
                                    ("ms 표시", self.var_imm, self.MS_LABELS, 34),
                                    ("선 연해짐", self.var_ifade, self.FADE_LABELS, 30),
                                    ("글자 연해짐", self.var_itfade, self.TFADE_LABELS, 30)):
            pf = tk.Frame(t_ind, bg=BG)
            pf.pack(fill="x", padx=8, pady=1)
            self._lbl(pf, txt, 9, width=14, anchor="w").pack(side="left")
            cbx = ttk.Combobox(pf, textvariable=var, values=list(labels.values()), state="readonly", width=w,
                               font=(FONT, 9))
            cbx.pack(side="left", padx=2)
            cbx.bind("<<ComboboxSelected>>", lambda e: self.on_indicator())
        # 인디케이터 범위 설명은 따로 한 줄 (길어도 옆으로 밀지 않게)
        self.lbl_range = self._lbl(t_ind, "", 9, fg=DIM, anchor="w", justify="left", width=1)
        self.lbl_range.pack(fill="x", padx=8, pady=(6, 2))
        if self.obs_edition:
            self._hint(t_ind, "OBS → 소스 추가 → 브라우저 → [주소 복사]한 주소를 URL에 붙여넣기. "
                              "OBS에서 소스를 늘리거나 줄이지 말고 '크기 %'로 바꾸세요 (줄이면 옅은 선이 사라져요).")
        else:
            self._hint(t_ind, "인디케이터 위에서 드래그하면 옮겨지고, 휠로 폭이 바뀌어요. 방향키 1px, Shift+방향키 10px.")

        # ===== 탭 4: 오답노트·기록
        self._hint(t_rec, "오답노트: 지금까지 친 곡을 박자·패턴별로 나눠서 어디서 밀리고 당기는지 알려 줘요.")
        fc = WrapFrame(t_rec, bg=BG)
        fc.pack(fill="x", padx=8, pady=2)
        fc.add(self._btn(fc, "오답노트 (코치) 열기", self.on_coach, bold=True), 0, 4)
        self._hint(t_rec, "기록 저장을 켜면 곡이 끝날 때 결과 화면 사진과 그 곡의 기록 CSV를 곡 이름으로 저장해요.")
        fr2 = WrapFrame(t_rec, bg=BG)
        fr2.pack(fill="x", padx=8, pady=2)
        self.var_rec = tk.IntVar(value=1 if self.cfg.get("record_save") else 0)
        fr2.add(tk.Checkbutton(fr2, text="기록 저장 (결과 화면 + 곡 CSV)", variable=self.var_rec,
                               command=self.on_record, **ck), 0, 4)
        fr2.add(self._btn(fr2, "기록 폴더", self.on_record_dir), 4)
        fr3 = WrapFrame(t_rec, bg=BG)
        fr3.pack(fill="x", padx=8, pady=(6, 2))
        fr3.add(self._btn(fr3, "기록 불러오기", self.on_load), 0, 4)
        fr3.add(self._btn(fr3, "CSV 저장 (지금까지)", self.on_csv), 0, 4)

        # ===== 탭 5: 설정
        self._hint(t_set, "처음 한 번만 맞추면 되는 것들이에요.")
        fs1 = WrapFrame(t_set, bg=BG)
        fs1.pack(fill="x", padx=8, pady=2)
        for txt, cmd in (("키 설정", self.on_keys), ("영역 보정", self.on_calib), ("미리보기", self.on_preview)):
            fs1.add(self._btn(fs1, txt, cmd), 0, 4)
        self._lbl(t_set, "스킨", 9, True, anchor="w").pack(fill="x", padx=8, pady=(8, 0))
        fs2 = WrapFrame(t_set, bg=BG)
        fs2.pack(fill="x", padx=8, pady=2)
        self.var_dark = tk.IntVar(value=1 if self.cfg.get("dark_notes") else 0)
        fs2.add(tk.Checkbutton(fs2, text="어두운 노트 스킨", variable=self.var_dark, command=self.on_dark, **ck), 4)
        self.var_ring = tk.IntVar(value=1 if self.cfg.get("ring_notes") else 0)
        fs2.add(tk.Checkbutton(fs2, text="동그란 노트 스킨", variable=self.var_ring, command=self.on_ring, **ck), 4)
        self.var_big = tk.IntVar(value=1 if self.cfg.get("big_effects") else 0)
        fs2.add(tk.Checkbutton(fs2, text="판정 이펙트가 큰 스킨", variable=self.var_big, command=self.on_big, **ck), 4)
        self._lbl(t_set, "창 · 캡처", 9, True, anchor="w").pack(fill="x", padx=8, pady=(8, 0))
        fs3 = WrapFrame(t_set, bg=BG)
        fs3.pack(fill="x", padx=8, pady=2)
        pf = tk.Frame(fs3, bg=BG)
        self._lbl(pf, "창 투명도 %", 9).pack(side="left")
        self.var_alpha = tk.IntVar(value=int(round(float(self.cfg["ui"].get("alpha", 0.92)) * 100)))
        sa = tk.Spinbox(pf, from_=40, to=100, increment=5, width=4, textvariable=self.var_alpha,
                        command=self.on_alpha, bg="#262a35", fg=FG, buttonbackground="#262a35",
                        relief="flat", font=(FONT, 9))
        sa.pack(side="left", padx=2)
        sa.bind("<Return>", lambda e: self.on_alpha())
        fs3.add(pf, 0, 10)
        self.var_thr = tk.IntVar(value=1 if self.cfg.get("capture_thread", True) else 0)
        fs3.add(tk.Checkbutton(fs3, text="새 캡처 방식", variable=self.var_thr, command=self.on_capture_thread, **ck), 4)
        self.var_focus = tk.IntVar(value=1 if self.cfg.get("focus_only", True) else 0)
        fs3.add(tk.Checkbutton(fs3, text="게임 창이 맨 앞일 때만 키 기록", variable=self.var_focus,
                               command=self.on_focus_only, **ck), 4)

        # 마지막에 보던 탭으로 열기
        try:
            ti = int(self.cfg["ui"].get("tab", 0))
            if 0 <= ti < len(tabs):
                self.nb.select(tabs[ti])
        except Exception:
            pass
        self._tabs = tabs
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.on_tab())

        # 맨 아래: 곡마다 자주 누르는 것만
        b = self._btnbar
        for txt, cmd in (("초기화", self.on_reset), ("일시정지", self.on_pause)):
            btn = self._btn(b, txt, cmd)
            b.add(btn, 2, 2)
            if txt == "일시정지":
                self.btn_pause = btn

    def window_rect(self):
        """이 창이 화면에서 차지하는 곳 (테두리·제목 표시줄 몫 포함): 결과 화면 글자 인식 때 가리는 데 씀"""
        try:
            r = self.root
            return (r.winfo_rootx() - 12, r.winfo_rooty() - 45, r.winfo_width() + 24, r.winfo_height() + 57)
        except Exception:
            return None

    def _on_resize(self, ev):
        if ev.widget is self.root:
            wl = max(300, ev.width - 30)
            for w in (self.lbl_status, self.lbl_margin, self.lbl_sync, self.lbl_in, self.lbl_spread,
                      self.lbl_graph, self.lbl_range, self.lbl_banner, *self._wrap_lbls):
                try:
                    w.configure(wraplength=wl)
                except Exception:
                    pass

    @staticmethod
    def _key_of(labels, text, default):
        for k, v in labels.items():
            if v == text:
                return k
        return default

    @staticmethod
    def _recent_text(n):
        return "전체" if not n else "최근 %d" % n

    # ------------------------------------------------------------ 이벤트
    def _fs_presets(self):
        """[(이름, 범위 ms)]. 측정으로 확인: 하드 = 노멀의 1/2 (±20.90, 단계 8.33), 맥스 = 1/4 (±10.0~10.2, 단계 4.3).
        config.json의 fs_presets 에서 바꿀 수 있음"""
        p = self.cfg.get("fs_presets") or {}
        normal = 125.0 / 3.0
        hard = float(p.get("hard", round(normal / 2, 2)))
        mx = float(p.get("max", round(normal / 4, 2)))
        return [("노멀 판정", round(normal, 2)), ("하드 판정", hard), ("맥스 판정", mx)]

    def _fs_preset_name(self, v):
        for n, pv in self._fs_presets():
            if abs(v - pv) < 0.005:
                return n
        return "사용자 설정"

    def on_fs_preset(self, pvar, var):
        """100% 범위 / FAST/SLOW 범위 옆 판정 종류 칸 (서로 따로 움직임)"""
        name = pvar.get()
        for n, pv in self._fs_presets():
            if n == name:
                var.set(pv)
                self.on_settings()
                return
        pvar.set(self._fs_preset_name(float(var.get())))   # '사용자 설정' 고르면 숫자 그대로

    def on_settings(self):
        try:
            self.cfg["judge_setting_ms"] = int(self.var_judge.get())
            self.cfg["output_setting_ms"] = int(self.var_out.get())
            self.cfg["window_ms"] = round(float(self.var_win.get()), 2)
            self.cfg["fs_ms"] = round(float(self.var_fs.get()), 2)
        except (tk.TclError, ValueError):
            return
        try:
            self.var_fsp.set(self._fs_preset_name(self.cfg["fs_ms"]))
            self.var_winp.set(self._fs_preset_name(self.cfg["window_ms"]))
        except Exception:
            pass
        self.cfg["measure"] = self.var_measure.get()
        self.cfg["recent"] = 0          # '최근 N개' 옵션은 없앰 (초기화로 끊어 쓰면 됨)
        self.ctrl.save_cfg()
        self.refresh()

    def on_calib_manual(self):
        try:
            v = round(max(-40.0, min(40.0, float(self.var_calib.get()))), 1)
        except (tk.TclError, ValueError):
            return
        if abs(v - float(self.cfg.get("calib_ms", 0.0))) < 0.05:
            return
        self.ses.set_calib(v)
        self.ses.suggest = None
        self.ctrl.save_cfg()
        self.refresh()

    def on_indicator(self):
        try:
            io = self.cfg.setdefault("indicator", {})
            new = {"enabled": bool(self.var_ind.get()),
                   "seconds": max(0.5, min(10.0, float(self.var_in.get()))),
                   "flash_sec": round(max(0.3, min(5.0, float(self.var_fl.get()))), 1),
                   "bar": bool(self.var_ibar.get()),
                   "fs_mode": self._key_of(self.FS_LABELS, self.var_ifm.get(), "fs"),
                   "ms_mode": self._key_of(self.MS_LABELS, self.var_imm.get(), "off"),
                   "fade": self._key_of(self.FADE_LABELS, self.var_ifade.get(), "weak"),
                   "text_fade": self._key_of(self.TFADE_LABELS, self.var_itfade.get(), "end"),
                   "text_scale": round(max(50, min(250, int(float(self.var_its.get())))) / 100.0, 2),
                   "swap_colors": bool(self.var_iswap.get())}
        except (tk.TclError, ValueError):
            return
        moved = (new["enabled"] != io.get("enabled", True)
                 or abs(new["text_scale"] - float(io.get("text_scale", 1.0))) > 1e-6)   # 글자 크기 → 창 높이가 바뀜
        io.update(new)
        for k in ("fs_text", "ms_all", "fs_label_win"):      # 예전 방식 설정은 지움
            io.pop(k, None)
        self.ctrl.save_cfg()
        if self.indicator is not None:
            self.indicator.update(force=True)
        if self.obs is not None:
            self.obs.update(force=True)
        if moved and not self.obs_edition:
            self.schedule_restart()

    def on_obs(self):
        try:
            sc = max(50, min(400, int(float(self.var_obss.get())))) / 100.0
        except (tk.TclError, ValueError):
            return
        oo = self.cfg.setdefault("obs", {})
        oo["enabled"] = bool(self.var_obs.get())
        oo["scale"] = sc
        oo.pop("width", None)
        self.ctrl.save_cfg()
        if self.obs is not None:
            self.obs.update(force=True)
        self.refresh()

    def on_obs_copy(self):
        if self.obs is None or not self.obs.url:
            self.set_status(msg="OBS 인디케이터를 먼저 켜 주세요.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.obs.url)
        self.set_status(msg="복사했어요: %s  (OBS → 소스 추가 → 브라우저 → URL에 붙여넣기)" % self.obs.url)

    def on_obs_open(self):
        if self.obs is None or not self.obs.url:
            self.set_status(msg="OBS 인디케이터를 먼저 켜 주세요.")
            return
        import webbrowser
        webbrowser.open(self.obs.url)

    def on_focus_only(self):
        on = bool(self.var_focus.get())
        self.cfg["focus_only"] = on
        fo = getattr(self.ctrl, "focus_only", None)
        if fo is not None:
            fo.value = 1 if on else 0
        self.ctrl.save_cfg()

    def schedule_restart(self):
        # 인디케이터가 덮는 줄은 분석에서 '가려짐'으로 처리하므로 위치가 바뀌면 캡처를 다시 시작
        if self._restart_job is not None:
            self.root.after_cancel(self._restart_job)
        self._restart_job = self.root.after(800, self.ctrl.restart_capture)

    def on_ind_move(self):
        if self.indicator is None:
            return
        on = not self.indicator.moving
        if on and not self.var_ind.get():
            self.var_ind.set(1)
            self.on_indicator()
        self.indicator.set_moving(on)
        self.btn_move.configure(text="이동 끝" if on else "인디케이터 이동",
                                bg=OKC if on else "#262a35", fg="#111" if on else FG)

    def on_capture_thread(self):
        """새 캡처 방식(캡처 전용 스레드가 프레임 시각 기록) 켜기/끄기 → 캡처 다시 시작"""
        cb = self.cfg.setdefault("calib_by_capture", {})
        cb["thread" if self.cfg.get("capture_thread", True) else "old"] = float(self.cfg.get("calib_ms", 0.0))
        self.cfg["capture_thread"] = bool(self.var_thr.get())
        # 캡처 방식마다 화면 시각 기준이 조금 다를 수 있어서 영점 보정값을 따로 기억
        self.ses.set_calib(float(cb.get("thread" if self.cfg["capture_thread"] else "old", 0.0)))
        self.ses.calib_songs = []
        self.ctrl.save_cfg()
        self.ses.log("capture_thread", int(self.cfg["capture_thread"]))
        self.ctrl.restart_capture()

    def on_tab(self):
        """고른 탭을 기억해 두었다가 다음에 그 탭으로 연다"""
        try:
            i = self._tabs.index(self.root.nametowidget(self.nb.select()))
        except Exception:
            return
        if self.cfg["ui"].get("tab") != i:
            self.cfg["ui"]["tab"] = i
            self.ctrl.save_cfg()

    def on_alpha(self):
        try:
            a = max(40, min(100, int(self.var_alpha.get()))) / 100.0
        except (tk.TclError, ValueError):
            return
        self.cfg["ui"]["alpha"] = a
        try:
            self.root.attributes("-alpha", a)
        except Exception:
            pass
        self.ctrl.save_cfg()

    def _paint_mode(self):
        cur = self.cfg.get("mode", "auto")
        for v, b in self.mode_btns.items():
            b.configure(bg=OKC if v == cur else "#262a35", fg="#111" if v == cur else FG)

    def on_dark(self):
        self.cfg["dark_notes"] = bool(self.var_dark.get())
        self.ctrl.save_cfg()
        self.schedule_restart()

    def on_record(self):
        self.cfg["record_save"] = bool(self.var_rec.get())
        self.ctrl.save_cfg()
        if self.cfg["record_save"]:
            from . import record
            self.set_status(msg="기록 저장 켜짐: 곡이 끝나 결과 화면을 읽으면 '%s'에 자동 저장해요." % record.record_dir(self.cfg))

    def on_record_dir(self):
        from . import record
        d = record.record_dir(self.cfg)
        try:
            os.makedirs(d, exist_ok=True)
            os.startfile(d)
        except Exception as e:
            messagebox.showerror("기록 폴더", "폴더를 열지 못했어요: %r" % (e,))

    def on_big(self):
        self.cfg["big_effects"] = bool(self.var_big.get())
        self.ctrl.save_cfg()
        self.schedule_restart()

    def on_ring(self):
        self.cfg["ring_notes"] = bool(self.var_ring.get())
        self.ctrl.save_cfg()
        self.schedule_restart()

    def on_mode(self, v):
        if v == self.cfg.get("mode", "auto"):
            return
        self.cfg["mode"] = v
        self._paint_mode()
        self.ctrl.save_cfg()
        self.ses.rebuild()
        self.ctrl.restart_capture()

    def on_gmode(self):
        """게임 모드 = 프리셋: 판정·출력 타이밍과 영점을 그 모드에 저장된 값으로 바꿈"""
        new = self.var_gmode.get()
        if new == self.cfg.get("game_mode"):
            return
        self.ses.switch_gmode(new)
        try:
            self.var_judge.set(int(self.cfg.get("judge_setting_ms", 0)))
            self.var_out.set(int(self.cfg.get("output_setting_ms", 0)))
            self.var_calib.set(round(float(self.cfg.get("calib_ms", 0.0)), 1))
        except Exception:
            pass
        self.ses.suggest = None
        self.ctrl.save_cfg()
        self.set_status(msg="%s 설정: 판정 타이밍 %+d · 출력 타이밍 %+d · 영점 보정 %+.1f ms" % (
            new, int(self.cfg.get("judge_setting_ms", 0)), int(self.cfg.get("output_setting_ms", 0)),
            float(self.cfg.get("calib_ms", 0.0))))
        self.refresh()

    def _apply_zero_preset(self, force=False):
        """버튼 모드가 바뀌면 그 버튼의 영점으로 (버튼별 따로 쓰기를 켠 버튼만 전용 값, 나머지는 공통 값)"""
        if self.ses.sync_calib():
            try:
                self.var_calib.set(self.cfg["calib_ms"])
            except Exception:
                pass
            self.ctrl.save_cfg()
            m = self.ses.active_mode()
            self.set_status(msg="영점 보정 %+.1f ms (%s)" % (self.cfg["calib_ms"], self.ses.calib_label(m)))
        try:
            m = self.ses.active_mode() or self.cfg.get("mode")
            on = [k for k, v in self.var_split.items() if v.get()]
            if on:
                parts = ["공통 %+.1f" % self.ses.calib_common()] + [
                    "%s %+.1f" % (k, self.ses.calib_for(k)) for k in on]
                self.lbl_calib_src.configure(text="지금 %s 영점 사용 · %s" % (
                    self.ses.calib_label(m) if m in ("4B", "5B", "6B", "8B") else "공통", " / ".join(parts)))
            else:
                self.lbl_calib_src.configure(text="모든 버튼이 같은 영점을 써요")
        except Exception:
            pass

    def on_calib_split(self, m):
        on = bool(self.var_split[m].get())
        self.cfg.setdefault("calib_split", {})[m] = on
        if on and m not in (self.cfg.get("calib_btn") or {}):
            # 처음 켤 때는 지금 공통 값에서 시작
            self.cfg.setdefault("calib_btn", {})[m] = round(self.ses.calib_common(), 1)
        if "calib_common" not in self.cfg:
            self.cfg["calib_common"] = float(self.cfg.get("calib_ms", 0.0))
        self.ses.update_zero()
        self._apply_zero_preset()
        self.ctrl.save_cfg()
        self.refresh()

    def on_reset(self):
        self.ses.reset()
        self.refresh()

    def on_pause(self):
        self.ses.paused = not self.ses.paused
        self.btn_pause.configure(text="다시 시작" if self.ses.paused else "일시정지")

    def on_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=C.app_dir(),
                                         initialfile=time.strftime("dmjudge_%Y%m%d_%H%M%S.csv"),
                                         filetypes=[("CSV", "*.csv")])
        if p:
            self.ses.export_csv(p)
            rp = p[:-4] + "_raw.csv" if p.lower().endswith(".csv") else p + "_raw.csv"
            try:
                self.ses.export_raw(rp)
                self.set_status(msg="저장: %s (+ 진단용 %s)" % (os.path.basename(p), os.path.basename(rp)))
            except Exception as e:
                self.set_status(msg="진단 기록 저장 실패: %r" % (e,))

    def on_calib(self):
        self.ctrl.calibrate()

    def on_preview(self):
        if not self.cfg.get("geometry"):
            messagebox.showinfo("미리보기", "먼저 '영역 보정'을 해주세요.")
            return
        self.ctrl.preview()

    def on_close(self):
        self.cfg["ui"]["geometry"] = self.root.geometry()
        self.ctrl.save_cfg()
        self.ctrl.shutdown()
        self.root.destroy()

    def on_apply_suggest(self):
        self.ses.apply_suggestion()
        self.ctrl.save_cfg()
        self.refresh()

    def on_dismiss(self):
        self.ses.dismiss_suggestion()
        self.refresh()

    def _show_banner(self):
        sg = self.ses.suggest
        if sg is None:
            if self._banner_on:
                self.banner.pack_forget()
                self._banner_on = False
            return
        b, lo, hi, n = sg
        cur = float(self.cfg.get("calib_ms", 0.0))
        self.lbl_banner.configure(
            text="⚠ 결과 화면 %d곡 기준 영점 보정 추천: %+.1f ms  (맞는 범위 %+.1f ~ %+.1f)\n"
                 "지금 %+.1f ms는 이 범위 밖이라 게임과 FAST/SLOW 개수가 어긋나요." % (n, b, lo, hi, cur))
        if not self._banner_on:
            self.banner.pack(fill="x", padx=8, pady=(0, 6))
            self._banner_on = True

    def on_load(self):
        from . import record
        rd = record.record_dir(self.cfg)
        p = filedialog.askopenfilename(initialdir=rd if os.path.isdir(rd) else C.app_dir(), title="저장한 기록 불러오기",
                                       filetypes=[("분석기 기록", "*.csv"), ("모든 파일", "*.*")])
        if not p:
            return
        if not p.endswith("_raw.csv"):                     # 결과 CSV를 골랐으면 같은 이름의 진단 기록을 찾음
            q = p[:-4] + "_raw.csv"
            if os.path.exists(q):
                p = q
        try:
            from .session import replay_raw
            cfg, songs = replay_raw(p)
        except Exception as e:
            messagebox.showerror("기록 불러오기", "불러오지 못했어요: %s" % (e,))
            return
        if not songs:
            messagebox.showinfo("기록 불러오기", "이 기록에는 끝까지 친 곡이 없어요.")
            return
        return RecordViewer(self, os.path.basename(p), cfg, songs)

    def on_game_calib(self):
        GameCalibDialog(self)

    def on_coach(self):
        """방금 끝난 곡의 오답노트 (곡이 끝난 뒤 계산 — 플레이 중 부하 없음)"""
        CoachWindow(self)

    # ------------------------------------------------------------ 키 설정
    def on_keys(self):
        KeyDialog(self)

    def key_event(self, kid, is_down):
        """키 입력 프로세스에서 온 키 (바인딩 대기 중이면 여기서 소비)"""
        if self.bind_req is not None and is_down:
            cb = self.bind_req
            self.bind_req = None
            cb(kid)
            return True
        return False

    # ------------------------------------------------------------ 표시
    def set_status(self, **kw):
        self.status.update(kw)

    def refresh(self):
        try:
            self._apply_zero_preset()
        except Exception:
            pass
        try:
            self._show_banner()
        except Exception:
            pass
        try:   # 다른 곳(게임 결과 보정, 보정 적용 등)에서 바뀐 영점 보정값을 칸에 반영 (입력 중이면 건드리지 않음)
            if self.root.focus_get() is not self.sp_calib:
                cb = round(float(self.cfg.get("calib_ms", 0.0)), 1)
                if abs(float(self.var_calib.get()) - cb) > 0.05:
                    self.var_calib.set(cb)
        except Exception:
            pass
        try:
            from .ind_state import bar_range
            fs = self.ses.fs_range()
            if self.indicator is not None and self.indicator.moving:
                self.lbl_range.configure(text="이동 중: 막대를 끌어서 옮기기 (기어 가운데 근처에서 자동으로 딱 맞춰짐) · "
                                              "막대를 한 번 클릭한 뒤 방향키 1px, Shift+방향키 10px · 휠 = 폭")
            else:
                t = "막대 양끝 ±%.0f ms (FAST/SLOW 범위 ±%.1f에 맞춰 자동)" % (bar_range(self.cfg, fs), fs)
                if self.obs is not None:
                    from .obs_server import obs_size
                    if self.obs.error:
                        t += "\nOBS: " + self.obs.error
                    elif self.obs.running:
                        w, h = obs_size(self.cfg)
                        con = " · 연결됨" if self.obs.hub and self.obs.hub.clients > 0 else ""
                        t += "\nOBS 주소 %s · 브라우저 소스 크기 %d × %d (크기 100%% = 원래 인디케이터와 같음, OBS에서 늘리거나 줄이지 말고 '크기 %%'로)%s" % (
                            self.obs.url, w, h, con)
                self.lbl_range.configure(text=t)
        except Exception:
            pass
        s = self.ses.stats()
        o = s["overall"]
        win = float(self.cfg.get("window_ms", 41.6))
        js = int(self.cfg.get("judge_setting_ms", 0))
        out = int(self.cfg.get("output_setting_ms", 0))
        mode = s["mode"] or ("판별 중" if self.cfg.get("mode") == "auto" else self.cfg.get("mode"))
        st = self.status
        warn = ""
        if 0 < st["fps"] < 55:
            warn = "  ⚠ 캡처가 느려요"
        elif st["thr"]:
            # 캡처 스레드가 시각을 따로 찍으므로 분석이 느려도 시각은 정확. 너무 많이 건너뛸 때만 알림
            # 게임이 화면 주사율보다 빠르게 그리면 캡처 프레임이 그만큼 많아져 건너뛰는 건 정상.
            # 실제로 분석한 프레임이 초당 90개 아래로 떨어질 때만 알림
            done = st["fps"] * (1.0 - min(1.0, st["skip"]))
            if st["fps"] > 0 and done < 90 and st["speed"] > 0.05:   # 결과·선곡 화면(노트 없음)에선 알리지 않음
                warn = "  ⚠ 분석 프레임이 적어요 (초당 %.0f개, PC 부하)" % done
        elif st["fps"] > 0 and st["proc"] > 0.8 * 1000.0 / st["fps"]:
            warn = "  ⚠ 처리 지연(%.1fms) — 정확도가 떨어질 수 있어요" % st["proc"]
        try:   # 노트 하나가 둘로 쪼개져 잡히는 스킨(고리 모양 동그란 노트) 알림
            m_ = self.ses.active_mode()
            mt_ = self.ses.matchers.get(m_) if m_ else None
            if (mt_ is not None and not self.cfg.get("ring_notes") and mt_.phantom > 40
                    and mt_.phantom > 0.15 * (len(mt_.hits) + 1)):
                warn += "  ⚠ 노트가 둘로 쪼개져 보여요 — 동그란 노트 스킨이면 '동그란 노트 스킨'을 체크하세요"
        except Exception:
            pass
        diag = ""
        if s["mode"] is None and st["lc"] is not None:
            n, sc = st["lc"]
            diag = "\n레인 판별: 샘플 %d/800 · 점수 %s" % (n, " ".join("%sB=%.2f" % (k, v) for k, v in sorted(sc.items())) or "-")
        vtxt = ""
        try:
            nv, vr = self.ses.speed_changes()
            if nv:
                vtxt = " · 변속 감지 %d회 (%.2f~%.2f)" % (nv, vr[0], vr[1])
        except Exception:
            pass
        how = ""
        if s["mode"] is not None and self.cfg.get("mode") == "auto" and self.ses.lanes_guess is None:
            how = "(키 입력으로 판별)"
        self.lbl_status.configure(
            text="모드 %s%s · 캡처 %.0f fps · 노트 속도 %.2f px/ms%s · 키 입력 %s%s%s%s" % (
                mode, how, st["fps"], st["speed"], vtxt, st["keys"], warn, diag,
                ("\n" + st["msg"]) if st["msg"] else ""))
        if not o.get("n"):
            self.lbl_line.configure(text="기록 없음 — 곡을 플레이하면 바로 표시돼요")
            for w in (self.lbl_spread, self.lbl_fs, self.lbl_in, self.lbl_rec, self.lbl_margin, self.lbl_sync):
                w.configure(text="")
            self._draw_graph(None, win)
            self._fill_tree(s, win)
            return
        mj = o["median_j"]                 # 판정 기준 중심 (보정값 포함)
        # FAST/SLOW 개수는 곡마다 0부터 (다른 줄은 모을수록 정확해지는 값이라 [초기화] 전까지 누적)
        sf = self.ses.song_fs()
        if sf is None:
            self.lbl_fs.configure(text="이번 곡  FAST 0  ·  SLOW 0    (±%.1f ms 밖)" % self.ses.fs_range())
        else:
            self.lbl_fs.configure(text="이번 곡  FAST %d  ·  SLOW %d    (±%.1f ms 밖, %d노트)" % (
                sf[0], sf[1], self.ses.fs_range(), sf[2]))
        side = "빠름" if mj < 0 else "느림"
        self.lbl_line.configure(
            text="나만의 판정선  %s ms  (%s)" % (fmt_ms(mj), side),
            fg=EARLY if mj < -2 else LATE if mj > 2 else FG)
        self.lbl_spread.configure(text="흔들림  ±%.1f ms    (화면 기준 평균 %s ms)" % (o["robust_std"], fmt_ms(o["mean_e"])))
        self.lbl_in.configure(text="±%.1f ms 안  %.1f%%    (n=%d)" % (win, o["in_now"] * 100, o["n"]))
        delta = o["best_setting"] - js
        if self.cfg.get("measure", "eye") == "eye":
            rec = "추천  판정 타이밍 %+d  →  %.1f%%" % (o["best_setting"], o["in_best"] * 100)
            if delta == 0:
                rec += "  (지금이 최적)"
        else:
            # 귀판정: 판정 타이밍은 그대로 두고, 출력 타이밍 = 판정 타이밍 + n 으로 알려 줌
            ro = out - delta
            rec = "추천  출력 타이밍 %+d  (= 판정 타이밍 %+d  %s %d)  →  %.1f%%" % (
                ro, js, "+" if ro - js >= 0 else "−", abs(ro - js), o["in_best"] * 100)
            if delta == 0:
                rec += "  (지금이 최적)"
        self.lbl_rec.configure(text=rec)
        self.lbl_margin.configure(text="추천값 적용 시 여유: 빠른 쪽 %.0f ms / 느린 쪽 %.0f ms%s" % (
            o["margin_early"], o["margin_late"],
            "   ⚠ 흔들림이 커서 설정만으로는 다 못 넣어요" if o["robust_std"] > win / 2 else ""))
        cb = float(self.cfg.get("calib_ms", 0.0))
        zs = ("영점 보정 %+.1f ms" % cb) if cb else "영점 보정 없음"
        lat = self.ses.lat[-50:]
        ls = ("    · 입력→표시 %.1f ms" % (sorted(lat)[len(lat) // 2])) if lat else ""
        self.lbl_sync.configure(text="귀판정 싱크: 지금 출력 = 판정 %s %d ms    · %s%s" % (
            "+" if out - js >= 0 else "−", abs(out - js), zs, ls))
        self._draw_graph(s["hist"], win)
        self._fill_tree(s, win)

    def on_graph(self):
        self.cfg["graph"] = self.graph_var.get()
        self.ctrl.save_cfg()
        self.refresh()

    def on_song_end(self):
        """결과 화면으로 곡이 끝났을 때 (그래프가 그 곡 전체를 보여주도록 다시 그림)"""
        self.refresh()

    def _set_all_view(self, on):
        """2개 같이 보기: 통계표를 숨기고 그래프 2개를 세로로"""
        if on == self._all_view:
            return
        self._all_view = on
        if on:
            self.tree.pack_forget()
            self.cv2.pack(padx=8, pady=2, fill="x", after=self.cv)
        else:
            self.cv2.pack_forget()
            self.tree.pack(fill="both", expand=True, padx=8, pady=4, after=self.cv)

    def _draw_graph(self, hist, win):
        g = self.graph_var.get()
        two = GRAPH_NAMES.index(g) >= 4 if g in GRAPH_NAMES else False
        self._set_all_view(two)
        try:
            self.cv.delete("all")
            self.cv2.delete("all")
            self.lbl_graph.configure(text=self._draw_choice(g, self.cv, self.cv2, win, hist, titles=False) or "")
        except Exception as e:          # 그래프 오류로 창 전체가 멈추지 않게
            self.cv.delete("all")
            self.lbl_graph.configure(text="그래프 오류: %r" % (e,))

    def _fill_tree(self, s, win):
        t = self.tree
        t.delete(*t.get_children())
        def row(name, o, etc=""):
            if not o.get("n"):
                return
            t.insert("", "end", text=name, values=(
                o["n"], fmt_ms(o["median_j"]), "±%.1f" % o["robust_std"], o["fast"], o["slow"],
                "%.1f%%" % (o["in_now"] * 100), etc))
        for g, o in sorted(s["groups"], key=lambda x: x[0]):
            row("■ " + g, o)
        for m, ch, keys, o, miss in s["channels"]:
            chat = sum(s["chatter"].get(k, 0) for k in keys)
            kn = "/".join(key_name(k) for k in keys) or "?"
            row("  %s %s" % (kn, C.CH_LABEL.get(ch, ch)), o, "%d / %d" % (miss, chat))


class KeyDialog:
    def __init__(self, app: App):
        self.app = app
        cfg = app.cfg
        top = self.top = tk.Toplevel(app.root)
        top.title("키 설정")
        top.configure(bg=BG)
        top.attributes("-topmost", True)
        f = tk.Frame(top, bg=BG)
        f.pack(fill="x", padx=10, pady=8)
        tk.Label(f, text="모드", bg=BG, fg=FG, font=(FONT, 10)).pack(side="left")
        self.var_m = tk.StringVar(value=(app.ses.active_mode() or "8B"))
        cb = ttk.Combobox(f, textvariable=self.var_m, values=C.MODES, width=5, state="readonly")
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        tk.Label(f, text="'추가'를 누르고 원하는 키를 누르세요. 한 레인에 키 여러 개 가능.",
                 bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left", padx=6)
        self.body = tk.Frame(top, bg=BG)
        self.body.pack(fill="both", padx=10, pady=(0, 10))
        self.draw()

    def draw(self):
        for w in self.body.winfo_children():
            w.destroy()
        m = self.var_m.get()
        mc = self.app.cfg["modes"][m]
        for i, ch in enumerate(C.mode_channels(m)):
            keys = mc["keymap"].get(ch, [])
            if isinstance(keys, str):
                keys = [keys]
            tk.Label(self.body, text=C.CH_LABEL[ch], bg=BG, fg=FG, font=(FONT, 10), width=11,
                     anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            lab = tk.Label(self.body, text=" / ".join(key_name(k) for k in keys) or "(없음)", bg="#262a35",
                           fg=FG, font=(FONT, 10), width=16, anchor="w")
            lab.grid(row=i, column=1, padx=4)
            tk.Button(self.body, text="추가", command=lambda ch=ch, lab=lab: self.listen(m, ch, lab),
                      bg="#262a35", fg=FG, relief="flat", font=(FONT, 9)).grid(row=i, column=2, padx=2)
            tk.Button(self.body, text="지우기", command=lambda ch=ch: self.clear(m, ch),
                      bg="#262a35", fg=FG, relief="flat", font=(FONT, 9)).grid(row=i, column=3, padx=2)
            gv = tk.StringVar(value=mc["groups"].get(ch, ""))
            e = tk.Entry(self.body, textvariable=gv, width=8, bg="#262a35", fg=FG, insertbackground=FG,
                         relief="flat", font=(FONT, 10))
            e.grid(row=i, column=4, padx=4)
            e.bind("<FocusOut>", lambda ev, ch=ch, gv=gv: self.set_group(m, ch, gv.get()))
            e.bind("<Return>", lambda ev, ch=ch, gv=gv: self.set_group(m, ch, gv.get()))
        tk.Label(self.body, text="오른쪽 칸 = 손 그룹 이름(통계 묶음)", bg=BG, fg=DIM,
                 font=(FONT, 9)).grid(row=99, column=0, columnspan=5, sticky="w", pady=(6, 0))

    def listen(self, m, ch, lab):
        lab.configure(text="키를 누르세요…", fg=OKC)

        def done(kid):
            km = self.app.cfg["modes"][m]["keymap"]
            cur = km.get(ch, [])
            if isinstance(cur, str):
                cur = [cur]
            # 같은 모드 안에서 다른 레인에 쓰던 키면 거기서 뺀다
            for c2, ks in km.items():
                if isinstance(ks, list) and kid in ks and c2 != ch:
                    ks.remove(kid)
            if kid not in cur:
                cur.append(kid)
            km[ch] = cur
            self.apply()
        self.app.bind_req = done

    def clear(self, m, ch):
        self.app.cfg["modes"][m]["keymap"][ch] = []
        self.apply()

    def set_group(self, m, ch, g):
        self.app.cfg["modes"][m]["groups"][ch] = g.strip() or "기타"
        self.app.ctrl.save_cfg()

    def apply(self):
        self.app.ctrl.save_cfg()
        self.app.ses.update_keymap()
        self.draw()


class GameCalibDialog:
    """결과 화면으로 모은 곡들로 영점 보정 추천 (곡이 끝나면 결과 화면 숫자를 자동으로 읽어 모은다)"""

    def __init__(self, app: App):
        self.app = app
        top = self.top = tk.Toplevel(app.root)
        top.title("영점 보정 (결과 화면)")
        top.configure(bg=BG)
        top.attributes("-topmost", True)
        msg = ("곡을 끝까지 치고 결과 화면이 뜨면 분석기가 MAX 100%~BREAK 숫자를 자동으로 읽어 여기에 모아요.\n"
               "3곡 이상 모이면, 곡마다 '게임의 100% 아닌 개수'와 '분석기의 범위 밖 개수'가 가장 잘 맞는\n"
               "영점 보정값과 그 범위를 계산해요 (100% 범위 ±41.67 ms, 경계 근처 노트가 많은 어려운 곡일수록 정확).\n"
               "더블클릭: 그 곡을 계산에서 빼기/넣기 · 모은 곡은 분석기를 껐다 켜도 남아요.")
        tk.Label(top, text=msg, bg=BG, fg=FG, font=(FONT, 10), justify="left").pack(padx=12, pady=(10, 6), anchor="w")
        cols = ("when", "mode", "n", "game", "brk", "now", "rec", "use")
        tv = self.tv = ttk.Treeview(top, columns=cols, show="headings", height=9)
        for c, t, w in zip(cols, ("시각", "모드", "노트", "게임 100%아님", "BREAK", "분석기(지금)", "분석기(추천)", "계산"),
                           (88, 96, 52, 96, 54, 84, 84, 44)):
            tv.heading(c, text=t)
            tv.column(c, width=w, anchor="center")
        tv.pack(padx=12, pady=4, fill="x")
        tv.bind("<Double-1>", self.toggle)
        self.res = tk.Label(top, text="", bg=BG, fg=OKC, font=(FONT, 11, "bold"), justify="left")
        self.res.pack(padx=12, pady=4, anchor="w")
        # 그룹별 (버튼 모드 × 게임 모드): 루머 확인용
        self.grp = tk.Label(top, text="", bg=BG, fg=FG, font=(FONT, 9), justify="left")
        self.grp.pack(padx=12, pady=(0, 4), anchor="w")
        tk.Label(top, text="버튼별로 영점을 따로 쓰려면 메인 창 [판정·영점] 탭의 '버튼별 영점 따로'에서 체크하세요. "
                           "게임 모드마다 영점도 따로 기억해요.",
                 bg=BG, fg=DIM, font=(FONT, 9)).pack(padx=12, anchor="w")
        b = tk.Frame(top, bg=BG)
        b.pack(padx=12, pady=(2, 4), anchor="w")
        for t, c in (("추천값 적용", self.apply), ("모은 곡 비우기", self.forget), ("보정 해제(0)", self.clear),
                     ("닫기", top.destroy)):
            tk.Button(b, text=t, command=c, bg="#262a35", fg=FG, relief="flat", font=(FONT, 9),
                      padx=8).pack(side="left", padx=3)
        f = tk.Frame(top, bg=BG)
        f.pack(padx=12, pady=(2, 10), anchor="w")
        tk.Label(f, text="결과 화면을 못 읽었을 때: 방금 곡의 100% 아닌 개수(90%~1% 합, BREAK 제외)", bg=BG, fg=DIM,
                 font=(FONT, 9)).pack(side="left")
        self.vn = tk.IntVar(value=0)
        tk.Spinbox(f, from_=0, to=999, width=5, textvariable=self.vn, font=(FONT, 9)).pack(side="left", padx=4)
        numeric_only(f)
        tk.Button(f, text="직접 추가", command=self.manual, bg="#262a35", fg=FG, relief="flat", font=(FONT, 9),
                  padx=6).pack(side="left")
        self._show()

    def _show(self, extra=""):
        ses, cfg = self.app.ses, self.app.cfg
        m = ses.active_mode() or cfg.get("mode")
        z = ses.zero_for(m)
        for i in self.tv.get_children():
            self.tv.delete(i)
        for i, r in enumerate(ses.results):
            zr = ses.zero_for(r.get("mode"))
            best = zr["best"] if zr else None
            self.tv.insert("", "end", iid=str(i), values=(
                r.get("when", ""), r.get("mode", "") + {"래더": " 래더", "버서스 매치": " 버서스", "미션": " 미션"}.get(
                    r.get("gmode"), "") + {"hard": " 하드", "max": " 맥스", "custom": " 사용자"}.get(
                    r.get("judge", "normal"), ""), r.get("n", ""), r["non100"],
                "-" if r.get("brk") is None else r["brk"], ses.predicted_out(r),
                "-" if best is None else ses.predicted_out(r, best), "O" if r.get("use", True) else "-"))
        cb = float(cfg.get("calib_ms", 0.0))
        src = ses.calib_label(m)
        if z is None or z.get("n", 0) < 1:
            txt = "[%s] 모은 곡 없음 · 지금 영점 보정 %+.1f ms" % (src, cb)
        elif z.get("ambiguous"):
            txt = ("아직 정할 수 없어요: %s ms가 비슷하게 맞아요 (%d곡)\n"
                   "'100%% 아닌 개수'만으로는 FAST 쪽으로 밀린 값과 SLOW 쪽으로 밀린 값을 못 가려요. "
                   "곡을 더 모으거나, 게임 FAST/SLOW 표시를 켜고 치면 정해져요. 지금 %+.1f ms" % (
                       " / ".join("%+.1f" % v for v in z["ambiguous"]), z["n"], cb))
        else:
            ok = "  ✔ 지금 값이 범위 안이에요" if z["lo"] <= cb <= z["hi"] else "  → 바꾸는 게 좋아요"
            few = "  (3곡 이상 모으면 믿을 만해요)" if z["n"] < 3 else ""
            txt = "[%s] 추천 영점 보정 %+.1f ms  (맞는 범위 %+.1f ~ %+.1f, %d곡)%s\n지금 %+.1f ms%s" % (
                src, z["best"], z["lo"], z["hi"], z["n"], few, cb, ok)
            if z.get("by_text"):
                txt += "\n(개수만으론 %s ms도 맞았지만, 게임 FAST/SLOW 글자 방향으로 골랐어요)" % (
                    " / ".join("%+.1f" % v for v in z.get("alt", [])))
        self.res.configure(text=(extra + "\n" if extra else "") + txt)
        self.grp.configure(text=self._group_text())

    def _group_text(self):
        """버튼별 추천 영점과 맞는 범위 (따로 쓰기 여부와 상관없이 비교용, 지금 게임 모드 곡만). 범위가 안 겹치면 '차이 있음'"""
        ses = self.app.ses
        zb = getattr(ses, "zero_btn", {}) or {}
        if not zb:
            return "버튼별: 아직 없음"
        lines = ["버튼별 (%s · 노멀 판정 곡만):" % (self.app.cfg.get("game_mode") or "프리스타일")]
        for k in ses.BTN_MODES:
            z = zb.get(k)
            if z is None:
                continue
            lines.append("  %s : %+.1f ms (맞는 범위 %+.1f ~ %+.1f, %d곡) · 지금 %s %+.1f%s" % (
                k, z["best"], z["lo"], z["hi"], z["n"], "전용" if ses.calib_split_on(k) else "공통", ses.calib_for(k),
                "  · 아직 못 정함 (%s)" % " / ".join("%+.1f" % v for v in z["ambiguous"]) if z.get("ambiguous") else ""))
        ok = [z for z in zb.values() if z["n"] >= 3 and not z.get("ambiguous")]
        if len(ok) >= 2:
            sep = any(a["hi"] < b["lo"] or b["hi"] < a["lo"] for a in ok for b in ok)
            lines.append("  → " + ("범위가 안 겹치는 버튼이 있어요: 버튼별로 따로 쓰는 게 나을 수 있어요" if sep else
                                   "3곡 이상 모인 버튼끼리 범위가 겹쳐요: 같은 영점을 써도 돼요"))
        else:
            lines.append("  → 비교하려면 버튼마다 3곡 이상 필요해요")
        return "\n".join(lines)

    def toggle(self, ev=None):
        sel = self.tv.selection()
        if not sel:
            return
        r = self.app.ses.results[int(sel[0])]
        r["use"] = not r.get("use", True)
        from .session import save_results
        save_results(self.app.ses.results)
        self.app.ses.update_zero()
        self._show()

    def _set(self, v):
        self.app.ses.set_calib(v)
        self.app.ctrl.save_cfg()
        self.app.refresh()

    def apply(self):
        app = self.app
        m = app.ses.active_mode() or app.cfg.get("mode")
        src = app.ses.calib_label(m)
        z = app.ses.zero_for(m)
        if z is None:
            self._show("[%s] 모은 곡이 없어요." % src)
            return
        if z.get("ambiguous"):
            self._show("[%s] 아직 FAST 쪽/SLOW 쪽 값을 못 가려서 적용하지 않았어요." % src)
            return
        self._set(z["best"])
        self._show("[%s] 영점 보정 %+.1f ms 적용" % (src, z["best"]))

    def manual(self):
        try:
            n = int(self.vn.get())
        except (tk.TclError, ValueError):
            return
        rec = self.app.ses.add_manual_result(n)
        self._show("추가했어요." if rec else "방금 친 곡 기록이 없어요 (노트 100개 이상인 곡을 끝까지 쳐주세요).")

    def forget(self):
        from .session import save_results
        self.app.ses.results = []
        save_results([])
        self.app.ses.update_zero()
        self._show("모은 곡을 비웠어요.")

    def clear(self):
        self._set(0.0)
        self._show("보정을 해제했어요.")


class RecordViewer(GraphDraw):
    """저장한 기록(_raw.csv)을 다시 흘려 넣어 곡별로 그래프를 보는 창"""

    def __init__(self, app, name, cfg, songs):
        from .session import SongView
        self.app, self.cfg, self.songs = app, cfg, songs
        self.views = [SongView(sg, cfg) for sg in songs]
        self.label_prefix = "불러온 곡"
        top = self.top = tk.Toplevel(app.root)
        top.title("기록 보기 - " + name)
        top.configure(bg=BG)
        f = tk.Frame(top, bg=BG)
        f.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(f, text="곡", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")
        self.song_names = []
        for i, sg in enumerate(songs):
            d = (sg["end"] - sg["start"]) / 1000.0
            r = sg.get("result")
            rs = (" · 결과 100%%아님 %d, BREAK %d" % (sum(r[1:11]), r[11])) if r else ""
            self.song_names.append("%d. %s · %d:%02d · %d노트%s" % (i + 1, sg["mode"], int(d) // 60, int(d) % 60, sg["n"], rs))
        self.song_var = tk.StringVar(value=self.song_names[-1])
        cb = ttk.Combobox(f, textvariable=self.song_var, values=self.song_names, state="readonly", width=48, font=(FONT, 9))
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        f2 = tk.Frame(top, bg=BG)
        f2.pack(fill="x", padx=8, pady=2)
        tk.Label(f2, text="그래프", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")
        self.graph_names = app.graph_names
        self.graph_var = tk.StringVar(value=self.graph_names[4])     # 기본 = 곡 흐름 + 밀도
        cg = ttk.Combobox(f2, textvariable=self.graph_var, values=self.graph_names, state="readonly", width=30, font=(FONT, 9))
        cg.pack(side="left", padx=6)
        cg.bind("<<ComboboxSelected>>", lambda e: self.draw())
        self.lbl = tk.Label(top, text="", bg=BG, fg=FG, font=(FONT, 10), anchor="nw", justify="left",
                            width=1, height=2, wraplength=600)
        self.lbl.pack(fill="x", padx=8, pady=2)
        self.cvs = [tk.Canvas(top, width=560, height=150, bg="#101217", highlightthickness=0) for _ in range(2)]
        for c in self.cvs:
            c.pack(fill="x", padx=8, pady=2)
        fb = tk.Frame(top, bg=BG)
        fb.pack(fill="x", padx=8, pady=(2, 8))
        tk.Button(fb, text="닫기", command=top.destroy, bg="#262a35", fg=FG, relief="flat",
                  font=(FONT, 9), padx=8).pack(side="right")
        tk.Button(fb, text="이 곡 오답노트", command=self.on_coach, bg="#262a35", fg=FG, relief="flat",
                  font=(FONT, 9), padx=8).pack(side="right", padx=4)
        top.after(80, self.draw)

    def on_coach(self):
        from . import coach
        i = self.song_names.index(self.song_var.get()) if self.song_var.get() in self.song_names else len(self.songs) - 1
        sg = self.songs[i]
        try:
            rep = coach.from_song(sg, self.cfg)
        except Exception as ex:
            messagebox.showerror("오답노트", "분석하지 못했어요: %r" % (ex,))
            return
        CoachWindow(self.app, rep=rep, cfg=self.cfg, title="불러온 곡 %d" % (i + 1))

    def draw(self):
        from .analysis import histogram
        i = self.song_names.index(self.song_var.get()) if self.song_var.get() in self.song_names else len(self.views) - 1
        self.ses = self.views[i]
        sg = self.songs[i]
        win = float(self.cfg.get("fs_ms", 41.67) or 41.67)
        e = self.ses.judge_errors()
        nf, ns = int((e < -win).sum()), int((e > win).sum())
        r = sg.get("result")
        gtxt = ("   ·   게임: 100%% 아님 %d, BREAK %d" % (sum(r[1:11]), r[11])) if r else ""
        import numpy as np
        self.lbl.configure(text="분석기: FAST %d · SLOW %d (±%.2f 밖) · 중앙 %+.1f ms · 흔들림 ±%.1f%s" % (
            nf, ns, win, float(np.median(e)), float(np.subtract(*np.percentile(e, [84, 16])) / 2), gtxt))
        g = self.graph_var.get()
        hist = histogram(e)
        for c in self.cvs:
            c.delete("all")
        try:
            two = g in GRAPH_NAMES and GRAPH_NAMES.index(g) >= 4
            if two:
                self.cvs[1].pack(fill="x", padx=8, pady=2)
            else:
                self.cvs[1].pack_forget()
            self._draw_choice(g, self.cvs[0], self.cvs[1], win, hist, titles=True)
        except Exception as ex:
            self.lbl.configure(text="그래프 오류: %r" % (ex,))


class CoachWindow:
    """오답노트 창: 어디서 · 무엇이 틀렸나 · 부족한 점 · 이렇게 고치기 · 다음 목표"""

    def __init__(self, app, rep=None, cfg=None, title=None):
        from . import coach
        self.app = app
        self.coach = coach
        self.cfg = cfg or app.cfg
        rev = trend = pool = None
        if rep is None:                      # 측정 중인 세션의 마지막 곡
            ses = app.ses
            try:
                rep, st = coach.from_session(ses)
            except Exception as ex:
                rep, st = None, None
                messagebox.showerror("오답노트", "분석하지 못했어요: %r" % (ex,))
            adv = coach.advise(rep, self.cfg) if rep else []
            if rep:
                hist = getattr(ses, "coach_hist", None)
                if hist is None:
                    hist = ses.coach_hist = coach.load_history()
                prev = None
                for h in reversed(hist):
                    if h.get("mode") == rep["mode"] and (st is None or h.get("start") is None
                                                          or abs(h["start"] - st) >= 3000.0):
                        prev = h
                        break
                coach.record(hist, rep, adv, judge=ses.judge_kind(), start_t=st)
                coach.save_history(hist)
                rev = coach.review(prev, rep, adv) if prev else None
                trend, _ = coach.trend_advice(hist, rep["mode"])
                pool = coach.pooled(hist, rep["mode"])
            title = title or "방금 친 곡"
        else:
            adv = coach.advise(rep, self.cfg)
            title = title or "불러온 곡"
        self.lines = coach.report_text(rep, adv, self.cfg, trend=trend, title=title, rev=rev, pool=pool)

        top = self.top = tk.Toplevel(app.root)
        top.title("오답노트 - " + title)
        top.configure(bg=BG)
        top.geometry("680x760")
        fr = tk.Frame(top, bg=BG)
        fr.pack(fill="both", expand=True, padx=8, pady=(8, 2))
        sb = tk.Scrollbar(fr)
        sb.pack(side="right", fill="y")
        tx = self.tx = tk.Text(fr, bg="#101217", fg=FG, wrap="word", relief="flat", padx=12, pady=10,
                               yscrollcommand=sb.set, font=(FONT, 10), spacing2=2, cursor="arrow")
        tx.pack(side="left", fill="both", expand=True)
        sb.config(command=tx.yview)
        tx.tag_configure("h", font=(FONT, 12, "bold"), foreground=OKC, spacing1=14, spacing3=4)
        tx.tag_configure("t", font=(FONT, 11, "bold"), foreground=FG, spacing1=10, spacing3=2)
        tx.tag_configure("p", font=(FONT, 10), foreground=FG, lmargin1=4, lmargin2=4)
        tx.tag_configure("dim", font=(FONT, 9), foreground=DIM, lmargin1=4, lmargin2=4, spacing3=2)
        tx.tag_configure("k", font=(FONT, 10), foreground=FG, tabs=(118,), lmargin1=8, lmargin2=126)
        tx.tag_configure("klab", font=(FONT, 9, "bold"), foreground="#9ec3ff")
        tx.tag_configure("row", font=(FONT, 9), foreground=FG, tabs=(190, 240, 300, 360, 470), lmargin1=8)
        tx.tag_configure("rowh", font=(FONT, 9, "bold"), foreground=DIM, tabs=(190, 240, 300, 360, 470), lmargin1=8)
        fb = tk.Frame(top, bg=BG)
        fb.pack(fill="x", padx=8, pady=(2, 8))
        tk.Button(fb, text="닫기", command=top.destroy, bg="#262a35", fg=FG, relief="flat",
                  font=(FONT, 9), padx=8).pack(side="right")
        self.btn_src = tk.Button(fb, text="근거 자료 보기", command=self.toggle_src, bg="#262a35", fg=FG,
                                 relief="flat", font=(FONT, 9), padx=8)
        self.btn_src.pack(side="right", padx=4)
        tk.Label(fb, text="플레이 중에는 계산하지 않아요 (곡이 끝난 뒤에만)", bg=BG, fg=DIM,
                 font=(FONT, 8)).pack(side="left")
        self.src_on = False
        self.render(self.lines)

    def render(self, lines):
        tx = self.tx
        tx.configure(state="normal")
        tx.delete("1.0", "end")
        first_row = True
        for row in lines:
            st = row[0]
            if st == "k":
                lab, text = row[1], row[2]
                tx.insert("end", (lab or "") + "\t", ("k", "klab"))
                tx.insert("end", text + "\n", "k")
            elif st == "row":
                tag = "rowh" if first_row or row[1][1] == "노트" else "row"
                first_row = False
                tx.insert("end", "\t".join(row[1]) + "\n", tag)
            else:
                if st == "h":
                    first_row = True
                tx.insert("end", row[1] + "\n", st if st in ("h", "t", "p", "dim") else "p")
        tx.configure(state="disabled")
        tx.yview_moveto(0)

    def toggle_src(self):
        self.src_on = not self.src_on
        self.btn_src.configure(text="오답노트로 돌아가기" if self.src_on else "근거 자료 보기")
        self.render(self.coach.sources_text() if self.src_on else self.lines)
