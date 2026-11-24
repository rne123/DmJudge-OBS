"""디맥 판정 분석기 - 실행 진입점 (Windows)."""
import multiprocessing as mp
import queue
import sys
import time


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # 고해상도 화면에서 좌표 어긋남 방지
    except Exception:
        pass
    try:
        import ctypes
        # 타이머 정밀도 1ms (기본 15.6ms면 화면 갱신이 최대 15ms 늦어짐) + 우선순위 '보통보다 높음'
        ctypes.windll.winmm.timeBeginPeriod(1)
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
    except Exception:
        pass
    import tkinter as tk
    from tkinter import messagebox

    from dmjudge import config as C
    from dmjudge.session import Session
    from dmjudge.ui import App
    from dmjudge.keys_win import key_process_main
    from dmjudge.capture import capture_process_main, grab_fullscreen

    cfg = C.load()
    C.save(cfg)
    ses = Session(cfg)
    key_q, cap_q, cmd_q = mp.Queue(), mp.Queue(), mp.Queue()
    key_stop, cap_stop = mp.Event(), mp.Event()
    procs = {"key": None, "cap": None}

    class Ctrl:
        def save_cfg(self):
            C.save(cfg)

        def start_capture(self):
            if not cfg.get("geometry"):
                app.set_status(msg="'영역 보정'으로 플레이 영역과 판정선을 먼저 지정해 주세요.")
                return
            cap_stop.clear()
            p = mp.Process(target=capture_process_main, args=(cap_q, cap_stop, cfg, cmd_q), daemon=True)
            p.start()
            procs["cap"] = p

        def stop_capture(self):
            p = procs.get("cap")
            if p is not None:
                cap_stop.set()
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)
            procs["cap"] = None

        def restart_capture(self):
            self.stop_capture()
            self.start_capture()

        def calibrate(self):
            from dmjudge.calibrate import run_calibration
            self.stop_capture()
            root.withdraw()
            ind.close()                      # 보정 화면에 인디케이터가 찍히지 않게
            try:
                time.sleep(0.3)
                g = run_calibration(cfg, lambda: grab_fullscreen(cfg.get("monitor", 0)))
                if g:
                    cfg["geometry"] = g
                    C.save(cfg)
                    app.set_status(msg="")
            except Exception as e:
                messagebox.showerror("영역 보정", "보정 중 오류: %r" % (e,))
            finally:
                root.deiconify()
                ind.update(force=True)
                self.start_capture()

        def preview(self):
            from dmjudge.calibrate import run_preview
            self.stop_capture()
            try:
                run_preview(cfg)
            except Exception as e:
                messagebox.showerror("미리보기", "오류: %r" % (e,))
            finally:
                self.start_capture()

        def shutdown(self):
            ind.close()
            obs.close()
            self.stop_capture()
            key_stop.set()
            p = procs.get("key")
            if p is not None:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)

    root = tk.Tk()
    ctrl = Ctrl()
    app = App(root, cfg, ses, ctrl)
    from dmjudge.edition import is_obs
    from dmjudge.obs_server import ObsIndicator
    overlay = None
    if not is_obs():                     # OBS판에는 indicator.py(게임 위에 띄우는 창)가 아예 없음
        from dmjudge.indicator import Indicator
        overlay = Indicator(root, cfg, ses)
        overlay.on_save = ctrl.save_cfg
        overlay.on_moved = app.schedule_restart
    obs = ObsIndicator(cfg, ses, track_latency=overlay is None)
    app.indicator = overlay
    app.obs = obs

    class Inds:
        """게임 위 오버레이(일반판만) + OBS 브라우저 소스를 한꺼번에"""

        def update(self, force=False):
            err = None
            if overlay is not None:
                try:
                    overlay.update(force)
                except Exception as e:
                    err = e
            obs.update(force)
            if err is not None:
                raise err

        def close(self):
            if overlay is not None:
                overlay.close()

    ind = Inds()

    import os
    focus_only = mp.Value("b", 1 if cfg.get("focus_only", True) else 0)
    ctrl.focus_only = focus_only
    kp = mp.Process(target=key_process_main, args=(key_q, key_stop, focus_only, os.getpid()), daemon=True)
    kp.start()
    procs["key"] = kp
    ctrl.start_capture()

    last_refresh = [0.0]

    def poll():
        pressed = False
        try:
            while True:
                m = key_q.get_nowait()
                if m[0] == "key":
                    _, kid, down, t = m
                    if not app.key_event(kid, down):
                        ses.on_key(kid, down, t)
                        pressed = pressed or bool(down)
                elif m[0] == "info":
                    app.set_status(keys="정상")
                elif m[0] == "focus":
                    app.set_status(keys="정상" if m[1] else "게임 창 밖 (키 무시 중)")
                elif m[0] == "error":
                    app.set_status(keys="오류", msg=m[1])
        except queue.Empty:
            pass
        if pressed:
            # 누른 순간의 기록은 이미 만들어졌으니 캡처 메시지 처리보다 먼저 바로 그린다
            try:
                ind.update()
            except Exception as e:
                app.set_status(msg="인디케이터 오류: %r" % (e,))
        try:
            while True:
                m = cap_q.get_nowait()
                if m[0] == "arr":
                    ses.on_arrival(m[1], m[2], m[3], m[4], m[5], m[6] if len(m) > 6 else None)
                elif m[0] == "pred":
                    ses.on_pred(m[1], m[2])
                elif m[0] == "fstext":
                    ses.on_fstext(m[1], m[2])
                elif m[0] == "lanes":
                    ses.on_lanes(m[1])
                elif m[0] == "lc":
                    app.set_status(lc=(m[1], m[2]))
                elif m[0] == "result":
                    rec = ses.on_result(m[1], m[2])
                    if rec is not None:
                        z = ses.zero
                        zt = ("  · 영점 보정 추천 %+.1f (%d곡)" % (z["best"], z["n"])) if z and z.get("n", 0) >= 3 else \
                             ("  · 영점 추천까지 %d곡 더" % (3 - len(ses.results))) if len(ses.results) < 3 else ""
                        app.set_status(msg="결과 화면 읽음: MAX 100%% %d · 100%% 아님 %d · BREAK %d  (분석기 범위 밖 %d)%s" % (
                            rec["game"][0], rec["non100"], rec["brk"], ses.predicted_out(rec), zt))
                        app.on_song_end()
                elif m[0] == "songend":
                    ses.on_songend(m[1])
                    app.on_song_end()
                elif m[0] == "songstart":
                    ses.log("songstart", round(m[1], 2))
                elif m[0] == "vchg":
                    ses.on_vchg(m[1], m[2], m[3], m[4])
                elif m[0] == "seg":
                    ses.on_segment(m[1])
                elif m[0] == "cap":
                    fps = m[1]
                    sk = m[5] if len(m) > 5 else 0
                    app.set_status(fps=fps, speed=m[2], proc=m[4] if len(m) > 4 else 0.0,
                                   skip=(sk / (fps * 0.5)) if fps > 0 else 0.0,
                                   thr=bool(m[6]) if len(m) > 6 else False)
                    ses.log("cap", round(fps, 1), round(m[4] if len(m) > 4 else 0.0, 2), sk,
                            int(bool(m[6])) if len(m) > 6 else 0)
                elif m[0] == "error":
                    app.set_status(msg=m[1][:300])
                elif m[0] == "info":
                    app.set_status(msg="")
        except queue.Empty:
            pass
        ses.tick()
        if ses.dump_requests:
            for tq in ses.dump_requests:
                try:
                    cmd_q.put(("dump", tq))
                except Exception:
                    pass
            ses.dump_requests = []
        try:
            ind.update()
        except Exception as e:
            app.set_status(msg="인디케이터 오류: %r" % (e,))
        now = time.time()
        if now - last_refresh[0] > 0.5:
            last_refresh[0] = now
            try:
                app.refresh()
            except Exception as e:  # 표시 오류가 측정을 멈추지 않게
                app.set_status(msg="표시 오류: %r" % (e,))
        root.after(1, poll)

    root.after(100, poll)
    root.mainloop()
    # 끝낼 때 자식 프로세스(캡처·키 입력)가 남아 있으면 exe 임시 폴더를 못 지워
    # "Failed to remove temporary directory" 경고가 뜬다 → 전부 확실히 정리
    try:
        ctrl.shutdown()
    except Exception:
        pass
    for p in mp.active_children():
        try:
            p.terminate()
            p.join(timeout=1)
        except Exception:
            pass


def _run_logged():
    """바로가기(pythonw)로 실행하면 오류가 화면에 안 보이므로, 오류가 나면 error_log.txt에 남기고 알림창을 띄운다."""
    try:
        main()
    except Exception:
        import ctypes
        import os
        import traceback
        txt = traceback.format_exc()
        try:
            from dmjudge.config import app_dir
            path = os.path.join(app_dir(), "error_log.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
        except Exception:
            path = None
        try:
            ctypes.windll.user32.MessageBoxW(
                None, "오류로 종료됐어요.\n\n" + txt[-1500:] +
                ("\n\n자세한 내용: " + path if path else "") +
                "\n\nGitHub Issues에 이 내용을 올려 주세요.", "DmJudge 오류", 0x10)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    mp.freeze_support()
    _run_logged()
