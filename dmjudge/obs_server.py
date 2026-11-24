"""OBS 브라우저 소스용 인디케이터.

내 PC 안에서만 열리는 작은 웹 서버(127.0.0.1)를 띄우고, 인디케이터 그림(투명 PNG)을
바뀔 때마다 밀어 보낸다(SSE). OBS에서 [브라우저 소스]에 주소를 넣으면 방송·녹화 화면에만 보이고,
게임 화면 위에는 아무것도 띄우지 않는다. 다른 PC나 인터넷에서는 접속할 수 없다(127.0.0.1에만 열림).
"""
from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8777
FALLBACK_GEAR_W = 500     # 영역 보정 전이면 1080p 기본 기어 폭으로 계산

PAGE = b"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>DmJudge indicator</title>
<style>
html,body{margin:0;padding:0;background:transparent!important;overflow:hidden;width:100%;height:100%}
img{display:block;width:100vw;height:100vh;object-fit:contain}
</style></head><body><img id="i" alt="">
<script>
'use strict';
const img = document.getElementById('i');
function connect(){
  const es = new EventSource('/events');
  es.onmessage = (e) => { img.src = 'data:image/png;base64,' + e.data; };
  es.onerror = () => { img.removeAttribute('src'); };
}
connect();
</script></body></html>"""


def obs_opts(cfg):
    from .edition import is_obs
    # OBS판은 기본으로 켜짐, 일반판은 기본으로 꺼짐
    o = {"enabled": is_obs(), "port": DEFAULT_PORT, "scale": 1.0}
    o.update(cfg.get("obs", {}))
    return o


def base_size(cfg):
    """기존 디맥냥이의 게임 위 인디케이터와 똑같은 크기 (기어 폭 × 막대 폭 비율, 높이 58 + 글자 크기)"""
    from .capture import indicator_size
    from .ind_state import opts_of
    o = opts_of(cfg)
    g = cfg.get("geometry")
    fw = (g["x1"] - g["x0"]) if g else FALLBACK_GEAR_W
    w = int(fw * max(0.2, min(1.5, float(o["width_frac"]))))
    return max(40, w), int(indicator_size(o)[1])


def obs_scale(cfg):
    try:
        return max(0.5, min(4.0, float(obs_opts(cfg).get("scale", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def obs_size(cfg):
    """브라우저 소스 크기 (폭, 높이) = 원래 크기 × 크기 %"""
    bw, bh = base_size(cfg)
    sc = obs_scale(cfg)
    return max(8, int(round(bw * sc))), max(8, int(round(bh * sc)))


def to_png(canvas, size=None):
    """ind_render.Canvas(premultiplied RGBA) → 투명 PNG 바이트.
    size가 다르면 프로그램 안에서 크기를 바꾼다: 줄일 때는 면적 평균(INTER_AREA)이라
    얇고 옅은 선도 사라지지 않는다 (OBS에서 줄이면 얇은 선이 지워짐)."""
    import cv2
    import numpy as np
    p = canvas.px
    if size is not None and (size[0], size[1]) != (p.shape[1], p.shape[0]):
        interp = cv2.INTER_AREA if size[0] < p.shape[1] else cv2.INTER_LINEAR
        p = cv2.resize(p, (int(size[0]), int(size[1])), interpolation=interp)   # premultiplied 상태로 (가장자리 정확)
    a = p[..., 3:4]
    rgb = np.where(a > 1e-6, p[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([rgb[..., ::-1], a], axis=2)          # BGRA (straight alpha)
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG 변환 실패")
    return buf.tobytes()


class _Hub:
    """가장 최근 그림 하나만 들고 있다가 접속한 OBS들에 보낸다"""

    def __init__(self):
        self.cond = threading.Condition()
        self.rev = 0
        self.b64 = ""
        self.closed = False
        self.clients = 0                          # 지금 연결된 OBS/브라우저 수 (없으면 그리지 않음)

    def publish(self, png):
        b = base64.b64encode(png).decode("ascii")
        with self.cond:
            self.b64 = b
            self.rev += 1
            self.cond.notify_all()

    def wait(self, rev, timeout=10.0):
        with self.cond:
            self.cond.wait_for(lambda: self.rev != rev or self.closed, timeout=timeout)
            return self.rev, self.b64


def _make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(PAGE, "text/html; charset=utf-8")
            if path == "/frame.png":
                with hub.cond:
                    b = hub.b64
                return self._send(base64.b64decode(b) if b else b"", "image/png")
            if path == "/state.json":
                return self._send(json.dumps({"rev": hub.rev}).encode(), "application/json")
            if path != "/events":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            rev = -1
            with hub.cond:
                hub.clients += 1
            try:
                while not hub.closed:
                    nrev, b = hub.wait(rev)
                    if hub.closed:
                        break
                    if nrev != rev and b:
                        rev = nrev
                        data = "data: " + b + "\n\n"
                    else:
                        data = ": ping\n\n"           # 연결 확인용 (끊긴 OBS 정리)
                    self.wfile.write(data.encode("ascii"))
                    self.wfile.flush()
            except (OSError, ValueError):
                pass
            finally:
                with hub.cond:
                    hub.clients -= 1

    return Handler


class ObsIndicator:
    """메인 루프에서 update()를 부르면 새로 그릴 게 있을 때만 그림을 만들어 보낸다."""

    def __init__(self, cfg, session, track_latency=False):
        from .ind_state import IndModel
        self.cfg, self.ses = cfg, session
        self.model = IndModel(track_latency=track_latency)
        self.hub = None
        self.httpd = None
        self.thread = None
        self.port = None
        self.error = None
        self._size = None
        self._watched = False
        self._jcv = threading.Condition()
        self._job = None
        self._stop = False
        self._wthread = None

    # ------------------------------------------------------------ 서버
    @property
    def running(self):
        return self.httpd is not None

    @property
    def url(self):
        return "http://127.0.0.1:%d/" % self.port if self.port else ""

    def start(self):
        if self.httpd is not None:
            return True
        base = int(obs_opts(self.cfg).get("port", DEFAULT_PORT))
        hub = _Hub()
        last_err = None
        for port in range(base, base + 10):     # 다른 프로그램이 쓰고 있으면 다음 번호
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(hub))
                break
            except OSError as e:
                last_err = e
        else:
            self.error = "OBS 서버를 열 수 없어요 (포트 %d~%d 사용 중): %s" % (base, base + 9, last_err)
            return False
        httpd.daemon_threads = True
        self.hub, self.httpd, self.port, self.error = hub, httpd, port, None
        self.thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.2), daemon=True)
        self.thread.start()
        if self._wthread is None or not self._wthread.is_alive():
            self._stop = False
            self._wthread = threading.Thread(target=self._worker, daemon=True)
            self._wthread.start()
        self.model.invalidate()
        return True

    def close(self):
        if self.httpd is None:
            return
        hub, httpd = self.hub, self.httpd
        self.httpd = self.hub = None
        self.port = None
        with self._jcv:
            self._stop = True
            self._job = None
            self._jcv.notify_all()
        self._wthread = None
        with hub.cond:
            hub.closed = True
            hub.cond.notify_all()
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass

    # ------------------------------------------------------------ 그리기
    def update(self, force=False):
        if not obs_opts(self.cfg)["enabled"]:
            self.close()
            return
        if not self.start():
            return
        size = obs_size(self.cfg)
        if size != self._size:
            self._size = size
            force = True
        if self.hub.clients <= 0:
            self._watched = False
            return                                # 보는 곳(OBS)이 없으면 그리지 않음 → 부하 없음
        if not self._watched:
            self._watched = True
            force = True                          # 방금 접속했으면 바로 한 장
        f = self.model.frame(self.cfg, self.ses, force=force)
        if f is None:
            return
        # 그리기·PNG 변환(수 ms)은 별도 스레드에서: 키·노트 처리 루프를 붙잡지 않게. 밀리면 가장 최근 것만 그림
        job = (f, size, base_size(self.cfg), self.hub)
        with self._jcv:
            self._job = job
            self._jcv.notify()

    def _worker(self):
        from .ind_render import render
        while True:
            with self._jcv:
                while self._job is None and not self._stop:
                    self._jcv.wait(1.0)
                if self._stop:
                    return
                f, size, (bw, bh), hub = self._job
                self._job = None
            try:
                o = f["o"]
                sc = size[0] / float(bw)
                if sc < 0.95:
                    # 줄이면 1~2px 선이 옅어지므로 그만큼 진하게 (최대 2배)
                    o = dict(o)
                    o["strength"] = float(o.get("strength", 1.0)) * min(2.0, 1.0 / max(sc, 0.3))
                img = render(bw, bh, f["vis"], f["R"], f["fs"], f["win"], f["fl"], o,
                             fl_ms=f["FL"], last=f["last"], scale=sc)   # 실제 크기로 바로 그림 (글자·선 선명)
                hub.publish(to_png(img, size))
            except Exception as e:                # 그림 오류가 측정을 멈추지 않게
                self.error = "OBS 그림 오류: %r" % (e,)
