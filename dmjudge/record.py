"""기록 저장: 곡이 끝나고 결과 화면을 읽으면 그 화면(PNG)과 그 곡만의 진단 기록(_raw.csv)을 자동으로 저장한다.

- 플레이 중에는 아무것도 하지 않는다. 결과 화면 숫자가 확정된 뒤, 별도 스레드에서 한 번만 돈다.
- 곡 이름: 결과 화면의 제목·작곡가 글자를 Windows 기본 OCR(Windows.Media.Ocr)로 읽고 songs.py로 수록곡 DB와 맞춘다.
  · 추가 설치 없음: Windows PowerShell 5.1에서 WinRT OCR을 부른다. 게임 프로세스·메모리는 건드리지 않는다.
  · 확실하지 않으면 파일 이름 앞에 '확인필요_'를 붙이고 가장 비슷한 후보 이름을 쓴다.
- 난이도: 결과 화면 제목 아래 배지(NORMAL/HARD/MAXIMUM/SC)의 글자 폭과 색으로 판별한다.
- 버튼: 분석기가 판별한 모드(4B/5B/6B/8B)를 쓴다.
- 저장 이름: 제목_6B_SC_20261125_2001.png / 같은 이름_raw.csv ([기록 불러오기]로 다시 열 수 있음)
"""
from __future__ import annotations

import bisect
import csv
import json
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from . import songs
from .result_ocr import to_1080

DIFFS = ("NM", "HD", "MX", "SC")


# ------------------------------------------------------------ 난이도 배지
def difficulty(g):
    """1080p 결과 화면 → 'NM'/'HD'/'MX'/'SC' 또는 None.
    배지 안 글자 폭(SC 18 · HARD 45 · NORMAL 69 · MAXIMUM 83px)으로 고르고, 색 계열(NM·HD 노랑~주황 / SC·MX 자홍~빨강)이 맞을 때만."""
    b = g[86:104, 709:799]
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 120) & (hsv[..., 2] > 150)
    if m.sum() < 40:
        return None
    cols = np.where(m.sum(0) >= 2)[0]
    if not len(cols):
        return None
    w = int(cols.max() - cols.min() + 1)
    h = float(np.median(hsv[m][:, 0]))
    warm = h <= 35            # 노랑~주황
    pink = 145 <= h <= 179    # 자홍~빨강
    if 10 <= w <= 30 and pink:
        return "SC"
    if 35 <= w <= 56 and warm:
        return "HD"
    if 58 <= w <= 76 and warm:
        return "NM"
    if 77 <= w <= 92 and pink:
        return "MX"
    return None


# ------------------------------------------------------------ 글자 영역 (OCR에 넘길 그림)
def title_masked(g):
    """제목(흰 글자)만 남긴 검은 글자/흰 바탕, 3배 확대"""
    roi = g[10:52, 772:1500]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 185) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    cols = np.where(m.sum(0) > 0)[0]
    if len(cols):
        m = m[:, :min(m.shape[1], cols.max() + 12)]
    m = cv2.resize(255 - m, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(m, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _gray(roi):
    r = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.resize(255 - r, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(r, 20, 20, 20, 20, cv2.BORDER_REPLICATE)


def title_gray(g):
    return _gray(g[10:52, 772:1500])


def composer_gray(g):
    """작곡가(회색 작은 글씨)는 배경에 따라 밝기가 달라 색으로 거르지 않고 명암만 늘린다"""
    return _gray(g[47:71, 772:1300])


# ------------------------------------------------------------ Windows OCR (PowerShell → WinRT)
_PS = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
  $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) {
  $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op))
  $task.Wait(-1) | Out-Null
  $task.Result
}
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
$engines = @()
foreach ($tag in @('ko', 'en-US', 'en', 'ja')) {
  try {
    $lang = New-Object Windows.Globalization.Language $tag
    if ([Windows.Media.Ocr.OcrEngine]::IsLanguageSupported($lang)) {
      $e = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
      if ($e -ne $null) { $engines += ,@($tag, $e) }
    }
  } catch {}
}
if ($engines.Count -eq 0) {
  $e = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
  if ($e -ne $null) { $engines += ,@('user', $e) }
}
$out = @()
foreach ($p in $args) {
  $f = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile])
  $s = Await ($f.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
  $d = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($s)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bmp = Await ($d.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $texts = @()
  foreach ($pair in $engines) {
    try {
      $r = Await ($pair[1].RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
      $texts += ($r.Lines | ForEach-Object { $_.Text }) -join ' '
    } catch { $texts += '' }
  }
  $s.Dispose()
  $out += ,@($texts)
}
ConvertTo-Json -InputObject @($out) -Compress -Depth 4
"""


def ocr_windows(images, timeout=20.0):
    """그림 목록 → 그림마다 [언어별 글자...]. 실패하면 None (Windows가 아니거나 OCR 언어가 없을 때)"""
    if os.name != "nt":
        return None
    tmp = tempfile.mkdtemp(prefix="dmjudge_ocr_")
    paths = []
    try:
        for i, im in enumerate(images):
            p = os.path.join(tmp, "t%d.png" % i)
            cv2.imwrite(p, im)
            paths.append(p)
        ps = os.path.join(tmp, "ocr.ps1")
        with open(ps, "w", encoding="utf-8-sig") as f:
            f.write(_PS)
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                            "-File", ps] + paths, capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = r.stdout.decode("utf-8", "replace").strip().lstrip("﻿")
        if not out:
            return None
        d = json.loads(out)
        # PowerShell은 원소가 하나인 배열을 풀어 버리기도 해서 모양을 맞춘다
        if isinstance(d, str):
            d = [[d]]
        return [[x] if isinstance(x, str) else list(x or []) for x in d]
    except Exception:
        return None
    finally:
        for p in paths + [os.path.join(tmp, "ocr.ps1")]:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass


def identify(g, button, ocr=ocr_windows):
    """1080p 결과 화면 → {"diff", "title", "song", "sure", "texts"}.
    sure=False면 song은 가장 비슷한 후보(없을 수도 있음)."""
    diff = difficulty(g)
    info = {"diff": diff, "title": None, "song": None, "sure": False, "texts": None}
    res = ocr([title_masked(g), title_gray(g), composer_gray(g)])
    if not res or len(res) < 3:
        return info
    titles = [t for t in res[0] + res[1] if t and t.strip()]
    comps = [t for t in res[2] if t and t.strip()]
    info["texts"] = (titles, comps)
    comp = max(comps, key=len) if comps else None
    best, cand = None, None
    for t in titles:                         # 언어·그림마다 읽은 글자 중 DB와 가장 잘 맞는 것
        b = songs.best(t, comp, button, diff)
        if b is not None and (best is None or b[1] > best[1]):
            best = b
        m = songs.match(t, comp, button, diff, 1)
        if m and (cand is None or m[0][1] > cand[1]):
            cand = m[0]
    if best is not None:
        info.update(song=best[0], title=best[0]["name"], sure=True)
    elif cand is not None and cand[1] >= 0.5:
        info.update(song=cand[0], title=cand[0]["name"])
    elif titles:
        info["title"] = max(titles, key=len)
    return info


# ------------------------------------------------------------ 곡 하나만의 진단 기록
def raw_slice(raw, t0, t1):
    """세션 원본 기록(raw)에서 곡 하나 부분만 잘라낸다. [기록 불러오기]가 곡을 다시 만들 수 있게
    곡 시작 직전의 마지막 'seg'(곡 바뀜)부터, 레인 판별('lanes')이 조각 안에 없으면 직전 것을 앞에 붙인다."""
    if not raw:
        return []
    ts = [r[0] for r in raw]
    i0 = bisect.bisect_left(ts, t0)
    i1 = bisect.bisect_right(ts, t1)
    lo = bisect.bisect_left(ts, t0 - 60000.0)
    for j in range(i0 - 1, lo - 1, -1):     # 곡 시작 전 1분 안의 마지막 곡 바뀜부터
        if raw[j][1] in ("seg", "reset"):
            i0 = j
            break
    out = list(raw[i0:i1])
    if not any(r[1] == "lanes" for r in out):
        for j in range(i0 - 1, max(-1, i0 - 200000), -1):
            if raw[j][1] == "lanes":
                out.insert(0, (out[0][0] if out else raw[j][0],) + tuple(raw[j][1:]))
                break
            if raw[j][1] == "reset":
                break
    return out


def write_raw(path, cfg, rows):
    keys = {k: v for k, v in cfg.items() if k not in ("colors",)}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["#cfg", json.dumps(keys, ensure_ascii=False)])
        w.writerow(["#colors", json.dumps(cfg.get("colors", {}), ensure_ascii=False)])
        for ev in rows:
            w.writerow(ev)


def record_dir(cfg):
    from .config import app_dir
    d = (cfg.get("record_dir") or "").strip()
    return d if d else os.path.join(app_dir(), "기록")


def _unique(base):
    p, k = base, 2
    while os.path.exists(p + ".png") or os.path.exists(p + "_raw.csv"):
        p = "%s_%d" % (base, k)
        k += 1
    return p


def blank_cover(frame, cover, monitor=0):
    """분석기 창(항상 위)이 결과 화면 제목 위에 겹치면 그 부분을 검게 지운 사본 (글자 인식용, 저장하는 사진은 원본).
    창 좌표는 화면 좌표라서 첫 번째 모니터(0번)일 때만 그대로 맞는다."""
    if not cover or int(monitor or 0) != 0:
        return frame
    x, y, w, h = [int(v) for v in cover]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame
    out = frame.copy()
    out[y0:y1, x0:x1] = 0
    return out


def save_song(frame, rows, cfg, button, when=None, ocr=ocr_windows, cover=None):
    """결과 화면 + 곡 기록 저장. 반환: (저장한 파일 이름(확장자 없음), 정보 dict)"""
    g = to_1080(blank_cover(frame, cover, cfg.get("monitor", 0)))
    info = identify(g, button, ocr)
    d = record_dir(cfg)
    os.makedirs(d, exist_ok=True)
    title = info["title"] or "곡이름모름"
    name = songs.safe_filename(title)
    if len(name) > 60:
        name = name[:60].rstrip()
    if not info["sure"]:
        name = "확인필요_" + name
    stamp = time.strftime("%Y%m%d_%H%M", time.localtime(when or time.time()))
    parts = [name, button or "모드모름", info["diff"] or "난이도모름", stamp]
    base = _unique(os.path.join(d, "_".join(parts)))
    ok, buf = cv2.imencode(".png", frame)
    if ok:
        with open(base + ".png", "wb") as f:      # 한글 경로도 되게 (cv2.imwrite는 한글 경로에서 실패)
            f.write(buf.tobytes())
    write_raw(base + "_raw.csv", cfg, rows)
    if info.get("song") is not None and info["diff"] and button:
        pi = songs.pattern_info(info["song"], button, info["diff"])
        if pi:
            info["level"], info["floor"], info["tags"] = pi
    return base, info
