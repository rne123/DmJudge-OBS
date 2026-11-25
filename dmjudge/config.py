"""설정 파일(config.json) 읽기/쓰기. exe 옆에 저장된다."""
from __future__ import annotations

import copy
import json
import os
import sys

from .tracker import DEFAULT_COLORS

CHANNELS = ["L1", "L2", "L3", "L4", "L5", "L6", "TL", "TR", "SL", "SR"]
MODES = ["4B", "5B", "6B", "8B"]


def mode_channels(mode):
    n = {"4B": 4, "5B": 5, "6B": 6, "8B": 6}[mode]
    ch = ["L%d" % (i + 1) for i in range(n)]
    if mode == "8B":
        ch += ["TL", "TR"]
    return ch + ["SL", "SR"]
CH_LABEL = {
    "L1": "레인 1", "L2": "레인 2", "L3": "레인 3", "L4": "레인 4", "L5": "레인 5", "L6": "레인 6",
    "TL": "트리거 왼쪽", "TR": "트리거 오른쪽", "SL": "사이드 왼쪽", "SR": "사이드 오른쪽",
}

DEFAULTS = {
    "monitor": 0,
    "lanes": 6,
    "geometry": None,            # 보정 도구로 지정: x0, x1, top, occl_y, judge_y (화면 픽셀)
    "big_effects": False,        # 판정 이펙트가 큰 스킨: 추적 한계선을 영역 높이 30%까지 올림
    "colors": DEFAULT_COLORS,
    "mode": "auto",              # auto(자동 판별) / 4B / 5B / 6B / 8B
    "modes": {                   # 모드별: 채널 → 키 목록(스캔코드, e = 확장키), 손 그룹
        "4B": {"lanes": 4,
               "keymap": {"L1": ["20"], "L2": ["21"], "L3": ["4B"], "L4": ["4C"],   # D F Num4 Num5
                          "SL": ["39"], "SR": ["4E"]},                               # Space, Num+
               "groups": {"L1": "왼손", "L2": "왼손", "L3": "오른손", "L4": "오른손",
                          "SL": "사이드", "SR": "사이드"}},
        "5B": {"lanes": 5,
               "keymap": {"L1": ["1F"], "L2": ["20"], "L3": ["21", "4B"], "L4": ["4C"], "L5": ["4D"],  # S D (F/Num4) Num5 Num6
                          "SL": ["39"], "SR": ["52"]},                               # Space, Num0
               "groups": {"L1": "왼손", "L2": "왼손", "L3": "가운데", "L4": "오른손", "L5": "오른손",
                          "SL": "사이드", "SR": "사이드"}},
        "6B": {"lanes": 6,
               "keymap": {"L1": ["1F"], "L2": ["20"], "L3": ["21"], "L4": ["4B"], "L5": ["4C"], "L6": ["4D"],
                          "SL": ["39"], "SR": ["52"]},                               # Space, Num0
               "groups": {"L1": "왼손", "L2": "왼손", "L3": "왼손", "L4": "오른손", "L5": "오른손",
                          "L6": "오른손", "SL": "사이드", "SR": "사이드"}},
        "8B": {"lanes": 6,
               "keymap": {"L1": ["1F"], "L2": ["20"], "L3": ["21"], "L4": ["4B"], "L5": ["4C"], "L6": ["4D"],
                          "TL": ["39"], "TR": ["4De"],                               # Space, →
                          "SL": ["3A"], "SR": ["4E"]},                               # CapsLock, Num+
               "groups": {"L1": "왼손", "L2": "왼손", "L3": "왼손", "L4": "오른손", "L5": "오른손",
                          "L6": "오른손", "TL": "트리거", "TR": "트리거", "SL": "사이드", "SR": "사이드"}},
    },
    "judge_setting_ms": 0,       # 게임의 판정 타이밍 설정값
    "output_setting_ms": 0,      # 게임의 노트 출력 타이밍 설정값
    "window_ms": 41.67,          # MAX 100% 범위 (±) = 2.5/60초 (사용자 결과 화면으로 확인)
    "fs_ms": 41.67,              # FAST/SLOW 범위 (±ms). 이 밖이면 FAST/SLOW로 세고 인디케이터가 반짝임
    "dark_notes": False,         # 어두운/반투명 노트 스킨이면 켜기 (밝기 기준을 낮춤)
    "calib_ms": 0.0,             # 영점 보정값 ([게임 결과로 보정]에서 설정)
    "calib_split": {},           # 버튼별 영점 따로 쓰기 {"4B": True, ...}: 켠 버튼은 calib_btn 값, 나머지는 calib_common
    "calib_btn": {},             # 버튼별 영점 값 {"4B": 6.5, ...}
    "match_window_ms": 150,      # 키 입력과 노트를 짝지을 최대 거리 (게임은 ±127ms 입력도 SLOW로 판정함)
    "match_early_ms": 190,       # 일찍 친 쪽 최대 거리 (실측: 게임은 약 -185ms까지 FAST로 받음)
    "match_v": 5,
    "capture_bias_ms": 0.0,      # 캡처 지연 보정(고급)
    "chatter_ms": 25,
    "recent": 0,                 # 0 = 전체, N = 최근 N개만
    "record_save": False,        # [기록 저장]: 곡이 끝나면 결과 화면 PNG + 그 곡 _raw.csv를 자동 저장
    "record_dir": "",            # 비우면 프로그램 폴더의 '기록' 폴더
    "measure": "eye",            # eye(눈판정) / ear(귀판정)
    "focus_only": True,          # 게임 창(제목에 DJMAX) 또는 분석기 창이 맨 앞일 때만 키 기록
    "ui": {"alpha": 0.92, "geometry": ""},
    "indicator": {               # 기어 위 EARLY/LATE 인디케이터
        "enabled": True,
        "y_frac": 0.45,          # 판정선에서 위로 플레이 영역 높이의 몇 % 지점
        "zone_frac": 0.416,      # 가운데 띠(FAST/SLOW 범위)의 길이 비율 (고정). 양끝 ±ms = 범위 ÷ 이 값
        "seconds": 3.0,          # 최근 몇 초 동안 친 노트만 표시
        "width_frac": 0.45,      # 플레이 영역 폭 대비 막대 폭
        "height": 58,
        "style_v": 2,
    },
}


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path():
    return os.path.join(app_dir(), "config.json")


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load():
    p = config_path()
    data = {}
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    # v0.3.0의 자동 영점이 바꿔 놓은 보정값은 되돌린다 (기능 제거)
    if "auto_zero" in data:
        if data.get("auto_zero"):
            data["calib_ms"] = 0.0
        data.pop("auto_zero", None)
    data.pop("fs_text", None)
    if data.get("match_v", 1) < 2:
        # v0.3.22: 게임이 ±100ms 넘는 입력도 판정하는 것이 확인됨 → 짝짓기 범위 확대
        if float(data.get("match_window_ms", 100)) <= 100:
            data["match_window_ms"] = 150
        data["match_v"] = 2
    if data.get("match_v", 1) < 3:
        data["match_early_ms"] = 190
        data["match_v"] = 3
    if data.get("match_v", 1) < 4:
        # v0.3.28: MAX 100% 범위 = 41.67 (2.5/60초). 결과 화면 등급 경계로 확인. 41.6으로 두었던 것만 바꿈
        for k in ("window_ms", "fs_ms"):
            if abs(float(data.get(k, 41.67)) - 41.6) < 0.05:
                data[k] = 41.67
        data["match_v"] = 4
    if data.get("match_v", 1) < 5:
        # v0.3.29: 범위 칸이 소수 한 자리로 저장해 41.67 → 41.7이 되던 것 되돌림
        for k in ("window_ms", "fs_ms"):
            if abs(float(data.get(k, 41.67)) - 41.7) < 0.005:
                data[k] = 41.67
        data["match_v"] = 5
    col = data.get("colors")
    if isinstance(col, dict) and col.get("colors_v", 1) < 2:
        # v0.3.5: 다양한 노트 스킨 대응 기본값으로 갱신
        col.update({"lane_frac": 0.45, "gap_frac": 0.35, "colors_v": 2})
    ind = data.get("indicator")
    if isinstance(ind, dict) and ind.get("style_v", 1) < 2:
        ind.update({"width_frac": 0.45, "height": 58, "style_v": 2})
    if isinstance(ind, dict):
        ind.pop("range_ms", None)
    return _merge(DEFAULTS, data)


def save(cfg):
    p = config_path()
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
