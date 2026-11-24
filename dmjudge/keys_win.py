"""Windows Raw Input 키 로거 (별도 프로세스에서 실행).

- 게임이 포커스를 가져도 입력을 받도록 RIDEV_INPUTSINK 사용
- 스캔코드 + E0 플래그로 키를 구분 → 넘패드 6과 → 방향키, NumLock 상태와 무관하게 구분됨
- 시간은 time.perf_counter() (QPC, 화면 캡처와 같은 시계)
- 키 반복(누르고 있을 때 반복 입력)은 걸러서 '눌림 시작'만 보냄
- focus_only가 켜져 있으면 DJMAX 창(제목에 DJMAX) 또는 분석기 자신의 창이 맨 앞에 있을 때만 보냄
  (다른 프로그램에서 치는 키는 이 프로세스 밖으로 나가지 않음). 게임 프로세스를 열거나 건드리지 않고
  맨 앞 창의 제목만 본다.
메시지: ("key", key_id, is_down, t_ms), ("focus", 게임 창 여부)
"""
from __future__ import annotations

import time

KEY_NAMES = {
    "01": "Esc", "0F": "Tab", "1C": "Enter", "39": "Space", "3A": "CapsLock",
    "2A": "LShift", "36": "RShift", "1D": "LCtrl", "1De": "RCtrl", "38": "LAlt", "38e": "RAlt",
    "1E": "A", "1F": "S", "20": "D", "21": "F", "22": "G", "23": "H", "24": "J", "25": "K",
    "26": "L", "27": ";", "28": "'", "10": "Q", "11": "W", "12": "E", "13": "R", "14": "T",
    "15": "Y", "16": "U", "17": "I", "18": "O", "19": "P", "2C": "Z", "2D": "X", "2E": "C",
    "2F": "V", "30": "B", "31": "N", "32": "M", "33": ",", "34": ".", "35": "/",
    "35e": "Num/", "37": "Num*", "4A": "Num-", "4E": "Num+", "1Ce": "NumEnter",
    "47": "Num7", "48": "Num8", "49": "Num9", "4B": "Num4", "4C": "Num5", "4D": "Num6",
    "4F": "Num1", "50": "Num2", "51": "Num3", "52": "Num0", "53": "Num.",
    "48e": "↑", "50e": "↓", "4Be": "←", "4De": "→",
    "47e": "Home", "4Fe": "End", "49e": "PgUp", "51e": "PgDn", "52e": "Ins", "53e": "Del",
    "3B": "F1", "3C": "F2", "3D": "F3", "3E": "F4", "3F": "F5", "40": "F6", "41": "F7",
    "42": "F8", "43": "F9", "44": "F10", "57": "F11", "58": "F12",
    "02": "1", "03": "2", "04": "3", "05": "4", "06": "5", "07": "6", "08": "7", "09": "8",
    "0A": "9", "0B": "0", "0C": "-", "0D": "=", "1A": "[", "1B": "]", "2B": "\\", "29": "`",
}


def key_name(kid: str) -> str:
    return KEY_NAMES.get(kid, kid)


GAME_TITLE = "DJMAX"


def run_key_logger(q, stop_evt=None, focus_only=None, allow_pid=None):
    import ctypes
    from ctypes import wintypes as wt

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    LRESULT = wt.LPARAM
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [("style", wt.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int), ("hInstance", wt.HINSTANCE), ("hIcon", wt.HICON),
                    ("hCursor", wt.HANDLE), ("hbrBackground", wt.HBRUSH),
                    ("lpszMenuName", wt.LPCWSTR), ("lpszClassName", wt.LPCWSTR)]

    class RAWINPUTDEVICE(ctypes.Structure):
        _fields_ = [("usUsagePage", wt.USHORT), ("usUsage", wt.USHORT),
                    ("dwFlags", wt.DWORD), ("hwndTarget", wt.HWND)]

    class RAWINPUTHEADER(ctypes.Structure):
        _fields_ = [("dwType", wt.DWORD), ("dwSize", wt.DWORD),
                    ("hDevice", wt.HANDLE), ("wParam", wt.WPARAM)]

    class RAWKEYBOARD(ctypes.Structure):
        _fields_ = [("MakeCode", wt.USHORT), ("Flags", wt.USHORT), ("Reserved", wt.USHORT),
                    ("VKey", wt.USHORT), ("Message", wt.UINT), ("ExtraInformation", wt.ULONG)]

    class RAWINPUT_KB(ctypes.Structure):
        _fields_ = [("header", RAWINPUTHEADER), ("keyboard", RAWKEYBOARD)]

    WM_INPUT = 0x00FF
    RID_INPUT = 0x10000003
    RIM_TYPEKEYBOARD = 1
    RIDEV_INPUTSINK = 0x00000100
    RI_KEY_BREAK, RI_KEY_E0, RI_KEY_E1 = 1, 2, 4
    HWND_MESSAGE = wt.HWND(-3)

    user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.DefWindowProcW.restype = LRESULT
    user32.GetRawInputData.argtypes = [wt.HANDLE, wt.UINT, ctypes.c_void_p,
                                       ctypes.POINTER(wt.UINT), wt.UINT]
    user32.GetRawInputData.restype = wt.UINT
    user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.HWND,
                                       wt.HMENU, wt.HINSTANCE, wt.LPVOID]
    user32.CreateWindowExW.restype = wt.HWND
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    user32.RegisterClassW.restype = wt.ATOM
    user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wt.UINT, wt.UINT]
    user32.RegisterRawInputDevices.restype = wt.BOOL
    user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
    user32.GetMessageW.restype = wt.BOOL
    user32.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
    user32.PeekMessageW.restype = wt.BOOL
    user32.MsgWaitForMultipleObjects.argtypes = [wt.DWORD, ctypes.c_void_p, wt.BOOL, wt.DWORD, wt.DWORD]
    user32.MsgWaitForMultipleObjects.restype = wt.DWORD
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
    kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wt.HMODULE
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wt.HWND
    user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    user32.GetWindowThreadProcessId.restype = wt.DWORD

    fg_cache = {}                 # 창 → (게임 창인지, 확인한 시각)
    fg_last = [None]
    title_buf = ctypes.create_unicode_buffer(256)

    def focus_ok(t):
        """맨 앞 창이 DJMAX(제목) 또는 분석기 자신이면 True"""
        if focus_only is None or not focus_only.value:
            return True
        h = user32.GetForegroundWindow()
        if not h:
            ok = False
        else:
            key = int(h)
            c = fg_cache.get(key)
            if c is not None and t - c[1] < 2000.0:
                ok = c[0]
            else:
                pid = wt.DWORD(0)
                user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                if allow_pid is not None and pid.value == allow_pid:
                    ok = True
                else:
                    n = user32.GetWindowTextW(h, title_buf, 256)
                    ok = n > 0 and GAME_TITLE in title_buf.value.upper()
                if len(fg_cache) > 64:
                    fg_cache.clear()
                fg_cache[key] = (ok, t)
        if ok != fg_last[0]:
            fg_last[0] = ok
            q.put(("focus", ok))
        return ok

    sent_down = set()             # 밖으로 보낸 '누름' (뗌은 짝이 되는 누름을 보냈을 때만 보냄)

    down_state = {}
    buf = RAWINPUT_KB()
    hdr_size = ctypes.sizeof(RAWINPUTHEADER)

    def wndproc(hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            t = time.perf_counter() * 1000.0
            size = wt.UINT(ctypes.sizeof(buf))
            r = user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, ctypes.byref(buf),
                                       ctypes.byref(size), hdr_size)
            if r != 0xFFFFFFFF and r > 0 and buf.header.dwType == RIM_TYPEKEYBOARD:
                kb = buf.keyboard
                if not (kb.Flags & RI_KEY_E1) and kb.MakeCode not in (0, 0xFF):
                    kid = "%02X" % kb.MakeCode + ("e" if kb.Flags & RI_KEY_E0 else "")
                    if kid not in ("2Ae", "36e"):          # 가짜 Shift(확장키 보조) 제외
                        is_down = not (kb.Flags & RI_KEY_BREAK)
                        if down_state.get(kid) != is_down:  # 반복 입력 제거
                            down_state[kid] = is_down
                            if is_down:
                                if focus_ok(t):
                                    sent_down.add(kid)
                                    q.put(("key", kid, True, t))
                            elif kid in sent_down:
                                sent_down.discard(kid)
                                q.put(("key", kid, False, t))
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    proc = WNDPROC(wndproc)
    hinst = kernel32.GetModuleHandleW(None)
    wc = WNDCLASSW()
    wc.lpfnWndProc = proc
    wc.hInstance = hinst
    wc.lpszClassName = "DmJudgeRawInput"
    user32.RegisterClassW(ctypes.byref(wc))
    hwnd = user32.CreateWindowExW(0, wc.lpszClassName, "dmjudge", 0, 0, 0, 0, 0,
                                  HWND_MESSAGE, None, hinst, None)
    if not hwnd:
        q.put(("error", "키 입력 창 생성 실패: %d" % ctypes.get_last_error()))
        return
    rid = RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd)
    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
        q.put(("error", "Raw Input 등록 실패: %d" % ctypes.get_last_error()))
        return
    q.put(("info", "키 입력 준비 완료"))

    msg = wt.MSG()
    QS_ALLINPUT, PM_REMOVE = 0x04FF, 0x0001
    while stop_evt is None or not stop_evt.is_set():
        user32.MsgWaitForMultipleObjects(0, None, False, 200, QS_ALLINPUT)
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


def key_process_main(q, stop_evt, focus_only=None, allow_pid=None):
    try:
        # 이 프로세스의 우선순위를 높여 타임스탬프 지연을 줄임
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
    except Exception:
        pass
    try:
        run_key_logger(q, stop_evt, focus_only, allow_pid)
    except Exception as e:  # pragma: no cover
        q.put(("error", "키 로거 오류: %r" % (e,)))
