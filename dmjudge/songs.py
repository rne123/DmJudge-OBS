"""곡 맞추기: 결과 화면에서 읽은(OCR) 제목·작곡가를 수록곡 DB와 비교해 가장 비슷한 곡을 고른다.

- DB: songdb.json — V-ARCHIVE OPEN API의 곡 목록(/db/v2/songs.json)을 받아 둔 사본.
  곡마다 제목·작곡가·DLC·패턴(버튼 → 난이도 → [레벨, 층, 태그])만 있고 채보 데이터는 없다.
  (V-ARCHIVE: "누구나 OPEN API로 서비스·도구를 자유롭게 만들어 쓸 수 있음", 데이터 정확성·유지는 보장 안 됨)
- DB가 없으면 songlist.tsv(제목만)로 대신한다.
- OCR은 몇 글자 틀리거나 빠뜨리기 쉬우므로, 대소문자·띄어쓰기·기호를 지운 뒤 글자 순서 유사도로 비교한다.
- 제목이 같은 다른 곡(Voyage 등)은 작곡가로, 버튼·난이도 패턴이 없는 곡은 후보에서 뺀다.
- 확실하지 않으면 None → 사용자에게 확인을 받는다.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata

_DB = None


def _here(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def norm(s):
    """비교용: 전각→반각, 소문자, (feat. ...) 같은 꼬리 정보와 기호·띄어쓰기 제거"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\(?feat\.[^)]*\)?", "", s)
    s = re.sub(r"[\s\W_]+", "", s)
    return s


def load():
    """[{id, name, composer, dlc, p, n(정규화 제목), nc(정규화 작곡가)}]"""
    global _DB
    if _DB is not None:
        return _DB
    out = []
    try:
        with open(_here("songdb.json"), encoding="utf-8") as f:
            for s in json.load(f)["songs"]:
                s = dict(s)
                s["n"], s["nc"] = norm(s["name"]), norm(s.get("composer", ""))
                out.append(s)
    except (OSError, ValueError, KeyError):
        try:
            with open(_here("songlist.tsv"), encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("# ") or "\t" not in ln:
                        continue
                    pack, title = ln.rstrip("\n").split("\t", 1)
                    out.append({"id": None, "name": title, "composer": "", "dlc": pack, "p": {},
                                "n": norm(title), "nc": ""})
        except OSError:
            pass
    _DB = out
    return _DB


def _score(q, n):
    if not q or not n:
        return 0.0
    if q == n:
        return 1.0
    r = difflib.SequenceMatcher(None, q, n, autojunk=False).ratio()
    if len(q) >= 4 and n.startswith(q):          # 제목 앞부분만 읽혔거나(길어서 잘림) 줄여 쓴 경우
        r = max(r, 0.80 + 0.2 * len(q) / len(n))
    return r


def match(title, composer=None, button=None, diff=None, top=3):
    """OCR 글자 → [(곡 dict, 점수)] 점수 높은 순.
    작곡가를 읽었으면 제목 점수에 조금 더해 주고(같은 제목 다른 곡 구분), 버튼·난이도 패턴이 없는 곡은 뺀다."""
    q = norm(title)
    qc = norm(composer) if composer else ""
    res = []
    for s in load():
        if button and s["p"]:
            pb = s["p"].get(button)
            if not pb or (diff and diff not in pb):
                continue
        res.append((s, _score(q, s["n"])))
    res.sort(key=lambda x: -x[1])
    # 작곡가는 OCR이 자주 틀리므로(회색 작은 글씨), 제목 점수가 거의 같은 후보끼리 순서를 정할 때만 쓴다
    if qc and len(res) > 1:
        near = [r for r in res if res[0][1] - r[1] < 0.06]
        if len(near) > 1:
            cs = sorted(((s, sc, _score(qc, s["nc"])) for s, sc in near), key=lambda x: -x[2])
            if cs[0][2] >= 0.6 and cs[0][2] - cs[1][2] >= 0.2:
                res = [(cs[0][0], max(sc for _, sc, _ in cs) + 0.06)] + [(s, sc) for s, sc, _ in cs[1:]] + res[len(near):]
    return res[:top]


def best(title, composer=None, button=None, diff=None, min_score=0.75, min_gap=0.06):
    """확실할 때만 (곡 dict, 점수), 아니면 None.
    1등이 min_score 이상이고 2등과 min_gap 이상 차이 나야 확실하다고 본다."""
    r = match(title, composer, button, diff, 2)
    if not r or r[0][1] < min_score:
        return None
    if len(r) > 1 and r[0][1] - r[1][1] < min_gap:
        return None
    return r[0]


def pattern_info(song, button, diff):
    """(레벨, 층 이름, 태그 목록) — 없으면 None"""
    try:
        lv, fl, tags = song["p"][button][diff]
        return lv, fl, tags
    except (KeyError, TypeError, ValueError):
        return None


def safe_filename(title):
    """파일 이름에 못 쓰는 글자만 바꾼다"""
    return re.sub(r'[\\/:*?"<>|]', "_", title).strip() or "곡"
