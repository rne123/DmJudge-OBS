"""리듬게임 코치: 곡 하나의 기록으로 채보 모양을 다시 만들고, 패턴·손·레인별로 판정을 나눠 보고 조언을 만든다.

- 플레이 중에는 아무것도 하지 않는다 (곡이 끝난 뒤, 또는 [코치 조언] 버튼을 누를 때만 계산).
- 입력: 노트 도달(레인, 시각, 롱노트 여부), 짝지어진 판정 오차, 키 입력, 변속 감지 시각.
- 조언마다 [데이터] 근거 수치와 [근거] 출처(커뮤니티 공략 / 연구)를 같이 붙인다.
  원인은 '가능성'으로만 말한다 — 데이터는 '어디서 어떻게 틀어지는지'까지는 정확하지만, '왜'는 추정이다.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np

# ------------------------------------------------------------------ 출처 (조언에 붙는 근거)
SOURCES = {
    "sms": ("박자에 맞춰 두드리는 연구에서 '평균 어긋남(치우침)'과 '흔들림(표준편차)'은 따로 보는 두 지표이고, "
            "흔들림이 안정성의 지표. 훈련된 음악가일수록 흔들림이 작다.",
            "Repp & Su (2013) Sensorimotor synchronization 리뷰", "https://link.springer.com/article/10.3758/s13423-012-0371-2"),
    "nma": ("사람은 박자보다 수십 ms 일찍 치는 경향(음의 평균 어긋남)이 있고, 덜 일찍 치는 사람일수록 흔들림도 작다.",
            "Psychological Research (2018) 음의 어긋남과 흔들림의 관계", "https://link.springer.com/10.1007/s00426-018-1043-2"),
    "phase": ("어긋남이 생기면 다음 타격에서 일부(보통 절반 정도)만 고친다. 고치는 힘이 약하면 어긋남이 여러 노트에 걸쳐 "
              "물결처럼 이어진다(연속 노트끼리 오차가 비슷해짐).",
              "Repp & Keller / Vorberg 위상 보정 모델", "https://haskinslabs.org/sites/default/files/files/Reprints/HL1496.pdf"),
    "visual": ("내려오는(움직이는) 시각 자극에 맞추기는 깜빡이는 자극보다 훨씬 정확하고 소리에 가깝지만, "
               "같은 조건에서 소리에 맞출 때가 여전히 덜 흔들린다.",
               "Hove·Iversen·Zhang·Repp (2013), Iversen 외 (2015)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4255154"),
    "guidance": ("매 시도마다 즉시 결과를 보여주면 연습 중엔 잘하지만, 그 정보에 기대게 되어 장기 학습엔 오히려 불리할 수 있다 "
                 "(가이던스 가설). 요약 피드백·가끔 보는 피드백이 대안.",
                 "Winstein·Pohl·Lewthwaite, guidance hypothesis", "https://experts.nau.edu/en/publications/effects-of-physical-guidance-and-knowledge-of-results-on-motor-le/"),
    "challenge": ("실력에 비해 너무 쉽거나 너무 어려운 과제는 학습이 덜 된다. 실력에 맞는 '적당히 어려운' 지점이 있다.",
                  "Guadagnoli & Lee (2004) Challenge Point Framework", "https://en.wikipedia.org/wiki/Challenge_point_framework"),
    "spacing": ("같은 연습량이면 몰아서 하는 것보다 나눠서 하는 편이 성과가 좋다(메타분석 효과크기 0.46).",
                "Donovan & Radosevich (1999) 분산 연습 메타분석", "https://www.gwern.net/docs/spaced-repetition/1999-donovan.pdf"),
    "sleep": ("손가락 순서 과제는 잠을 잔 뒤 기억이 굳는다는 연구가 많다(다만 '실력이 더 오른다'는 부분은 반론도 있음).",
              "수면과 운동 순서 학습 연구 (찬반 모두 있음)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10474136"),
    "warmup": ("쉬었다 다시 하면 처음 몇 분은 실력이 일시적으로 떨어진다(워밍업 감소). 다시 시작하기 직전에 "
               "'같은 동작' 자체를 해 주는 것이 가장 효과적이었다.",
               "Adams (1961) 워밍업 감소 리뷰, Anshel & Wrisberg (1993)", "https://coachsci.sdsu.edu/csa/vol122/anshel.htm"),
    "fatigue": ("손가락을 최대로 빠르게 두드리면 10~30초 만에 속도가 떨어진다(주로 중추 피로).",
                "Scientific Reports (2018) 손가락 두드리기 피로", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6008331"),
    "focus": ("몸 동작(손가락)보다 동작의 결과(노트·소리)에 주의를 두면 낫다는 메타분석이 있지만, 출판 편향을 보정하면 "
              "효과가 거의 없다는 반론도 있다. '손을 일일이 신경 쓰지 않기' 정도로만 참고.",
              "Chua 외 (2021) 메타분석 / McKay 외 (2024) 반론", "https://digitalscholarship.unlv.edu/kns_fac_articles/428"),
    "choking": ("압박을 받으면 잘 익은 동작을 의식적으로 하나하나 통제하려 들어 오히려 무너진다(명시적 감시 이론).",
                "Beilock & Carr, choking under pressure", "https://louisville.edu/psychology/decaro/lab/publications-2/decaro_thomas_albert_beilock_2011"),
    "bimanual": ("양손 타이밍은 서로 묶여 있고(양손으로 치면 오히려 덜 흔들림), 주로 쓰는 손과 다른 손의 차이는 "
                 "훈련(드러머·음악가)으로 줄어든다.",
                 "Helmuth & Ivry (1996), Fujii 외 드러머 연구", "https://ivrylab.berkeley.edu/files/organized_pubs_pdfs/1996_helmuth_ivry_jephpp.pdf"),
    "chord": ("피아니스트도 '동시에' 친 음이 실제로는 어긋나며, 같은 손 안의 어긋남은 작고 양손 사이 어긋남이 더 크다.",
              "Goebl (2001) 피아노 동시 타건 어긋남", "https://iwk.mdw.ac.at/goebl/papers/Goebl_JASA2001_melodyLead.pdf"),
    "deliberate": ("의도적 연습량은 게임 실력 차이의 약 26%를 설명한다(중요하지만 전부는 아님).",
                   "Macnamara·Hambrick·Oswald (2014) 메타분석", "https://pubmed.ncbi.nlm.nih.gov/24986855/"),
    # ---- BMS(발광·Overjoy·개전)·IIDX·LR2·beatoraja 커뮤니티 (v0.4.5 조사)
    "iidx_judge": ("판정 타이밍은 리듬감보다 눈(판정선에 겹치는 순간)으로, 노트 수가 적당한 쉬운 곡으로 맞춘다. 빽빽한 곡은 노트를 줍느라 "
                   "시선이 올라가 FAST가 늘고, 거기에 맞춰 판정을 옮기면 쉬운 곡에서 SLOW만 나오는 악순환. FAST/SLOW 비율은 참고만 하고, "
                   "한 번 정하면 그날 컨디션·곡마다 바꾸지 말고 시선·배속으로 몸을 맞춘다.",
                   "THE SAFARI — 판정 타이밍 조정 요령 (IIDX)", "https://the-safari.com/2279"),
    "iidx_fasttrap": ("빽빽한 곡을 치기 쉽게 판정을 계속 옮겼더니 쉬운 곡에서 FAST가 쏟아졌다. 판정은 쉬운 부분에 맞추고, 그 판정으로 "
                      "빽빽한 부분도 칠 수 있는 힘(강하게 누르는 연타)을 기르는 쪽이 실력이 늘었다.",
                      "X-JIN — FAST 판정이 쏟아지는 함정 (IIDX)", "https://quasar5134.hatenablog.com/entry/2020/03/06/135452"),
    "iidx_slow": ("SLOW가 많은 흔한 원인: 판정 타이밍이 안 맞음, 배속·가리개가 지나침, 실력보다 어려워 인식이 늦음, 시선이 너무 아래, "
                  "피로·식사·졸음. 배속은 버틸 수 있는 한계보다 조금 아래로 정하고 자주 바꾸지 않는다.",
                  "SLOW가 많이 나는 원인과 대처 (IIDX·BMS)", "https://www.otomata01.com/2022-01-19-220120/"),
    "iidx_cond": ("인식이 또렷한 날은 FAST 쪽, 지쳐서 보고 치기까지 늦어지는 날은 SLOW 쪽으로 쏠린다(늘 치던 같은 기기 기준).",
                  "IIDX 컨디션과 FAST/SLOW (중전 블로그)", "https://blog.goo.ne.jp/abeta0730/e/791c2cfdad079438db203956f8fc5e39"),
    "iidx_deliberate": ("결과 화면만으로는 원인을 모른다. 플레이를 녹화해 보니 연타를 너무 세게 누르는 버릇이 있었는데 감각만으론 몰랐다. "
                        "막연히 치지 말고 그날 과제와 연습 곡을 미리 정하고, 편한 영역에서 조금 벗어나 연습한다.",
                        "rice Place — 비트매니아와 한계적 연습", "https://scrapbox.io/jigsaw/%E3%83%93%E3%83%BC%E3%83%88%E3%83%9E%E3%83%8B%E3%82%A2%E3%81%A8%E3%80%8C%E9%99%90%E7%95%8C%E7%9A%84%E7%B7%B4%E7%BF%92%E3%80%8D_-_rice_Place%E3%81%AE%E6%97%A5%E8%A8%98"),
    "iidx_jack": ("연타(縦連)는 손가락 끝에 힘을 주지 말고 손목 스냅·팔뚝으로 '튕기듯' 친다. 누르는 힘보다 빨리 떼는 것을 의식한다. "
                  "같은 레인 연타와 다른 노트를 같은 방식으로 치면 리듬이 덜 무너진다.",
                  "IIDX 연타 공략 (SEIRYU, ひとみ 외)", "https://seiryu1210.xyz/entry/2020/07/12/003105"),
    "oh_stream": ("스트림·버스트에서 밀리는 느낌이 들면, 노트를 못 읽어서가 아니라면 대개 키를 필요 이상 오래 누르고 있어서다. "
                  "밀린다고 더 빨리 치려 하면 힘이 들어가 더 지친다.",
                  "oh! 포럼 — 스트림·버스트 팁 (키를 오래 누르면 밀림)", "https://osu.ppy.sh/community/forums/posts/4655285"),
    "iidx_relax": ("손가락은 팔을 늘어뜨렸을 때처럼 살짝 굽힌 모양으로. 중간 속도(BPM 150 안팎)에선 손목 힘을 완전히 빼고, "
                   "빠른 곡에서만 위치를 잡을 만큼 힘을 준다. 팔꿈치에서 손끝까지 일직선이면 손목에 힘이 들어간 것.",
                   "GIORNO — 힘 빼고 누르기 (IIDX)", "https://giorno-iidx.hatenablog.com/entry/2023/10/18/201931"),
    "iidx_yoko": ("동시치기는 노트를 하나씩 보지 말고 가로 모양으로 한 번에 읽는다(横認識). 시야를 넓게 쓰면 모양과 타이밍 인식이 함께 "
                  "안정되고, 시선이 한 점에 몰리면 반사적으로 FAST 쪽으로 쏠린다.",
                  "高空なの — 보면을 제대로 인식하기 (IIDX)", "https://note.com/nano_takazora/n/n15891405170b"),
    "bms_uya": ("가장 효과가 컸던 연습은 패턴(테마)별로 나눠 하기. 빠듯한 곡을 이지로 밀어붙이면 노트를 대충 보는 버릇이 생겨서, "
                "쉬운 곡 스코어(AAA) 노리기로 바로잡았다. 판정을 좁혀서 치는 연습도 효과가 컸다. 팔꿈치를 너무 내리면 건초염 위험.",
                "うや — BMS 의미 있었던 연습·없었던 연습 (발광)", "https://note.com/uya_otoge/n/ncfe62e8acf18"),
    "bms_elma": ("2~3일 치고 하루 쉬기, 하루 1.5~2시간. 워밍업으로 단계적으로 올리고, 빠른 16분이 또렷이 쳐지는지·강한 연타가 들어가는지·"
                 "배속이 어색하지 않은지 확인한 뒤 어려운 곡으로. 같은 곡은 하루 2번까지. 녹화로 약점을 찾고, 손에 이상이 오면 바로 멈춘다.",
                 "elma — 발광개전 합격까지 연습 (BMS)", "https://note.com/exelma/n/nb567f3a746c4"),
    "bms_hstk": ("약한 패턴은 될 때까지 따로 했고, 같은 곡만 반복하기보다 여러 패턴을 겪는 쪽이 효과가 컸다. 칠 수 있는 경계를 찾아 "
                 "조금씩 올린다. 배속을 조금 낮춰 오래 쓰니 보는 힘과 스코어가 좋아졌다.",
                 "はしたか — 발광개전까지 한 것 (BMS)", "https://hstk-mega.hatenablog.com/entry/2020/07/04/063712"),
    "bms_jose": ("지력은 여러 축(긴 발광, 짧은 연타, 강하게 누르는 연타, 고속 등)으로 나뉘고 축마다 치는 법이 달라서 한 번에 한 축씩 연습한다. "
                 "짧은 연타는 손가락 힘을 빼고 팔뚝·스냅으로. 같은 장르만 매일 하면 근육이 지쳐 다치기 쉬우니 날마다 나눈다.",
                 "BMS 연습 방침 지론 (발광개전)", "http://jose.gg-blog.com/%E9%9F%B3%E3%82%B2%E3%83%BC%E6%8C%81%E8%AB%96/bms%E7%B7%B4%E7%BF%92%E6%96%B9%E9%87%9D%E6%8C%81%E8%AB%96"),
    "bms_djamn": ("늘 같은 곡들로 워밍업을 고정하니 실력이 빨리 늘었다. 워밍업에 약한 장르를 넣으면 그대로 연습이 된다. "
                  "강하게 누르는 연타는 하루만 쉬어도 떨어진다.",
                  "DJ DJAMN — 2년 만에 Overjoy 합격한 연습법 (BMS)", "https://note.com/bms_amn/n/n65ec4c7f30b3"),
    "bms_tight": ("연습 모드에서 판정을 좁혀(타이밍을 엄격하게) 치는 것을 꾸준히 하면 타이밍이 좋아진다. 어려운 구간만 골라 반복할 수도 있다.",
                  "beatoraja 영어 가이드 (wcko87) — 연습 모드", "https://github-wiki-see.page/m/wcko87/beatoraja-english-guide/wiki/Practice-Mode"),
    "bms_dmrv": ("BMS는 원하는 패턴 경향을 골라 연습할 수 있고, 4·5·6키 난이도표는 디맥으로 실력을 옮기기 좋다. "
                 "높은(좁은) 판정으로 연습하면 디맥으로 돌아왔을 때 판정이 안정된다.",
                 "디맥 리스펙트V 갤 — 실력을 늘리고 싶은 사람들을 위한 BMS 가이드 (2025)",
                 "https://gall.dcinside.com/mgallery/board/view/?id=djmaxrespect&no=978593"),
    "lr2_setup": ("수직동기(VSync)는 끄고, 오디오 버퍼는 소리가 깨지지 않는 선에서 최대한 낮게. 판정 자동 조정 값에 그대로 맞추지 말고 "
                  "내가 정확히 쳤을 때 빛나는 위치를 찾는다.",
                  "Tiny Diary — LR2에서 높은 점수가 나오는 설정", "https://sonchou.hatenadiary.org/entry/20090428/1240845132"),
    "gaze": ("리듬게임 숙련자는 화면 한 지점에 시선을 고정해 두고 친다.",
             "小原·藤波 (2010) 시선 추적으로 본 리듬게임 숙련", "https://www.ai-gakkai.or.jp/jsai2010/webprogram/2010/pdf/164.pdf"),
    "ur": ("oh!의 UR은 판정 오차의 표준편차×10. '일관성'과 '정확도(치우침)'는 다른 것.",
           "oh! wiki — Unstable rate", "https://osu.ppy.sh/wiki/en/Gameplay/Unstable_rate"),
    "oh_acc": ("정확도 연습은 이미 잘 치는 쉬운 곡·BPM이 일정한 곡에서. 체력 구간 정확도와 기술 구간 정확도는 따로 연습해야 한다. "
                "치우침은 판정 바(오차 막대)로 확인.",
                "oh!man! 정확도 가이드 (ERA Zenith 외)", "https://osu.ppy.sh/community/forums/posts/8726685"),
    "oh_4k": ("세게 친다고 빨라지지 않는다(작동점 근처에서 가볍게). 연타(잭)는 적당한 긴장으로 움직임 거리를 조절. "
               "너무 어려운 곡만 하면 나쁜 습관이 굳는다. 약한 기술부터. 손목이 타는 느낌이면 멈추고 스트레칭, 저리면 중단. "
               "한 번에 4시간 넘게 하지 말고 1~2시간 집중 후 쉬기.",
               "oh!man! 4K 심화 가이드", "https://osu.ppy.sh/community/forums/posts/8681787"),
    "etterna": ("설정(배속·영점)은 일정하게 유지하고 특정 패턴 때문에 바꾸지 말 것. 영점은 편한 곡에서 계속 남는 치우침일 때만. "
                "시선 고정점. 손 동작을 일일이 생각하지 말 것. 단기 실력은 들쭉날쭉하지만 장기적으로는 오른다.",
                "Improving in Etterna", "https://overthefraildream.net/writings/etterna-improvement"),
    "bandori": ("실력보다 어려운 곡에서 틀리지 않으려고 서두르면 대부분 '얼리'가 뜬다. 이럴 땐 싱크를 건드리면 안 된다. "
                "쉬운 채보로 판정 감을 몸에 익히고, 고레벨 꼴박 대신 한 단계 낮은 곡들을 고르게.",
                "방도리 갤 판정 향상 가이드", "https://gall.dcinside.com/m/bang_dream/3145476"),
    "sdvx_early": ("'일찍 치는 것'과 '빠르게 치는 것'은 다르다. 판정선·속도를 함부로 바꾸지 말고 여유 있는 난이도에서 "
                   "정확한 타점 연습을.",
                   "사볼 갤 판정 관련 글", "https://gall.dcinside.com/sdvx/970851"),
    "sdvx_phys": ("특정 손가락 순서(예: 약지-엄지-검지)를 여러 속도로 '일정한 간격·정확한 순서'로 치는 연습. "
                  "덜 쓰는 손(특히 왼손)은 따로 연습. 전체 얼리/레이트보다 '특정 패턴에서 터지는지'를 볼 것.",
                  "사볼 갤 리듬 실력 늘리는 방법", "https://gall.dcinside.com/sdvx/1073136"),
    "react_read": ("노트를 '읽지' 않고 '반응'해서 치면, 쉬운 곡에선 일찍, 빽빽한 곡에선 늦게 치게 된다.",
                   "oh! 포럼 (Lights)", "https://osu.ppy.sh/community/forums/posts/7309922"),
    "rush": ("어려운 구간에서 빨라지는 건 '패닉 반응'. 빠른 구간일수록 머릿속으로는 '느리게, 여유 있게'를 떠올리고, "
             "느린 템포에서 메트로놈으로 연습.",
             "Rushing for musicians (Cellofun)", "https://cellofun.eu/home/cello-blog-encyclopaedia/mechanics-technique/basic-principles-of-technique/fast-and-tricky-walk-jog-or-sprint/rushing/"),
    "dc_basic": ("정확도 97%(S) 언저리가 나오는 곡이 연습하기 좋은 곡. 매일 여러 곡을 돌리고, 정확도가 처지는 곡은 다시 친다.",
                 "디맥 갤 — 뉴비를 위한 실력 상승 정석", "https://gall.dcinside.com/ez2djmax/179123"),
    "dc_lowlv": ("판정 연습은 자기 레벨보다 1~2단계 낮은 곡으로. 음악에서 악기 소리를 따로 들어 박자감을 기르고, 약한 패턴을 따로 연습.",
                 "디맥 갤 — 판정 연습 관련 글", "https://gall.dcinside.com/ez2djmax/94752"),
    "dc_skill2": ("연타(잭)는 손가락이 '돌아오는 시간'이 문제라 손목을 살짝 들고 떨듯이. 계단은 대부분 박자감 문제라 박 단위로 나눠 볼 것. "
                  "롱노트를 잡고 있으면 다른 손가락이 둔해진다. 한 곡 반복(꼴박)은 빨리 늘지만 나쁜 습관도 굳으니 여러 곡 돌리기와 섞을 것.",
                  "디맥 갤 — 실력 늘리는 법 2", "https://gall.dcinside.com/ez2djmax/179897"),
    "dc_finger": ("손가락을 살짝 세워서 칠 것(눕히면 체력·판정 모두 손해, 관절에도 부담). 손등 힘은 빼되, 너무 빼면 늦게 눌린다.",
                  "디맥 리스펙트V 갤 — 처리력 관련 글", "https://gall.dcinside.com/m/djmaxrespect/924077"),
    "dc_setup": ("VSync·G-Sync·FreeSync 끄기, 프레임 제한 해제, Reflex 켜기, 오디오 버퍼 낮게. 판정·출력 타이밍은 함부로 건드리지 말 것. "
                 "정확도 90% 초반이 나오는 곡은 판정 연습엔 너무 어렵고, 95% 이상 나오는 곡을 고를 것.",
                 "디맥 리스펙트V 갤 — 닥터디붕 가이드", "https://gall.dcinside.com/m/djmaxrespect/1220094"),
    "dc_speed": ("배속이 너무 느리면 일찍 치기 쉽다(3~4.5 정도 권장). 판정 연습은 95~99%가 나오는 곡에서. 판정선은 건드리지 말 것.",
                 "디맥 리스펙트V 갤 — 판정 연습 관련 글", "https://gall.dcinside.com/m/djmaxrespect/824226"),
    "dc_rotate": ("높은 레벨과 낮은 레벨을 번갈아 치고, 여러 곡 돌리기에 조금 어려운 곡을 섞는다는 의견과, 못 깨는 곡을 파야 처리력이 "
                  "올라 판정도 따라온다는 반대 의견이 함께 있다. 공통점: 어느 한쪽만 하지 말 것.",
                  "디맥 리스펙트V 갤 — 실력 향상 글 두 편", "https://gall.dcinside.com/m/djmaxrespect/106614"),
    "sdvx_hold": ("롱노트를 잡고 트릴을 칠 때 롱노트 잡은 손가락에 힘이 너무 들어가 다른 손가락이 막히는 경우가 많다.",
                  "사볼 갤 — 롱잡트릴 관련 글", "https://gall.dcinside.com/sdvx/1319107"),
    "ez2on_stac": ("쉬운 노트를 자꾸 놓치면 가장 쉬운 4K 난이도로 리듬을 들으며 치고, 키를 짧게 끊어 치는(스타카토) 연습을 할 것.",
                   "이지투온 갤 — 쉬운 노트 실수 관련 글", "https://gall.dcinside.com/m/ez2on/115627"),
    "arca_sound": ("적정 난이도 곡에서 자기 키음을 들어 보고, 박자에 맞춰 치는지 그냥 '노트를 받아먹고' 있는지 확인할 것.",
                   "아카라이브 리듬게임 채널", "https://arca.live/b/mgamm/73315224"),
    "arca_hum": ("소리를 끄고 흥얼거리며 친 걸 녹음해 들어 보면 어디서 박자가 틀어지는지 알 수 있다(오답노트식 점검).",
                 "아카라이브 프로젝트 세카이 채널", "https://arca.live/b/projectsekal/51194195"),
    "arca_early": ("얼리가 많다고 판정선을 계속 올리지 말 것. 얼리는 처리가 버겁거나 뭉개 칠 때 나온다. 음악에 맞춰 칠 것.",
                   "아카라이브 리듬게임 채널", "https://arca.live/b/mgamm/78389105"),
    "arca_gaze": ("노트가 0.4~0.6초 정도 보이는 배속이 무난하고, 시선 고정이 가장 중요한 기본기. 패턴을 눈으로 쫓아가면 실수가 연쇄로 터진다.",
                  "아카라이브 리듬게임·프로젝트 세카이 채널", "https://arca.live/b/mgamm/31156296"),
    "lane_cover": ("배속을 올리고 가리개(레인 커버)를 쓰면 시선이 판정선 근처에 고정된다.",
                   "방도리 갤 — 가리개 관련 글", "https://gall.dcinside.com/m/bang_dream/4014201"),
    "health": ("손·손목이 아프면 참고 계속하지 말 것. 1~2시간마다 자리에서 일어나 몇 분 움직이고, 플레이 전 손·손목 준비운동.",
               "Cleveland Clinic / Mayo Clinic e스포츠 건강 가이드", "https://health.clevelandclinic.org/hand-stretches-for-gamers/"),
    "kbd": ("기계식 스위치 접점이 튀면 한 번 누른 게 두 번 들어간다(채터). 키보드 문제이므로 연습으로 고칠 수 없다.",
            "oh! 포럼 / 디바운스 설명", "https://osu.ppy.sh/community/forums/posts/4342419"),
}

# ------------------------------------------------------------------ 패턴 이름
TAG_NAME = {
    "single": "느린 단타",
    "stream": "빠른 연속 단타",
    "chord2": "2개 동치",
    "chord3": "3개 이상 동치",
    "jack": "같은 레인 연타",
    "trill": "트릴",
    "stair": "계단",
    "hold": "롱노트 잡은 채 치기",
    "long": "롱노트 시작",
    "side": "사이드",
    "trigger": "트리거",
    "burst": "밀도 높은 구간",
    "calm": "밀도 낮은 구간",
    "bpm": "변속 직후 3초",
}
TAG_ORDER = ["single", "stream", "chord2", "chord3", "jack", "trill", "stair", "hold", "long", "side",
             "trigger", "burst", "calm", "bpm"]

KEY_NAME = {"1E": "A", "1F": "S", "20": "D", "21": "F", "22": "G", "23": "H", "24": "J", "25": "K", "26": "L",
            "27": ";", "2C": "Z", "2D": "X", "2E": "C", "2F": "V", "30": "B", "31": "N", "32": "M", "39": "Space",
            "3A": "CapsLock", "2A": "LShift", "36": "RShift", "1D": "LCtrl", "1De": "RCtrl", "38": "LAlt",
            "38e": "RAlt", "52": "Num0", "4F": "Num1", "50": "Num2", "51": "Num3", "4B": "Num4", "4C": "Num5",
            "4D": "Num6", "47": "Num7", "48": "Num8", "49": "Num9", "4E": "Num+", "4A": "Num-", "37": "Num*",
            "35e": "Num/", "1Ce": "NumEnter", "1C": "Enter", "4De": "→", "4Be": "←", "48e": "↑", "50e": "↓",
            "10": "Q", "11": "W", "12": "E", "13": "R", "14": "T", "15": "Y", "16": "U", "17": "I", "18": "O",
            "19": "P", "53": "Num.", "0F": "Tab"}

LANES_OF = {"4B": 4, "5B": 5, "6B": 6, "8B": 6}
CHORD_TOL = 8.0       # 이 ms 안에 도달한 노트는 동시치기
FAST_GAP = 180.0      # 연타·트릴·계단으로 보는 최대 간격 (ms)
STREAM_GAP = 150.0
SLOW_GAP = 250.0
MIN_N = 20            # 패턴 비교에 필요한 최소 노트 수
# 박자 기준 (곡 BPM을 추정해 '몇 분음표 간격인지'로 패턴을 나눔. 격자가 안 맞는 구간만 위 ms 기준으로)
STREAM_BEAT = 0.36    # 앞 노트와 1/3박(셋잇단 8분) 이하 → 연속 단타 (16분·24분·32분 포함)
SLOW_BEAT = 0.70      # 3/4박(점8분) 이상 → 드문 노트
FAST_BEAT = 0.27      # 16분(1/4박) 이하이거나
FAST_MS = 145.0       # 손이 실제로 이 ms 안에 다시 눌러야 하면(BPM이 높은 곡의 8분 등) → 연타·트릴·계단
GRID_MIN_SCORE = 0.85  # 이 이상 격자에 맞는 구간만 박자 기준 사용
GRID_TOL_MS = 4.0      # 격자에 '맞는다'고 보는 오차 (기록 시각은 박자 위치와 표준편차 1 ms 안팎)
RESCUE_KEY_MS = 80.0   # 곡이 끝난 뒤: 확신도가 낮아 버려진 노트 후보 중, 이 안에 같은 레인 키를 누른 것은 노트로 살림


def josa(word, pair):
    """받침에 맞는 조사: josa('트릴', '을를') → '트릴을'"""
    ch = word.rstrip(" )")[-1:] if word else ""
    if ch and not ("가" <= ch <= "힣"):
        # 숫자·영문 끝: 읽는 소리의 받침으로 (5 = 오, 6 = 육, L = 엘 …)
        has = ch in "013678" or ch.upper() in "LMNR"
        if pair == "으로로":
            return word + ("으로" if has and ch.upper() != "L" and ch not in "178" else "로")
        return word + (pair[0] if has else pair[1])
    if "가" <= ch <= "힣":
        has = (ord(ch) - 0xAC00) % 28 != 0
        if pair == "으로로":
            return word + ("으로" if has and (ord(ch) - 0xAC00) % 28 != 8 else "로")
        return word + (pair[0] if has else pair[1])
    return word + pair[0] if pair != "으로로" else word + "으로"


def key_name(k):
    try:
        from .keys_win import KEY_NAMES          # 앱의 키 이름 표가 있으면 그것과 같게
        if str(k) in KEY_NAMES:
            return KEY_NAMES[str(k)]
    except Exception:
        pass
    return KEY_NAME.get(str(k), str(k))


def lane_label(mode, ch, cfg):
    names = {"SL": "왼쪽 사이드", "SR": "오른쪽 사이드", "TL": "왼쪽 트리거", "TR": "오른쪽 트리거"}
    base = names.get(ch, ("레인 %s" % ch[1:]) if ch.startswith("L") else ch)
    keys = ((cfg or {}).get("modes", {}).get(mode, {}).get("keymap", {}) or {}).get(ch)
    if isinstance(keys, str):
        keys = [keys]
    if keys:
        base += " (%s)" % "/".join(key_name(k) for k in keys)
    return base


def hand_of(mode, ch, cfg):
    g = ((cfg or {}).get("modes", {}).get(mode, {}).get("groups", {}) or {}).get(ch, "")
    if g:
        return g
    if ch in ("SL", "SR"):
        return "사이드"
    if ch in ("TL", "TR"):
        return "트리거"
    try:
        n = LANES_OF[mode]
        i = int(ch[1:])
        if n % 2 == 1 and i == (n + 1) // 2:
            return "가운데"
        return "왼손" if i <= n / 2.0 else "오른손"
    except (ValueError, KeyError):
        return "기타"


# ------------------------------------------------------------------ 원본 기록에서 곡 하나 꺼내기
def _f(x):
    return float(x)


def collect(raw, start, end, lanes, live=False, low=None):
    """진단 기록(세션 raw 튜플 또는 _raw.csv 행)에서 [start, end] 구간의 노트 도달·키·변속을 꺼낸다.
    live=True면 끝에서부터 거꾸로 보다가 구간보다 충분히 앞이면 멈춘다 (40만 줄 전체를 보지 않게)."""
    arrs, keys, vch = [], [], []
    lo, hi = start - 2000.0, end + 2000.0
    it = reversed(raw) if live else raw
    for r in it:
        try:
            if len(r) < 3:
                continue
            k = r[1]
            if live:
                stamp = _f(r[0])
                if stamp < lo - 8000.0:
                    break
            if k == "arr":
                t = _f(r[4])
                if lo <= t <= hi and int(_f(r[2])) == lanes:
                    cf = _f(r[5])
                    if cf >= 1.0:
                        arrs.append((t, str(r[3]), str(r[6]) in ("1", "True")))
                    elif cf > 0 and low is not None:
                        low.append((t, str(r[3]), str(r[6]) in ("1", "True")))
            elif k == "key":
                t = _f(r[4])
                if lo <= t <= hi:
                    keys.append((t, str(r[2]), str(r[3]) in ("1", "True")))
            elif k == "vchg":
                t = _f(r[3])
                if lo <= t <= hi and int(_f(r[2])) == lanes:
                    vch.append(t)
        except (ValueError, IndexError, TypeError):
            continue
    arrs.sort()
    keys.sort()
    vch.sort()
    return arrs, keys, vch


# ------------------------------------------------------------------ 곡이 끝난 뒤 정밀화 (실시간 측정·인디케이터와 무관)
def rescue_arrs(arrs, low, keys, hits, cfg, mode):
    """확신도가 낮아 버려진 도달 후보(low: [(시각, 채널, 롱노트?)]) 중, 그 레인 키를 실제로 누른 것만 노트로 되살린다.
    (실제 채보 18판 대조: 놓친 노트 420 → 31, 가짜 +23)
    반환: (arrs, hits, 살린 수). 살린 노트의 오차는 가장 가까운 '아직 안 쓴' 키 누름으로 잰다."""
    if not low:
        return arrs, hits, 0
    km = ((cfg or {}).get("modes", {}).get(mode, {}).get("keymap", {})) or {}
    k2c = {}
    for ch, ks in km.items():
        for k in ([ks] if isinstance(ks, str) else ks):
            k2c[str(k).upper()] = ch
    downs = defaultdict(list)
    for t, k, dn in keys:
        ch = k2c.get(str(k).upper())
        if dn and ch:
            downs[ch].append(t)
    used = defaultdict(list)                     # 이미 노트와 짝지어진 누름
    for (ch, t2), es in hits.items():
        used[ch].append(t2 + es)
    conf = defaultdict(list)
    for t, ch, _ in arrs:
        conf[ch].append(t)
    for d in (downs, used, conf):
        for ch in d:
            d[ch].sort()
    import bisect

    def near(lst, t, w):
        i = bisect.bisect_left(lst, t - w)
        return i < len(lst) and lst[i] <= t + w

    out = list(arrs)
    hits = dict(hits)
    lastc = {}
    n = 0
    for t, ch, lg in sorted(low):
        if near(conf.get(ch, []), t, 40.0) or (ch in lastc and t - lastc[ch] < 30.0):
            continue
        lastc[ch] = t
        cand = [d for d in downs.get(ch, []) if abs(d - t) <= RESCUE_KEY_MS and not near(used.get(ch, []), d, 1.0)]
        if not cand:
            continue
        d = min(cand, key=lambda v: abs(v - t))
        out.append((t, ch, lg))
        hits[(ch, round(t, 2))] = d - t
        used[ch].append(d)
        used[ch].sort()
        n += 1
    out.sort()
    return out, hits, n


def _grid_fit(gaps, bpms, with_dev=False):
    """간격들이 이 BPM의 16분·24분·32분 격자에 얼마나 맞는지 (점수, 복잡도[, 평균 어긋남])"""
    tick = (60000.0 / bpms / 48.0)[:, None]        # 한 박 = 48칸 (16분=12, 24분=8, 32분=6)
    r = gaps[None, :] / tick
    d6 = np.abs(r - 6.0 * np.round(r / 6.0))
    d8 = np.abs(r - 8.0 * np.round(r / 8.0))
    d12 = np.abs(r - 12.0 * np.round(r / 12.0)) * tick
    dev = np.minimum(d6, d8) * tick
    raw = (dev <= GRID_TOL_MS).mean(axis=1)        # 격자에서 몇 ms 안에 떨어지는 간격 비율
    # BPM이 높을수록 격자가 촘촘해 우연히 맞는 비율도 커지므로 그만큼 빼고 본다 (한 박 24칸 중 6자리 × 허용 폭)
    chance = np.minimum(0.95, GRID_TOL_MS / (2.0 * tick[:, 0]))
    score = (raw - chance) / (1.0 - chance)
    cx = (d12 > GRID_TOL_MS).mean(axis=1)          # 16분 격자 밖(셋잇단·32분) 비율
    if with_dev:
        return score, cx, np.minimum(dev, GRID_TOL_MS).mean(axis=1)
    return score, cx


def beat_grid(times, win=8000.0, step=4000.0):
    """곡 안 시간 구간별 BPM 추정 → [(구간 시작, 끝, BPM, 맞는 정도)].
    변속곡도 구간마다 따로 잡는다. 같은 패턴이 BPM과 상관없이 같은 이름(몇 분음표)으로 묶이게 하는 데 쓴다."""
    t = np.asarray(sorted(times), dtype=float)
    if len(t) < 12:
        return []
    bpms = np.arange(60.0, 400.0, 0.1)        # (맞는 범위를 찾은 뒤 0.02 간격으로 다시 다듬음)
    out = []
    a = t[0]
    while a < t[-1]:
        sel = t[(t >= a) & (t < a + win)]
        if len(sel) >= 8:
            g = np.diff(sel)
            g = g[(g > 25.0) & (g < 2000.0)]
            if len(g) >= 6:
                sc, cx = _grid_fit(g, bpms)
                best = float(sc.max())
                ok = np.where(sc >= best - 0.03)[0]
                # 같은 리듬이 BPM 절반·두 배로도 맞으므로: 디맥에서 흔한 118~300 안에서, 16분 격자로 가장 단순하게
                # 읽히는 것(셋잇단·32분이 적은 것), 비슷하면 낮은 BPM
                inr = [i for i in ok if 118.0 <= bpms[i] < 300.0] or list(ok)
                cmin = min(float(cx[i]) for i in inr)
                i = min([i for i in inr if cx[i] <= cmin + 0.05], key=lambda j: bpms[j])
                # 맞는 범위 안에서 어긋남이 가장 작은 BPM으로 다듬기
                near_ = np.arange(bpms[i] * 0.96, bpms[i] * 1.04, 0.02)
                s2, _, dv = _grid_fit(g, near_, True)
                good = s2 >= s2.max() - 0.02
                j = int(np.argmin(np.where(good, dv, 1e9)))
                out.append((float(a), float(a + win), float(near_[j]), float(s2[j]), g))
        a += step
    # 같은 리듬은 4/3배·3/2배·2배 BPM으로도 맞을 수 있고(16분 ↔ 셋잇단), 노트가 드문 구간은 엉뚱한 BPM도 우연히 맞는다.
    # 그래서 곡에서 가장 많이 잡힌 BPM부터 차례로, 그 BPM이 구간에 거의 똑같이 맞으면 그것으로 통일한다.
    # (진짜 변속 구간은 그 BPM이 안 맞으므로 그대로 남는다)
    cnt = defaultdict(int)
    for w in out:
        if w[3] >= GRID_MIN_SCORE:
            cnt[round(w[2] * 2) / 2] += 1
    common = [b for b, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:4]]
    res = []
    for a, b, bpm, sc, g in out:
        for ci, cb in enumerate(common):
            if abs(cb - bpm) < 0.6:
                break
            s3, _ = _grid_fit(g, np.array([cb], dtype=float))
            if s3[0] >= sc - (0.12 if ci == 0 else 0.05) and s3[0] >= GRID_MIN_SCORE - 0.1:
                bpm, sc = float(cb), float(s3[0])
                break
        res.append((a, b, bpm, sc))
    return res


def beat_at(grid, t):
    """시각 t의 (BPM, 맞는 정도) — 겹치는 구간 중 가장 잘 맞는 것. 없으면 (None, 0)"""
    best = (None, 0.0)
    for a, b, bpm, sc in grid:
        if a <= t < b and sc > best[1]:
            best = (bpm, sc)
    return best


NOTE_VALUE = [(1 / 8, "32분"), (1 / 6, "24분"), (1 / 4, "16분"), (1 / 3, "셋잇단 8분"), (1 / 2, "8분"),
              (2 / 3, "셋잇단 4분"), (3 / 4, "점8분"), (1.0, "4분"), (2.0, "2분")]


def note_value(beats):
    """박 단위 간격 → '16분' 같은 이름 (가장 가까운 것, 4% 안쪽만)"""
    if beats is None:
        return None
    v, nm = min(NOTE_VALUE, key=lambda x: abs(math.log(beats / x[0])))
    return nm if abs(beats / v - 1.0) <= 0.04 else None


# ------------------------------------------------------------------ 통계 도우미
def _stats(e, miss=0):
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return {"n": 0, "miss": int(miss), "mean": 0.0, "sd": 0.0, "abs": 0.0, "out": 0.0}
    # 흔들림 = (84백분위 - 16백분위) / 2 — 앱의 다른 곳과 같은 정의. 몇 개 튄 노트에 덜 휘둘린다
    if n >= 10:
        p16, p84 = np.percentile(e, [16, 84])
        sd = float((p84 - p16) / 2.0)
    else:
        sd = float(e.std(ddof=1)) if n > 1 else 0.0
    return {"n": int(n), "miss": int(miss), "mean": float(e.mean()), "sd": sd,
            "abs": float(np.abs(e).mean()), "out": 0.0}


def _cmp(a, b):
    """a 그룹이 기준 b와 비교해 평균이 얼마나 다른지 (차이, z)"""
    if a["n"] < 2 or b["n"] < 2:
        return 0.0, 0.0
    d = a["mean"] - b["mean"]
    se = math.sqrt(a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"]) or 1e-9
    return d, d / se


def _expect_spread(sizes, sd):
    """노트마다 따로(독립적으로) 흔들릴 때 동치 안 '가장 이른 것~가장 늦은 것' 차이의 중앙값 (모의실험)"""
    if not sizes or sd <= 0:
        return 0.0
    rng = np.random.default_rng(0)
    ks = rng.choice(np.array(sizes), 3000)
    x = rng.normal(0, sd, (3000, int(max(sizes))))
    r = [float(np.ptp(x[i, :k])) for i, k in enumerate(ks)]
    return float(np.median(r))


def _robust_mean(v):
    """(평균, 개수, 흔들림) — 크게 튄 값은 가장자리로 눌러서"""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else 0.0, len(v), 0.0)
    md = float(np.median(v))
    p16, p84 = np.percentile(v, [16, 84])
    r = max(2.0, float(p84 - p16) / 2.0)
    w = np.clip(v, md - 3 * r, md + 3 * r)
    return (float(w.mean()), len(v), float(w.std(ddof=1)))


def fmt(v):
    return "%+.1f" % v


# ------------------------------------------------------------------ 분석
DWELL_LONG = 0.5     # 다음 노트(같은 레인)까지 간격의 이 비율 넘게 누르고 있으면 '오래 누름'
DWELL_SHORT = 0.35   # 이보다 짧으면 '짧게 뗌'
DWELL_BINS = ((60.0, 150.0), (150.0, 250.0), (250.0, 400.0))   # 간격이 비슷한 것끼리 비교


def _dwell_stats(pairs):
    """오래 누른 뒤 다음 노트 오차 − 짧게 뗀 뒤 다음 노트 오차 (간격 구간별로 나눠 가중 평균, 크게 튄 값은 눌러서)"""
    num = den = var = 0.0
    nl = ns = 0
    longs, shorts = [], []
    for lo, hi in DWELL_BINS:
        L = [b["e"] for g, r, b in pairs if lo <= g < hi and r >= DWELL_LONG]
        S = [b["e"] for g, r, b in pairs if lo <= g < hi and r < DWELL_SHORT]
        if len(L) >= 6 and len(S) >= 6:
            L_, S_ = np.clip(L, -60, 60), np.clip(S, -60, 60)
            w = 1.0 / (1.0 / len(L_) + 1.0 / len(S_))
            num += w * (float(np.mean(L_)) - float(np.mean(S_)))
            den += w
            var += w * w * (float(np.var(L_)) / len(L_) + float(np.var(S_)) / len(S_))
            nl += len(L_)
            ns += len(S_)
            longs += list(L_)
            shorts += list(S_)
    if den <= 0:
        return None
    return {"d": num / den, "se": (var ** 0.5) / den or 1e-9, "n_long": nl, "n_short": ns,
            "long_mean": float(np.mean(longs)), "short_mean": float(np.mean(shorts))}


def analyze(arrs, hits, keys, vch, cfg, mode, js_calib, win=None, grid=None):
    """arrs: [(도달 시각, 채널, 롱노트?)] (신뢰도 1 이상, 이 곡 레인 수)
    hits: {(채널, round(도달,2)): 화면 기준 오차}  (짝지어진 것)
    keys: [(시각, 키, 누름?)], vch: [변속 시각]
    js_calib: 판정 타이밍 + 영점 보정 (판정 기준 오차 = 화면 기준 + 이 값)
    반환: 보고서 dict (조언 만들기의 입력)"""
    win = float(win if win is not None else (cfg or {}).get("window_ms", 125.0 / 3.0))
    km = ((cfg or {}).get("modes", {}).get(mode, {}).get("keymap", {})) or {}
    key2ch = {}
    for ch, ks in km.items():
        for k in ([ks] if isinstance(ks, str) else ks):
            key2ch[str(k)] = ch

    # 노트 목록
    N = []
    for t, ch, lg in arrs:
        es = hits.get((ch, round(t, 2)))
        N.append({"t": t, "ch": ch, "long": lg, "hit": es is not None,
                  "e": (es + js_calib) if es is not None else None,
                  "press": (t + es) if es is not None else None,
                  "hand": hand_of(mode, ch, cfg), "tags": set()})
    if len(N) < 30:
        return None
    nl = LANES_OF.get(mode, 6)

    # 동시치기 묶음
    groups, cur = [], [N[0]]
    for nt in N[1:]:
        if nt["t"] - cur[0]["t"] <= CHORD_TOL:
            cur.append(nt)
        else:
            groups.append(cur)
            cur = [nt]
    groups.append(cur)
    gt = [g[0]["t"] for g in groups]
    if grid is None:                 # (시험용으로 정답 격자를 넣을 수 있게)
        grid = beat_grid(gt)

    def beats(gap, t):
        bpm, sc = beat_at(grid, t)
        if bpm is None or sc < GRID_MIN_SCORE or gap >= 1e8:
            return None
        return gap / (60000.0 / bpm)

    def is_fast(gap, t):
        b = beats(gap, t)
        return (b <= FAST_BEAT or gap <= FAST_MS) if b is not None else gap <= FAST_GAP
    for gi, g in enumerate(groups):
        main = [x for x in g if x["ch"].startswith("L")]
        gp = gt[gi] - gt[gi - 1] if gi > 0 else 1e9
        gn = gt[gi + 1] - gt[gi] if gi + 1 < len(groups) else 1e9
        for x in g:
            x["gi"], x["gsize"], x["gap_p"], x["gap_n"] = gi, len(main), gp, gn
            x["beat_p"] = beats(gp, gt[gi])
            if x["ch"] in ("SL", "SR"):
                x["tags"].add("side")
            elif x["ch"] in ("TL", "TR"):
                x["tags"].add("trigger")
            if x["long"]:
                x["tags"].add("long")
            if len(main) == 2 and x["ch"].startswith("L"):
                x["tags"].add("chord2")
            elif len(main) >= 3 and x["ch"].startswith("L"):
                x["tags"].add("chord3")

    # 같은 레인 연타 (+ 연타 사슬 안 순서)
    last = {}
    for x in N:
        p = last.get(x["ch"])
        x["jack_pos"] = 0
        if p is not None and is_fast(x["t"] - p["t"], x["t"]) and x["ch"].startswith("L"):
            x["tags"].add("jack")          # 연타는 '다시 누르는' 노트만 (첫 노트는 들어가는 노트라 따로 봄)
            p["jack_pos"] = max(p["jack_pos"], 1)
            x["jack_pos"] = p["jack_pos"] + 1
            x["jack_gap"] = x["t"] - p["t"]
            x["jack_nv"] = note_value(beats(x["t"] - p["t"], x["t"]))
        last[x["ch"]] = x

    # 단일 레인 묶음의 흐름에서 트릴·계단 찾기
    singles = [(gi, g[0]) for gi, g in enumerate(groups) if len(g) == 1 and g[0]["ch"].startswith("L")]

    def runs(pred, minlen):
        out, cur = [], []
        for k in range(len(singles)):
            if cur and pred(cur, singles[k]):
                cur.append(singles[k])
            else:
                if len(cur) >= minlen:
                    out.append(cur)
                cur = [singles[k]]
        if len(cur) >= minlen:
            out.append(cur)
        return out

    def lane(x):
        return int(x["ch"][1:])

    def trill_pred(cur, nx):
        gi, x = nx
        pgi, p = cur[-1]
        if gi != pgi + 1 or not is_fast(x["t"] - p["t"], x["t"]) or x["ch"] == p["ch"]:
            return False
        if len(cur) >= 2:
            return x["ch"] == cur[-2][1]["ch"]
        return True

    trills = runs(trill_pred, 4)
    for r in trills:
        for _, x in r:
            x["tags"].add("trill")

    def stair_pred(cur, nx):
        gi, x = nx
        pgi, p = cur[-1]
        if gi != pgi + 1 or not is_fast(x["t"] - p["t"], x["t"]):
            return False
        d = lane(x) - lane(p)
        if abs(d) != 1:
            return False
        if len(cur) >= 2:
            return d == lane(p) - lane(cur[-2][1])
        return True

    stairs = runs(stair_pred, 3)
    for r in stairs:
        d = lane(r[1][1]) - lane(r[0][1])
        for _, x in r:
            x["tags"].add("stair")
            x["stair_dir"] = d

    # 롱노트 잡고 있는 동안 다른 레인 치기
    downs = defaultdict(list)
    ups = defaultdict(list)
    for t, k, dn in keys:
        ch = key2ch.get(k)
        if ch is None:
            continue
        (downs if dn else ups)[ch].append(t)
    holds = []
    for x in N:
        if x["long"] and x["hit"]:
            u = [t for t in ups.get(x["ch"], []) if t > x["press"] + 20]
            if u and u[0] - x["press"] > 120:
                holds.append((x["press"] + 30.0, u[0] - 30.0, x["ch"], x["hand"]))
    if holds:
        for x in N:
            for a, b, ch, hd in holds:
                if a <= x["t"] <= b and x["ch"] != ch:
                    x["tags"].add("hold")
                    x["hold_same_hand"] = (hd == x["hand"])
                    break

    # 키를 누르고 있던 시간(누름→뗌). 같은 레인에 다음 노트가 빨리 올 때 오래 누르고 있으면 다음 누름이 밀리는지
    # (IIDX 연타 공략 '누르는 힘보다 빨리 떼기', oh! 포럼 '필요 이상 오래 누르면 스트림이 밀림'). 롱노트는 뺌.
    import bisect as _bs
    ups_s = {ch: sorted(v) for ch, v in ups.items()}
    for x in N:
        if x["hit"] and not x["long"] and x["ch"].startswith("L"):
            u = ups_s.get(x["ch"], [])
            i = _bs.bisect_left(u, x["press"] - 5.0)
            if i < len(u) and 0.0 <= u[i] - x["press"] < 1000.0:
                x["dwell"] = u[i] - x["press"]
    bylane = defaultdict(list)
    for x in N:
        if x["ch"].startswith("L"):
            bylane[x["ch"]].append(x)
    dw_pairs = []        # (다음 노트까지 간격, 누름 비율, 다음 노트)
    for ch, xs in bylane.items():
        for a, b in zip(xs, xs[1:]):
            g = b["t"] - a["t"]
            if a.get("dwell") is None or not b["hit"] or b["long"] or not (60.0 <= g <= 400.0):
                continue
            dw_pairs.append((g, a["dwell"] / g, b))
            if a["dwell"] / g >= DWELL_LONG:
                b["tags"].add("after_long_press")

    ts = np.array([x["t"] for x in N])
    nps = (np.searchsorted(ts, ts + 500.0) - np.searchsorted(ts, ts - 500.0)).astype(float)
    hi_d = max(10.0, float(np.percentile(nps, 80)))
    lo_d = float(np.percentile(nps, 40))
    for x, d in zip(N, nps):
        x["nps"] = float(d)
        if d >= hi_d:
            x["tags"].add("burst")
        elif d <= lo_d:
            x["tags"].add("calm")

    # 변속 직후
    for x in N:
        if any(0 <= x["t"] - v <= 3000.0 for v in vch):
            x["tags"].add("bpm")

    # 연속 단타 / 느린 단타(기준)
    special = {"jack", "trill", "stair", "hold", "side", "trigger", "chord2", "chord3", "long"}
    for x in N:
        if x["tags"] & special:
            continue
        b = x.get("beat_p")
        fast = (b <= STREAM_BEAT) if b is not None else x["gap_p"] <= STREAM_GAP
        slow = (b >= SLOW_BEAT) if b is not None else x["gap_p"] >= SLOW_GAP
        if fast and x["ch"].startswith("L"):
            x["tags"].add("stream")
        elif slow and x["ch"].startswith("L") and "burst" not in x["tags"]:
            x["tags"].add("single")

    # ---- 통계
    H = [x for x in N if x["hit"]]
    E = np.array([x["e"] for x in H])
    allst = _stats(E, len(N) - len(H))
    allst["out"] = float((np.abs(E) > win).mean())
    # 비교용 오차는 크게 튄 노트(짝짓기 실수·노트 잘못 봄 등)를 가장자리로 눌러서 (평균이 몇 개에 끌려가지 않게)
    med = float(np.median(E))
    rs = max(3.0, allst["sd"])
    for x in H:
        x["e_raw"] = x["e"]
        x["e"] = float(min(med + 3 * rs, max(med - 3 * rs, x["e"])))
    base_notes = [x for x in H if "single" in x["tags"]]
    base_name = "느린 단타"
    if len(base_notes) < 40:
        base_notes = [x for x in H if not (x["tags"] & (special | {"burst"}))]
        base_name = "특별한 패턴 없는 노트"
    if len(base_notes) < 30:
        base_notes = H
        base_name = "전체 노트"
    base = _stats([x["e"] for x in base_notes],
                  sum(1 for x in N if not x["hit"] and ("single" in x["tags"] if base_name == "느린 단타" else True)))
    base["n_all"] = len(base_notes) + base["miss"]

    # 손 전체 치우침(예: 오른손이 다 늦음)이 패턴 비교에 섞이지 않게, 레인 노트는 손별 치우침을 빼고 비교
    def _is_chord(x):
        return bool(x["tags"] & {"chord2", "chord3"})
    mn = [x["e"] for x in H if x["ch"].startswith("L") and not _is_chord(x)]
    mn_mu = float(np.mean(mn)) if mn else 0.0
    hand_adj = {}
    for hd_ in ("왼손", "오른손", "가운데"):
        v = [x["e"] for x in H if x["hand"] == hd_ and x["ch"].startswith("L") and not _is_chord(x)]
        hand_adj[hd_] = (float(np.mean(v)) - mn_mu) if len(v) >= 60 else 0.0
    for x in H:
        x["ea"] = x["e"] - (hand_adj.get(x["hand"], 0.0) if x["ch"].startswith("L") else 0.0)

    def tagstat(tag):
        """이 패턴 노트 vs 이 곡의 나머지 노트 (한 곡 안에서 비교하므로 곡 전체 치우침·영점은 빠진다)"""
        xs = [x for x in N if tag in x["tags"]]
        rs = [x for x in N if tag not in x["tags"]]
        hs = [x["e"] for x in xs if x["hit"]]
        s = _stats(hs, sum(1 for x in xs if not x["hit"]))
        r = _stats([x["e"] for x in rs if x["hit"]], sum(1 for x in rs if not x["hit"]))
        s["n_all"] = len(xs)
        r["n_all"] = len(rs)
        if hs:
            s["out"] = float((np.abs(np.array(hs)) > win).mean())
        # 차이는 손 치우침을 뺀 값으로
        d, z = _cmp(_stats([x["ea"] for x in xs if x["hit"]]), _stats([x["ea"] for x in rs if x["hit"]]))
        s["d"], s["z"] = d, z
        s["rest"] = r
        return s

    tags = {t: tagstat(t) for t in TAG_ORDER}

    # 레인·손
    lanes = {}
    for ch in sorted({x["ch"] for x in N}):
        xs = [x for x in N if x["ch"] == ch]
        hs = [x["e"] for x in xs if x["hit"]]
        s = _stats(hs, sum(1 for x in xs if not x["hit"]))
        s["n_all"] = len(xs)
        s["d"], s["z"] = _cmp(s, allst)
        s["hand"] = hand_of(mode, ch, cfg)
        lanes[ch] = s
    # 동치가 아닌 노트만으로 본 레인 치우침 (동치 속 어긋남과 섞이지 않게)
    def is_chord(x):
        return bool(x["tags"] & {"chord2", "chord3"})
    nc_all = [x["e"] for x in H if not is_chord(x) and x["ch"].startswith("L")]
    nc_mu = float(np.mean(nc_all)) if nc_all else 0.0
    lane_bias = {}
    for ch in lanes:
        v = [x["e"] for x in H if x["ch"] == ch and not is_chord(x)]
        lanes[ch]["nc"] = _stats(v)
        lane_bias[ch] = (float(np.mean(v)) - nc_mu) if len(v) >= 15 else 0.0
    hands = {}
    for hd in sorted({x["hand"] for x in N}):
        xs = [x for x in N if x["hand"] == hd]
        hs = [x["e"] for x in xs if x["hit"]]
        s = _stats(hs, sum(1 for x in xs if not x["hit"]))
        s["n_all"] = len(xs)
        s["nc"] = _stats([x["e"] for x in xs if x["hit"] and not is_chord(x)])
        hands[hd] = s

    # 동치: 누른 시각 퍼짐, 레인별 늦게 누르는 경향, 양손 중 먼저 나가는 손
    spreads, lane_off, lr, sizes_ = [], defaultdict(list), [], []
    for g in groups:
        main = [x for x in g if x["ch"].startswith("L")]
        if len(main) < 2 or not all(x["hit"] for x in main):
            continue
        pr = np.array([x["press"] for x in main])
        spreads.append(float(pr.max() - pr.min()))
        sizes_.append(len(main))
        for i_, x in enumerate(main):
            # 같은 동치의 '다른' 노트들보다 얼마나 늦게/먼저 눌렀나 — 레인 자체 치우침으로 설명되는 만큼은 뺀다
            oth = [y for j_, y in enumerate(main) if j_ != i_]
            mo = np.mean([y["press"] for y in oth])
            eb = np.mean([lane_bias.get(y["ch"], 0.0) for y in oth])
            lane_off[x["ch"]].append(x["press"] - mo - (lane_bias.get(x["ch"], 0.0) - eb))
        L = [x["press"] for x in main if x["hand"] == "왼손"]
        R = [x["press"] for x in main if x["hand"] == "오른손"]
        if L and R:
            lr.append(float(np.mean(L) - np.mean(R)))
    chord = {"n": len(spreads), "spread": float(np.median(spreads)) if spreads else 0.0,
             "spread_p90": float(np.percentile(spreads, 90)) if spreads else 0.0,
             "lane_off": {ch: _robust_mean(v) for ch, v in lane_off.items()},
             "lr": (float(np.mean(lr)), len(lr), float(np.std(lr, ddof=1)) if len(lr) > 1 else 0.0) if lr else None,
             "expect": _expect_spread(sizes_, float(allst["sd"]))}   # 손가락이 서로 따로 흔들릴 때 예상 퍼짐(중앙값)

    # 트릴: 손별 / 레인별 평균
    tr_hand, tr_lane = defaultdict(list), defaultdict(list)
    for r in trills:
        cross = len({x["hand"] for _, x in r}) == 2      # 양손 트릴 (왼손 레인 ↔ 오른손 레인)
        for _, x in r:
            if x["hit"]:
                if cross:
                    tr_hand[x["hand"]].append(x["e"])
                else:
                    tr_lane[x["ch"]].append(x["e"])      # 한 손 트릴: 손가락끼리 비교
    lane_mu = {}
    for x in H:
        lane_mu.setdefault(x["ch"], []).append(x["e"])
    lane_mu = {c: float(np.mean(v)) for c, v in lane_mu.items()}
    tr_res = {c: _stats([e - lane_mu[c] for e in v]) for c, v in tr_lane.items()}
    trill = {"hand": {h: _stats(v) for h, v in tr_hand.items()},
             "lane": {c: _stats(v) for c, v in tr_lane.items()}, "lane_res": tr_res, "runs": len(trills)}

    # 연타 사슬 안 순서별
    jpos = defaultdict(list)
    for x in H:
        if x.get("jack_pos", 0) >= 1:
            jpos[min(x["jack_pos"], 4)].append(x["e"])
    jack = {k: _stats(v) for k, v in jpos.items()}

    # 계단 방향: 손 안쪽(가운데로) / 바깥쪽
    sdir = defaultdict(list)
    for x in H:
        d = x.get("stair_dir")
        if not d:
            continue
        i = int(x["ch"][1:])
        center = (nl + 1) / 2.0
        inward = (i < center and d > 0) or (i > center and d < 0)
        sdir["안쪽" if inward else "바깥쪽"].append(x["e"])
    stair = {k: _stats(v) for k, v in sdir.items()}

    # 롱노트 잡은 손과 같은 손 / 다른 손
    hd_s = defaultdict(list)
    for x in N:
        if "hold" in x["tags"]:
            hd_s["같은 손" if x.get("hold_same_hand") else "다른 손"].append(x)
    hold = {}
    rest_h = _stats([x["e"] for x in H if "hold" not in x["tags"]])
    for k, xs in hd_s.items():
        s = _stats([x["e"] for x in xs if x["hit"]], sum(1 for x in xs if not x["hit"]))
        s["n_all"] = len(xs)
        s["d"], s["z"] = _cmp(s, rest_h)
        s["rest"] = rest_h
        hold[k] = s

    # 곡 흐름: 3등분 / 처음 40노트 / 실수 직후 / 연속 노트 오차 상관
    third = []
    if len(H) >= 90:
        for part in np.array_split(np.arange(len(H)), 3):
            third.append(_stats(E[part]))
    start = None
    if len(H) >= 160:
        start = {"first": _stats(E[:40]), "rest": _stats(E[40:])}
        a1, a2 = np.abs(E[:40]), np.abs(E[40:])
        start["z"] = float((a1.mean() - a2.mean()) / (math.sqrt(a1.var(ddof=1) / len(a1) + a2.var(ddof=1) / len(a2)) or 1e-9))
    bad = [i for i, x in enumerate(N) if (not x["hit"]) or abs(x["e_raw"]) > win]
    after = set()
    for i in bad:
        for j in range(i + 1, min(len(N), i + 7)):
            after.add(j)
    after -= set(bad)
    ea = [abs(N[j]["e"]) for j in after if N[j]["hit"]]
    eo = [abs(x["e"]) for j, x in enumerate(N) if x["hit"] and j not in after and j not in bad]
    pz = 0.0
    if len(ea) >= 5 and len(eo) >= 5:
        pz = float((np.mean(ea) - np.mean(eo)) / (math.sqrt(np.var(ea, ddof=1) / len(ea) + np.var(eo, ddof=1) / len(eo)) or 1e-9))
    post = {"z": pz, "after": float(np.mean(ea)) if ea else 0.0, "other": float(np.mean(eo)) if eo else 0.0,
            "n_err": len(bad), "n_after": len(ea)}
    # 패턴별 평균을 뺀 나머지로 연속 상관 (빽빽한 구간이 통째로 이른 것 같은 '패턴 효과'는 빼고 순수 흐름만)
    ac1 = None
    if len(H) >= 100:
        res = []
        for x in H:
            t0 = next((t for t in ("jack", "trill", "stair", "chord3", "chord2", "hold", "stream", "single")
                       if t in x["tags"]), None)
            mu = tags[t0]["mean"] if t0 and tags[t0]["n"] >= 10 else allst["mean"]
            res.append((x["t"], x["e"] - mu))
        a, b = [], []
        for (t1, r1), (t2, r2) in zip(res, res[1:]):
            if 0 < t2 - t1 <= 600:
                a.append(r1)
                b.append(r2)
        if len(a) >= 60 and np.std(a) > 0 and np.std(b) > 0:
            ac1 = float(np.corrcoef(a, b)[0, 1])

    # 연속한 노트 오차가 서로 비슷하면(물결) 노트 수만큼 독립적인 정보가 아니다 → 확실성(z)을 그만큼 낮춘다
    rho = min(0.8, max(0.0, ac1 or 0.0))
    infl = math.sqrt((1 + rho) / (1 - rho))
    for t in tags.values():
        t["z"] /= infl
    for v in hold.values():
        v["z"] /= infl

    # 밀도와 치우침의 관계 (기울기: 초당 노트 1개 늘 때 ms)
    slope = None
    if len(H) >= 100:
        d = np.array([x["nps"] for x in H])
        if d.std() > 1.0:
            slope = float(np.polyfit(d, E, 1)[0])

    # 키 입력: 남는 누름(빈 누름), 채터
    gkeys = set(key2ch)
    t0, t1 = N[0]["t"] - 300.0, N[-1]["t"] + 300.0
    kd = [(t, k) for t, k, dn in keys if dn and k in gkeys and t0 <= t <= t1]
    lastup, chat = {}, 0
    for t, k, dn in keys:
        if k not in gkeys:
            continue
        if dn:
            u = lastup.get(k)
            if u is not None and t - u < 25.0:
                chat += 1
        else:
            lastup[k] = t
    extra = max(0, len(kd) - len(H))

    # 밀도 효과를 패턴별로 나눠서 (빽빽한 구간이 연타·트릴 투성이라 생긴 차이인지, 밀도 자체 때문인지)
    def prim(x):
        for t in ("jack", "trill", "stair", "chord3", "chord2", "hold", "side", "trigger", "long", "stream", "single"):
            if t in x["tags"]:
                return t
        return "etc"
    num = den = 0.0
    for p_ in {prim(x) for x in H}:
        bh = [x["e"] for x in H if prim(x) == p_ and "burst" in x["tags"]]
        ch_ = [x["e"] for x in H if prim(x) == p_ and "calm" in x["tags"]]
        if len(bh) >= 8 and len(ch_) >= 8:
            w = 1.0 / (1.0 / len(bh) + 1.0 / len(ch_))
            num += w * (np.mean(bh) - np.mean(ch_))
            den += w
    dens_strat = float(num / den) if den > 0 else None

    # 키 누름 시간 요약
    dws = [x["dwell"] for x in H if x.get("dwell") is not None]
    fast_r = [r for g, r, b in dw_pairs if g <= 250.0]
    dwell = {"n": len(dws), "med": float(np.median(dws)) if dws else None,
             "n_pairs": len(dw_pairs), "fast_ratio_med": float(np.median(fast_r)) if fast_r else None,
             "long_frac": (float(np.mean([r >= DWELL_LONG for g, r, b in dw_pairs])) if dw_pairs else None),
             "cmp": _dwell_stats(dw_pairs)}

    # 박자 요약 (보고서 머리말용)
    bl = [beat_at(grid, x["t"]) for x in N]
    okb = [round(b) for b, sc in bl if b is not None and sc >= GRID_MIN_SCORE]
    bcount = defaultdict(int)
    for b in okb:
        bcount[b] += 1
    bpm_main = sorted(bcount.items(), key=lambda kv: -kv[1])[:3]
    jv = defaultdict(list)
    for x in N:
        if "jack" in x["tags"]:
            jv[x.get("jack_nv") or "기타"].append(x.get("jack_gap", 0.0))
    gridsum = {"ok": len(okb) / max(1, len(N)), "bpms": [b for b, _ in bpm_main],
               "jack": sorted(((k, len(v), float(np.mean(v))) for k, v in jv.items()), key=lambda r: -r[1])}

    t_first = N[0]["t"]
    notes = [{"t": x["t"] - t_first, "ch": x["ch"], "e": x["e"], "hand": x["hand"], "tags": x["tags"],
              "jp": x.get("jack_pos", 0), "sd": x.get("stair_dir", 0), "hs": x.get("hold_same_hand"),
              "nv": note_value(x.get("beat_p"))} for x in N]
    return {"notes": notes, "game": None,
            "mode": mode, "win": win, "n": len(N), "hit": len(H), "all": allst, "base": base, "base_name": base_name,
            "tags": tags, "lanes": lanes, "hands": hands, "chord": chord, "trill": trill, "jack": jack,
            "stair": stair, "hold": hold, "third": third, "start": start, "post": post, "ac1": ac1, "dwell": dwell,
            "slope": slope, "dens_hi": hi_d, "dens_strat": dens_strat, "infl": infl, "extra": extra, "chatter": chat, "presses": len(kd),
            "dur": (N[-1]["t"] - N[0]["t"]) / 1000.0, "n_bpm": len(vch), "grid": gridsum, "rescued": 0}


# ------------------------------------------------------------------ 조언 만들기
def mmss(ms):
    s = max(0, int(round(ms / 1000.0)))
    return "%d:%02d" % (s // 60, s % 60)


def spans(rep, pred, bad, top=3, gap=3000.0):
    """pred(노트)에 해당하면서 bad(노트)인 노트가 몰린 곡 구간 (첫 노트 기준 시간) 상위 몇 개"""
    ts = [x["t"] for x in rep.get("notes", []) if pred(x) and bad(x)]
    if not ts:
        return []
    grp, cur = [], [ts[0]]
    for t in ts[1:]:
        if t - cur[-1] <= gap:
            cur.append(t)
        else:
            grp.append(cur)
            cur = [t]
    grp.append(cur)
    big = [g for g in grp if len(g) >= 3]
    grp = big if big else [g for g in grp if len(g) >= 2][:1]
    grp.sort(key=lambda g: -len(g))
    out = []
    for g in sorted(grp[:top], key=lambda g: g[0]):
        out.append("%s~%s (%d개)" % (mmss(g[0]), mmss(g[-1]), len(g)) if g[-1] - g[0] >= 1000 else
                   "%s 부근 (%d개)" % (mmss(g[0]), len(g)))
    return out


def _adv(rid, score, title, where, wrong, lack, todo, goal, src, kind="fix", metric=None, spans_=None):
    return {"id": rid, "score": float(score), "title": title, "where": where, "spans": spans_ or [],
            "wrong": wrong, "lack": lack, "todo": list(todo), "goal": goal, "src": list(src), "kind": kind,
            "metric": metric}


# 패턴별 오답 문장: (부족한 점, [지침들])
PAT_TEXT = {
    "single": {
        "late": ("노트가 드문 곳에서 박자를 미리 예측하지 않고, 노트가 판정선에 온 걸 '보고 나서' 누르고 있어요.",
                 ["드문 노트일수록 음악 박자를 속으로 세고, 박자에 맞춰 누르세요.",
                  "노트를 기다리지 말고, 판정선에 닿는 '순간'을 예상해서 같이 누르세요.",
                  "가끔 화면보다 자기 키음을 들어 보세요. 키음이 박자에 붙는지 귀로 확인할 수 있어요."]),
        "early": ("노트가 드문 곳에서 기다리지 못하고 앞당겨 누르고 있어요.",
                  ["드문 노트는 판정선 정중앙에 닿을 때까지 기다렸다가 누르세요.",
                   "음악 박자에 맞춰 누르면 앞당기는 버릇이 줄어요.",
                   "배속이 너무 느리면 일찍 치기 쉬워요. 배속이 지나치게 낮지 않은지도 보세요."])},
    "stream": {
        "late": ("빠른 흐름을 미리 읽지 못하고 뒤따라가고 있어요.",
                 ["시선을 판정선보다 조금 위에 고정하고, 흐름을 4~8개 덩어리로 읽으세요.",
                  "흐름을 '하나씩' 치지 말고 리듬 덩어리로 기억해서 치세요."]),
        "early": ("흐름이 빨라지면 앞당겨 치고(달리고) 있어요.",
                  ["빠른 구간일수록 '여유 있게'를 떠올리고 음악 박자에 맞추세요.",
                   "비슷한 흐름이 나오는 한 단계 쉬운 곡에서 판정선 정중앙을 맞추는 연습을 하세요."])},
    "chord2": {
        "late": ("동시에 누를 손 모양을 늦게 잡아서 망설이고 있어요.",
                 ["동치가 보이면 미리 두 손가락을 키 위에 올려 두고 '한 번에' 내리세요."]),
        "early": ("동치를 판정선에 닿기 전에 덮치듯 누르고 있어요.",
                  ["동치도 판정선 정중앙에 닿는 순간에 누르세요.",
                   "동치는 보통 음악의 강한 박(킥·스네어)에 있어요. 그 소리에 맞추세요."])},
    "chord3": {
        "late": ("손가락을 많이 쓰는 동치에서 준비가 늦어요.",
                 ["큰 동치는 손 모양을 미리 잡고 손 전체로 한 번에 누르세요."]),
        "early": ("큰 동치에서 서둘러 덮치고 있어요.",
                  ["큰 동치일수록 손 전체를 판정선에 닿는 순간에 내리세요.",
                   "강한 박 소리에 맞춰 누르세요."])},
    "jack": {
        "late": ("같은 키를 다시 누를 때 손가락이 늦게 떨어져요(끝까지 꾹 누르거나 힘이 들어감).",
                 ["키를 바닥까지 세게 누르지 말고, 작동점 근처에서 가볍게 튕기세요. 세게 쳐도 빨라지지 않아요.",
                  "손가락은 살짝 세우고 손등 힘은 빼세요. 빠른 연타는 손목을 살짝 들고 떨듯이 치면 손가락이 빨리 돌아와요.",
                  "연타 구간만 따로, 느린 곡에서 일정한 간격으로 연습하세요."]),
        "early": ("연타를 박자 없이 뭉개서(마구) 누르고 있어요.",
                  ["연타 개수와 박자를 소리로 세면서 치세요.",
                   "뭉개서 넘기는 버릇은 읽는 습관까지 망쳐요. 칠 수 있는 속도의 곡에서 정확히 치세요."])},
    "trill": {
        "late": ("두 손가락이 고르게 번갈아 나가지 못하고 한쪽이 늦게 따라와요.",
                 ["느린 템포에서 두 손가락을 '똑같은 간격'으로 번갈아 치는 연습부터 하세요.",
                  "아래 손별·레인별 오답도 같이 보세요."]),
        "early": ("트릴이 점점 빨라지고(달리고) 있어요.",
                  ["트릴은 손이 리듬을 만들어 달리기 쉬워요. 음악 박자에 맞춰 일정하게 치세요."])},
    "stair": {
        "late": ("손가락 순서가 한 박씩 밀리고 있어요.",
                 ["계단은 대부분 박자감 문제예요. 박 단위로 나눠 세면서 치세요.",
                  "계단 순서(예: 1-2-3-4)를 느린 속도에서 '일정한 간격·정확한 순서'로 연습하세요."]),
        "early": ("계단을 한 번에 쓸어내리듯 굴려서 앞당기고 있어요.",
                  ["손가락 하나하나가 따로 박자를 맞추도록 천천히 연습하세요."])},
    "long": {
        "late": ("롱노트를 '잡는' 동작이 단타보다 느려요.", ["롱노트 머리도 단타처럼 판정선에 닿는 순간에 누르세요."]),
        "early": ("롱노트를 미리 잡고 있어요.", ["롱노트 머리도 판정선 정중앙에 닿을 때 누르세요."])},
    "side": {
        "late": ("엄지(사이드)는 덜 쓰여서 늦게 나가요.",
                 ["사이드가 많은 곡을 골라 엄지만 박자에 맞추는 연습을 하세요."]),
        "early": ("사이드를 미리 누르고 있어요.", ["사이드도 판정선 정중앙에 닿을 때 누르세요."])},
    "trigger": {
        "late": ("트리거 키에 손이 늦게 가요.", ["트리거가 많은 곡에서 트리거만 박자에 맞추는 연습을 하세요."]),
        "early": ("트리거를 미리 누르고 있어요.", ["트리거도 판정선 정중앙에 맞추세요."])},
    "bpm": {
        "late": ("속도가 바뀐 뒤 새 박자에 늦게 적응해요.",
                 ["변속 지점을 미리 외우고, 바뀐 박자를 소리로 먼저 잡으세요. 화면 속도보다 음악이 더 믿을 만해요."]),
        "early": ("변속 직후 서두르고 있어요.", ["변속 지점에서 새 박자를 귀로 먼저 잡고 치세요."])},
}
PAT_SRC = {"single": ["react_read", "visual", "arca_sound", "dc_speed"], "stream": ["react_read", "rush", "arca_gaze", "bms_jose"],
           "chord2": ["chord", "iidx_yoko"], "chord3": ["chord", "iidx_yoko"], "jack": ["oh_4k", "dc_skill2", "dc_finger", "iidx_jack"],
           "trill": ["sdvx_phys", "rush"], "stair": ["sdvx_phys", "dc_skill2"],
           "long": ["oh_4k"], "side": ["sdvx_phys"], "trigger": ["sdvx_phys"], "bpm": ["phase", "visual"]}
# 패턴별 오답에는 '그 패턴만 따로 연습'(BMS 테마별 연습·4/5/6키 난이도표) 근거를 같이 붙인다
for _k in PAT_SRC:
    if _k != "bpm":
        PAT_SRC[_k] = PAT_SRC[_k] + ["bms_uya", "bms_dmrv"]


def advise(rep, cfg=None):
    """보고서 → 오답 목록 (중요한 순서). kind: fix(오답) / good(잘한 점) / info(참고)"""
    if not rep:
        return []
    out = []
    mode = rep["mode"]
    win = rep["win"]
    a = rep["all"]
    T = rep["tags"]
    dev = win * 0.4            # 이 정도 넘게 벗어난 노트를 '틀린 노트'로 보고 구간을 찾는다
    IF = rep.get("infl", 1.0)  # 오차가 물결처럼 이어지는 만큼 확실성을 낮춤

    # 놓침을 믿어도 되는지: 게임 결과(BREAK)와 비교
    g = rep.get("game")
    brk = int(g[11]) if g and len(g) > 11 else None
    miss_ok = None if brk is None else (a["miss"] <= max(3, 2 * brk))
    miss_note = "" if miss_ok else " (분석기 놓침에는 노트를 잘못 본 경우가 섞일 수 있어요)"

    def where_tag(tag, k):
        sign = 1 if k == "late" else -1

        def bad(x):
            return x["e"] is None or (x["e"] - rest_mean) * sign >= dev
        rest_mean = T[tag]["rest"]["mean"] if "rest" in T[tag] else a["mean"]
        return spans(rep, lambda x: tag in x["tags"], bad)

    # 1) 전체 치우침
    m = a["mean"]
    sg = T.get("single", {})
    ref = sg["mean"] if sg.get("n", 0) >= 40 else None
    hh = rep["hands"]
    hands_same = all(hh[h]["n"] < 60 or (hh[h]["mean"] * m > 0 and abs(hh[h]["mean"]) >= 3.0)
                     for h in ("왼손", "오른손") if h in hh)
    if abs(m) >= 4.0 and a["n"] >= 200 and hands_same and (ref is None or (ref * m > 0 and abs(ref) >= 3.0)):
        k = "일찍(FAST 쪽)" if m < 0 else "늦게(SLOW 쪽)"
        out.append(_adv("bias", abs(m) * 2.0, "곡 전체가 %s 치우침" % k, "곡 전체 (패턴과 상관없이 고르게)",
                        "평균 %s ms%s." % (fmt(m), (", 느린 단타에서도 %s ms" % fmt(ref)) if ref is not None else ""),
                        "손 습관보다는 영점(판정 타이밍) 설정과 내 감각이 어긋나 있을 가능성이 커요.",
                        ["편한 곡 2~3곡에서도 같은 쪽으로 치우치는지 확인하세요.",
                         "계속 그러면 [영점 보정 (결과 화면)]의 추천값으로 맞추세요.",
                         "특정 패턴에서만 치우친다면 영점은 건드리지 마세요.",
                         "한 번 맞춘 뒤에는 그날 컨디션이나 곡마다 바꾸지 마세요."],
                        "편한 곡에서 평균 ±2 ms 안 (지금 %s ms)" % fmt(m), ["etterna", "oh_acc", "bandori", "iidx_judge", "iidx_fasttrap"],
                        metric=round(m, 1)))
    elif abs(m) >= 4.0 and ref is not None and abs(ref) < 2.5:
        out.append(_adv("bias_pattern", abs(m), "평균 치우침은 패턴 탓 (영점은 그대로)", "곡 전체",
                        "전체 평균 %s ms인데 느린 단타는 %s ms." % (fmt(m), fmt(ref)),
                        "쉬운 곳은 정확한데 특정 패턴에서 한쪽으로 몰려서 평균이 끌려갔어요.",
                        ["아래 패턴별 오답을 고치세요. 이럴 때 영점을 옮기면 잘 치던 곳이 틀어져요."],
                        "", ["bandori", "etterna", "iidx_fasttrap"], "info"))

    # 2) 흔들림 수준
    sd = a["sd"]
    if a["n"] >= 100:
        r = sd / max(1.0, win)
        if r <= 0.25:
            out.append(_adv("sd_good", 3, "흔들림이 아주 작아요", "곡 전체",
                            "흔들림 ±%.1f ms (100%% 범위 ±%.1f의 %.0f%%)." % (sd, win, r * 100), "",
                            ["이제는 치우침과 패턴별 약점을 다듬는 단계예요."], "", ["sms", "ur"], "good"))
        elif r >= 0.45:
            out.append(_adv("sd_high", 6 + (r - 0.45) * 30, "판정이 넓게 흩어짐", "곡 전체",
                            "흔들림 ±%.1f ms (100%% 범위 ±%.1f의 %.0f%%), 100%% 밖 %.0f%%."
                            % (sd, win, r * 100, a["out"] * 100),
                            "흔들림은 영점으로 못 고쳐요. 곡이 실력보다 어려워 처리에 급급하거나, 박자를 '보고 반응'해서 치는 비중이 커요.",
                            ["정확도 연습은 풀콤이 여유 있는 한두 단계 낮은 곡, BPM이 일정한 곡으로 하세요.",
                             "화면만 보지 말고 음악 박자에도 맞추세요.",
                             "가끔 게임 판정을 하드로 바꿔 쉬운 곡을 치면 대충 치는 버릇이 줄어요."],
                            "흔들림 ±%.0f ms 이하 (지금 ±%.1f)" % (sd * 0.9, sd), ["sms", "challenge", "oh_acc", "visual", "dc_lowlv",
                                                                         "bms_uya", "bms_tight"],
                            metric=round(sd, 1)))

    # 3) 곡 난이도
    miss_n = brk if brk is not None else a["miss"]
    miss_r = miss_n / max(1, rep["n"])
    if (miss_r >= 0.03 and (miss_ok is not False)) or a["out"] >= 0.25:
        out.append(_adv("too_hard", 5 + miss_r * 60 + a["out"] * 10, "정확도 연습용으로는 어려운 곡", "곡 전체",
                        "%s %d개 (%.1f%%), 100%% 밖 %.0f%%." % ("게임 BREAK" if brk is not None else "놓침", miss_n,
                                                            miss_r * 100, a["out"] * 100),
                        "처리하기 바쁜 곡에서는 '일단 누르기'가 먼저라 판정 습관이 망가지기 쉬워요.",
                        ["이 곡은 클리어·처리력 연습으로 두세요.",
                         "판정 연습은 BREAK가 거의 없고 정확도 95~99%가 나오는 곡에서 하세요.",
                         "어려운 곡만 반복(꼴박)하지 말고 한 단계 낮은 곡 여러 개와 섞어서 치세요."],
                        "", ["challenge", "bandori", "dc_setup", "dc_speed", "dc_rotate", "bms_uya"]))

    # 4) 밀도
    burst, calm = T["burst"], T["calm"]
    if burst["n"] >= MIN_N and calm["n"] >= MIN_N:
        d, z = _cmp(burst, calm)
        z /= IF
        ds = rep.get("dens_strat")
        # 같은 패턴끼리 비교해도 차이가 남아 있어야 '밀도 때문' (아니면 그 구간의 패턴 오답으로 이미 나옴)
        if abs(d) >= 4.0 and abs(z) >= 3.0 and (ds is None or (ds * d > 0 and abs(ds) >= 3.0)):
            k = "late" if d > 0 else "early"
            sign = 1 if d > 0 else -1
            sp = spans(rep, lambda x: "burst" in x["tags"],
                       lambda x: x["e"] is None or (x["e"] - calm["mean"]) * sign >= dev)
            if k == "early":
                out.append(_adv("rush", abs(d) * 1.6 + min(abs(z), 8) * 0.3, "빽빽한 구간에서 서둘러 일찍 침",
                                "밀도 높은 구간 (초당 %.0f노트 이상)" % rep["dens_hi"],
                                "평균 %s ms, 한산한 구간 %s ms → %.1f ms 일찍 (노트 %d개). 같은 패턴끼리 비교해도 %.1f ms 일찍."
                                % (fmt(burst["mean"]), fmt(calm["mean"]), abs(d), burst["n"], abs(ds if ds is not None else d)),
                                "노트가 몰리면 놓칠까 봐 '시간을 벌려고' 앞당기는 패닉 반응이에요. 일찍 치는 것 ≠ 빠르게 치는 것.",
                                ["빽빽해질수록 속으로 '여유 있게, 소리에 맞춰'를 떠올리세요.",
                                 "비슷한 밀도의 쉬운 곡에서 판정선 정중앙을 맞추는 연습을 하세요.",
                                 "이 때문에 영점·판정선을 바꾸지 마세요."],
                                "빽빽한 구간과 한산한 구간 차이 ±3 ms 안 (지금 %s)" % fmt(d), ["rush", "bandori", "arca_early", "iidx_judge"],
                                metric=round(d, 1), spans_=sp))
            else:
                out.append(_adv("lag", abs(d) * 1.6 + min(abs(z), 8) * 0.3, "빽빽한 구간에서 늦게 따라감",
                                "밀도 높은 구간 (초당 %.0f노트 이상)" % rep["dens_hi"],
                                "평균 %s ms, 한산한 구간 %s ms → %.1f ms 늦게 (노트 %d개). 같은 패턴끼리 비교해도 %.1f ms 늦게."
                                % (fmt(burst["mean"]), fmt(calm["mean"]), abs(d), burst["n"], abs(ds if ds is not None else d)),
                                "노트를 미리 읽지 못하고 판정선 근처에서 '보고 반응'해서 치고 있어요.",
                                ["시선을 판정선보다 조금 위에 고정해 미리 읽으세요.",
                                 "노트를 하나씩이 아니라 모양(덩어리)으로 보세요.",
                                 "배속은 패턴마다 바꾸지 말고 한 값으로 꾸준히 쓰세요."],
                                "차이 ±3 ms 안 (지금 %s)" % fmt(d), ["react_read", "arca_gaze", "etterna", "iidx_slow", "iidx_yoko"],
                                metric=round(d, 1), spans_=sp))
        if burst["n"] >= 40 and burst["sd"] >= calm["sd"] * 1.35 and burst["sd"] - calm["sd"] >= 3:
            out.append(_adv("burst_sd", 4 + (burst["sd"] / max(1, calm["sd"]) - 1.35) * 8, "빽빽한 구간에서 판정이 크게 흔들림",
                            "밀도 높은 구간", "흔들림 ±%.1f ms (한산한 구간 ±%.1f ms)." % (burst["sd"], calm["sd"]),
                            "처리 한계 근처라 손이 따라가기 바쁜 상태예요.",
                            ["그 밀도를 여유 있게 칠 수 있는 낮은 레벨 곡으로 '정확히' 치는 시간을 늘리세요.",
                             "체력 구간 정확도와 기술 구간 정확도는 따로 연습해야 늘어요."],
                            "흔들림 ±%.0f ms 이하 (지금 ±%.1f)" % (max(calm["sd"], burst["sd"] * 0.85), burst["sd"]),
                            ["oh_acc", "challenge", "bms_jose"], metric=round(burst["sd"], 1),
                            spans_=spans(rep, lambda x: "burst" in x["tags"],
                                         lambda x: x["e"] is None or abs(x["e"] - burst["mean"]) >= win * 0.6)))

    # 5) 패턴별
    for tag in ("single", "stream", "chord2", "chord3", "jack", "trill", "stair", "long", "side", "trigger", "bpm"):
        s = T.get(tag)
        if not s or s["n"] < MIN_N:
            continue
        name = TAG_NAME[tag]
        r = s["rest"]
        d, z = s["d"], s["z"]
        if abs(d) >= 4.0 and abs(z) >= 3.0:
            k = "late" if d > 0 else "early"
            lack, todo = PAT_TEXT[tag][k]
            sc = abs(d) * 1.4 + min(abs(z), 8) * 0.3 + math.log10(s["n"])
            if tag == "bpm":
                sc *= 0.6
                lack += " (분석기도 변속 지점에선 측정이 몇 ms 덜 정확해요.)"
            out.append(_adv("pat_%s_%s" % (tag, k), sc, "%s %s 침" % (josa(name, "을를"), "늦게" if d > 0 else "일찍"), name,
                            "평균 %s ms, 이 곡 나머지 노트 %s ms → %.1f ms %s (노트 %d개)."
                            % (fmt(s["mean"]), fmt(r["mean"]), abs(d), "늦게" if d > 0 else "일찍", s["n"]),
                            lack, todo, "다음 곡에서 나머지와 차이 ±3 ms 안 (지금 %s)" % fmt(d), PAT_SRC[tag],
                            metric=round(d, 1), spans_=where_tag(tag, k)))
        if s["n"] >= 30 and s["sd"] >= r["sd"] * 1.35 and s["sd"] - r["sd"] >= 3.0:
            lack = ("속도가 바뀐 뒤 새 박자에 맞추는 동안 흔들려요 (분석기도 변속 지점에선 측정이 몇 ms 덜 정확해요)."
                    if tag == "bpm" else "이 패턴에 아직 익숙하지 않아 칠 때마다 타이밍이 달라져요.")
            out.append(_adv("sd_" + tag, (1.5 if tag == "bpm" else 2.5) + (s["sd"] - r["sd"]) * 0.35,
                            "%s에서 판정이 흔들림" % name, name,
                            "흔들림 ±%.1f ms, 나머지 노트 ±%.1f ms (노트 %d개)." % (s["sd"], r["sd"], s["n"]), lack,
                            ["이 패턴이 많이 나오는 쉬운 곡에서 정확히 치는 연습을 하세요.",
                             "약한 패턴부터 메우면 전체 실력이 같이 올라요."],
                            "흔들림 ±%.0f ms 이하 (지금 ±%.1f)" % (max(r["sd"], s["sd"] * 0.85), s["sd"]),
                            ["oh_4k", "sdvx_phys"], metric=round(s["sd"], 1),
                            spans_=spans(rep, lambda x, tg=tag: tg in x["tags"],
                                         lambda x, mu=s["mean"]: x["e"] is None or abs(x["e"] - mu) >= win * 0.6)))
        mr = s["miss"] / max(1, s["n_all"])
        rr = r["miss"] / max(1, r["n_all"])
        if miss_ok is not False and s["miss"] >= 3 and mr >= 0.03 and mr >= 2.5 * max(rr, 0.004):
            out.append(_adv("miss_" + tag, 4 + mr * 50, "%s에서 많이 놓침" % name, name,
                            "놓침 %d / %d개 (%.1f%%), 나머지 %.1f%%%s." % (s["miss"], s["n_all"], mr * 100, rr * 100, miss_note),
                            "타이밍보다 '읽기'나 손 배치 문제일 가능성이 커요.",
                            ["이 패턴이 나오는 낮은 레벨 곡으로 모양에 먼저 익숙해지세요.",
                             "쉬운 노트까지 놓친다면 가장 쉬운 난이도에서 리듬을 들으며, 키를 짧게 끊어 치는 연습을 하세요."],
                            "놓침 %d개 이하 (지금 %d)" % (s["miss"] // 2, s["miss"]), ["oh_4k", "react_read", "ez2on_stac"],
                            metric=s["miss"], spans_=spans(rep, lambda x, tg=tag: tg in x["tags"], lambda x: x["e"] is None)))

    # 6) 연타 사슬: 두 번째 → 세 번째 이후
    J = rep["jack"]
    if J.get(2, {}).get("n", 0) >= 15 and J.get(3, {}).get("n", 0) >= 15:
        dj = J[3]["mean"] - J[2]["mean"]
        sej = math.sqrt(J[3]["sd"] ** 2 / J[3]["n"] + J[2]["sd"] ** 2 / J[2]["n"]) or 1e-9
        if abs(dj) >= 5.0 and abs(dj) / sej / IF >= 3.0:
            lt = dj > 0
            out.append(_adv("jack_drift", abs(dj) * 1.2, "연타가 이어질수록 %s" % ("점점 늦어짐" if lt else "점점 빨라짐"),
                            "같은 레인 3연타 이상",
                            "두 번째 노트 %s ms → 세 번째 이후 %s ms." % (fmt(J[2]["mean"]), fmt(J[3]["mean"])),
                            ("키를 끝까지 누르고 늦게 떼거나, 힘이 들어가서 누를 때마다 조금씩 밀려요." if lt else
                             "연타가 이어지면 박자보다 빨리 달려요."),
                            (["가볍게 튕기듯, 작동점 근처에서 치세요. 누르는 힘보다 빨리 떼는 걸 의식하세요.",
                              "손가락 끝에 힘을 주지 말고 손목 스냅·팔뚝으로 치세요.",
                              "손목·팔뚝이 타는 느낌이면 멈추고 풀어 주세요."] if lt else
                             ["연타 개수를 소리로 세면서 일정한 간격으로 치세요."]),
                            "두 번째와 세 번째 이후 차이 ±3 ms 안 (지금 %s)" % fmt(dj),
                            ["oh_4k", "fatigue", "iidx_jack", "bms_jose"] if lt else ["rush"], metric=round(dj, 1),
                            spans_=spans(rep, lambda x: x["jp"] >= 3,
                                         lambda x: x["e"] is None or (x["e"] - J[2]["mean"]) * (1 if lt else -1) >= dev)))

    # 6-1) 키를 오래 누른 뒤 같은 레인 다음 노트가 늦는지
    Dw = (rep.get("dwell") or {}).get("cmp")
    if Dw and Dw["n_long"] >= 20 and Dw["n_short"] >= 20 and Dw["d"] >= 4.0 and Dw["d"] / Dw["se"] / IF >= 3.0:
        dwi = rep["dwell"]
        out.append(_adv("long_press", Dw["d"] * 1.3, "키를 오래 누른 다음 노트가 늦음",
                        "같은 레인에 다음 노트가 빨리 오는 곳 (연타·스트림)",
                        "다음 노트까지 간격의 절반 넘게 누르고 있던 뒤 %s ms, 짧게 뗀 뒤 %s ms → %.1f ms 늦게 "
                        "(오래 누른 경우 %d번, 간격이 비슷한 것끼리 비교). 빠른 곳에서 누르고 있는 시간은 보통 간격의 %.0f%%."
                        % (fmt(Dw["long_mean"]), fmt(Dw["short_mean"]), Dw["d"], Dw["n_long"],
                           (dwi.get("fast_ratio_med") or 0) * 100),
                        "키를 끝까지 꾹 누르고 늦게 떼면 손가락이 돌아올 시간이 모자라 다음 누름이 밀려요. "
                        "빽빽해지면 힘이 들어가 더 오래 누르게 되기 쉬워요.",
                        ["누르자마자 떼는 느낌으로 짧게 치세요. 누르는 힘보다 빨리 떼는 걸 의식하세요.",
                         "손가락 끝에 힘을 빼고 손목 스냅·팔뚝으로 튕기듯 치세요."],
                        "오래 누른 뒤와 짧게 뗀 뒤 차이 ±3 ms 안 (지금 %s)" % fmt(Dw["d"]),
                        ["oh_stream", "iidx_jack", "oh_4k"], metric=round(Dw["d"], 1),
                        spans_=spans(rep, lambda x: "after_long_press" in x["tags"],
                                     lambda x: x["e"] is None or x["e"] - Dw["short_mean"] >= dev)))

    # 7) 트릴: 손별·손가락별
    th = rep["trill"]["hand"]
    if "왼손" in th and "오른손" in th and th["왼손"]["n"] >= 15 and th["오른손"]["n"] >= 15:
        d = th["왼손"]["mean"] - th["오른손"]["mean"]
        se = math.sqrt(th["왼손"]["sd"] ** 2 / th["왼손"]["n"] + th["오른손"]["sd"] ** 2 / th["오른손"]["n"]) or 1e-9
        if abs(d) >= 6.0 and abs(d) / se / IF >= 3.0:
            late = "왼손" if d > 0 else "오른손"
            out.append(_adv("trill_hand", abs(d) * 1.3, "양손 트릴에서 %s이 늦음" % late, "양손 트릴 (%s)" % late,
                            "트릴 안에서 왼손 %s ms, 오른손 %s ms → %s이 %.1f ms 늦게."
                            % (fmt(th["왼손"]["mean"]), fmt(th["오른손"]["mean"]), late, abs(d)),
                            "한 손이 리듬을 끌고 %s이 따라가는 상태예요. 덜 쓰는 손이 늦는 경우가 흔해요." % late,
                            ["%s만으로 같은 리듬을 일정한 간격으로 치는 연습을 따로 하세요." % late,
                             "양손 트릴은 느린 템포에서 두 손 간격을 똑같이 맞추는 것부터."],
                            "양손 차이 ±3 ms 안 (지금 %.1f ms)" % abs(d), ["bimanual", "sdvx_phys"], metric=round(abs(d), 1),
                            spans_=spans(rep, lambda x: "trill" in x["tags"] and x["hand"] == late,
                                         lambda x: x["e"] is None or x["e"] - th["왼손" if late == "오른손" else "오른손"]["mean"] >= dev
                                         if True else False)))
    tl = rep["trill"]["lane"]
    tr = rep["trill"]["lane_res"]      # 레인 자체의 치우침을 뺀, 트릴에서만 생기는 차이
    for hd in ("왼손", "오른손"):
        chs = [c for c in tr if rep["lanes"].get(c, {}).get("hand") == hd and tr[c]["n"] >= 15]
        if len(chs) >= 2:
            ms_ = sorted(chs, key=lambda c: tr[c]["mean"])
            hi_, lo_ = tr[ms_[-1]], tr[ms_[0]]
            d = hi_["mean"] - lo_["mean"]
            se = math.sqrt(hi_["sd"] ** 2 / hi_["n"] + lo_["sd"] ** 2 / lo_["n"]) or 1e-9
            if d >= 7.0 and d / se / IF >= 3.0:
                lab = lane_label(mode, ms_[-1], cfg)
                out.append(_adv("trill_finger_" + ms_[-1], d * 1.1, "%s 트릴에서 %s 늦음" % (hd, josa(lab, "이가")),
                                "%s 트릴 (%s ↔ %s)" % (hd, lab, lane_label(mode, ms_[0], cfg)),
                                "트릴 안에서 %s %s ms, %s %s ms (각 레인 평소 치우침을 빼도 %.1f ms 차이)."
                                % (lab, fmt(tl[ms_[-1]]["mean"]), lane_label(mode, ms_[0], cfg), fmt(tl[ms_[0]]["mean"]), d),
                                "같은 손 안에서 이 손가락이 덜 독립적으로 움직여요.",
                                ["이 두 손가락 트릴을 천천히, 간격을 똑같이 치는 연습을 하세요."],
                                "두 레인 차이 ±3 ms 안 (지금 %.1f ms)" % d, ["sdvx_phys"], metric=round(d, 1)))

    # 8) 동치
    C = rep["chord"]
    if C["n"] >= 15:
        if C["spread"] >= 12.0 and C["spread"] >= C["expect"] * 1.3:
            out.append(_adv("chord_spread", 3 + C["spread"] * 0.25, "동치에서 손가락이 따로 내려옴", "동치 전체",
                            "동치 안 누른 시각 차이 중앙값 %.1f ms (10개 중 1개는 %.0f ms 이상). "
                            "손가락이 서로 상관없이 흔들릴 때 예상되는 %.1f ms보다도 커요."
                            % (C["spread"], C["spread_p90"], C["expect"]),
                            "손가락을 하나씩 굴리듯 내리거나 손 모양을 늦게 잡아요.",
                            ["동치는 손 모양을 미리 만들어 두고 한 번에 내리세요.",
                             "노트를 하나씩 보지 말고 가로 모양으로 한 번에 읽으세요."],
                            "차이 중앙값 %.0f ms 이하 (지금 %.1f)" % (max(8.0, C["spread"] * 0.8), C["spread"]), ["chord", "iidx_yoko"],
                            metric=round(C["spread"], 1)))
        for ch, (mu, n, s_) in C["lane_off"].items():
            if n < 15 or s_ <= 0:
                continue
            zz = mu / (s_ / math.sqrt(n))
            if abs(mu) >= 6.0 and abs(zz) >= 3.5:      # 레인 여러 개를 동시에 보므로 더 엄격하게
                lab = lane_label(mode, ch, cfg)
                out.append(_adv("chord_lane_" + ch, abs(mu) * 1.1,
                                "동치에서 %s %s 누름" % (josa(lab, "을를"), "늦게" if mu > 0 else "먼저"), "동치 속 %s" % lab,
                                "같은 동치의 다른 노트보다 평균 %s ms (동치 %d개, 이 레인의 평소 치우침은 뺀 값)." % (fmt(mu), n),
                                "이 손가락이 다른 손가락보다 %s 내려와요." % ("늦게" if mu > 0 else "먼저"),
                                ["이 레인이 들어간 동치를 천천히 '동시에' 누르는 연습을 하세요."],
                                "±3 ms 안 (지금 %s)" % fmt(mu), ["chord", "sdvx_phys"], metric=round(mu, 1)))
        if C["lr"] and C["lr"][1] >= 15:
            mu, n, s_ = C["lr"]
            if abs(mu) >= 6.0 and s_ > 0 and abs(mu) / (s_ / math.sqrt(n)) >= 3.0:
                late = "왼손" if mu > 0 else "오른손"
                out.append(_adv("chord_lr", abs(mu) * 1.2, "양손 동치에서 %s이 늦게 내려옴" % late, "양손 동치",
                                "왼손이 오른손보다 평균 %s ms (동치 %d개)." % (fmt(mu), n),
                                "한 손이 이끌고 %s이 따라가요. 양손 사이 어긋남은 한 손 안보다 원래 큰 편이에요." % late,
                                ["양손 동치만 나오는 느린 구간에서 두 손을 같이 내리는 연습을 하세요."],
                                "양손 차이 ±3 ms 안 (지금 %.1f ms)" % abs(mu), ["chord", "bimanual"], metric=round(abs(mu), 1)))

    # 9) 손·레인
    hd = rep["hands"]
    if "왼손" in hd and "오른손" in hd and hd["왼손"]["n"] >= 60 and hd["오른손"]["n"] >= 60:
        d, z = _cmp(hd["왼손"], hd["오른손"])
        if abs(d) >= 5.0 and abs(z) / IF >= 3.0:
            late = "왼손" if d > 0 else "오른손"
            other = "오른손" if late == "왼손" else "왼손"
            out.append(_adv("hand_bias", abs(d) * 1.3, "%s이 전반적으로 늦음" % late, late,
                            "왼손 평균 %s ms, 오른손 %s ms → %.1f ms 차이." % (fmt(hd["왼손"]["mean"]), fmt(hd["오른손"]["mean"]), abs(d)),
                            "주로 쓰는 손과 다른 손의 타이밍 차이는 흔하고, 연습으로 줄어들어요.",
                            ["%s 쪽 레인이 많은 패턴에서 %s만 박자에 맞추는 연습을 하세요." % (late, late)],
                            "양손 차이 ±3 ms 안 (지금 %.1f ms)" % abs(d), ["bimanual", "sdvx_phys", "bms_hstk"], metric=round(abs(d), 1),
                            spans_=spans(rep, lambda x: x["hand"] == late,
                                         lambda x: x["e"] is None or x["e"] - hd[other]["mean"] >= dev)))
        w = "왼손" if hd["왼손"]["sd"] > hd["오른손"]["sd"] else "오른손"
        o = "오른손" if w == "왼손" else "왼손"
        if hd[w]["sd"] >= hd[o]["sd"] * 1.3 and hd[w]["sd"] - hd[o]["sd"] >= 3.0:
            out.append(_adv("hand_sd", 3 + (hd[w]["sd"] - hd[o]["sd"]) * 0.4, "%s 판정이 더 흔들림" % w, w,
                            "흔들림 %s ±%.1f ms, %s ±%.1f ms." % (w, hd[w]["sd"], o, hd[o]["sd"]),
                            "%s 연습량이 부족한 경우가 많아요." % w, ["%s만 쓰는 패턴을 따로 연습하세요." % w],
                            "%s 흔들림 ±%.0f ms 이하 (지금 ±%.1f)" % (w, hd[o]["sd"] * 1.1, hd[w]["sd"]),
                            ["sdvx_phys", "bimanual"], metric=round(hd[w]["sd"], 1)))
    L = rep["lanes"]
    for ch, s0 in L.items():
        s = dict(s0)
        lab = lane_label(mode, ch, cfg)
        mr = s["miss"] / max(1, s["n_all"])
        if miss_ok is not False and s["miss"] >= 4 and mr >= 0.03 and mr >= 3 * (a["miss"] / max(1, rep["n"])):
            out.append(_adv("lane_miss_" + ch, 4 + mr * 40, "%s에서 많이 놓침" % lab, lab,
                            "놓침 %d / %d개 (%.1f%%)%s." % (s["miss"], s["n_all"], mr * 100, miss_note),
                            "그 손가락이 노트를 놓치거나, 키보드가 그 키 조합을 동시에 못 받는 경우일 수 있어요.",
                            ["특정 동치에서만 빠진다면 키보드 동시 입력(N키 롤오버)을 확인하세요."],
                            "놓침 %d개 이하 (지금 %d)" % (s["miss"] // 2, s["miss"]), ["kbd"], metric=s["miss"],
                            spans_=spans(rep, lambda x, c=ch: x["ch"] == c, lambda x: x["e"] is None)))
        sn = s0.get("nc") or s0          # 동치 밖 노트로만
        hs0 = hd.get(s["hand"])
        hs = (hs0 or {}).get("nc") or hs0
        if sn["n"] < 40 or not hs or hs["n"] - sn["n"] < 60:
            continue
        no = hs["n"] - sn["n"]
        mo = (hs["mean"] * hs["n"] - sn["mean"] * sn["n"]) / no      # 같은 손 다른 레인들 평균
        d = sn["mean"] - mo
        s["mean"] = sn["mean"]
        se = math.sqrt(sn["sd"] ** 2 / sn["n"] + hs["sd"] ** 2 / no) or 1e-9
        if abs(d) >= 6.0 and abs(d / se) / IF >= 3.0 and s["hand"] in ("왼손", "오른손", "가운데"):
            out.append(_adv("lane_" + ch, abs(d) * 1.1, "%s %s 침" % (josa(lab, "을를"), "늦게" if d > 0 else "일찍"),
                            "%s (%s)" % (lab, s["hand"]),
                            "동치 밖 노트에서 이 레인 %s ms, %s 다른 레인 %s ms → %.1f ms %s." % (fmt(s["mean"]), s["hand"], fmt(mo),
                                                                          abs(d), "늦게" if d > 0 else "일찍"),
                            "이 키를 맡은 손가락 하나의 습관이에요 (키가 멀거나 손가락 힘이 약한 경우도).",
                            ["이 레인이 섞인 패턴을 느리게, 박자에 맞춰 연습하세요.", "키 배치가 불편한지도 확인하세요."],
                            "같은 손 다른 레인과 ±3 ms 안 (지금 %s)" % fmt(d), ["sdvx_phys"], metric=round(d, 1)))

    # 10) 롱노트 잡은 채
    for k, s in rep["hold"].items():
        if s["n"] >= MIN_N and abs(s["d"]) >= 5.0 and abs(s["z"]) >= 3.0:
            same = (k == "같은 손")
            sign = 1 if s["d"] > 0 else -1
            out.append(_adv("hold_" + k, abs(s["d"]) * 1.3,
                            "롱노트 잡은 채 %s으로 칠 때 %s" % (k, "늦음" if s["d"] > 0 else "서두름"),
                            "롱노트를 누르고 있는 동안 %s 노트" % k,
                            "평균 %s ms, 나머지 노트 %s ms → %.1f ms %s (노트 %d개)."
                            % (fmt(s["mean"]), fmt(s["rest"]["mean"]), abs(s["d"]), "늦게" if s["d"] > 0 else "일찍", s["n"]),
                            ("한 손가락이 키를 누르고 있으면 같은 손 다른 손가락까지 같이 굳어(따로 못 놀아) 타이밍이 틀어져요."
                             if same else "한 손이 롱노트를 잡고 있으면 다른 손까지 같이 긴장해 타이밍이 틀어져요."),
                            ["롱노트를 잡은 손가락은 힘을 빼고 '얹어 두기'만 하세요. 많은 사람이 여기에 힘을 너무 줘서 다른 손가락이 막혀요.",
                             "롱노트를 잡고 %s 손가락으로 치는 패턴만 천천히 연습하세요." % ("같은 손 다른" if same else "다른 손")],
                            "나머지와 차이 ±3 ms 안 (지금 %s)" % fmt(s["d"]), ["sdvx_hold", "dc_skill2", "bimanual"], metric=round(s["d"], 1),
                            spans_=spans(rep, lambda x, sm=same: "hold" in x["tags"] and bool(x["hs"]) == sm,
                                         lambda x, mu=s["rest"]["mean"], sg_=sign: x["e"] is None or (x["e"] - mu) * sg_ >= dev)))

    # 11) 계단 방향
    S = rep["stair"]
    if len(S) == 2 and all(v["n"] >= 15 for v in S.values()):
        d = S["안쪽"]["mean"] - S["바깥쪽"]["mean"]
        se = math.sqrt(S["안쪽"]["sd"] ** 2 / S["안쪽"]["n"] + S["바깥쪽"]["sd"] ** 2 / S["바깥쪽"]["n"]) or 1e-9
        if abs(d) >= 6.0 and abs(d) / se / IF >= 3.0:
            ref_m = T["stair"]["rest"]["mean"] if "rest" in T["stair"] else a["mean"]
            k = max(S, key=lambda q: abs(S[q]["mean"] - ref_m))
            dd = S[k]["mean"] - ref_m
            out.append(_adv("stair_dir", abs(d) * 1.1, "%s 계단에서 %s" % ("손 안쪽(가운데)으로 가는" if k == "안쪽" else "손 바깥쪽으로 가는", "늦음" if dd > 0 else "서두름"),
                            "계단 · %s 방향 (안쪽 = 손가락이 키보드 가운데 쪽으로)" % k,
                            "안쪽 계단 %s ms, 바깥쪽 계단 %s ms, 나머지 노트 %s ms."
                            % (fmt(S["안쪽"]["mean"]), fmt(S["바깥쪽"]["mean"]), fmt(ref_m)),
                            "한 방향으로 쓸어내리듯 치면 앞당겨지고, 손가락 순서가 어색하면 밀려요.",
                            ["%s 방향 계단만 천천히, 손가락 하나하나 일정한 간격으로 연습하세요." % k],
                            "두 방향 차이 ±3 ms 안 (지금 %.1f ms)" % abs(d), ["sdvx_phys"], metric=round(abs(d), 1)))

    # 12) 곡 흐름
    th3 = rep["third"]
    dur = rep.get("dur", 0) * 1000.0
    if len(th3) == 3:
        if th3[2]["sd"] >= th3[0]["sd"] * 1.25 and th3[2]["sd"] - th3[0]["sd"] >= 3.0:
            out.append(_adv("fatigue", 3 + (th3[2]["sd"] - th3[0]["sd"]) * 0.5, "곡 후반에 흔들림이 커짐",
                            "곡 마지막 1/3 (%s 이후)" % mmss(dur * 2 / 3),
                            "흔들림 앞부분 ±%.1f → 뒷부분 ±%.1f ms." % (th3[0]["sd"], th3[2]["sd"]),
                            "지구력이 떨어지거나 집중이 풀려요. 곡 후반이 원래 어려운 곡일 수도 있어요.",
                            ["지구력은 조금씩 늘리세요.", "손목·팔뚝이 타는 느낌이면 멈추고 풀어 주고, 저리면 그만하세요."],
                            "뒷부분 흔들림 ±%.0f ms 이하 (지금 ±%.1f)" % (th3[0]["sd"] * 1.1, th3[2]["sd"]),
                            ["fatigue", "oh_4k", "health", "bms_elma", "iidx_slow"], metric=round(th3[2]["sd"], 1)))
        dm = th3[2]["mean"] - th3[0]["mean"]
        se3 = math.sqrt(th3[2]["sd"] ** 2 / th3[2]["n"] + th3[0]["sd"] ** 2 / th3[0]["n"]) or 1e-9
        if abs(dm) >= 4.0 and abs(dm) / se3 / IF >= 3.0:
            out.append(_adv("drift_song", abs(dm), "곡 후반으로 갈수록 %s" % ("늦어짐" if dm > 0 else "빨라짐"),
                            "곡 마지막 1/3 (%s 이후)" % mmss(dur * 2 / 3),
                            "평균 앞부분 %s → 뒷부분 %s ms." % (fmt(th3[0]["mean"]), fmt(th3[2]["mean"])),
                            ("지치면서 손이 처져요." if dm > 0 else "긴장이 쌓이며 서둘러요."),
                            (["곡 후반에도 손에 힘을 빼고 가볍게 치세요."] if dm > 0 else ["후반에 '여유 있게'를 떠올리세요."]),
                            "앞뒤 차이 ±3 ms 안 (지금 %s)" % fmt(dm), ["fatigue"] if dm > 0 else ["rush", "choking"],
                            metric=round(dm, 1)))
    st = rep["start"]
    if st and st["first"]["n"] >= 30:
        f, r = st["first"], st["rest"]
        if f["abs"] >= r["abs"] * 1.3 and f["abs"] - r["abs"] >= 3.0 and st.get("z", 9) >= 3.0:
            out.append(_adv("warmup", 3 + (f["abs"] - r["abs"]) * 0.4, "곡 시작 부분이 덜 정확함", "곡 처음 40노트",
                            "평균 오차 크기 %.1f ms, 나머지 %.1f ms." % (f["abs"], r["abs"]),
                            "쉬었다 다시 하면 처음 몇 분은 감이 덜 돌아와요(워밍업 감소). 첫 곡이었다면 특히 그래요.",
                            ["본격적으로 치기 전에 쉬운 곡 하나로 손을 풀어 주세요. 같은 동작으로 푸는 게 가장 효과적이에요.",
                             "워밍업 곡은 늘 같은 순서로 정해 두고, 빠른 16분이 또렷이 쳐지는지 확인한 뒤 어려운 곡으로 가세요."],
                            "처음 40노트 오차 크기 %.0f ms 이하 (지금 %.1f)" % (r["abs"] * 1.1, f["abs"]), ["warmup", "bms_elma", "bms_djamn"],
                            metric=round(f["abs"], 1)))
    po = rep["post"]
    if (po["n_after"] >= 30 and po["other"] > 0 and po["after"] >= po["other"] * 1.3 and po["after"] - po["other"] >= 3
            and po.get("z", 9) >= 3.0):
        out.append(_adv("post_err", 3 + (po["after"] - po["other"]) * 0.5, "실수 직후 몇 노트가 같이 무너짐",
                        "실수(놓침·100% 밖) 바로 뒤 6노트",
                        "평균 오차 크기 %.1f ms, 평소 %.1f ms (실수 %d번)." % (po["after"], po["other"], po["n_err"]),
                        "실수를 의식하면서 손을 하나하나 통제하려 들면 오히려 무너지기 쉬워요.",
                        ["틀린 노트는 바로 잊고 다음 노트·소리에 주의를 두세요.", "손가락을 일일이 의식하지 마세요."],
                        "실수 뒤 오차 크기 %.0f ms 이하 (지금 %.1f)" % (po["other"] * 1.1, po["after"]),
                        ["choking", "arca_gaze", "etterna"], metric=round(po["after"], 1)))
    if rep["ac1"] is not None and rep["ac1"] >= 0.35:
        out.append(_adv("waves", 3 + (rep["ac1"] - 0.35) * 15, "어긋남이 물결처럼 여러 노트 이어짐", "곡 전체",
                        "연속한 노트끼리 오차 상관 %.2f (0에 가까울수록 노트마다 바로 바로잡는 것)." % rep["ac1"],
                        "한 번 어긋난 박자를 바로 고치지 못하고 몇 노트 끌고 가요. 화면만 보고 칠 때 이런 경향이 커질 수 있어요.",
                        ["곡의 박자(소리)를 기준으로 삼으세요. 소리에 맞출 때가 화면에만 맞출 때보다 덜 흔들려요.",
                         "적정 난이도 곡에서 자기 키음이 박자에 붙는지 들어 보세요."],
                        "상관 0.25 이하 (지금 %.2f)" % rep["ac1"], ["phase", "visual", "arca_sound"], metric=round(rep["ac1"], 2)))

    # 13) 키보드
    if rep["chatter"] >= 3:
        out.append(_adv("chatter", 6 + rep["chatter"], "한 번 눌렀는데 두 번 들어간 흔적", "키보드",
                        "떼고 25ms 안에 다시 눌린 입력 %d번." % rep["chatter"],
                        "키보드 스위치 접점이 튀는 '채터'일 가능성이 커요. 연습으로 고칠 수 없어요.",
                        ["같은 키에서 반복되면 키보드 설정(디바운스)이나 스위치를 확인하세요."], "", ["kbd"]))
    if rep["presses"] >= 200 and rep["extra"] / max(1, rep["hit"]) >= 0.08:
        out.append(_adv("mash", 3 + rep["extra"] / max(1, rep["hit"]) * 30, "노트 없는데 누른 키가 많음", "곡 전체",
                        "게임 키 누름 %d번 중 노트와 짝지어지지 않은 누름 약 %d번." % (rep["presses"], rep["extra"]),
                        "모르는 구간을 뭉개서(마구 눌러) 넘기는 습관일 수 있어요.",
                        ["뭉개기는 읽기 습관을 망쳐요. 안 보이는 구간은 더 쉬운 곡에서 읽는 연습을 하세요."],
                        "빈 누름 %d번 이하" % (rep["extra"] // 2), ["oh_4k", "arca_early"], metric=rep["extra"]))

    # 14) 잘한 점
    good = []
    for tag in ("jack", "trill", "stair", "chord2", "chord3", "stream", "side", "burst"):
        s = T.get(tag)
        if s and s["n"] >= 40 and abs(s["d"]) <= 2.0 and s["sd"] <= s["rest"]["sd"] * 1.1:
            good.append(TAG_NAME[tag])
    if good:
        out.append(_adv("good_pat", 2, "흔들리지 않는 패턴: " + ", ".join(good), "",
                        "나머지 노트와 평균·흔들림 차이가 거의 없어요.", "", ["지금처럼 유지하세요."], "", [], "good"))

    out = _dedupe(out)
    out.sort(key=lambda x: (x["kind"] != "fix", -x["score"]))
    return out


def _dedupe(out):
    """같은 버릇이 여러 오답으로 겹쳐 나오지 않게 (더 큰 원인 하나로)"""
    ids = {x["id"]: x for x in out}
    drop = set()
    hb = ids.get("hand_bias")
    if hb:
        late = hb["title"].split("이")[0]
        for k in ("trill_hand", "chord_lr"):
            x = ids.get(k)
            if x and late in x["title"] and abs(x["metric"] or 0) < abs(hb["metric"] or 0) + 5:
                drop.add(k)
    if hb:
        # 손 하나가 통째로 늦으면, 손이 없는 사이드·트리거는 '평균'과 비교해 반대로 보인다 → 그 정도면 뺌
        for k in ("pat_side_early", "pat_side_late", "pat_trigger_early", "pat_trigger_late"):
            x = ids.get(k)
            if x and abs(x["metric"] or 0) <= abs(hb["metric"] or 0) / 2.0 + 3.0:
                drop.add(k)
    return [x for x in out if x["id"] not in drop]


# ------------------------------------------------------------------ 여러 곡 합쳐 보기 (한 곡에선 노트가 적어 안 보이던 버릇)
def _se(d, z):
    return abs(d / z) if z else 0.0


def measures(rep, cfg=None):
    """곡마다 저장해 두고 여러 곡을 합쳐 다시 검정할 값: {이름: [차이 ms, 표준오차, 노트 수]}"""
    M = {}
    if not rep:
        return M
    T = rep["tags"]
    for t in ("single", "stream", "chord2", "chord3", "jack", "trill", "stair", "long", "side", "trigger"):
        s = T.get(t)
        if s and s["n"] >= 10 and s["z"]:
            M["pat_" + t] = [round(s["d"], 2), round(_se(s["d"], s["z"]), 2), s["n"]]
    b, c = T["burst"], T["calm"]
    if b["n"] >= 10 and c["n"] >= 10:
        d, z = _cmp(b, c)
        if z:
            M["burst"] = [round(d, 2), round(_se(d, z), 2), b["n"]]
    hd = rep["hands"]
    if "왼손" in hd and "오른손" in hd and hd["왼손"]["n"] >= 30 and hd["오른손"]["n"] >= 30:
        d, z = _cmp(hd["왼손"], hd["오른손"])
        if z:
            M["hand"] = [round(d, 2), round(_se(d, z), 2), hd["왼손"]["n"] + hd["오른손"]["n"]]
    th = rep["trill"]["hand"]
    if "왼손" in th and "오른손" in th and th["왼손"]["n"] >= 5 and th["오른손"]["n"] >= 5:
        d, z = _cmp(th["왼손"], th["오른손"])
        if z:
            M["trill_hand"] = [round(d, 2), round(_se(d, z), 2), th["왼손"]["n"] + th["오른손"]["n"]]
    for ch, (mu, n, s_) in rep["chord"]["lane_off"].items():
        if n >= 5 and s_ > 0:
            M["chord_lane_" + ch] = [round(mu, 2), round(s_ / math.sqrt(n), 2), n]
    if rep["chord"]["lr"] and rep["chord"]["lr"][1] >= 5 and rep["chord"]["lr"][2] > 0:
        mu, n, s_ = rep["chord"]["lr"]
        M["chord_lr"] = [round(mu, 2), round(s_ / math.sqrt(n), 2), n]
    for k, s in rep["hold"].items():
        if s["n"] >= 5 and s["z"]:
            M["hold_" + k] = [round(s["d"], 2), round(_se(s["d"], s["z"]), 2), s["n"]]
    for ch, s0 in rep["lanes"].items():
        sn = s0.get("nc")
        hs = (hd.get(s0["hand"]) or {}).get("nc")
        if sn and hs and sn["n"] >= 10 and hs["n"] - sn["n"] >= 10 and s0["hand"] in ("왼손", "오른손", "가운데"):
            no = hs["n"] - sn["n"]
            mo = (hs["mean"] * hs["n"] - sn["mean"] * sn["n"]) / no
            se = math.sqrt(sn["sd"] ** 2 / sn["n"] + hs["sd"] ** 2 / no)
            if se > 0:
                M["lane_" + ch] = [round(sn["mean"] - mo, 2), round(se, 2), sn["n"]]
    J = rep["jack"]
    if J.get(2, {}).get("n", 0) >= 5 and J.get(3, {}).get("n", 0) >= 5:
        d, z = _cmp(J[3], J[2])
        if z:
            M["jack_drift"] = [round(d, 2), round(_se(d, z), 2), J[3]["n"]]
    Dw = (rep.get("dwell") or {}).get("cmp")
    if Dw and Dw["n_long"] >= 5 and Dw["n_short"] >= 5 and Dw["se"] > 0:
        M["long_press"] = [round(Dw["d"], 2), round(Dw["se"], 2), Dw["n_long"]]
    return M


def _mname(k, d, mode=None, cfg=None):
    """합쳐 보기 결과 이름과 할 일"""
    late = d > 0
    if k.startswith("pat_"):
        t = k[4:]
        lack, todo = PAT_TEXT[t]["late" if late else "early"]
        return "%s %s 침" % (josa(TAG_NAME[t], "을를"), "늦게" if late else "일찍"), lack, todo
    if k == "burst":
        return ("빽빽한 구간에서 늦게 따라감" if late else "빽빽한 구간에서 서둘러 일찍 침",
                "노트를 미리 읽지 못하고 반응해서 쳐요." if late else "노트가 몰리면 시간을 벌려고 앞당겨요.",
                ["시선을 조금 위에 고정하고 덩어리로 읽으세요."] if late else ["빽빽해질수록 '여유 있게, 소리에 맞춰'."])
    if k == "hand":
        h = "왼손" if late else "오른손"
        return "%s이 전반적으로 늦음" % h, "두 손 타이밍 차이예요.", ["%s만 박자에 맞추는 연습을 하세요." % h]
    if k == "trill_hand":
        h = "왼손" if late else "오른손"
        return "양손 트릴에서 %s이 늦음" % h, "번갈아 칠 때 %s이 따라가요." % h, ["%s만으로 같은 리듬을 일정한 간격으로 연습하세요." % h]
    if k == "chord_lr":
        h = "왼손" if late else "오른손"
        return "양손 동치에서 %s이 늦게 내려옴" % h, "한 손이 이끌고 다른 손이 따라가요.", ["양손 동치를 느리게 같이 내리는 연습을 하세요."]
    if k.startswith("chord_lane_"):
        lab = lane_label(mode, k[11:], cfg) if mode else "레인 " + k[12:]
        return ("동치에서 %s %s 누름" % (josa(lab, "을를"), "늦게" if late else "먼저"),
                "이 손가락이 동치에서 따로 내려와요.", ["이 레인이 들어간 동치를 천천히 '동시에' 누르세요."])
    if k.startswith("lane_"):
        lab = lane_label(mode, k[5:], cfg) if mode else "레인 " + k[6:]
        return ("%s %s 침" % (josa(lab, "을를"), "늦게" if late else "일찍"),
                "이 키를 맡은 손가락 하나의 습관이에요.", ["이 레인이 섞인 패턴을 느리게 박자에 맞춰 연습하세요."])
    if k.startswith("hold_"):
        return ("롱노트 잡은 채 %s으로 칠 때 %s" % (k[5:], "늦음" if late else "서두름"),
                "롱노트를 잡은 손가락 때문에 다른 손가락까지 굳어요.",
                ["롱노트 잡은 손가락은 힘을 빼고 얹어 두기만, 다른 손가락만 박자에 맞추는 연습을 하세요."])
    if k == "jack_drift":
        return ("연타가 이어질수록 %s" % ("늦어짐" if late else "빨라짐"),
                "누를 때마다 조금씩 밀려요." if late else "박자보다 빨리 달려요.",
                ["가볍게 튕기듯 치세요."] if late else ["연타 개수를 소리로 세면서 치세요."])
    if k == "long_press":
        if late:
            return ("키를 오래 누른 다음 노트가 늦음",
                    "키를 꾹 누르고 늦게 떼면 손가락이 돌아올 시간이 모자라 같은 레인 다음 누름이 밀려요.",
                    ["누르자마자 떼는 느낌으로 짧게 치세요. 누르는 힘보다 빨리 떼는 걸 의식하세요."])
        return "키를 오래 누른 다음 노트가 오히려 일찍", "", []
    return k, "", []


def pooled(hist, mode, cur_flags=(), last=10):
    """같은 모드 최근 곡들을 합쳐 다시 검정: 곡 하나로는 노트가 적어 확실하지 않던 버릇을 찾는다"""
    H = [h for h in hist if h.get("mode") == mode and h.get("ms")][-last:]
    if len(H) < 3:
        return []
    acc = defaultdict(list)
    for h in H:
        for k, v in h["ms"].items():
            if v[1] > 0:
                acc[k].append(v)
    out = []
    for k, vs in acc.items():
        if len(vs) < 3:
            continue
        w = np.array([1.0 / v[1] ** 2 for v in vs])
        d = float((w * np.array([v[0] for v in vs])).sum() / w.sum())
        se = float(1.0 / math.sqrt(w.sum()))
        n = int(sum(v[2] for v in vs))
        if abs(d) >= 3.0 and abs(d) / se >= 3.5:
            out.append((k, d, se, n, len(vs)))
    out.sort(key=lambda x: -abs(x[1]) / x[2])
    return out


# ------------------------------------------------------------------ 기록 (곡별 요약) — 오답 다시 보기용
def history_path():
    from .config import app_dir
    return os.path.join(app_dir(), "coach_history.json")


def load_history():
    try:
        with open(history_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def save_history(h):
    try:
        p = history_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(h[-400:], f, ensure_ascii=False)
        os.replace(tmp, p)
    except Exception:
        pass


def summary(rep, adv, judge="normal", start_t=None):
    T = rep["tags"]
    return {"when": time.strftime("%Y-%m-%d %H:%M"), "ts": time.time(), "mode": rep["mode"], "judge": judge,
            "start": start_t, "n": rep["n"], "mean": round(rep["all"]["mean"], 2), "sd": round(rep["all"]["sd"], 2),
            "out": round(rep["all"]["out"], 4), "miss": rep["all"]["miss"],
            "tags": {t: [T[t]["n"], round(T[t]["mean"], 1), round(T[t]["sd"], 1), T[t]["miss"]]
                     for t in TAG_ORDER if T[t]["n_all"] > 0},
            "flags": [x["id"] for x in adv if x["kind"] == "fix"][:12],
            "ms": measures(rep),
            "fx": {x["id"]: [x["title"], x["metric"], x["goal"]] for x in adv if x["kind"] == "fix"}}


def record(hist, rep, adv, judge="normal", start_t=None):
    s = summary(rep, adv, judge, start_t)
    if start_t is not None:
        for i, h in enumerate(hist):
            if h.get("start") is not None and abs(h["start"] - start_t) < 3000.0 and h.get("mode") == s["mode"]:
                hist[i] = s
                return hist
    hist.append(s)
    return hist


def _flag_tag(fid):
    """오답 id → 그 오답을 확인하는 데 필요한 패턴 (이번 곡에 그 패턴이 적으면 '확인 못 함')"""
    for t in TAG_ORDER:
        if fid.startswith("pat_%s_" % t) or fid in ("sd_" + t, "miss_" + t):
            return t
    return {"rush": "burst", "lag": "burst", "burst_sd": "burst", "jack_drift": "jack", "trill_hand": "trill",
            "stair_dir": "stair", "chord_spread": "chord2", "chord_lr": "chord2", "long_press": "jack"}.get(fid)


def review(prev, rep, adv):
    """지난 오답 다시 보기: 지난 곡에서 나온 오답이 이번 곡에서 어떻게 됐는지"""
    out = []
    if not prev:
        return out
    now = {x["id"]: x for x in adv if x["kind"] == "fix"}
    for fid, v in (prev.get("fx") or {}).items():
        title, met = v[0], v[1]
        if fid in now:
            m2 = now[fid]["metric"]
            if met is not None and m2 is not None:
                better = abs(m2) < abs(met) * 0.8
                out.append(("남음", title, "지난번 %s → 이번 %s%s" % (met, m2, " (나아지는 중)" if better else "")))
            else:
                out.append(("남음", title, "이번 곡에도 나왔어요"))
        else:
            tg = _flag_tag(fid)
            if tg and rep["tags"].get(tg, {}).get("n", 0) < MIN_N:
                out.append(("모름", title, "이번 곡엔 그 패턴이 적어 확인하지 못했어요"))
            else:
                out.append(("해결", title, "이번 곡에선 나오지 않았어요"))
    order = {"남음": 0, "해결": 1, "모름": 2}
    out.sort(key=lambda x: order[x[0]])
    return out


def trend_advice(hist, mode=None):
    """최근 기록 흐름: 꾸준한 오답, 흔들림 추이, 오래 칠 때 나빠지는지"""
    H = [h for h in hist if (mode is None or h.get("mode") == mode)]
    out = []
    if len(H) < 3:
        return out, H
    recent = H[-10:]
    cnt = defaultdict(int)
    names = {}
    for h in recent:
        for f in set(h.get("flags", [])):
            cnt[f] += 1
            names[f] = (h.get("fx") or {}).get(f, [f])[0]
    pers = [(f, c) for f, c in cnt.items() if c >= 3 and c >= len(recent) * 0.4]
    pers.sort(key=lambda x: -x[1])
    if pers:
        out.append(("자주 틀리는 문제 (최근 %d곡)" % len(recent),
                    ", ".join("%s %d곡" % (names[f], c) for f, c in pers[:6]),
                    "한 곡에서만 나온 건 곡 탓일 수 있지만, 여러 곡에서 반복되면 습관이에요. 이것부터 연습하세요."))
    same = [h for h in H if h.get("judge") == H[-1].get("judge")]
    if len(same) >= 10:
        a = [h["sd"] for h in same[-5:]]
        b = [h["sd"] for h in same[-10:-5]]
        d = float(np.mean(a) - np.mean(b))
        out.append(("흔들림 추이 (같은 판정, 최근 5곡 vs 그 전 5곡)",
                    "±%.1f → ±%.1f ms (%s)" % (np.mean(b), np.mean(a), "좋아짐" if d < -0.5 else "나빠짐" if d > 0.5 else "비슷함"),
                    "곡 난이도에 따라 달라지니 몇 곡 단위로 길게 보세요. 단기적으로는 들쭉날쭉해도 길게 보면 늘어요."))
        ms = [h["mean"] for h in same[-10:]]
        mm, sm = float(np.mean(ms)), float(np.std(ms, ddof=1))
        if abs(mm) >= 3.0 and sm > 0 and abs(mm) / (sm / math.sqrt(len(ms))) >= 3.0:
            out.append(("곡마다 같은 쪽으로 치우침", "최근 10곡 평균 %s ms (곡끼리 차이 ±%.1f)." % (fmt(mm), sm),
                        "여러 곡에서 꾸준히 같은 쪽이면 영점 문제일 수 있어요. [영점 보정 (결과 화면)]의 추천을 확인하세요."))
    sess, cur = [], [H[0]]
    for h in H[1:]:
        if h.get("ts") and cur[-1].get("ts") and h["ts"] - cur[-1]["ts"] <= 1200:
            cur.append(h)
        else:
            sess.append(cur)
            cur = [h]
    sess.append(cur)
    long_s = [s for s in sess if len(s) >= 8]
    if long_s:
        s = long_s[-1]
        k = len(s) // 3
        early = float(np.mean([h["sd"] for h in s[:k]]))
        late = float(np.mean([h["sd"] for h in s[-k:]]))
        if late >= early * 1.15 and late - early >= 1.5:
            out.append(("오래 칠수록 흔들림이 커짐 (한 번에 %d곡)" % len(s),
                        "처음 %d곡 ±%.1f → 마지막 %d곡 ±%.1f ms." % (k, early, k, late),
                        "쉬지 않고 오래 치면 지쳐서 정확도가 떨어져요. 1~2시간 집중 후 쉬고, 나눠서 치는 편이 학습에도 좋아요."))
    return out, H


GENERAL_TIPS = [
    ("판정 연습 곡 고르기",
     "판정 연습은 정확도 95~99%가 나오는 곡(자기 레벨보다 1~2단계 낮은 곡)에서. 90% 초반이 나오는 곡은 처리력 연습용이에요. "
     "한 곡 반복과 여러 곡 돌리기는 섞어서 하세요.",
     ["dc_basic", "dc_setup", "dc_speed", "dc_lowlv", "dc_rotate"]),
    ("귀로 확인하기",
     "가끔 화면 말고 자기 키음이 음악 박자에 붙는지 들어 보세요. 소리를 끄고 흥얼거리며 친 걸 녹음해 들어 보는 방법도 있어요.",
     ["arca_sound", "arca_hum"]),
    ("인디케이터는 가끔 끄고 쳐 보세요",
     "매 노트 결과를 보여주는 피드백에 계속 기대면 실제 감각이 덜 자라요. 가끔 끄고 친 뒤 곡이 끝나고 이 오답노트를 보세요.",
     ["guidance"]),
    ("나눠서, 꾸준히",
     "같은 시간이면 하루에 몰아서 길게보다 여러 날 나눠서 하는 편이 실력이 잘 남아요. 연습 뒤 잠도 손 기억을 굳히는 데 도움이 된다는 연구가 많아요.",
     ["spacing", "sleep"]),
    ("설정은 한 번 정하면 그대로",
     "배속·영점은 패턴마다 바꾸지 말고 한 값으로 꾸준히. 특정 패턴에서만 치우치는 건 설정이 아니라 습관 문제예요.",
     ["etterna", "sdvx_early", "arca_early", "iidx_judge"]),
    ("판정은 쉬운 곡으로 맞추고, 정하면 몸을 맞추기",
     "영점·판정은 노트 수가 적당한 쉬운 곡(풀콤이 여유 있는 곡) 기준으로 맞추세요. 빽빽한 곡은 노트를 줍느라 시선이 올라가 FAST가 늘고, "
     "거기에 맞춰 옮기면 쉬운 곡에서 반대로 어긋나요. FAST/SLOW 비율은 참고만 하고, 한 번 정하면 그날 컨디션·곡마다 바꾸지 말고 "
     "시선·배속으로 몸을 맞추세요. 자동으로 맞춰 주는 값을 그대로 믿지 말고 내 감각과 함께 보세요.",
     ["iidx_judge", "iidx_fasttrap", "lr2_setup"]),
    ("FAST/SLOW는 그날 상태도 알려줘요",
     "집중이 잘 되는 날은 FAST 쪽, 지치거나 졸리거나 시선이 너무 아래면 SLOW 쪽으로 쏠리기 쉬워요. 그날만 SLOW가 많으면 설정보다 휴식이 먼저예요.",
     ["iidx_cond", "iidx_slow"]),
    ("시선은 한 곳에",
     "숙련자는 화면 한 지점에 시선을 고정하고 쳐요. 노트가 0.4~0.6초쯤 보이는 배속이 무난하고, 패턴을 눈으로 쫓아가면 실수가 연쇄로 터져요. "
     "가리개(레인 커버)도 도움이 돼요. 배속은 버틸 수 있는 한계보다 조금 느리게 정해 두면 보는 힘이 늘어요.",
     ["gaze", "arca_gaze", "lane_cover", "iidx_yoko", "iidx_slow", "bms_hstk"]),
    ("녹화해서 돌려 보기",
     "연타를 너무 세게 누르는 버릇 같은 건 감각만으로는 알아채기 어려워요. 가끔 플레이를 녹화해 느리게 돌려 보고, "
     "연습할 땐 그날 과제와 칠 곡을 미리 정해 두세요.",
     ["iidx_deliberate", "bms_elma"]),
    ("가끔은 좁은 판정으로",
     "가끔 게임 판정을 하드로 바꿔 쉬운 곡을 치면 대충 치는 버릇이 줄어요. BMS를 한다면 4·5·6키 난이도표에서 약한 패턴 곡만 골라 연습할 수 있어요.",
     ["bms_tight", "bms_uya", "bms_dmrv"]),
    ("같은 곡은 하루 두 번까지, 며칠 치면 하루 쉬기",
     "같은 곡을 계속 반복하면 얻는 게 줄어요. 여러 곡을 돌리고, 하루 1.5~2시간 안에서, 2~3일 치면 하루 쉬세요. "
     "같은 종류(예: 강한 연타)만 매일 하지 말고 날마다 나누세요.",
     ["bms_elma", "bms_hstk", "bms_jose"]),
    ("PC 설정 먼저",
     "VSync·G-Sync 끄기, 프레임 제한 해제, 오디오 버퍼 낮게. 판정·출력 설정은 영점 맞출 때 말고는 건드리지 마세요.",
     ["dc_setup", "lr2_setup"]),
    ("손 모양",
     "손가락은 살짝 세우고 손등 힘은 빼세요. 너무 힘을 빼면 오히려 늦게 눌려요. 중간 속도에선 손목 힘을 완전히 빼고, "
     "빠른 곡에서만 위치를 잡을 만큼 힘을 주세요.",
     ["dc_finger", "iidx_relax"]),
    ("손이 아프면 멈추기",
     "손목이 타는 느낌이면 쉬고 풀어 주고, 저리거나 아프면 그날은 그만하세요. 1~2시간마다 일어나 몇 분 움직이세요. "
     "팔꿈치를 너무 낮추고 손목만 꺾어 치면 건초염 위험이 있어요.",
     ["health", "oh_4k", "bms_elma", "bms_uya"]),
]


# ------------------------------------------------------------------ 오답노트 글
def report_text(rep, adv, cfg=None, trend=None, title="이 곡", rev=None, pool=None):
    """코치 창에 보여줄 글: [(스타일, 글)]  스타일 = h(제목) t(소제목) p(본문) k(항목 이름) dim(작게) mono(표)"""
    L = []
    if not rep:
        return [("h", "분석할 곡이 없어요"),
                ("p", "곡을 끝까지 친 뒤(노트 30개 이상)에 다시 눌러 주세요. 기록 보기 창에서는 고른 곡을 분석해요.")]
    a = rep["all"]
    mode = rep["mode"]
    L.append(("h", "오답노트 · %s · %s · %d노트 · %s" % (title, mode, rep["n"], mmss(rep["dur"] * 1000))))
    g = rep.get("game")
    gt = (" · 게임 결과 100%%아님 %d, BREAK %d" % (sum(g[1:11]), g[11])) if g else ""
    L.append(("p", "평균 %s ms · 흔들림 ±%.1f ms · 100%% 밖 %.1f%%%s" % (fmt(a["mean"]), a["sd"], a["out"] * 100, gt)))
    gd = rep.get("grid") or {}
    if gd.get("bpms"):
        line = "박자: BPM 약 %s · 노트 %.0f%%가 박자 격자에 맞아 패턴을 '몇 분음표'로 나눴어요" % (
            " / ".join(str(b) for b in gd["bpms"]), gd.get("ok", 0) * 100)
        jk = [r for r in gd.get("jack", []) if r[0] != "기타"][:2]
        if jk:
            line += " · 연타는 " + ", ".join("%s %d개(간격 %.0f ms)" % r for r in jk)
        if rep.get("rescued"):
            line += " · 확신도가 낮던 노트 %d개를 키 누름으로 확인해 살림" % rep["rescued"]
        L.append(("dim", line + "."))
    L.append(("dim", "평균 = 치우침(음수 = 일찍), 흔들림 = 일관성. 시간은 첫 노트부터 잰 곡 안 위치예요."))

    if rev:
        L.append(("h", "지난 오답 다시 보기"))
        mark = {"남음": "✗ 아직", "해결": "✓ 해결", "모름": "· 확인 못 함"}
        for st, ti, det in rev:
            L.append(("p", "%s  %s — %s" % (mark[st], ti, det)))

    fixes = [x for x in adv if x["kind"] == "fix"]
    L.append(("h", "이번 곡 오답 (중요한 순서)"))
    if not fixes:
        L.append(("p", "눈에 띄게 틀린 곳이 없어요. 아래 패턴표에서 세부를 보세요."))
    for i, x in enumerate(fixes[:8]):
        L.append(("t", "오답 %d. %s" % (i + 1, x["title"])))
        wh = x["where"] + (("  —  " + ", ".join(x["spans"])) if x["spans"] else "")
        L.append(("k", "어디서", wh))
        L.append(("k", "무엇이 틀렸나", x["wrong"]))
        if x["lack"]:
            L.append(("k", "부족한 점", x["lack"]))
        for j, d in enumerate(x["todo"]):
            L.append(("k", "이렇게 고치기" if j == 0 else "", "%d) %s" % (j + 1, d)))
        if x["goal"]:
            L.append(("k", "다음 목표", x["goal"]))
        if x["src"]:
            L.append(("dim", "근거: " + " / ".join(SOURCES[s][1] for s in x["src"] if s in SOURCES)))
    if len(fixes) > 8:
        L.append(("dim", "그 밖에 작은 오답 %d개: %s" % (len(fixes) - 8, ", ".join(x["title"] for x in fixes[8:]))))
    for x in [x for x in adv if x["kind"] == "info"]:
        L.append(("t", "참고: " + x["title"]))
        L.append(("p", x["wrong"] + " " + " ".join(x["todo"])))
    goods = [x for x in adv if x["kind"] == "good"]
    if goods:
        L.append(("h", "잘하고 있는 점"))
        for x in goods:
            L.append(("p", "✓ %s — %s" % (x["title"], x["wrong"])))
    if pool:
        cur = [x["id"] for x in fixes]
        alias = {"hand": "hand_bias", "burst": ("rush", "lag")}

        def shown(k):
            a_ = alias.get(k, k)
            a_ = a_ if isinstance(a_, tuple) else (a_,)
            return any(c == q or c.startswith(q + "_") for c in cur for q in a_)
        pool = [x for x in pool if not shown(x[0])]
    if pool:
        L.append(("h", "여러 곡 합쳐 본 숨은 버릇"))
        L.append(("dim", "곡 하나로는 노트가 적어 확실하지 않던 차이를 최근 같은 모드 곡들로 합쳐서 다시 확인한 거예요."))
        for k, d, se, n, ns in pool[:6]:
            ti, lack, todo = _mname(k, d, mode, cfg)
            L.append(("t", ti))
            L.append(("k", "무엇이 틀렸나", "최근 %d곡을 합치면 %s ms (노트 %d개, 오차 범위 ±%.1f)." % (ns, fmt(d), n, 2 * se)))
            if lack:
                L.append(("k", "부족한 점", lack))
            for j, t_ in enumerate(todo):
                L.append(("k", "이렇게 고치기" if j == 0 else "", "%d) %s" % (j + 1, t_)))
    if trend:
        L.append(("h", "최근 기록 흐름"))
        for t, d, c in trend:
            L.append(("t", t))
            L.append(("p", d))
            L.append(("dim", c))
    L.append(("h", "패턴별 판정 (이 곡의 나머지 노트와 비교)"))
    L.append(("row", ("패턴", "노트", "평균", "흔들림", "나머지와 차이", "놓침")))
    for t in TAG_ORDER:
        s = rep["tags"][t]
        if s["n_all"] < 5:
            continue
        L.append(("row", (TAG_NAME[t], str(s["n_all"]), fmt(s["mean"]), "±%.1f" % s["sd"], fmt(s["d"]), str(s["miss"]))))
    L.append(("h", "손·레인별"))
    L.append(("row", ("구분", "노트", "평균", "흔들림", "", "놓침")))
    for hd, s in rep["hands"].items():
        L.append(("row", (hd, str(s["n_all"]), fmt(s["mean"]), "±%.1f" % s["sd"], "", str(s["miss"]))))
    for ch, s in rep["lanes"].items():
        L.append(("row", (lane_label(mode, ch, cfg), str(s["n_all"]), fmt(s["mean"]), "±%.1f" % s["sd"], "",
                          str(s["miss"]))))
    C = rep["chord"]
    if C["n"]:
        L.append(("dim", "동치 %d개: 누른 시각 차이 중앙값 %.1f ms (손가락이 서로 따로 흔들릴 때 예상 %.1f ms)."
                  % (C["n"], C["spread"], C["expect"])))
    dwi = rep.get("dwell") or {}
    if dwi.get("med") is not None:
        L.append(("dim", "키를 누르고 있던 시간 중앙값 %.0f ms%s." % (
            dwi["med"], (" · 같은 레인 다음 노트가 빨리 오는 곳에서는 간격의 %.0f%%, 절반 넘게 누른 경우 %.0f%%"
                         % (dwi["fast_ratio_med"] * 100, (dwi["long_frac"] or 0) * 100))
            if dwi.get("fast_ratio_med") is not None else "")))
    L.append(("h", "늘 지키면 좋은 것"))
    for t, d, s in GENERAL_TIPS:
        L.append(("t", t))
        L.append(("p", d))
        L.append(("dim", "근거: " + " / ".join(SOURCES[k][1] for k in s)))
    L.append(("h", "읽는 법과 한계"))
    L.append(("dim", "패턴은 기록된 노트 도달로 채보를 다시 만들어 나눈 것이라 실제 채보 이름과 조금 다를 수 있어요. "
                     "노트 20개 미만인 패턴은 비교하지 않고, 우연으로 보기 어려운 차이(약 99%)만 오답으로 적어요. "
                     "'부족한 점'은 데이터로 확인한 게 아니라 공략·연구에서 흔한 원인이에요. 롱노트 끝(뗄 때) 판정은 재지 않아요. "
                     "게임 결과보다 분석기 놓침이 많으면 놓침 관련 오답은 빼요."))
    return L


def sources_text():
    L = [("h", "오답노트의 근거 자료")]
    for k, (desc, name, url) in SOURCES.items():
        L.append(("t", name))
        L.append(("p", desc))
        L.append(("dim", url))
    return L


# ------------------------------------------------------------------ 세션·기록에서 바로 쓰는 입구
def from_session(ses, start=None, end=None):
    """측정 세션의 마지막 곡 보고서 → (보고서, 곡 시작 시각)"""
    m = ses.active_mode()
    g = ses._last_song_hits()
    if g is None or len(g[2]) < 30:
        return None, None
    start, end, hs = g
    m = hs[0][1] or m
    if m is None:
        return None, None
    lanes = LANES_OF.get(m, 6)
    raw = ses.raw[-150000:]           # 측정 스레드가 계속 덧붙이므로 복사본으로 (곡 하나는 보통 3만 줄 안쪽)
    low = []
    arrs, keys, vch = collect(raw, start, end, lanes, live=True, low=low)
    hits = {(str(h[2]), round(h[0], 2)): h[3] for h in hs if h[1] == m}
    arrs = [x for x in arrs if start - 1500 <= x[0] <= end + 500]
    low = [x for x in low if start - 1500 <= x[0] <= end + 500]
    arrs, hits, nres = rescue_arrs(arrs, low, keys, hits, ses.cfg, m)
    js_c = float(ses.cfg.get("judge_setting_ms", 0)) + float(ses.cfg.get("calib_ms", 0.0))
    rep = analyze(arrs, hits, keys, vch, ses.cfg, m, js_c)
    if rep is not None:
        rep["rescued"] = nres
        rt = getattr(ses, "_last_result_t", None)
        if rt is not None and end <= rt <= end + 120000.0 and ses.results:
            rep["game"] = ses.results[-1].get("game")
    return rep, start


def from_song(sg, cfg):
    """기록 보기([기록 불러오기])의 곡 하나 보고서"""
    arrs = sg.get("arrs") or []
    if not arrs:
        return None
    hits = {(str(h[2]), round(h[0], 2)): h[3] for h in sg["hits"]}
    js_c = float(sg.get("js", 0.0)) + float(sg.get("calib", 0.0))
    arrs, hits, nres = rescue_arrs(arrs, sg.get("arrs_low") or [], sg.get("keys") or [], hits, cfg, sg["mode"])
    rep = analyze(arrs, hits, sg.get("keys") or [], sg.get("vchg") or [], cfg, sg["mode"], js_c)
    if rep is not None:
        rep["game"] = sg.get("result")
        rep["rescued"] = nres
    return rep
