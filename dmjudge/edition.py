"""어떤 판인지: indicator.py(게임 위에 띄우는 창)가 들어 있으면 일반판, 없으면 OBS판.

OBS판에는 게임 화면 위에 창을 띄우는 코드 자체가 없다. 인디케이터는 OBS 브라우저 소스로만 볼 수 있다.
"""
import os


def is_obs():
    return not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "indicator.py"))


def name():
    return "OBS판" if is_obs() else "일반판"
