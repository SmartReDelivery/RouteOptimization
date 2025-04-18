# input_generator.py
import random
from typing import List
from .utils import (
    Location,
    TimeWindows,
    TimeWindowAllowedList,
    HOURS_IN_WINDOW,
)


def generate_locations(N: int, W: float, H: float) -> List[Location]:
    """デポ(0,0)とN個のランダムな配送先座標を生成する"""
    locations: List[Location] = [(0.0, 0.0)]  # Index 0 is the depot
    for _ in range(N):
        locations.append((random.uniform(0, W), random.uniform(0, H)))
    return locations


def generate_time_windows(N: int) -> TimeWindows:
    """
    N個の各配送先にランダムなタイムウィンドウ(8時-21時の1時間スロット)を割り当てる。
    各配送先には少なくとも1つの許容時間帯があるようにする。
    """
    time_windows: TimeWindows = []
    num_slots = HOURS_IN_WINDOW  # 8-9, 9-10, ..., 20-21 の 13 スロット

    for _ in range(N):
        while True:
            # ランダムに各時間帯を許容(1)または非許容(0)にする
            # (例: 各時間帯が許容される確率を 50% とする)
            allowed_list: TimeWindowAllowedList = [
                random.choice([0, 1]) for _ in range(num_slots)
            ]
            # 少なくとも1つは許容時間帯があることを保証
            if sum(allowed_list) > 0:
                time_windows.append(allowed_list)
                break
    return time_windows
