# input_generator.py
import random
from typing import List

import numpy as np

from .utils import (
    HOURS_IN_WINDOW,
    Location,
    TimeWindowAllowedList,
    TimeWindows,
)


def generate_locations(N: int, W: float, H: float, mode="random") -> List[Location]:
    """デポ(0,0)とN個の配送先座標を生成する"""
    if mode == "random":
        locations: List[Location] = [(0.0, 0.0)]  # Index 0 is the depot
        for _ in range(N):
            locations.append(
                (random.uniform(-W / 2, W / 2), random.uniform(-H / 2, H / 2))
            )
        return locations
    if mode == "double_circle":
        # 2つの同心円
        radius_sm = np.sqrt(W**2 + H**2) / 4 * 3
        radius_lg = np.sqrt(W**2 + H**2) / 2
        theta_sm = np.linspace(0, 2 * np.pi, N // 2, endpoint=False)
        theta_lg = np.linspace(0, 2 * np.pi, (N + 1) // 2, endpoint=False)
        locations: List[Location] = [(0.0, 0.0)]
        for theta in theta_sm:
            x = radius_sm * np.cos(theta)
            y = radius_sm * np.sin(theta)
            locations.append((x, y))
        for theta in theta_lg:
            x = radius_lg * np.cos(theta)
            y = radius_lg * np.sin(theta)
            locations.append((x, y))
        assert len(locations) == N + 1, "Number of locations does not match N"
        return locations
    raise ValueError(f"Unknown mode: {mode}")


def generate_time_windows(N: int, mode="random", locations=None) -> TimeWindows:
    """
    N個の各配送先にランダムなタイムウィンドウ(8時-21時の1時間スロット)を割り当てる。
    各配送先には少なくとも1つの許容時間帯があるようにする。
    """
    time_windows: TimeWindows = []
    num_slots = HOURS_IN_WINDOW  # 8-9, 9-10, ..., 20-21 の 13 スロット

    if mode == "random":
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
    if mode == "proportional":
        # 各配送先に対して、時間帯を均等に割り当てる
        n = 3  # 3時間ごとに許容時間帯を設定, ストライド: 1
        for i in range(N):
            allowed_list: TimeWindowAllowedList = [0 for _ in range(num_slots)]
            if locations is None:
                shift = i * (num_slots - n + 1) // N
            else:
                theta = np.arctan2(locations[i][1], locations[i][0])
                shift = int(
                    np.floor((np.pi / 2 - theta) / (2 * np.pi) * (num_slots - n + 1))
                ) % (num_slots - n + 1)
                # print(
                #     f"Location {locations[i]}: shift = {shift}, theta = {np.pi / 2 - theta}"
                # )
            for i in range(0, n):
                allowed_list[shift + i] = 1
            time_windows.append(allowed_list)
        return time_windows
    if mode == "all":
        # すべての時間帯を許容する
        for _ in range(N):
            allowed_list: TimeWindowAllowedList = [1 for _ in range(num_slots)]
            time_windows.append(allowed_list)
        return time_windows
    raise ValueError(f"Unknown mode: {mode}")
