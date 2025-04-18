# local_search.py
import random
import itertools
from typing import List
from .utils import (
    Location,
    Route,
    TimeWindows,
    check_time_window_feasibility,
)


def apply_2opt(route: Route, i: int, j: int) -> Route:
    """経路に2-opt操作を適用する"""
    if i >= j:
        # print(f"Warning: 2-opt indices invalid i={i}, j={j}")
        return route  # 何もしない
    # 経路の一部 route[i+1:j+1] を逆順にする
    new_route = route[: i + 1] + route[j:i:-1] + route[j + 1 :]
    return new_route


def improve_route_local_search(
    initial_route: Route,
    locations: List[Location],
    time_windows: TimeWindows,
    V: float,
    max_iterations: int = 100,  # 改善が見られなくても終了する最大反復回数
) -> Route:
    """
    2-opt局所探索を用いて経路を改善する (First Improvement戦略)。

    Args:
        initial_route: 初期経路 [0, ..., 0]
        locations: 全地点の座標リスト
        time_windows: 各配送先の許容時間帯リスト
        V: 移動速度
        max_iterations: 最大反復回数

    Returns:
        Route: 改善された経路
    """
    current_route = list(initial_route)  # コピーを作成
    # num_locations = len(locations)
    # num_customers = len(current_route) - 2  # デポを除く顧客数

    # 初期コスト (時間枠考慮)
    feasible_init, current_cost, _ = check_time_window_feasibility(
        current_route, locations, time_windows, V
    )
    if not feasible_init:
        print("Warning: Initial route for local search is not feasible!")
        # 実行不可能な経路は改善できないのでそのまま返す
        return current_route

    print(f"Local Search starting with cost: {current_cost:.2f}")

    improvement_found = True
    iteration = 0
    while improvement_found and iteration < max_iterations:
        improvement_found = False
        iteration += 1
        # print(f"Iteration {iteration}...") # Debug

        # 2-optの辺の組み合わせを試す (デポ間の辺は変更しない)
        # i は 0 から N-1 まで (辺 (route[i], route[i+1]))
        # j は i+1 から N まで (辺 (route[j], route[j+1]))
        # デポ(0)を除外してインデックスを考える必要がある route = [0, c1, c2, ..., cn, 0] 長さ n+2
        # i は 0 から n   (route[i] と route[i+1] の間の辺)
        # j は i+1 から n (route[j] と route[j+1] の間の辺)
        indices = list(range(len(current_route) - 1))  # 0 to n
        random.shuffle(indices)  # 試す順番をランダム化

        # 辺 (route[i], route[i+1]) と (route[j], route[j+1]) を選ぶ
        # i は 0 から len-3 まで
        # j は i+2 から len-2 まで (i+1 の辺とは隣接しない)
        possible_pairs = list(itertools.combinations(range(len(current_route) - 1), 2))
        random.shuffle(possible_pairs)  # 試す順番をランダム化

        for i, j in possible_pairs:
            # i < j を保証し、隣接する辺(j=i+1)は除外
            if i > j:
                i, j = j, i  # swap
            if j == i + 1:
                continue
            # デポ発着の辺は変更しない (i=0 か j=len-2 か)
            # -> 2-opt ではデポ発着も変更して良い場合が多いが、今回は単純化のため除く
            # if i == 0 or j == len(current_route) - 2: continue

            # 2-opt操作を適用
            new_route_candidate = apply_2opt(current_route, i, j)

            # 時間枠適合性をチェック
            is_feasible, new_cost, _ = check_time_window_feasibility(
                new_route_candidate, locations, time_windows, V
            )

            if is_feasible and new_cost < current_cost:
                current_route = new_route_candidate
                current_cost = new_cost
                improvement_found = True
                # print(f"  Improved! New cost: {current_cost:.2f} with 2-opt ({i}, {j})") # Debug
                # First Improvement: 改善が見つかったら、次のイテレーションへ
                break  # 内側のループ (ペア探索) を抜ける

        # if not improvement_found:
        # print("No improvement found in this iteration.")

    print(
        f"Local Search finished after {iteration} iterations. Final cost: {current_cost:.2f}"
    )
    return current_route
