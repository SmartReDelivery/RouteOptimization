# insertion_heuristic.py
import random
from typing import List, Tuple, Optional
from .utils import (
    Location,
    Route,
    TimeWindows,
    DEPOT_INDEX,
    calculate_travel_time,
    check_time_window_feasibility,
    calculate_distance,
)


def find_best_insertion(
    current_route: Route,
    customer_to_insert: int,
    locations: List[Location],
    time_windows: TimeWindows,
    V: float,
) -> Optional[Tuple[int, float, Route]]:
    """
    指定された顧客を経路に挿入する最適な位置(コスト増加最小かつ時間枠適合)を見つける。

    Args:
        current_route: 現在の経路 [0, ..., 0]
        customer_to_insert: 挿入する顧客のインデックス
        locations: 全地点の座標リスト
        time_windows: 各配送先の許容時間帯リスト
        V: 移動速度

    Returns:
        Optional[Tuple[int, float, Route]]:
            - 挿入位置のインデックス (挿入後、customer_to_insertはこのインデックスになる)
            - 挿入によるコスト増加分
            - 挿入後の新しい経路
        挿入可能な位置がない場合は None
    """
    best_insertion_index = -1
    min_cost_increase = float("inf")
    best_new_route = None

    # 現在の経路コスト (時間枠無視)
    # current_cost = calculate_route_cost(current_route, locations, V) # これは不要かも

    # 各辺 (i, j) の間に customer_to_insert を挿入してみる
    for i in range(len(current_route) - 1):
        prev_node = current_route[i]
        next_node = current_route[i + 1]

        # 挿入後の経路候補を作成
        new_route_candidate = (
            current_route[: i + 1] + [customer_to_insert] + current_route[i + 1 :]
        )

        # 1. 時間枠の適合性をチェック
        is_feasible, _, _ = check_time_window_feasibility(
            new_route_candidate, locations, time_windows, V
        )

        if is_feasible:
            # 2. コスト増加分を計算
            # dist(i, k) + dist(k, j) - dist(i, j)
            cost_increase = (
                calculate_travel_time(
                    locations[prev_node], locations[customer_to_insert], V
                )
                + calculate_travel_time(
                    locations[customer_to_insert], locations[next_node], V
                )
                - calculate_travel_time(locations[prev_node], locations[next_node], V)
            )

            # 最良の挿入位置を更新
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_insertion_index = i + 1
                best_new_route = new_route_candidate
                # print(f"  Feasible insertion at index {i+1}, cost increase {cost_increase:.2f}") # Debug

    if best_new_route:
        return best_insertion_index, min_cost_increase, best_new_route
    else:
        # print(f"  No feasible insertion found for customer {customer_to_insert}") # Debug
        return None


def build_initial_route_insertion(
    locations: List[Location],
    time_windows: TimeWindows,
    V: float,
    start_node_strategy: str = "farthest",  # or "random"
) -> Optional[Route]:
    """
    挿入法を用いて初期経路を構築する。

    Args:
        locations: デポを含む全地点の座標リスト
        time_windows: 各配送先の許容時間帯リスト
        V: 移動速度
        start_node_strategy: 最初の顧客選択戦略 ('farthest' or 'random')

    Returns:
        Optional[Route]: 構築された経路。実行不可能な場合は None。
    """
    num_customers = len(locations) - 1
    if num_customers == 0:
        return [DEPOT_INDEX, DEPOT_INDEX]

    unvisited_customers = list(range(1, num_customers + 1))
    depot = locations[DEPOT_INDEX]

    # 1. 最初の顧客を選択
    first_customer = -1
    if start_node_strategy == "farthest":
        max_dist = -1
        for cust_idx in unvisited_customers:
            dist = calculate_distance(depot, locations[cust_idx])
            if dist > max_dist:
                max_dist = dist
                first_customer = cust_idx
    elif start_node_strategy == "random":
        first_customer = random.choice(unvisited_customers)
    else:
        raise ValueError(
            f"Invalid start_node_strategy: {start_node_strategy}. Use 'farthest' or 'random'."
        )

    if first_customer == -1:
        return None  # 起こらないはず

    # 2. 初期経路を作成 [デポ -> 最初の顧客 -> デポ]
    initial_route_candidate = [DEPOT_INDEX, first_customer, DEPOT_INDEX]
    is_feasible, _, _ = check_time_window_feasibility(
        initial_route_candidate, locations, time_windows, V
    )

    if not is_feasible:
        print(f"Error: Initial route [0, {first_customer}, 0] is not feasible.")
        # 代替の最初の顧客を探すか、失敗とする
        # ここでは単純に失敗とする
        return None

    current_route = initial_route_candidate
    unvisited_customers.remove(first_customer)
    # print(f"Initial feasible route: {current_route}") # Debug

    # 3. 残りの顧客を挿入
    while unvisited_customers:
        customer_to_insert = -1
        # 次に挿入する顧客を選択 (例: 未訪問の中からランダム)
        # より洗練された戦略 (例: 現在の経路から最も遠い顧客) も可能
        customer_to_insert = random.choice(unvisited_customers)
        # print(f"\nTrying to insert customer {customer_to_insert} into {current_route}") # Debug

        insertion_result = find_best_insertion(
            current_route, customer_to_insert, locations, time_windows, V
        )

        if insertion_result:
            _, _, new_route = insertion_result
            current_route = new_route
            unvisited_customers.remove(customer_to_insert)
            # print(f"Successfully inserted. New route: {current_route}") # Debug
        else:
            # 挿入できる場所がなかった場合
            print(
                f"Error: Could not find a feasible insertion position for customer {customer_to_insert}."
            )
            # この顧客を挿入できなかった -> 全体として失敗とする
            # (より頑健な実装では、別の未訪問顧客を試すなど)
            return None

    return current_route
