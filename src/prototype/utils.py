# utils.py
import math
import random
import time
from typing import List, Optional, Tuple

# 型エイリアス
Location = Tuple[float, float]
# ルート: デポ(0)で始まりデポ(0)で終わる訪問地点インデックスのリスト
Route = List[int]
# タイムウィンドウ: 各時間帯(8-9時, 9-10時, ..., 20-21時)に配送可能かを示すリスト(0 or 1)
TimeWindowAllowedList = List[int]
# 入力データ用: 配送先ごとのタイムウィンドウリスト
TimeWindows = List[TimeWindowAllowedList]
# 各地点への到着時刻
ArrivalTimes = List[Optional[float]]

# --- 定数 ---
SERVICE_TIME = 0.0  # 各配送先でのサービス時間 (今回は0)
DEPOT_INDEX = 0
START_HOUR = 8
END_HOUR = 21
HOURS_IN_WINDOW = END_HOUR - START_HOUR  # 13時間


def calculate_distance(loc1: Location, loc2: Location) -> float:
    """2点間のユークリッド距離を計算する"""
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def calculate_travel_time(loc1: Location, loc2: Location, V: float) -> float:
    """2点間の移動時間を計算する (単位: 分)"""
    if V <= 0:
        return float("inf")
    # 速度 V を時速と仮定し、時間を分で返す
    distance = calculate_distance(loc1, loc2)
    return (distance / V) * 60.0


def get_time_window_intervals(
    loc_index: int, time_windows: TimeWindows
) -> List[Tuple[float, float]]:
    """
    指定された配送先(loc_index > 0)の許容時間帯リストを分単位で返す。
    例: [ (480.0, 540.0), (600.0, 660.0), ... ]
    """
    if loc_index == DEPOT_INDEX:
        return []  # デポには時間枠なし

    allowed_intervals = []
    # time_windows は配送先1からNに対応するため、インデックス調整
    allowed_list = time_windows[loc_index - 1]
    for i, allowed in enumerate(allowed_list):
        if allowed == 1:
            start_min = (START_HOUR + i) * 60.0
            end_min = (START_HOUR + i + 1) * 60.0
            allowed_intervals.append((start_min, end_min))
    return allowed_intervals


def check_time_window_feasibility(
    route: Route,
    locations: List[Location],
    time_windows: TimeWindows,
    V: float,
    start_time: float = START_HOUR * 60.0,
) -> Tuple[bool, float, ArrivalTimes]:
    """
    経路が時間枠制約を満たすかチェックし、総移動時間と到着時刻を返す。

    Args:
        route: デポで始まりデポで終わる訪問順リスト [0, i, j, ..., 0]
        locations: デポを含む全地点の座標リスト
        time_windows: 各配送先の許容時間帯リスト (配送先1からNまで)
        V: 移動速度 (距離単位/時)
        start_time: デポ出発時刻 (分単位)

    Returns:
        Tuple[bool, float, ArrivalTimes]:
            - 適合性 (True/False)
            - 総移動時間 (分単位)
            - 各地点への到着時刻リスト (到達不可能な場合はNone)
    """
    current_time = start_time
    total_travel_time = 0.0
    # 到着時刻リストを初期化 (Noneで埋める)
    arrival_times: ArrivalTimes = [None] * len(locations)
    arrival_times[DEPOT_INDEX] = start_time  # デポの出発時刻を到着時刻として記録

    if not route or route[0] != DEPOT_INDEX or route[-1] != DEPOT_INDEX:
        print("Error: Route must start and end at the depot (index 0).")
        return False, 0.0, arrival_times

    for i in range(len(route) - 1):
        from_loc_idx = route[i]
        to_loc_idx = route[i + 1]

        # 前の地点の出発時刻が不明な場合は計算不能
        if current_time is None:
            print(
                f"Error: Cannot determine departure time from location {from_loc_idx}"
            )
            return False, total_travel_time, arrival_times

        # 移動時間を計算
        travel_t = calculate_travel_time(
            locations[from_loc_idx], locations[to_loc_idx], V
        )
        if travel_t == float("inf"):
            print(
                f"Error: Infinite travel time from {from_loc_idx} to {to_loc_idx} (V={V})"
            )
            return False, total_travel_time, arrival_times

        total_travel_time += travel_t
        arrival_time = current_time + travel_t
        arrival_times[to_loc_idx] = arrival_time  # 到着時刻を記録

        # 次の地点がデポ以外の場合、時間枠をチェック
        if to_loc_idx != DEPOT_INDEX:
            allowed_intervals = get_time_window_intervals(to_loc_idx, time_windows)

            # 許容時間帯がない場合は実行不可能
            if not allowed_intervals:
                # print(f"Debug: Location {to_loc_idx} has no allowed time window.")
                return False, total_travel_time, arrival_times

            # 到着時刻が許容される最も遅い時間よりも後かチェック
            latest_possible_end = max(interval[1] for interval in allowed_intervals)
            if arrival_time >= latest_possible_end:
                # print(f"Debug: Arrival at {to_loc_idx} ({arrival_time:.2f}) is after latest possible end ({latest_possible_end:.2f})")
                return False, total_travel_time, arrival_times

            # サービス可能な最も早い開始時刻を見つける
            possible_starts = []
            for start_min, end_min in allowed_intervals:
                # この区間でサービス開始できるか？
                # 開始可能時刻 = max(到着時刻, 区間開始時刻)
                potential_start = max(arrival_time, start_min)
                # 開始可能時刻が区間終了時刻より前ならOK
                if potential_start < end_min:
                    possible_starts.append(potential_start)

            if not possible_starts:
                # どの時間枠でもサービスを開始できない
                # print(f"Debug: Cannot start service at {to_loc_idx} arriving at {arrival_time:.2f}. Allowed: {allowed_intervals}")
                return False, total_travel_time, arrival_times

            # 可能な開始時刻の中で最も早いものを選択
            service_start_time = min(possible_starts)
            # wait_time = service_start_time - arrival_time

            # 出発時刻 = サービス開始時刻 + サービス時間
            departure_time = service_start_time + SERVICE_TIME
            current_time = departure_time

        else:  # 次の地点がデポの場合
            # デポには時間枠制約はないと仮定
            # デポ到着時刻が21時(1260分)を超えるかなどの制約はここに追加可能
            # if arrival_time > END_HOUR * 60.0:
            #     return False, total_travel_time, arrival_times
            current_time = arrival_time  # デポ到着で終了なので次の出発はない

    # 全ての地点を時間枠内に訪問できたら成功
    return True, total_travel_time, arrival_times


def calculate_route_cost(route: Route, locations: List[Location], V: float) -> float:
    """経路の総移動時間(コスト)を計算する (時間枠は考慮しない)"""
    total_time = 0.0
    for i in range(len(route) - 1):
        from_loc = locations[route[i]]
        to_loc = locations[route[i + 1]]
        total_time += calculate_travel_time(from_loc, to_loc, V)
    return total_time


def count_time_window_violations(
    route: Route,
    locations: List[Location],
    time_windows: TimeWindows,
    V: float,
    start_time: float = START_HOUR * 60.0,
    max_wait_time: float = 10.0,
) -> int:
    """
    指定された経路において、時間枠制約を満たせない配送先の数をカウントする。
    途中で違反があっても計算を続け、総違反数を返す。

    Args:
        route: デポで始まりデポで終わる訪問順リスト [0, i, j, ..., 0]
        locations: デポを含む全地点の座標リスト
        time_windows: 各配送先の許容時間帯リスト (配送先1からNまで)
        V: 移動速度 (距離単位/時)
        start_time: デポ出発時刻 (分単位)
        max_wait_time: 最大待機時間 (分単位)

    Returns:
        int: 時間枠制約違反の配送先数
    """
    current_time: Optional[float] = start_time  # 計算不能になった場合 None になる可能性
    violation_count = 0

    # --- 入力チェック ---
    if not route or route[0] != DEPOT_INDEX or route[-1] != DEPOT_INDEX:
        num_customers = len(locations) - 1
        print(
            "Warning: Invalid route structure provided to count_time_window_violations."
        )
        # 不正な経路の場合、全顧客が違反とみなす
        return num_customers if num_customers > 0 else 0

    if V <= 0:
        num_customers = len(locations) - 1
        print("Warning: Non-positive velocity (V) provided.")
        # 速度が0以下なら全顧客訪問不可
        return num_customers if num_customers > 0 else 0

    # --- 経路を順にたどる ---
    for i in range(len(route) - 1):
        from_loc_idx = route[i]
        to_loc_idx = route[i + 1]

        # 前の地点からの出発時刻が計算不能なら、以降も計算不能
        if current_time is None:
            # この地点が顧客なら違反カウント
            if to_loc_idx != DEPOT_INDEX:
                violation_count += 1
            continue  # 次の地点へ（current_time は None のまま）

        # --- 移動時間と到着時刻の計算 ---
        travel_t = calculate_travel_time(
            locations[from_loc_idx], locations[to_loc_idx], V
        )
        # 移動時間が無限大の場合（V=0など）、以降の地点は到達不能
        if travel_t == float("inf"):
            print(
                f"Warning: Infinite travel time from {from_loc_idx} to {to_loc_idx}. Subsequent locations marked as violations."
            )
            current_time = None  # 以降の計算を不能にする
            # この地点が顧客なら違反カウント
            if to_loc_idx != DEPOT_INDEX:
                violation_count += 1
            continue  # 次の地点へ

        arrival_time = current_time + travel_t

        # --- 時間枠チェック (次の地点がデポ以外の場合) ---
        departure_time = arrival_time  # 違反した場合やデポの場合のデフォルト出発時刻
        is_violation = False

        if to_loc_idx != DEPOT_INDEX:
            allowed_intervals = get_time_window_intervals(to_loc_idx, time_windows)

            # 1. 許容時間帯が存在しない場合 -> 違反
            if not allowed_intervals:
                is_violation = True
            else:
                # 2. 到着時刻が最も遅い許容終了時刻を過ぎている場合 -> 違反
                latest_possible_end = max(interval[1] for interval in allowed_intervals)
                if arrival_time >= latest_possible_end:
                    is_violation = True
                else:
                    # 3. どの許容時間帯でもサービスを開始できない場合 -> 違反
                    possible_starts = []
                    for start_min, end_min in allowed_intervals:
                        potential_start = max(arrival_time, start_min)
                        if potential_start < end_min:
                            possible_starts.append(potential_start)

                    if not possible_starts:
                        is_violation = True
                    else:
                        # 違反なしの場合: サービス開始時刻と出発時刻を計算
                        service_start_time = min(possible_starts)
                        waiting_time = service_start_time - arrival_time
                        # 最大待機時間を超える場合 -> 違反
                        if waiting_time > max_wait_time:
                            is_violation = True
                        else:
                            # サービス開始時刻が決まった場合、出発時刻を計算
                            departure_time = service_start_time + SERVICE_TIME

            # 違反があればカウント
            if is_violation:
                violation_count += 1
                # 違反した場合でも計算を続けるため、出発時刻は到着時刻とする
                # (サービス時間0なので、実質的に到着後すぐ出発したと仮定)
                departure_time = arrival_time + SERVICE_TIME

        # --- 次の地点への計算のため、現在時刻を更新 ---
        # (もし current_time が None になっていたら、departure_time も None のまま)
        current_time = departure_time if current_time is not None else None

    return violation_count


def show_time_windows(time_windows: TimeWindows):
    print("\n--- Time Windows ---")
    coutn_of_1 = sum(sum(1 for allowed in tw if allowed == 1) for tw in time_windows)
    coutn_of_0 = sum(sum(1 for allowed in tw if allowed == 0) for tw in time_windows)
    print(f"Total Time Windows: {coutn_of_1 + coutn_of_0}")
    print(f"  Allowed (1): {coutn_of_1}")
    print(f"  Not Allowed (0): {coutn_of_0}")
    print(f"  Ratio: {coutn_of_1 / (coutn_of_1 + coutn_of_0) * 100:.2f}%")
    print("Location Index | Time Window")
    print("---------------|----------------")
    for i, tw in enumerate(time_windows):
        tw_str = map(lambda x: " " if x == 0 else "=", tw)
        print(f" {str(i + 1).zfill(3)}{" "*11}| {''.join(tw_str)}")
    print("---------------|----------------")


def plot_route(route, locations, title, ax, cost=None):
    if cost is not None:
        title += f" (Cost: {cost:.2f} min)"

    depot = locations[DEPOT_INDEX]
    cust_x = [loc[0] for i, loc in enumerate(locations) if i != DEPOT_INDEX]
    cust_y = [loc[1] for i, loc in enumerate(locations) if i != DEPOT_INDEX]

    ax.scatter(cust_x, cust_y, c="blue", label="Customers")
    ax.scatter(depot[0], depot[1], c="red", marker="s", s=100, label="Depot")

    # 経路を描画
    route_x = [locations[i][0] for i in route]
    route_y = [locations[i][1] for i in route]
    ax.plot(route_x, route_y, "g-")

    # 地点番号を表示
    for i, loc in enumerate(locations):
        ax.text(loc[0], loc[1] + 0.5, str(i), fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")


def with_time_measurement(func):
    """
    デコレータ: 関数の実行時間を計測する
    """

    def wrapper(*args, msg="", **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        delta = end_time - start_time
        print(f"Execution time: {delta:.4f} seconds, for {func.__name__} {msg}")
        return delta, result

    return wrapper


def random_route(locations, time_windows, V):
    """
    ランダムな実行可能経路を生成する
    """
    for i in range(100000):
        route_tmp = list(range(1, len(locations)))
        random.shuffle(route_tmp)
        route_tmp.sort(key=lambda x: time_windows[x - 1].index(1))
        route = [0] + route_tmp + [0]
        f, _, _ = check_time_window_feasibility(route, locations, time_windows, V)
        if f:
            return route
    raise ValueError("No feasible route found in 100000 attempts.")
