# utils.py
import math
from typing import List, Tuple, Optional

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
