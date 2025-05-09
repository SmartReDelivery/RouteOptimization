# main.py
import time
from . import input_generator
from . import insertion_heuristic
from . import local_search
from . import utils

CONSIDER_TIME_WINDOW = True  # タイムウィンドウを考慮するかどうか

# --- パラメータ設定 ---
N = 100  # 配送先数 (ドキュメントは100だが、デモ用に小さく)
W = 10.0  # エリア幅 5km
H = 10.0  # エリア高さ 5km
V = 20.0  # 移動速度 20km/h

SEED = 42  # 乱数シード (再現性のため)
input_generator.random.seed(SEED)
insertion_heuristic.random.seed(SEED)
local_search.random.seed(SEED)


# --- 1. 入力の作成 ---
print("--- 1. Generating Input Data ---")
print(f"Locations: {N}, Area: {W}x{H}, Speed: {V} km/h, Seed: {SEED}")
start_gen = time.time()
locations = input_generator.generate_locations(N, W, H)
time_windows = input_generator.generate_time_windows(N)  # N個の配送先に対するTW
if not CONSIDER_TIME_WINDOW:
    time_windows_tmp = time_windows
    # タイムウィンドウを考慮しない場合は、すべての時間帯を許容する
    time_windows = input_generator.generate_time_windows_whenever(N)
end_gen = time.time()

print(f"Generated {N} locations and time windows in {end_gen - start_gen:.4f} seconds.")
# print(f"Depot: {locations[utils.DEPOT_INDEX]}")
# print("Locations (first 5):", locations[1:6])
# print("Time Windows (first 5):")
# for i in range(min(N, 5)):
#     print(f"  Loc {i+1}: {time_windows[i]} -> {utils.get_time_window_intervals(i+1, time_windows)}")

# --- 2. 初期経路構築 (挿入法) ---
print("\n--- 2. Building Initial Route (Insertion Heuristic) ---")
start_insert = time.time()
initial_route = insertion_heuristic.build_initial_route_insertion(
    locations, time_windows, V, start_node_strategy="farthest"
)
end_insert = time.time()


if initial_route:
    print(f"Initial route built in {end_insert - start_insert:.4f} seconds.")
    feasible_init, cost_init, arrivals_init = utils.check_time_window_feasibility(
        initial_route, locations, time_windows, V
    )
    print(f"Initial Route: {initial_route}")
    print(f"Feasible: {feasible_init}")
    if feasible_init:
        print(
            f"Cost (Total Travel Time): {cost_init / 60.0:.2f} hours ({cost_init:.2f} minutes)"
        )  # 時間単位で表示
    else:
        print("Initial route is infeasible!")
        # 実行不可能な場合は改善に進めない
        exit()

    # --- 3. 改善 (局所探索 2-opt) ---
    print("\n--- 3. Improving Route (Local Search 2-opt) ---")
    start_ls = time.time()
    improved_route = local_search.improve_route_local_search(
        initial_route, locations, time_windows, V
    )
    end_ls = time.time()

    print(f"Local search completed in {end_ls - start_ls:.4f} seconds.")
    feasible_imp, cost_imp, arrivals_imp = utils.check_time_window_feasibility(
        improved_route, locations, time_windows, V
    )
    print(f"Improved Route: {improved_route}")
    print(f"Feasible: {feasible_imp}")
    if feasible_imp:
        print(
            f"Cost (Total Travel Time): {cost_imp / 60.0:.2f} hours ({cost_imp:.2f} minutes)"
        )
        improvement_abs = cost_init - cost_imp
        improvement_rel = (improvement_abs / cost_init * 100) if cost_init > 0 else 0
        print(
            f"\nImprovement: {improvement_abs / 60.0:.2f} hours ({improvement_abs:.2f} minutes) "
            f"({improvement_rel:.2f}%)"
        )
    else:
        print("Improved route is infeasible! (Should not happen if LS starts feasible)")

    # --- 4. タイムウィンドウの違反数をカウント (タイムウィンドウを考慮しない場合) ---
    if not CONSIDER_TIME_WINDOW:
        time_windows_ = time_windows_tmp
        print("\n--- 4. Counting Time Window Violations ---")
        violation_count = utils.count_time_window_violations(
            improved_route, locations, time_windows_tmp, V
        )
        print(f"Time window violations: {violation_count}")

    # --- (Optional) 可視化 ---

    # # タイムウィンドウの表示
    utils.show_time_windows(time_windows)

    # # 経路の可視化
    # utils.plot_route(
    #     initial_route, locations, f"Initial Route (Cost: {cost_init/60.0:.2f} hrs)"
    # )
    # utils.plot_route(
    #     improved_route, locations, f"Improved Route (Cost: {cost_imp/60.0:.2f} hrs)"
    # )


else:
    print(
        f"Failed to build a feasible initial route after {end_insert - start_insert:.4f} seconds."
    )
