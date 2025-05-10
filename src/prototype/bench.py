# bench.py

import random

import matplotlib.pyplot as plt
from tqdm import tqdm

from . import (
    genetic_algorithm,
    input_generator,
    insertion_heuristic,
    local_search,
    utils,
)

# --- パラメータ設定 ---
N = 75  # 配送先数
W = 10.0  # エリア幅 km
H = 10.0  # エリア高さ km
V = 50.0  # 移動速度 km/h

SEED = 42  # 乱数シード (再現性のため)
random.seed(SEED)
input_generator.random.seed(SEED)
insertion_heuristic.random.seed(SEED)
local_search.random.seed(SEED)
utils.random.seed(SEED)


def run_benchmark(N, W, H, V):
    # locations_tmp = input_generator.generate_locations(N, W, H, mode="double_circle")
    locations_tmp = input_generator.generate_locations(N, W, H, mode="random")
    time_windows_tmp = input_generator.generate_time_windows(
        N, mode="proportional", locations=locations_tmp[1:]
    )
    time_windows_all_tmp = input_generator.generate_time_windows(N, mode="all")

    random_indices = list(range(1, N + 1))
    random.shuffle(random_indices)
    locations = [locations_tmp[i] for i in random_indices]
    locations.insert(0, locations_tmp[0])  # デポを最初に追加
    time_windows = [time_windows_tmp[i - 1] for i in random_indices]
    time_windows_all = [time_windows_all_tmp[i - 1] for i in random_indices]
    utils.show_time_windows(time_windows)

    # --- ランダム経路 + 遺伝的アルゴリズム ---
    random_ga_with_time = utils.with_time_measurement(random_ga)
    delta_ga_random, opt_ga_random = random_ga_with_time(
        locations, time_windows, V, population_size=100
    )
    feasible_ga_random, cost_ga_random, arrivals_ga_random = (
        utils.check_time_window_feasibility(opt_ga_random, locations, time_windows, V)
    )

    # --- 挿入法 + 2-opt ---
    insertion_2opt_with_time = utils.with_time_measurement(insertion_2opt)
    delta, opt = insertion_2opt_with_time(locations, time_windows, V)
    feasible, cost, arrivals = utils.check_time_window_feasibility(
        opt, locations, time_windows, V
    )
    delta_all, opt_all = insertion_2opt_with_time(locations, time_windows_all, V)
    feasible_all, cost_all, arrivals_all = utils.check_time_window_feasibility(
        opt_all, locations, time_windows_all, V
    )
    violation = utils.count_time_window_violations(opt_all, locations, time_windows, V)

    # --- 挿入法 + 遺伝的アルゴリズム ---
    insertion_ga_with_time = utils.with_time_measurement(insertion_ga)
    delta_ga, opt_ga = insertion_ga_with_time(locations, time_windows, V)
    feasible_ga, cost_ga, arrivals_ga = utils.check_time_window_feasibility(
        opt_ga, locations, time_windows, V
    )

    results = dict(
        # input
        locations=locations,
        time_windows=time_windows,
        time_windows_all=time_windows_all,
        # insertion_2opt
        initial_route=None,
        optimized_route=opt,
        optimized_route_all=opt_all,
        cost=cost,
        cost_all=cost_all,
        arrivals=arrivals,
        arrivals_all=arrivals_all,
        feasible=feasible,
        feasible_all=feasible_all,
        delta=delta,
        delta_all=delta_all,
        violation=violation,
        # insertion_ga
        optimized_route_ga=opt_ga,
        cost_ga=cost_ga,
        arrivals_ga=arrivals_ga,
        feasible_ga=feasible_ga,
        delta_ga=delta_ga,
        initial_route_ga=None,
        # random_ga
        optimized_route_ga_random=opt_ga_random,
        cost_ga_random=cost_ga_random,
        arrivals_ga_random=arrivals_ga_random,
        feasible_ga_random=feasible_ga_random,
        delta_ga_random=delta_ga_random,
        initial_route_ga_random=None,
    )

    return results


def insertion_2opt(locations, time_windows, V):
    ini = insertion_heuristic.build_initial_route_insertion(locations, time_windows, V)
    assert ini is not None, "Initial route is None"
    opt = local_search.improve_route_local_search(
        ini, locations, time_windows, V, max_iterations=1000
    )
    assert opt is not None, "Optimized route is None"
    return opt


def insertion_ga(locations, time_windows, V, population_size=100):
    inis = [
        insertion_heuristic.build_initial_route_insertion(
            locations, time_windows, V, start_node_strategy="random"
        )
        for _ in tqdm(range(population_size), desc="Generating Initial Routes")
    ]
    assert all(ini is not None for ini in inis), "Initial routes are None"
    opt = genetic_algorithm.genetic_algorithm(
        inis, locations, time_windows, V, max_generations=10000
    )
    assert opt is not None, "Optimized route is None"
    return opt


def random_ga(locations, time_windows, V, population_size=100):
    inis = [
        utils.random_route(locations, time_windows, V)
        for _ in tqdm(range(population_size), desc="Generating Initial Routes")
    ]
    assert all(ini is not None for ini in inis), "Initial routes are None"
    opt = genetic_algorithm.genetic_algorithm(
        inis, locations, time_windows, V, max_generations=10000
    )
    assert opt is not None, "Optimized route is None"
    return opt


if __name__ == "__main__":
    result = run_benchmark(N, W, H, V)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Route Optimization with Home Delivery Time Windows")
    utils.plot_route(
        result["optimized_route"],
        result["locations"],
        title="Insertion + 2-opt",
        ax=ax[0],
        cost=result["cost"],
    )
    utils.plot_route(
        result["optimized_route_ga_random"],
        result["locations"],
        title="Primitive GA",
        ax=ax[2],
        cost=result["cost_ga_random"],
    )
    utils.plot_route(
        result["optimized_route_ga"],
        result["locations"],
        title="Insertion + GA",
        ax=ax[1],
        cost=result["cost_ga"],
    )
    plt.show()
    print(result["violation"])
