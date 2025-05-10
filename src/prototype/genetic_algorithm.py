# genetic_algorithm.py

import random

from tqdm import tqdm

from . import utils


def genetic_algorithm(
    initial_routes,
    locations,
    time_windows,
    V,
    max_generations=100,
):
    population_size = len(initial_routes)

    min_cost_history = []
    mean_cost_history = []
    for g in tqdm(range(max_generations), desc="Genetic Algorithm Progress"):
        costs = []
        for route in initial_routes:
            # Calculate cost
            _, cost, _ = utils.check_time_window_feasibility(
                route, locations, time_windows, V
            )
            costs.append(cost)

        dices = list(range(population_size))
        random.shuffle(dices)
        for i, j in zip(dices[::2], dices[1::2]):
            parent1 = initial_routes[i]
            parent2 = initial_routes[j]
            r = random.random()
            if r < 1 / 3:
                # Order crossover
                offspring1 = order_crossover(parent1, parent2)
                offspring2 = order_crossover(parent2, parent1)
            elif r < 2 / 3:
                # Cut crossover
                offspring1, offspring2 = cut_crossover(parent1, parent2)
            else:
                # Uniform crossover
                offspring1 = uniform_crossover(parent1, parent2)
                offspring2 = uniform_crossover(parent2, parent1)
            mutated1 = mutation(offspring1)
            mutated2 = mutation(offspring2)
            f1, offspring1_cost, _ = utils.check_time_window_feasibility(
                offspring1, locations, time_windows, V
            )
            f2, offspring2_cost, _ = utils.check_time_window_feasibility(
                offspring2, locations, time_windows, V
            )
            f3, mutated1_cost, _ = utils.check_time_window_feasibility(
                mutated1, locations, time_windows, V
            )
            f4, mutated2_cost, _ = utils.check_time_window_feasibility(
                mutated2, locations, time_windows, V
            )
            if f1 and f3 and mutated1_cost < offspring1_cost:
                offspring1 = mutated1
                offspring1_cost = mutated1_cost
            if f2 and f4 and mutated2_cost < offspring2_cost:
                offspring2 = mutated2
                offspring2_cost = mutated2_cost
            if f1:
                initial_routes.append(offspring1)
                costs.append(offspring1_cost)
            if f2:
                initial_routes.append(offspring2)
                costs.append(offspring2_cost)
        # 上位 population_size のみを残す
        indices = sorted(range(len(costs)), key=lambda x: costs[x])
        initial_routes = [initial_routes[i] for i in indices[:population_size]]
        costs = [costs[i] for i in indices[:population_size]]

        # # --- Selection and Crossover ---
        # i, j = random.sample(range(population_size), 2)
        # if i == j:
        #     continue
        # parent1 = initial_routes[i]
        # parent2 = initial_routes[j]
        # if random.random() < 1 / 3:
        #     offspring1 = order_crossover(parent1, parent2)
        #     offspring2 = order_crossover(parent2, parent1)
        # elif random.random() < 1 / 2:
        #     offspring1, offspring2 = cut_crossover(parent1, parent2)
        # else:
        #     offspring1 = uniform_crossover(parent1, parent2)
        #     offspring2 = uniform_crossover(parent2, parent1)
        # parent1_cost = costs[i]
        # parent2_cost = costs[j]
        # min_cost = min(parent1_cost, parent2_cost)
        # k = 0
        # while k < 100:
        #     f1, offspring1_cost, _ = utils.check_time_window_feasibility(
        #         offspring1, locations, time_windows, V
        #     )
        #     f2, offspring2_cost, _ = utils.check_time_window_feasibility(
        #         offspring2, locations, time_windows, V
        #     )
        #     change_list = []
        #     if f1 and offspring1_cost < min_cost:
        #         change_list.append((offspring1, offspring1_cost))
        #     if f2 and offspring2_cost < min_cost:
        #         change_list.append((offspring2, offspring2_cost))
        #     if len(change_list) == 2:
        #         initial_routes[i] = change_list[0][0]
        #         initial_routes[j] = change_list[1][0]
        #         costs[i] = change_list[0][1]
        #         costs[j] = change_list[1][1]
        #     elif len(change_list) == 1:
        #         if parent1_cost < parent2_cost:
        #             initial_routes[i] = change_list[0][0]
        #             costs[i] = change_list[0][1]
        #         else:
        #             initial_routes[j] = change_list[0][0]
        #             costs[j] = change_list[0][1]
        #     if len(change_list) > 0:
        #         break
        #     k += 1

        min_cost = min(costs)
        mean_cost = sum(costs) / len(costs)
        min_cost_history.append(min_cost)
        mean_cost_history.append(mean_cost)

        if (
            g > 100
            and min_cost_history[-100] - min_cost < 1e-4
            and mean_cost_history[-100] - mean_cost < 1e-2
        ):
            break

    min_cost_index = costs.index(min(costs))
    best_route = initial_routes[min_cost_index]
    return best_route


def mutation(route):
    # 2-opt mutation
    i = random.randint(1, len(route) - 2)
    j = random.randint(i + 1, len(route) - 1)
    child = route[:i] + route[i:j][::-1] + route[j:]
    assert len(child[1:-1]) == len(set(child[1:-1])), "Child must be unique"
    return child


def order_crossover(parent1, parent2):
    # Order crossover
    parent1 = parent1[1:-1]
    parent2 = parent2[1:-1]
    size = len(parent1)
    assert size == len(parent2), "Parents must be of the same size"

    start = random.randint(0, size - 1)
    end = random.randint(start + 1, size)
    child = [None] * size
    child[start:end] = parent1[start:end]
    fill = [item for item in parent2 if item not in child]
    fill_index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[fill_index]
            fill_index += 1
    assert len(child) == len(set(child)), "Child must be unique"
    assert len(child) == size, "Child must be of the same size as parents"
    child = [0] + child + [0]
    return child


def cut_crossover(parent1, parent2):
    # Cut crossover
    parent1 = parent1[1:-1]
    parent2 = parent2[1:-1]
    size = len(parent1)
    assert size == len(parent2), "Parents must be of the same size"

    cut = random.randint(1, size - 1)
    child1 = parent1[:cut] + [x for x in parent2 if x not in parent1[:cut]]
    child2 = parent2[:cut] + [x for x in parent1 if x not in parent2[:cut]]
    if len(child1) != size:
        print("Child1:", child1)
        print("Parent1:", parent1)
        print("Parent2:", parent2)
        print("Cut:", cut)
    assert len(child1) == len(set(child1)), "Child must be unique"
    assert len(child2) == len(set(child2)), "Child must be unique"
    assert len(child1) == size, "Child must be of the same size as parents"
    assert len(child2) == size, "Child must be of the same size as parents"
    return [0] + child1 + [0], [0] + child2 + [0]


def uniform_crossover(parent1, parent2):
    # Uniform crossover
    parent1 = parent1[1:-1]
    parent2 = parent2[1:-1]
    size = len(parent1)
    assert size == len(parent2), "Parents must be of the same size"

    # assert len(parent1) == len(set(parent1)), "Parent1 must be unique"
    # assert len(parent2) == len(set(parent2)), "Parent2 must be unique"

    child = []
    stock = set()
    for i in range(size):
        p1 = parent1[i]
        p2 = parent2[i]
        if p1 not in child and p2 not in child:
            if random.random() < 0.5:
                p = p1
                stock.add(p2)
            else:
                p = p2
                stock.add(p1)
        elif p1 not in child and p2 in child:
            p = p1
        elif p1 in child and p2 not in child:
            p = p2
        else:
            p = random.choice(list(stock))
        child.append(p)
        if p in stock:
            stock.remove(p)

    assert len(child) == len(set(child)), "Child must be unique"
    assert len(child) == size, "Child must be of the same size as parents"
    child = [0] + child + [0]
    return child
