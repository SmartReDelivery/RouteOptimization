import random
import math
from tqdm import tqdm  # プログレスバーの表示
import matplotlib.pyplot as plt
import numpy as np
from max_availability import maximize_availability


random.seed(0)  # 乱数シードを固定


# ハーシュシュミット距離計算 (緯度経度から距離を計算)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球の半径 (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # 距離 (km)


# モックデータ生成
def generate_mock_data(num_deliveries):
    deliveries = []
    for i in range(num_deliveries):
        # ランダムな緯度経度
        lat = random.uniform(35.5, 35.8)  # 東京の一部の緯度
        lon = random.uniform(139.7, 139.9)  # 東京の一部の経度
        # ランダムな在宅確率（0.0〜1.0の間で1時間ごとの確率）
        availability = [random.uniform(0.0, 1.0) for _ in range(24)]  # 24時間の在宅確率
        deliveries.append({"lat": lat, "lon": lon, "availability": availability})
    return deliveries


# 配達ルートをランダムに生成
def generate_initial_population(deliveries, population_size):
    population = []
    num_deliveries = len(deliveries)
    for _ in range(population_size):
        # ランダムな順番で配達先を並べる
        individual = random.sample(range(num_deliveries), num_deliveries)
        population.append(individual)
    return population


def individual_hash(individual):
    return tuple(individual)


fitness_cache = {}


# 適応度の計算 (距離 + 在宅確率を考慮)
def fitness(individual, deliveries):
    if individual_hash(individual) in fitness_cache:
        return fitness_cache[individual_hash(individual)]

    total_distance = 0
    total_availability = 0
    num_deliveries = len(individual)

    for i in range(num_deliveries - 1):
        # 配達先間の距離を計算
        lat1, lon1 = deliveries[individual[i]]["lat"], deliveries[individual[i]]["lon"]
        lat2, lon2 = (
            deliveries[individual[i + 1]]["lat"],
            deliveries[individual[i + 1]]["lon"],
        )
        distance = haversine(lat1, lon1, lat2, lon2)
        total_distance += distance

        # 在宅確率の加算 (最高確率の時間帯を選択)
        best_availability, _ = best_time_to_deliver(deliveries, individual)
        total_availability += best_availability

    penalty = total_distance - total_availability  # 距離を最小化、在宅確率を最大化
    fitness_cache[individual_hash(individual)] = penalty
    return penalty


# 交叉操作
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + [x for x in parent2 if x not in parent1[:cut]]
    child2 = parent2[:cut] + [x for x in parent1 if x not in parent2[:cut]]
    return child1, child2


# 突然変異操作
def mutate(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]


# 遺伝的アルゴリズム
def genetic_algorithm(deliveries, population_size=50, max_generations=100):
    population = generate_initial_population(deliveries, population_size)
    best_fitness_values = []  # 各世代の最良適応度を保存 (収束判定用, グラフ表示用)

    for generation in tqdm(range(max_generations)):
        population.sort(key=lambda individual: fitness(individual, deliveries))
        best_fitness = fitness(population[0], deliveries)
        best_fitness_values.append(best_fitness)

        if is_converged(best_fitness_values):
            break

        next_generation = population[: population_size // 2]  # 上位50%選出
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation[:10], 2)  # 親選択
            child1, child2 = crossover(parent1, parent2)  # 交叉
            if random.random() < 0.1:  # 確率で突然変異
                mutate(child1)
            if random.random() < 0.1:
                mutate(child2)
            next_generation.extend([child1, child2])

        population = next_generation

    plt.title("Fitness over generations")
    plt.plot(best_fitness_values, "b-")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

    best_individual = population[0]
    return best_individual


# 与えられた配送順序の中で最適な配達時間を返す
def best_time_to_deliver(deliveries, individual):
    # A[i, j] = 時刻 i に配達するときの j 番目の配達先の在宅確率
    A = np.array(
        [[deliveries[j]["availability"][i] for j in individual] for i in range(24)]
    )
    # 配達時間を最適化
    best_availability, schedule = maximize_availability(A)

    return (best_availability, schedule)


def total_availability(availabilities, time_to_deliver, individual):
    total_availability = sum(
        availabilities[j][0] for j in individual[: time_to_deliver[0]]
    )
    for i in range(len(time_to_deliver - 1)):
        total_availability += sum(
            availabilities[j][i]
            for j in individual[time_to_deliver[i] : time_to_deliver[i + 1]]
        )
    i = len(time_to_deliver)
    total_availability += sum(
        availabilities[j][i] for j in individual[time_to_deliver[i] :]
    )
    return total_availability


# 収束判定
def is_converged(best_fitness_values):
    if len(best_fitness_values) < 10:
        return False
    diff_from_start = best_fitness_values[0] - best_fitness_values[-1]
    diff_from_decade = best_fitness_values[-10] - best_fitness_values[-1]
    converged = diff_from_decade < 1e-4 * diff_from_start
    if converged:
        print(
            f"Converged: diff from start={diff_from_start}, diff from decade={diff_from_decade}"
        )
    return converged


# 例: 5つの配達先データを生成
mock_deliveries = generate_mock_data(100)

# 出力確認
for i, delivery in enumerate(mock_deliveries):
    print(
        f"配達先 {i+1}: 緯度={delivery['lat']}, 経度={delivery['lon']}, \
            在宅確率=[{', '.join(map(str, delivery['availability'][:3]))}, ...](length: {len(delivery['availability'])})"
    )  # 最初の3時間分だけ表示

# 最適化を実行
best_route = genetic_algorithm(mock_deliveries)

# 最適な配達時間を計算
best_availability, best_schedule = best_time_to_deliver(mock_deliveries, best_route)
print(f"最大在宅確率: {best_availability}")
print(f"配達スケジュール: {best_schedule}")

# 最適化結果の表示
print(f"最適な配達ルート: {best_route}")
