import random
import math
import numpy as np

# ハーシュシュミット距離計算 (緯度経度から距離を計算)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球の半径 (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
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
        deliveries.append({
            'lat': lat,
            'lon': lon,
            'availability': availability
        })
    return deliveries

# 例: 5つの配達先データを生成
mock_deliveries = generate_mock_data(500)

# 出力確認
for i, delivery in enumerate(mock_deliveries):
    print(f"配達先 {i+1}: 緯度={delivery['lat']}, 経度={delivery['lon']}, 在宅確率={delivery['availability'][:3]}...")  # 最初の3時間分だけ表示

# 配達ルートをランダムに生成
def generate_initial_population(deliveries, population_size):
    population = []
    num_deliveries = len(deliveries)
    for _ in range(population_size):
        # ランダムな順番で配達先を並べる
        individual = random.sample(range(num_deliveries), num_deliveries)
        population.append(individual)
    return population

# 適応度の計算 (距離 + 在宅確率を考慮)
def fitness(individual, deliveries):
    total_distance = 0
    total_availability = 0
    num_deliveries = len(individual)
    
    for i in range(num_deliveries - 1):
        # 配達先間の距離を計算
        lat1, lon1 = deliveries[individual[i]]['lat'], deliveries[individual[i]]['lon']
        lat2, lon2 = deliveries[individual[i + 1]]['lat'], deliveries[individual[i + 1]]['lon']
        distance = haversine(lat1, lon1, lat2, lon2)
        total_distance += distance
        
        # 在宅確率の加算 (最高確率の時間帯を選択)
        best_availability = max(deliveries[individual[i]]['availability'])
        total_availability += best_availability
        
    return total_distance - total_availability  # 距離を最小化、在宅確率を最大化

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
def genetic_algorithm(deliveries, population_size=50, generations=100):
    population = generate_initial_population(deliveries, population_size)
    
    for generation in range(generations):
        population.sort(key=lambda ind: fitness(ind, deliveries))  # 適応度順にソート
        
        next_generation = population[:population_size // 2]  # 上位50%選出
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation[:10], 2)  # 親選択
            child1, child2 = crossover(parent1, parent2)  # 交叉
            if random.random() < 0.1:  # 確率で突然変異
                mutate(child1)
            if random.random() < 0.1:
                mutate(child2)
            next_generation.extend([child1, child2])
        
        population = next_generation
    
    best_individual = population[0]
    return best_individual

# 最適化を実行
best_route = genetic_algorithm(mock_deliveries)

# 最適化結果の表示
print(f"最適な配達ルート: {best_route}")