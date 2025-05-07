import random
import numpy as np
import matplotlib.pyplot as plt


# 计算两个城市之间的距离
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# 计算一条路径的总距离
def total_distance(path, cities):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(cities[path[i]], cities[path[i + 1]])
    dist += distance(cities[path[-1]], cities[path[0]])  # 回到起点
    return dist


# 生成初始种群
def generate_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population


# 选择操作（锦标赛选择）
def tournament_selection(population, cities, tournament_size):
    tournament = random.sample(population, tournament_size)
    best = min(tournament, key=lambda x: total_distance(x, cities))
    return best


# 交叉操作（有序交叉）
def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child[start:end]]
    index = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining[index]
            index += 1
    return child


# 变异操作（交换变异）
def swap_mutation(individual):
    index1, index2 = random.sample(range(len(individual)), 2)
    individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual


# 遗传算法主函数
def genetic_algorithm(cities, pop_size, generations, tournament_size, crossover_rate, mutation_rate):
    population = generate_population(pop_size, len(cities))
    best_path = None
    best_distance = float('inf')
    best_distances = []

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size):
            if random.random() < crossover_rate:
                parent1 = tournament_selection(population, cities, tournament_size)
                parent2 = tournament_selection(population, cities, tournament_size)
                child = ordered_crossover(parent1, parent2)
            else:
                child = tournament_selection(population, cities, tournament_size)

            if random.random() < mutation_rate:
                child = swap_mutation(child)

            new_population.append(child)

        population = new_population

        # 找到当前代的最优路径
        for individual in population:
            dist = total_distance(individual, cities)
            if dist < best_distance:
                best_distance = dist
                best_path = individual

        best_distances.append(best_distance)

    return best_path, best_distance, best_distances


# 示例使用
if __name__ == "__main__":
    # 随机生成一些城市的坐标
    num_cities = 20
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    # 遗传算法参数
    pop_size = 100
    generations = 200
    tournament_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.2

    best_path, best_distance, best_distances = genetic_algorithm(cities, pop_size, generations, tournament_size,
                                                                crossover_rate, mutation_rate)

    print(f"最优路径: {best_path}")
    print(f"最短距离: {best_distance}")

    # 绘制最优路径
    path_cities = [cities[i] for i in best_path]
    path_cities.append(path_cities[0])  # 回到起点
    x = [city[0] for city in path_cities]
    y = [city[1] for city in path_cities]
    plt.plot(x, y, '-o')
    plt.title(f"Shortest Distance: {best_distance:.2f}")
    plt.show()

    # 绘制每一代的最优距离变化
    plt.plot(best_distances)
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.title('Best Distance per Generation')
    plt.show()    