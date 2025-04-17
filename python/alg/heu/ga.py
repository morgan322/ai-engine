"Genetic Algorithms"
import random
import numpy as np

# 环境中易拉罐的位置
cans = [(2, 3), (5, 6), (8, 8), (1, 9)]

# 计算两点之间的距离
def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# 个体的表示：路径是一个易拉罐的索引列表
def create_individual():
    return random.sample(range(len(cans)), len(cans))

# 计算路径的总距离
def calculate_fitness(individual):
    total_distance = 0
    start = (0, 0)  # 假设机器人从原点开始
    for index in individual:
        total_distance += distance(start, cans[index])
        start = cans[index]
    total_distance += distance(start, (0, 0))  # 回到原点
    return total_distance

# 选择：选择适应度最好的个体
def select(population):
    population.sort(key=lambda x: x[1])  # 按适应度排序，最短路径优先
    return population[:2]  # 选择最好的两个个体

# 交叉：通过交换部分路径来生成新的个体
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
    child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
    return child1, child2

# 变异：随机改变一个位置
def mutate(individual):
    if random.random() < 0.1:  # 10%的变异概率
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

# 遗传算法主函数
def genetic_algorithm():
    population_size = 100
    generations = 500
    population = [(create_individual(), 0) for _ in range(population_size)]
    
    # 计算每个个体的适应度
    for i in range(population_size):
        population[i] = (population[i][0], calculate_fitness(population[i][0]))
    
    for generation in range(generations):
        print(f"Generation {generation}: Best fitness {population[0][1]}")
        parents = select(population)
        
        # 交叉生成新个体
        children = []
        while len(children) < population_size // 2:
            child1, child2 = crossover(parents[0][0], parents[1][0])
            children.append(child1)
            children.append(child2)
        
        # 变异
        for child in children:
            mutate(child)
        
        # 计算新个体的适应度
        population = [(child, calculate_fitness(child)) for child in children]
        
    best_individual = population[0]
    print(f"Best path: {best_individual[0]}, Fitness: {best_individual[1]}")

# 运行遗传算法
genetic_algorithm()
