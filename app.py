import random
import numpy as np 
from flask import Flask, render_template, request

app = Flask(__name__)

# ==========================================
# BAGIAN 1: TUGAS 3 (KNAPSACK)
# ==========================================

knapsack_items = {
    'A': {'weight': 7, 'value': 5},
    'B': {'weight': 2, 'value': 4},
    'C': {'weight': 1, 'value': 7},
    'D': {'weight': 9, 'value': 2},
}
knapsack_capacity = 15
knapsack_item_list = list(knapsack_items.keys())

def knapsack_decode(chromosome):
    total_weight = 0
    total_value = 0
    chosen_items = []
    for gene, name in zip(chromosome, knapsack_item_list):
        if gene == 1:
            total_weight += knapsack_items[name]['weight']
            total_value += knapsack_items[name]['value']
            chosen_items.append(name)
    return chosen_items, total_weight, total_value

def knapsack_fitness(chromosome):
    _, total_weight, total_value = knapsack_decode(chromosome)
    if total_weight <= knapsack_capacity:
        return total_value
    else:
        return 1e-6

def knapsack_genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, seed, elitism=True):
    random.seed(seed)
    n_genes = len(knapsack_item_list)
    pop = [[random.randint(0, 1) for _ in range(n_genes)] for _ in range(pop_size)]
    logs = []

    for g in range(generations):
        fitness_scores = [knapsack_fitness(ind) for ind in pop]
        best_idx = fitness_scores.index(max(fitness_scores))
        best_ind = pop[best_idx]
        items_chosen, w, v = knapsack_decode(best_ind)
        
        logs.append({
            'gen': g + 1,
            'chrom': str(best_ind),
            'items': ", ".join(items_chosen),
            'weight': w,
            'value': v,
            'fitness': fitness_scores[best_idx]
        })
        
        new_pop = []
        if elitism:
            new_pop.append(best_ind)
            
        while len(new_pop) < pop_size:
            parents = random.choices(pop, weights=fitness_scores, k=2)
            p1, p2 = parents[0], parents[1]
            if random.random() < crossover_rate:
                point = random.randint(1, n_genes - 1)
                child = p1[:point] + p2[point:]
            else:
                child = p1[:]
            if random.random() < mutation_rate:
                m_point = random.randint(0, n_genes - 1)
                child[m_point] = 1 - child[m_point]
            new_pop.append(child)
        pop = new_pop

    fitness_scores = [knapsack_fitness(ind) for ind in pop]
    best_idx = fitness_scores.index(max(fitness_scores))
    final_best = pop[best_idx]
    f_items, f_w, f_v = knapsack_decode(final_best)
    
    result = {
        'chromosome': str(final_best),
        'items': ", ".join(f_items),
        'weight': f_w,
        'value': f_v,
        'fitness': fitness_scores[best_idx]
    }
    return logs, result


# ==========================================
# BAGIAN 2: TUGAS 4 (TSP)
# ==========================================

TSP_CITIES = ['A', 'B', 'C', 'D', 'E']
TSP_CITY_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
TSP_DIST_MATRIX = [
    [0, 7, 5, 9, 9],
    [7, 0, 7, 2, 8],
    [5, 7, 0, 4, 3],
    [9, 2, 4, 0, 6],
    [9, 8, 3, 6, 0]
]

def tsp_route_distance(route_indices):
    d = 0
    n = len(route_indices)
    limit = 5
    for i in range(n):
        u = route_indices[i]
        v = route_indices[(i + 1) % n]
        if isinstance(u, str): u = TSP_CITY_MAP.get(u, 0)
        if isinstance(v, str): v = TSP_CITY_MAP.get(v, 0)
        u = int(u) % limit
        v = int(v) % limit
        d += TSP_DIST_MATRIX[u][v]
    return d

def tsp_create_individual():
    ind = [0, 1, 2, 3, 4]
    random.shuffle(ind)
    return ind

def tsp_ordered_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    current_p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while p2[current_p2_idx] in child:
                current_p2_idx += 1
                if current_p2_idx >= size: break
            if current_p2_idx < size:
                child[i] = p2[current_p2_idx]
    if None in child:
        missing = [x for x in range(size) if x not in child]
        for i in range(size):
            if child[i] is None:
                child[i] = missing.pop(0) if missing else 0
    return child

def tsp_swap_mutation(ind):
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]

def tsp_genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, seed, elitism=True):
    random.seed(seed)
    pop = [tsp_create_individual() for _ in range(pop_size)]
    logs = []
    
    for g in range(generations):
        pop = sorted(pop, key=lambda ind: tsp_route_distance(ind))
        best_ind = pop[0]
        best_dist = tsp_route_distance(best_ind)
        route_names = "-".join([TSP_CITIES[i % 5] for i in best_ind])
        
        logs.append({
            'gen': g + 1,
            'chrom': str(best_ind),
            'route': route_names,
            'distance': f"{best_dist:.2f}"
        })
        
        new_pop = []
        if elitism: new_pop.append(best_ind)
        while len(new_pop) < pop_size:
            parents = random.sample(pop, k=min(4, len(pop)))
            p1 = min(parents, key=lambda ind: tsp_route_distance(ind))
            parents = random.sample(pop, k=min(4, len(pop)))
            p2 = min(parents, key=lambda ind: tsp_route_distance(ind))
            
            if random.random() < crossover_rate:
                child = tsp_ordered_crossover(p1, p2)
            else:
                child = p1[:]
            if random.random() < mutation_rate:
                tsp_swap_mutation(child)
            new_pop.append(child)
        pop = new_pop

    pop = sorted(pop, key=lambda ind: tsp_route_distance(ind))
    final_best = pop[0]
    final_dist = tsp_route_distance(final_best)
    final_route = "-".join([TSP_CITIES[i % 5] for i in final_best]) + "-" + TSP_CITIES[final_best[0] % 5]

    result = {
        'chromosome': str(final_best),
        'route': final_route,
        'distance': final_dist
    }
    return logs, result

# ==========================================
# BAGIAN 3: TUGAS 5 (ANFIS SUGENO)
# ==========================================

def f1(x, y):
    return 0.1 * x + 0.1 * y + 0.1

def f2(x, y):
    return 10 * x + 10 * y + 10

def anfis_calculate(x, y):
    A1 = 0.5
    B1 = 0.1
    A2 = 0.25
    B2 = 0.039

    w1 = A1 * B1
    w2 = A2 * B2

    w_sum = w1 + w2
    if w_sum == 0:
        W1, W2 = 0, 0
    else:
        W1 = w1 / w_sum
        W2 = w2 / w_sum

    out1 = W1 * f1(x, y)
    out2 = W2 * f2(x, y)
    output = out1 + out2

    return {
        "A1": A1, "B1": B1, "A2": A2, "B2": B2,
        "w1": w1, "w2": w2,
        "W1": W1, "W2": W2,
        "f1_val": f1(x, y), "f2_val": f2(x, y),
        "out1": out1, "out2": out2,
        "final_output": output
    }

# ==========================================
# ROUTES FLASK
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tugas3', methods=['GET', 'POST'])
def tugas_3():
    generation_logs = None
    final_result = None
    settings = {'pop_size': 8, 'generations': 5, 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'seed': 16, 'elitism': True}

    if request.method == 'POST':
        pop_size = int(request.form['pop_size'])
        generations = int(request.form['generations'])
        crossover_rate = float(request.form['crossover_rate'])
        mutation_rate = float(request.form['mutation_rate'])
        seed = int(request.form['seed'])
        elitism = 'elitism' in request.form
        settings = request.form
        generation_logs, final_result = knapsack_genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, seed, elitism)

    return render_template('tugas3.html', logs=generation_logs, result=final_result, settings=settings)

@app.route('/tugas4', methods=['GET', 'POST'])
def tugas_4():
    generation_logs = None
    final_result = None
    settings = {'pop_size': 100, 'generations': 50, 'crossover_rate': 0.9, 'mutation_rate': 0.2, 'seed': 42, 'elitism': True}

    if request.method == 'POST':
        pop_size = int(request.form['pop_size'])
        generations = int(request.form['generations'])
        crossover_rate = float(request.form['crossover_rate'])
        mutation_rate = float(request.form['mutation_rate'])
        seed = int(request.form['seed'])
        elitism = 'elitism' in request.form
        settings = request.form
        generation_logs, final_result = tsp_genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, seed, elitism)

    return render_template('tugas4.html', logs=generation_logs, result=final_result, settings=settings)

@app.route('/tugas5', methods=['GET', 'POST'])
def tugas_5():
    result = None
    inputs = {'x': 3, 'y': 4}
    if request.method == 'POST':
        try:
            x = float(request.form['x'])
            y = float(request.form['y'])
            inputs = {'x': x, 'y': y}
            result = anfis_calculate(x, y)
        except ValueError:
            pass
    return render_template('tugas5.html', result=result, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)