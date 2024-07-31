import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination import get_termination

from generate_data import BicycleRouting
from plot import plot_problem

class StartFromZeroRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # Ensure starting point is always '0'
        I = np.where(X == 0)[1]
        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])
        return X

problem = BicycleRouting(15, 7, seed=42)

# Setup the NSGA-II algorithm
algorithm = NSGA2(
    pop_size=50,
    n_offsprings=10,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover(),
    repair=StartFromZeroRepair(),
    eliminate_duplicates=True
)

# Define the termination criterion
termination = get_termination("n_gen", 1005)

# Run the optimization
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

def extract_population(history, generation):
    if generation < len(history):
        return history[generation].pop.get("F")
    return None

def save_data(res):
    with open('results.json', 'w') as file:
        json.dump({
            'comment1': 'These are the raw results of pymoo NSGAII.',
            'comment2': 'The decisions are permutations but nodes are only visited until the first 7 nodes are visited, the rest are ignored and their order is meaningless',
            'comment3': 'The objectives are [Distance, Beauty, Safety], and for Beauty and Safety the sign is reversed because pymoo only minimizes',
            'comment4': 'This is the last population of NSGAII, presumably the best approximation of the Pareto front.',
            'comment5': 'To highlight the improvement we also include the objective results of the first generation.',
            'comment6': 'We left the lines as strings of lists intentionally to stay on a single line and be more readable, use json.loads() to recover the lists',
            'decisions': [str([int(e) for e in line]) for line in res.X],
            'objectives': [str([float(e) for e in line]) for line in res.F],
            'first_gen_objectives': [str([float(e) for e in line]) for line in extract_population(res.history, 0)]
        }, file, indent=4)

save_data(res)

print("Shape of res.F:", res.F.shape)
print("First few rows of res.F:", res.F[:40, :])

def correct_data(F):
    res = np.array(F)
    res[:, 1:3] = -F[:, 1:3]
    return res

def get_best(X, F, index):
    return X[np.argmin(F[:, index])]

print("First few rows of F corrected:", correct_data(res.F[:40, :]))

pop_200 = extract_population(res.history, 200)
pop_1000 = extract_population(res.history, 1000)

def get_non_dominated_points(data, axes):
    non_dominated = []
    for i, point in enumerate(data):
        is_dominated = False
        for other_point in data:
            if all(point[axes] >= other_point[axes]) and any(point[axes] > other_point[axes]):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append(i)
    return np.array(non_dominated)

def plot_non_dominated_points(data1, data2, axes, labels, title):
    fig, ax = plt.subplots()
    non_dominated1 = get_non_dominated_points(data1, axes)
    non_dominated2 = get_non_dominated_points(data2, axes)
    data1 = correct_data(data1)[:, axes]
    data2 = correct_data(data2)[:, axes]
    if len(non_dominated1) > 0:
        ax.scatter(data1[non_dominated1, 0], data1[non_dominated1, 1], color='blue', label='Approximate Pareto front after 200 generations')
    if len(non_dominated2) > 0:
        ax.scatter(data2[non_dominated2, 0], data2[non_dominated2, 1], color='red', label='Approximate Pareto front after 1000 generations')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_non_dominated_points(pop_200, pop_1000, [0, 1], ['Distance', 'Safety'], 'Distance vs Safety')
plot_non_dominated_points(pop_200, pop_1000, [0, 2], ['Distance', 'Beauty'], 'Distance vs Beauty')
plot_non_dominated_points(pop_200, pop_1000, [1, 2], ['Safety', 'Beauty'], 'Safety vs Beauty')

def plot_mean_fitness_history(res, generations, labels):
    means = np.zeros((len(generations), len(labels)))
    for i, generation in enumerate(generations):
        data = extract_population(res.history, generation)
        if data is not None:
            means[i, :] = correct_data(np.mean(data, axis=0, keepdims=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(generations, means[:, 0], label=labels[0])
    ax.set_title('Mean Fitness History')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Distance')
    ax2 = ax.twinx()
    ax2.set_ylabel('Mean Quality Score')
    ax2.plot(generations, means[:, 1], label=labels[1], color='C1')
    ax2.plot(generations, means[:, 2], label=labels[2], color='C2')
    
    ax.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

generations = list(range(0, 1001, 50))
plot_mean_fitness_history(res, generations, ['Distance', 'Safety', 'Beauty'])

plot_problem(problem, paths=[(get_best(res.X, res.F, 0), 'yellow'), (get_best(res.X, res.F, 1), 'green'), (get_best(res.X, res.F, 2), 'blue')])
