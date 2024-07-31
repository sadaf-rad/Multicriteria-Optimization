from matplotlib import pyplot as plt
import numpy as np

from generate_data import BicycleRouting

def plot_problem(problem, path=None, paths=[]):
    _map = (problem.get_map() - 1) / 4
    _map = np.array([np.zeros(_map[0].shape), _map[0], _map[1]])
    _map = np.moveaxis(_map, [0, 1, 2], [2, 0, 1])
    _map = 1 - (1 - _map) ** 1
    plt.imshow(_map, origin='lower', extent=(0, 1, 0, 1))

    o = problem.nodes[0]
    plt.scatter([o[0]], o[1], c='red')
    
    for o in problem.nodes[1:problem.len_route]:
        plt.scatter([o[0]], o[1], c='orange')
    
    for o in problem.nodes[problem.len_route:]:
        plt.scatter([o[0]], o[1], c='yellow')

    if path is not None:
        cutoff = problem.get_cutoff(path)
        x = [problem.nodes[path[i]][0] for i in range(cutoff + 1)]
        y = [problem.nodes[path[i]][1] for i in range(cutoff + 1)]
        plt.plot(x, y)
    
    for i, (path, color) in enumerate(paths):
        cutoff = problem.get_cutoff(path)
        x = [problem.nodes[path[i]][0] for i in range(cutoff + 1)]
        y = [problem.nodes[path[i]][1] for i in range(cutoff + 1)]
        plt.plot(x, y, '--', color=color, linewidth=2 * 0.8 ** i)

    plt.show(block=True)

if __name__ == '__main__':
    plot_problem(BicycleRouting(15, 7, seed=42))
