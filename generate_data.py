import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from pymoo.core.problem import ElementwiseProblem


class BicycleRouting(ElementwiseProblem):

    def __init__(self, n_nodes, len_route, n_attrib=2, n_control_points=None, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        if n_control_points is None:
            n_control_points = n_nodes

        self.n_nodes = n_nodes
        self.len_route = len_route
        self.n_attrib = n_attrib
        self.n_control_points = n_control_points

        self.nodes = np.random.random((n_nodes, 2))
        self.nodes_dist = cdist(self.nodes, self.nodes)
        
        self.control_points = np.random.random((n_attrib, n_control_points, 2))
        self.cp_values = np.random.random((n_attrib, n_control_points))
        self.cp_values = normalize_matrix(self.cp_values, 1, 5, axis=1)
        
        self.node_values = self.get_attribute_values(self.nodes)

        super().__init__(
            n_var=n_nodes,
            n_obj=1 + n_attrib,
            xl=0,
            xu=n_nodes,
            vtype=int,
            **kwargs
        )

    def get_attribute_values(self, locations):
        cp_values_broascast = np.broadcast_to(np.expand_dims(self.cp_values, axis=2), (self.n_attrib, self.n_control_points, len(locations)))
        cp_dist = cdist(self.control_points.reshape((-1, 2)), locations).reshape(self.n_attrib, self.n_control_points, len(locations))
        return np.average(cp_values_broascast, axis=1, weights=1 / cp_dist ** 2)

    def get_map(self, width=100):
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, width))
        x, y = x.reshape(-1), y.reshape(-1)
        xy = np.array([x, y])
        xy = np.swapaxes(xy, 0, 1)
        values = self.get_attribute_values(xy)
        return values.reshape(self.n_attrib, width, width)

    def _evaluate(self, x, out, *args, **kwargs):
        dist, attrib_values = self.eval(x)
        out['F'] = [dist] + [-a for a in attrib_values]     # minimize objective

    def eval(self, x):
        dist = 0
        attrib_values = [0 for _ in range(self.n_attrib)]
        not_visited = set(range(self.len_route))
        if x[0] in not_visited:
            not_visited.remove(x[0])
        
        for k in range(self.n_nodes - 1):
            i, j = x[k], x[k + 1]
            dist += self.nodes_dist[i, j]
            if j in not_visited:
                not_visited.remove(j)
            
            for l in range(self.n_attrib):
                attrib_values[l] += self.nodes_dist[i, j] * (self.node_values[l][i] + self.node_values[l][j]) / 2
            
            if not not_visited:
                break

        for l in range(self.n_attrib):
            attrib_values[l] /= dist

        return dist, attrib_values

    def get_cutoff(self, x):
        not_visited = set(range(self.len_route))
        if x[0] in not_visited:
            not_visited.remove(x[0])
        
        if not not_visited:
            return 0
        
        for k in range(self.n_nodes - 1):
            i, j = x[k], x[k + 1]
            
            if j in not_visited:
                not_visited.remove(j)
            
            if not not_visited:
                return k + 1
        return k + 1

def normalize_matrix(M, low, high, axis):
    return low + (M - np.min(M, axis=axis, keepdims=True)) / (np.max(M, axis=axis, keepdims=True) - np.min(M, axis=axis, keepdims=True)) * (high - low)


if __name__ == '__main__':
    problem = BicycleRouting(15, 7)
