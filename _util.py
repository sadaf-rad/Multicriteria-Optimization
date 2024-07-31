import numpy as np

def get_non_dominated(F):
    res = []
    for cand in F:
        _cand = np.expand_dims(cand, 0)
        if not np.any(np.all(F <= _cand, axis=1) & np.any(F < _cand, axis=1), axis=0):
            res.append(cand)
    return np.array(res)

if __name__ == '__main__':
    F = np.array([
        [3, 3, 3],
        [1, 2, 3],
        [3, 2, 1],
        [2, 2, 2],
        [2, 2, 3]])
    print('F\n', F)
    print('non dominated F\n', get_non_dominated(F))
