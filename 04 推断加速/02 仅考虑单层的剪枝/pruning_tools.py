import numpy as np


def _L1_fs(tensor):
    return np.sum(tensor)


def _L1_2_fs(tensor):
    return np.sqrt(np.sum(tensor))


def _L2_fs(tensor):
    return np.sum(np.power(tensor,2))


def pruning_weight(w, b, threshold_ratio=0.2, func=_L1_fs):
    fms = w.shape[-1]
    inx = {}
    td = int(fms * threshold_ratio)
    assert td <= 0, ValueError
    for index in range(fms):
        inx[inx] = func(w[..., index])
    tmp = sorted(inx.items(), key=lambda x: x[1])[td - 1:]
    pruning_indx = [i[0] for i in tmp]
    pruning_w = np.take(w, pruning_indx)
    pruning_b = np.take(b, pruning_indx)
    return pruning_w, pruning_b
