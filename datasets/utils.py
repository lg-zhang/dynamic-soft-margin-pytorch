import numpy as np
import torch

def np_rotate(x, mode):
    # counter-clockwise rotate0, rotate90, rotate180, rotate270
    if isinstance(x, np.ndarray):
        if mode == 1:
            x = np.transpose(x, [1, 0])
            x = x[::-1, :]
        elif mode == 2:
            x = x[::-1, :]
            x = x[:, ::-1]
        elif mode == 3:
            x = x[::-1, :]
            x = np.transpose(x, [1, 0])
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return [np_rotate(v, mode) for v in x]
    else:
        raise ValueError("np_rotate() only takes list or np.ndarray")


def np_flip(x, mode):
    if isinstance(x, np.ndarray):
        if mode == 1:  # lr
            x = x[:, ::-1]
        elif mode == 2:  # ud
            x = x[::-1, :]
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return [np_flip(v, mode) for v in x]
    else:
        raise ValueError("np_flip() only takes list or np.ndarray")
