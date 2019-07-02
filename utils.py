import numpy as np


def ones_for_bias_trick(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)