import numpy as np


def scalar_polynomial(X, degree):
    """

    :param X:
    :param degree:
    :return:
    """
    assert len(X.shape) <= 2, "scalar polynomial basis function only takes 1D input features"
    if len(X.shape) == 2:
        assert X.shape[1] == 1, "scalar polynomial basis function only takes 1D input features"
    degree = degree[0]
    X_expanded = np.zeros((X.shape[0], degree))

    for d in range(degree):
        X_expanded[:, d] = np.squeeze(np.power(X, d+1))

    return X_expanded

