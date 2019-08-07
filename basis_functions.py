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
    X_expanded = np.zeros((X.shape[0], degree))

    for d in range(degree):
        X_expanded[:, d] = np.squeeze(np.power(X, d+1))

    return X_expanded


def gaussian_rbf(X, centres, s):
    """
    Gaussian radial basis function expansion, with RBFs centred around points given in centres.
    :param X:
    :param s:
    :return:
    """
    if X.ndim != 2:
        X = np.expand_dims(X, 1)
    X2 = np.sum(X ** 2, 1)
    Z2 = np.sum(centres ** 2, 1)
    ones_Z = np.ones(centres.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, centres.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / s ** 2 * r2)
