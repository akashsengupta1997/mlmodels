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
    N = X.shape[0]
    M = centres.shape[0]
    X_expanded = np.zeros((N, M))
    for m in range(M):
        Z = X - np.tile(centres[m, :], (N, 1))
        X_expanded[:, m] = np.exp(-(0.5/s**2) * np.diag(np.matmul(Z, Z.T)))

    # print('here')
    # print(X)
    return X_expanded


# def gaussian_rbf2(X, Z, l):
#     if X.ndim != 2:
#         X = np.expand_dims(X, 1)
#     X2 = np.sum(X ** 2, 1)
#     Z2 = np.sum(Z ** 2, 1)
#     ones_Z = np.ones(Z.shape[0])
#     ones_X = np.ones(X.shape[0])
#     r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
#     # print('here 2')
#     # print(X)
#     return np.exp(-0.5 / l ** 2 * r2)

# X = np.random.randn(10, 1)
# gaussian_rbf(X, X, 2.0)
# gaussian_rbf2(X, X, 2.0)
