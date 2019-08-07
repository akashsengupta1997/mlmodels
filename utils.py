import numpy as np


def ones_for_bias_trick(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def split_train_val(X, y, train_ratio=0.75):
    num_samples = X.shape[0]
    shuffled_indices = np.random.permutation(num_samples)
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    num_training_samples = int(num_samples * train_ratio)
    X_train = X[:num_training_samples]
    X_val = X[num_training_samples:]
    y_train = y[:num_training_samples]
    y_val = y[num_training_samples:]

    return X_train, y_train, X_val, y_val
