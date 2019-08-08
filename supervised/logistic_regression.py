import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import ones_for_bias_trick


class LogisticRegressor():
    def __init__(self, basis_function, *basis_function_args):
        self.basis_function = basis_function
        self.basis_function_args= basis_function_args

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def compute_gradient(self, X, y, weights):
        sigmoid = self.sigmoid(np.dot(X, weights))
        return np.dot(X.T, y - sigmoid)

    def compute_log_likelihood(self, X, y, weights):
        Z = self.sigmoid(np.dot(X, weights))
        epsilon = np.finfo(float).eps
        Z = np.clip(Z, epsilon, 1.0-epsilon)

        ll_all = y * np.log(Z) + (1 - y) * np.log(1 - Z)
        return np.sum(ll_all)

    def fit(self, X_train, y_train, lr, epochs,  X_val, y_val, optimiser='gd',
            visualise_training=False):
        """

        :param X:
        :param y:
        :param lr:
        :param epochs:
        :param optimiser:
        :param visualise_training:
        :param train_test_ratio:
        :return:
        """
        assert optimiser in ['gd', 'newton'], "Invalid optimiser!"

        if self.basis_function is not None:
            X_train = self.basis_function(X_train, *self.basis_function_args)
            X_val = self.basis_function(X_val, *self.basis_function_args)

        X_train = ones_for_bias_trick(X_train)
        X_val = ones_for_bias_trick(X_val)

        weights = np.random.randn(X_train.shape[1]) * 0.5
        train_log_likelihoods = []
        test_log_likelihoods = []

        for epoch in range(epochs):
            print("Epoch:", epoch)
            gradient = self.compute_gradient(X_train, y_train, weights)
            weights = weights + lr * gradient
            if visualise_training:
                train_log_likelihoods.append(self.compute_log_likelihood(X_train,
                                                                         y_train,
                                                                         weights))
                test_log_likelihoods.append(self.compute_log_likelihood(X_val,
                                                                        y_val,
                                                                        weights))
        self.weights = weights
        if visualise_training:
            plt.figure()
            plt.plot(np.arange(1, epochs+1), train_log_likelihoods, label='Training')
            plt.plot(np.arange(1, epochs+1), test_log_likelihoods, label='Test')
            plt.legend()
            plt.show()

    def predict(self, X):
        """
        Returns probability of each input in X belonging to class 1.
        :param X: test input matrix - rows = test inputs, columns = features
        :return:
        """
        if self.basis_function is not None:
            X = self.basis_function(X, *self.basis_function_args)

        X = ones_for_bias_trick(X)
        return self.sigmoid(np.dot(X, self.weights))

    def predict_on_grid(self, xx, yy):
        """

        :param X:
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :return:
        """
        X_visualise = np.stack([xx, yy], axis=-1)
        X_visualise = np.reshape(X_visualise,
                                 (X_visualise.shape[0] * X_visualise.shape[1], -1))
        Z_visualise = self.predict(X_visualise)
        Z_visualise = Z_visualise.reshape(xx.shape)
        return Z_visualise

    def visualise_2d_input_scatter(self, X, y, show=False):
        plt.figure()
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', label='Class 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Class 1')
        if show:
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.show()

    def visualise_2d_input_contour(self, X, y):
        self.visualise_2d_input_scatter(X, y, show=False)
        axes = plt.gca()
        x_min, x_max = 1.2 * np.array(axes.get_xlim())
        y_min, y_max = 1.2 * np.array(axes.get_ylim())
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z_visualise = self.predict_on_grid(xx, yy)
        cs = plt.contour(xx, yy, Z_visualise, cmap='RdBu', linewidths=1, levels=10)
        plt.clabel(cs, fmt='%2.1f', colors='k', fontsize=14)
        axes.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def visualise_2d_input_surface(self, X, y):
        fig = plt.figure()
        axes = Axes3D(fig)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z_visualise = self.predict_on_grid(xx, yy)
        surf = axes.plot_surface(xx, yy, Z_visualise,cmap='RdBu',
                                 linewidth=0, antialiased=False)
        axes.scatter(X[y == 0, 0], X[y == 0, 1], zs=0, c='r', label='Class 0')
        axes.scatter(X[y == 1, 0], X[y == 1, 1], zs=0, c='b', label='Class 1')
        plt.show()





