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

    def split_training_test(self, X, y, ratio):
        total_samples = X.shape[0]
        num_training_samples = int(total_samples * ratio)
        train_indices = np.random.choice(total_samples, (num_training_samples, ), False)
        X_train = X[train_indices]
        y_train = y[train_indices]
        return X_train, y_train

    def fit(self, X, y, lr, epochs, optimiser='gd', visualise=False, train_test_ratio=0.7):
        """

        :param X:
        :param y:
        :param lr:
        :param optimiser:
        :param visualise:
        :return:
        """
        assert optimiser in ['gd', 'newton'], "Invalid optimiser!"
        X_orig = X
        if self.basis_function is not None:
            X = self.basis_function(X, *self.basis_function_args)

        X = ones_for_bias_trick(X)
        X_train, y_train = self.split_training_test(X, y, train_test_ratio)
        weights = np.random.randn(X.shape[1])

        for epoch in range(epochs):
            print("Epoch:", epoch)
            gradient = self.compute_gradient(X_train, y_train, weights)
            weights = weights + lr * gradient

        self.weights = weights

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




