import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork():
    def __init__(self, input_dims, layer1_nodes, layer2_nodes):
        self.W1 = np.random.randn(layer1_nodes, input_dims + 1) * 0.01
        self.W2 = np.random.randn(layer2_nodes, layer1_nodes + 1) * 0.01
        self.w_output = np.random.randn(layer2_nodes + 1, 1) * 0.01

    def ones_for_bias_trick(self, X):
        return np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

    def ones_for_bias_trick_batched(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1, 1))], axis=1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, y):
        return y * (1-y)

    def sigmoid_derivative_matrix(self, y):
        return np.diag(np.squeeze(self.sigmoid_derivative(y)))

    def sigmoid_derivative_matrix_batched(self, Y):
        diags = self.sigmoid_derivative(Y)
        der_matrix = np.zeros((Y.shape[0], Y.shape[1], Y.shape[1]))
        der_matrix[:, np.arange(Y.shape[1]), np.arange(Y.shape[1])] = np.squeeze(diags)
        return der_matrix

    def binary_crossentropy_loss(self, y_output, y_target):
        return -(y_target * np.log(y_output) + (1 - y_target) * np.log(1 - y_output))

    def binary_crossentropy_derivative(self, y_output, y_target):
        return (y_output - y_target)/(y_output * (1 - y_output))

    def binary_crossentropy_loss_batched(self, y_output, y_target):
        return np.mean(-(y_target * np.log(y_output) + (1 - y_target) * np.log(1 - y_output)))

    def binary_crossentropy_derivative_batched(self, y_output, y_target):
        return np.divide(y_output - y_target, y_output * (1 - y_output))

    def forward_pass(self, x):
        x1 = self.ones_for_bias_trick(x)
        z1 = np.dot(self.W1, x1)
        y1 = self.sigmoid(z1)

        x2 = self.ones_for_bias_trick(y1)
        z2 = np.dot(self.W2, x2)
        y2 = self.sigmoid(z2)

        x3 = self.ones_for_bias_trick(y2)
        z3 = np.dot(self.w_output.T, x3)
        y_output = self.sigmoid(z3)

        return y_output, y2, y1, x1

    def forward_pass_batched(self, X):
        """
        :param X: Input matrix X is B x D x 1 where D is input dimensionality, B is batch size!
        :return:
        """
        # print('X', X.shape)
        X1 = self.ones_for_bias_trick_batched(X)
        # print('X1', X1.shape)
        Z1 = np.matmul(self.W1, X1)
        # print('Z1', Z1.shape)
        Y1 = self.sigmoid(Z1)
        # print('Y1', Y1.shape)

        X2 = self.ones_for_bias_trick_batched(Y1)
        # print('X2', X2.shape)
        Z2 = np.matmul(self.W2, X2)
        # print('Z2', Z2.shape)
        Y2 = self.sigmoid(Z2)
        # print('Y2', Y2.shape)

        X3 = self.ones_for_bias_trick_batched(Y2)
        # print('X3', X3.shape)
        Z3 = np.matmul(self.w_output.T, X3)
        # print('Z3', Z3.shape)
        y_output = self.sigmoid(Z3)
        y_output = np.squeeze(y_output)
        # print('y_output', y_output.shape)

        return y_output, Y2, Y1, X1

    def backward_pass_batched(self, y_output, Y2, Y1, X1, y_target):
        X3 = self.ones_for_bias_trick_batched(Y2)
        # print('X3', X3.shape)
        X2 = self.ones_for_bias_trick_batched(Y1)
        # print('X2', X3.shape)

        dy_output = self.binary_crossentropy_derivative_batched(y_output, y_target)
        # print('dy_output', dy_output.shape)
        sigmoid_derivative3 = self.sigmoid_derivative(y_output)
        # print('sigmoid der 3', sigmoid_derivative3.shape)
        dz3 = np.expand_dims(np.expand_dims(sigmoid_derivative3 * dy_output, axis=-1), axis=-1)
        # print('dz3', dz3.shape)
        dw_output_batch = dz3 * X3
        # print('dw_output_batch', dw_output_batch.shape)
        dw_output = np.mean(dw_output_batch, axis=0)
        # print('dw_output', dw_output.shape)

        sigmoid_derivative_matrix2 = self.sigmoid_derivative_matrix_batched(Y2)
        # print('sigmoid der 2', sigmoid_derivative_matrix2.shape)
        dz2 = np.matmul(sigmoid_derivative_matrix2,
                        np.matmul(np.tile(self.w_output[:-1], [Y2.shape[0], 1, 1]), dz3))
        # print('dz2', dz2.shape)
        dW2_batch = np.matmul(dz2, np.transpose(X2, axes=[0, 2, 1]))
        # print('dW2 batch', dW2_batch.shape)
        dW2 = np.mean(dW2_batch, axis=0)
        # print('dW2', dW2.shape)

        sigmoid_derivative_matrix1 = self.sigmoid_derivative_matrix_batched(Y1)
        # print('sigmoid der 1', sigmoid_derivative_matrix1.shape)
        dz1 = np.matmul(sigmoid_derivative_matrix1,
                        np.matmul(np.tile(self.W2[:, :-1].T, [Y1.shape[0], 1, 1]), dz2))
        # print('dz1', dz1.shape)
        dW1_batch = np.matmul(dz1, np.transpose(X1, axes=[0, 2, 1]))
        # print('dW1 batch', dW1_batch.shape)
        dW1 = np.mean(dW1_batch, axis=0)
        # print('dW1', dW1.shape)

        return dw_output, dW1, dW2

    def backward_pass(self, y_output, y2, y1, x1, y_target):
        x3 = self.ones_for_bias_trick(y2)
        x2 = self.ones_for_bias_trick(y1)

        dy_output = self.binary_crossentropy_derivative(y_output, y_target)
        sigmoid_derivative3 = self.sigmoid_derivative(y_output)
        # dz3 = y_output - y_target
        dz3 = sigmoid_derivative3 * dy_output
        dw_output = dz3 * x3

        sigmoid_derivative_matrix2 = self.sigmoid_derivative_matrix(y2)
        dz2 = np.matmul(sigmoid_derivative_matrix2, self.w_output[:-1] * dz3)
        dW2 = np.matmul(dz2, x2.T)

        sigmoid_derivative_matrix1 = self.sigmoid_derivative_matrix(y1)
        dz1 = np.matmul(sigmoid_derivative_matrix1, np.matmul(self.W2[:, :-1].T, dz2))
        dW1 = np.matmul(dz1, x1.T)

        return dw_output, dW1, dW2

    def create_batches(self, X, y, batch_size):
        batch_indices = np.random.choice(np.arange(X.shape[0]), batch_size, replace=False)
        X_batch = X[batch_indices]
        X_batch = np.expand_dims(X_batch, axis=2)
        y_batch = y[batch_indices]
        return X_batch, y_batch

    def fit_batched(self, X, y, lr, epochs, steps_per_epoch, batch_size,
                    visualise_training=False):
        epoch_losses = []
        for epoch in range(epochs):
            losses = []
            for step in range(steps_per_epoch):
                X_batch, y_target_batch = self.create_batches(X, y, batch_size)
                y_output, Y2, Y1, X1 = self.forward_pass_batched(X_batch)
                dw_output, dW1, dW2 = self.backward_pass_batched(y_output, Y2, Y1, X1,
                                                                 y_target_batch)

                self.w_output = self.w_output - lr * dw_output
                self.W1 = self.W1 - lr * dW1
                self.W2 = self.W2 - lr * dW2
                loss = self.binary_crossentropy_loss_batched(y_output, y_target_batch)
                losses.append(loss)
            epoch_losses.append(np.mean(losses))

        if visualise_training:
            plt.figure()
            plt.plot(np.arange(1, epochs + 1), epoch_losses, label='Training')
            plt.legend()
            plt.show()

    def fit(self, X, y, lr, epochs, visualise_training=False):
        epoch_losses = []
        for epoch in range(epochs):
            losses = []
            for i in range(X.shape[0]):
                x = X[i, :]
                y_target = y[i]
                x = np.expand_dims(x, axis=-1)
                y_output, y2, y1, x1 = self.forward_pass(x)
                dw_output, dW1, dW2 = self.backward_pass(y_output, y2, y1, x1, y_target)

                self.w_output = self.w_output - lr * dw_output
                self.W1 = self.W1 - lr * dW1
                self.W2 = self.W2 - lr * dW2
                loss = self.binary_crossentropy_loss(y_output, y_target)
                losses.append(loss)
            epoch_losses.append(np.mean(losses))

        if visualise_training:
            plt.figure()
            plt.plot(np.arange(1, epochs + 1), epoch_losses, label='Training')
            plt.legend()
            plt.show()

    def predict_on_grid(self, xx, yy):
        batch_size = 10
        X_visualise = np.stack([xx, yy], axis=-1)
        X_visualise = np.reshape(X_visualise,
                                 (X_visualise.shape[0] * X_visualise.shape[1], -1))
        Z_visualise = []
        for i in range(int(X_visualise.shape[0]/batch_size)):
            x_visualise = np.expand_dims(X_visualise[batch_size*i:batch_size*(i+1), :], axis=2)
            forward_out, _, _, _ = self.forward_pass_batched(x_visualise)
            Z_visualise.append(forward_out)
        Z_visualise = np.array(Z_visualise)
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
        surf = axes.plot_surface(xx, yy, Z_visualise, cmap='RdBu',
                                 linewidth=0, antialiased=False)
        axes.scatter(X[y == 0, 0], X[y == 0, 1], zs=0, c='r', label='Class 0')
        axes.scatter(X[y == 1, 0], X[y == 1, 1], zs=0, c='b', label='Class 1')
        plt.show()
