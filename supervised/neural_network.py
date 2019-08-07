import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork():
    def __init__(self, input_dims, hidden_layer_nodes, hidden_layer_activations,
                 weight_init_var=0.1):
        self.num_hidden_layers = len(hidden_layer_nodes)
        self.hidden_layer_weights = self.initialise_weights(hidden_layer_nodes,
                                                            input_dims,
                                                            weight_init_var=weight_init_var)
        self.output_weights = np.random.randn(hidden_layer_nodes[-1] + 1, 1) * weight_init_var
        self.hidden_layer_activations = hidden_layer_activations

    def initialise_weights(self, layer_nodes, input_dims, weight_init_var):
        layer_weights = [None] * self.num_hidden_layers
        layer_weights[0] = np.random.randn(layer_nodes[0], input_dims + 1) * weight_init_var
        for layer in range(1, self.num_hidden_layers):
            layer_weights[layer] = np.random.randn(layer_nodes[layer],
                                                   layer_nodes[layer - 1] + 1) * weight_init_var

        return layer_weights

    def ones_for_bias_trick(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1, 1))], axis=1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, y):
        return y * (1-y)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, y):
        der = y
        der[der > 0] = 1
        return der

    def activation_derivative_matrix(self, Y, activation_derivative):
        diags = activation_derivative(Y)
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

    def forward_pass(self, X):
        hidden_layer_outputs = [None] * self.num_hidden_layers
        X = self.ones_for_bias_trick(X)

        for layer in range(self.num_hidden_layers):
            W = self.hidden_layer_weights[layer]
            Z = np.matmul(W, X)
            activation = self.hidden_layer_activations[layer]
            if activation == 'relu':
                Y = self.relu(Z)
            elif activation == 'sigmoid':
                Y = self.sigmoid(Z)
            hidden_layer_outputs[layer] = Y
            X = self.ones_for_bias_trick(Y)

        Z_final = np.matmul(self.output_weights.T, X)
        y_output = np.squeeze(self.sigmoid(Z_final))
        return hidden_layer_outputs, y_output

    def backward_pass(self, hidden_layer_outputs, y_output, y_target, X):
        hidden_layer_inputs = [self.ones_for_bias_trick(Y) for Y in
                               [X] + hidden_layer_outputs]
        hidden_layer_weight_updates = [None] * self.num_hidden_layers

        dy_output = self.binary_crossentropy_derivative_batched(y_output, y_target)
        final_activation_derivative = self.sigmoid_derivative(y_output)
        dz = np.expand_dims(np.expand_dims(final_activation_derivative * dy_output, axis=-1),
                            axis=-1)
        dw_output_batch = dz * hidden_layer_inputs[-1]
        dw_output = np.mean(dw_output_batch, axis=0)

        for layer in reversed(range(self.num_hidden_layers)):
            X = hidden_layer_inputs[layer]
            Y = hidden_layer_outputs[layer]
            activation = self.hidden_layer_activations[layer]
            if activation == 'relu':
                activation_derivative_matrix = self.activation_derivative_matrix(Y,
                                                                                 self.relu_derivative)
            elif activation == 'sigmoid':
                activation_derivative_matrix = self.activation_derivative_matrix(Y,
                                                                                 self.sigmoid_derivative)

            if layer == self.num_hidden_layers - 1:
                dz = np.matmul(activation_derivative_matrix,
                               np.matmul(np.tile(self.output_weights[:-1], [Y.shape[0], 1, 1]),
                                         dz))
            else:
                W = self.hidden_layer_weights[layer + 1]

                dz = np.matmul(activation_derivative_matrix,
                               np.matmul(np.tile(W[:, :-1].T, [Y.shape[0], 1, 1]), dz))

            dW_batch = np.matmul(dz, np.transpose(X, axes=[0, 2, 1]))
            dW = np.mean(dW_batch, axis=0)
            hidden_layer_weight_updates[layer] = dW

        return hidden_layer_weight_updates, dw_output

    def create_batches(self, X, y, batch_size):
        batch_indices = np.random.choice(np.arange(X.shape[0]), batch_size, replace=False)
        X_batch = X[batch_indices]
        X_batch = np.expand_dims(X_batch, axis=2)
        y_batch = y[batch_indices]
        return X_batch, y_batch

    def fit(self, X_train, y_train, lr, epochs, steps_per_epoch, batch_size, X_val, y_val,
            val_steps_per_epoch, val_batch_size, visualise_training=False):
        epoch_losses_train = []
        epoch_losses_val = []

        for epoch in range(epochs):
            losses_train = []
            losses_val = []
            for step in range(steps_per_epoch):
                X_batch, y_target_batch = self.create_batches(X_train, y_train, batch_size)
                hidden_layer_outputs, y_output = self.forward_pass(X_batch)
                hidden_layer_weight_updates, dw_output = self.backward_pass(hidden_layer_outputs,
                                                                            y_output,
                                                                            y_target_batch,
                                                                            X_batch)
                for layer in range(self.num_hidden_layers):
                    W = self.hidden_layer_weights[layer]
                    dW = hidden_layer_weight_updates[layer]
                    self.hidden_layer_weights[layer] = W - lr * dW
                self.output_weights = self.output_weights - lr * dw_output
                loss = self.binary_crossentropy_loss_batched(y_output, y_target_batch)
                losses_train.append(loss)
            epoch_losses_train.append(np.mean(losses_train))

            for step in range(val_steps_per_epoch):
                X_batch, y_target_batch = self.create_batches(X_val, y_val, val_batch_size)
                _, y_output = self.forward_pass(X_batch)
                loss = self.binary_crossentropy_loss_batched(y_output, y_target_batch)
                losses_val.append(loss)
            epoch_losses_val.append(np.mean(losses_val))

        if visualise_training:
            plt.figure()
            plt.plot(np.arange(1, epochs + 1), epoch_losses_train, label='Train')
            plt.plot(np.arange(1, epochs + 1), epoch_losses_val, label='Validation')
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
            # forward_out, _, _, _, _ = self.forward_pass(x_visualise)
            _, forward_out = self.forward_pass(x_visualise)
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
