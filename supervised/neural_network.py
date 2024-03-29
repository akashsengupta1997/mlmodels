import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import math


class NeuralNetwork():
    def __init__(self, input_dims, hidden_layer_nodes, hidden_layer_activations,
                 output_activation, weight_init_var=0.5, import_weights_file=None,
                 output_dims=1):
        self.num_hidden_layers = len(hidden_layer_nodes)
        self.output_activation = output_activation

        if import_weights_file is not None:
            with open(import_weights_file, 'rb') as f:
                self.hidden_layer_weights, self.output_weights, self.hidden_layer_activations = pickle.load(f)

        else:
            self.hidden_layer_weights = self.initialise_weights(hidden_layer_nodes,
                                                                input_dims,
                                                                weight_init_var=weight_init_var)
            if self.output_activation == 'sigmoid':
                self.output_weights = np.random.randn(hidden_layer_nodes[-1] + 1, 1) * weight_init_var
            elif self.output_activation == 'softmax':
                self.output_weights = np.random.randn(output_dims, hidden_layer_nodes[-1] + 1) * weight_init_var
            self.hidden_layer_activations = hidden_layer_activations

    def initialise_weights(self, layer_nodes, input_dims, weight_init_var):
        """
        Initalise neural network weights using 0-mean Gaussian pdf with variance equal to
        weight_unit_var.
        :param layer_nodes: number of nodes/neurons in each layer.
        :param input_dims: dimensionality of input.
        :param weight_init_var: variance of initialising Gaussian pdf.
        :return: list of initialiased layer weights.
        """
        layer_weights = [None] * self.num_hidden_layers
        layer_weights[0] = np.random.randn(layer_nodes[0], input_dims + 1) * weight_init_var
        for layer in range(1, self.num_hidden_layers):
            layer_weights[layer] = np.random.randn(layer_nodes[layer],
                                                   layer_nodes[layer - 1] + 1) * weight_init_var

        return layer_weights

    def ones_for_bias_trick(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1, 1))], axis=1)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, y):
        return y * (1-y)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, y):
        der = y
        der[der > 0] = 1
        return der

    def softmax(self, z):
        """
        Computes softmax of x in a numerically stable way (preventing NaNs due to too-large
        exponents).
        :param z: (batch size, num classes, 1) array.
        :return: softmax(z)
        """
        # All the weird expanding and tiling is necessary to deal with batches...
        num_classes = z.shape[1]
        shifted_z = z - np.tile(np.expand_dims(np.amax(z, axis=1), axis=-1),
                                [1, num_classes, 1])
        exps = np.exp(shifted_z)
        sum = np.tile(np.expand_dims(np.sum(exps, axis=1), axis=-1), [1, num_classes, 1])
        softmax = np.divide(exps, sum)
        return softmax

    def softmax_derivative_matrix(self, Y):
        """
        Creates activation derivative matrix for softmax (which is not one-to-one).
        :param Y: array containing batch of softmax outputs from (output) neural network layer.
        Has shape (batch_size, num_classes, 1)
        :return: Array containing batch of activation derivative matrices with shape
        (batch_size, num_classes, num_classes)
        """
        diags = Y * (1-Y)
        der_matrix = - np.matmul(Y, np.transpose(Y, axes=[0, 2, 1]))
        der_matrix[:, np.arange(Y.shape[1]), np.arange(Y.shape[1])] = np.squeeze(diags)
        return der_matrix

    def activation_derivative_matrix(self, Y, activation_derivative):
        """
        Creates activation derivative matrix for one-to-one activation functions, such as
        sigmoid and relu. Not for use with softmax (which is not one-to-one).
        :param Y: array containing batch of activations from neural network layer.
        :param activation_derivative: function to compute derivative, depending on activation
        used.
        :return: Array containing batch of activation derivative matrices with shape
        (batch_size, Y.shape[1], Y.shape[1])
        """
        diags = activation_derivative(Y)
        der_matrix = np.zeros((Y.shape[0], Y.shape[1], Y.shape[1]))
        der_matrix[:, np.arange(Y.shape[1]), np.arange(Y.shape[1])] = np.squeeze(diags)
        return der_matrix

    def binary_crossentropy_loss(self, y_output, y_target):
        """
        :param y_output: (batch size,) array of neural network outputs.
        :param y_target: (batch size,) array of target outputs.
        :return: Binary cross-entropy loss between batch of outputs and targets.
        """
        return np.mean(-(y_target * np.log(y_output) + (1 - y_target) * np.log(1 - y_output)))

    def binary_crossentropy_derivative(self, y_output, y_target):
        """
        :param y_output: (batch size,) array of neural network outputs.
        :param y_target: (batch size,) array of target outputs.
        :return: Derivative of binary cross-entropy loss w.r.t y_output
        (array of same shape as y_output)
        """
        return np.divide(y_output - y_target, y_output * (1 - y_output))

    def categorical_crossentropy_loss(self, y_output, y_target):
        """
        :param y_output: (batch size, num classes, 1) array of neural network outputs
        :param y_target: (batch size, num classes, 1) array of OHE target outputs
        :return: Multi-class cross-entropy loss between batch of outputs and targets.
        """
        y_target_transpose = np.transpose(y_target, axes=[0, 2, 1])
        log_y_output = np.log(y_output)
        return -np.mean(np.matmul(y_target_transpose, log_y_output))

    def categorical_crossentropy_derivative(self, y_output, y_target):
        """
        :param y_output: (batch size, num classes, 1) array of neural network outputs
        :param y_target: (batch size, num classes, 1) array of target outputs
        :return: Derivative of CCE loss w.r.t. y_output (array of same shape as y_output)
        """
        return -np.divide(y_target, y_output)

    def forward_pass(self, X):
        """
        Perform one forward pass of a single batch of training data.
        :param X: (batch_size, input_dims, 1) array of training data.
        :return: neural network output and outputs of intermediate hidden layers (to be used
        for backpropagation).
        """
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

        if self.output_activation == 'sigmoid':
            Z_final = np.matmul(self.output_weights.T, X)
            y_output = np.squeeze(self.sigmoid(Z_final))

        elif self.output_activation == 'softmax':
            Z_final = np.matmul(self.output_weights, X)
            y_output = self.softmax(Z_final)

        return hidden_layer_outputs, y_output

    def backward_pass(self, hidden_layer_outputs, y_output, y_target, X,
                      loss_function_derivative):
        """
        Perform one backward pass of a single batch of training data.
        :param hidden_layer_outputs: obtained during forward pass.
        :param y_output: neural network output, obtained during forward pass.
        :param y_target: target label.
        :param X: input training data.
        :param loss_function_derivative: function that outputs derivative of loss function
        w.r.t neural network output y_output.
        :return: derivatives of loss function w.r.t to weights - used for updating weights in
        gradient descent.
        """
        hidden_layer_inputs = [self.ones_for_bias_trick(Y) for Y in
                               [X] + hidden_layer_outputs]
        hidden_layer_weight_updates = [None] * self.num_hidden_layers

        dy_output = loss_function_derivative(y_output, y_target)
        if self.output_activation == 'sigmoid':
            final_activation_derivative = self.sigmoid_derivative(y_output)
            dz = np.expand_dims(
                np.expand_dims(final_activation_derivative * dy_output, axis=-1),
                axis=-1)
            dw_output_batch = dz * hidden_layer_inputs[-1]
            dw_output = np.mean(dw_output_batch, axis=0)
        elif self.output_activation == 'softmax':
            final_activation_derivative = self.softmax_derivative_matrix(y_output)
            dz = np.matmul(final_activation_derivative, dy_output)
            dw_output_batch = np.matmul(dz,
                                  np.transpose(hidden_layer_inputs[-1],
                                               [0, 2, 1]))
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
                if self.output_activation == 'sigmoid':
                    dz_dy_final = self.output_weights[:-1]
                elif self.output_activation == 'softmax':
                    dz_dy_final = self.output_weights[:, :-1].T
                dz = np.matmul(activation_derivative_matrix,
                               np.matmul(np.tile(dz_dy_final, [Y.shape[0], 1, 1]),
                                         dz))
            else:
                W = self.hidden_layer_weights[layer + 1]

                dz = np.matmul(activation_derivative_matrix,
                               np.matmul(np.tile(W[:, :-1].T, [Y.shape[0], 1, 1]), dz))

            dW_batch = np.matmul(dz, np.transpose(X, axes=[0, 2, 1]))
            dW = np.mean(dW_batch, axis=0)
            hidden_layer_weight_updates[layer] = dW

        return hidden_layer_weight_updates, dw_output

    def create_batches(self, X, y, batch_size, step):
        """
        Create sequential batches of data with size = batch_size.
        :param X: Input data array with shape (num_samples, input_dims, 1). Should be SHUFFLED
        to prevent ordering of sequential batches from being an issue.
        :param y: Target array with shape (num_samples,) or (num_samples, num_classes, 1).
        Should be SHUFFLED to prevent ordering of sequential batches from being an issue.
        :param batch_size: number of samples in one minibatch
        :param step: current step number in current epoch.
        :return: Batch of training samples and targets.
        """
        index_start = batch_size * step
        index_end = index_start + batch_size
        return np.expand_dims(X[index_start:index_end], axis=-1), y[index_start:index_end]

    def shuffle_training_data(self, X, y):
        """
        Shuffle input data and labels (along first axis)
        :param X: input data array
        :param y: target labels array.
        :return: Shuffled X and y.
        """
        num_samples = X.shape[0]
        shuffled_indices = np.random.permutation(num_samples)
        X = X[shuffled_indices]
        y = y[shuffled_indices]
        return X, y

    def fit(self, X_train, y_train, lr, epochs, batch_size, X_val, y_val, val_batch_size,
            loss_function, visualise_training=False, epochs_per_save=5, save_name=None,
            compute_accuracy=False, display_metrics=False):
        """
        Fit neural network to training data using stochastic gradient descent.
        :param X_train: (num_samples, input_dims, 1) array of training inputs.
        :param y_train: (num_samples,) or (num_samples, output_dims, 1) array of target labels
        :param lr: learning rate.
        :param epochs: number of training epochs.
        :param batch_size: minibatch size.
        :param X_val: (num_validation_samples, input_dims, 1) array of validation inputs.
        :param y_val: (num_validation)samples,) or (num_validation)samples, output_dims, 1)
        array of validation target labels
        :param val_batch_size: validation minibatch size.
        :param loss_function: loss function to be used between outputs and targets.
        :param visualise_training: bool flag - visualise loss/accuracy curves?
        :param epochs_per_save: how many epochs per save.
        :param save_name: save name string.
        :param compute_accuracy: bool flag - compute classification accuracy?
        :param display_metrics: bool flag - print loss/accuracy every epoch to console?
        """
        epoch_losses_train = []
        epoch_losses_val = []
        accs_train = []
        accs_val = []
        machine_eps = np.finfo(float).eps
        steps_per_epoch = math.ceil(X_train.shape[0]/float(batch_size))
        val_steps_per_epoch = math.ceil(X_val.shape[0]/float(val_batch_size))

        if loss_function == 'binary_crossentropy':
            loss_func = self.binary_crossentropy_loss
            loss_func_derivative = self.binary_crossentropy_derivative
        elif loss_function == 'categorical_crossentropy':
            loss_func = self.categorical_crossentropy_loss
            loss_func_derivative = self.categorical_crossentropy_derivative

        # Epochs
        for epoch in range(epochs):
            losses_train = []
            losses_val = []
            matches_train = 0
            matches_val = 0
            X_train, y_train = self.shuffle_training_data(X_train, y_train)

            # Batches
            for step in range(steps_per_epoch):
                X_batch, y_target_batch = self.create_batches(X_train, y_train, batch_size,
                                                              step)
                hidden_layer_outputs, y_output = self.forward_pass(X_batch)
                y_output = np.clip(y_output, machine_eps, 1.0-machine_eps)
                hidden_layer_weight_updates, dw_output = self.backward_pass(hidden_layer_outputs,
                                                                            y_output,
                                                                            y_target_batch,
                                                                            X_batch,
                                                                            loss_func_derivative)
                for layer in range(self.num_hidden_layers):
                    W = self.hidden_layer_weights[layer]
                    dW = hidden_layer_weight_updates[layer]
                    self.hidden_layer_weights[layer] = W - lr * dW
                self.output_weights = self.output_weights - lr * dw_output
                loss = loss_func(y_output, y_target_batch)
                losses_train.append(loss)
                if compute_accuracy:
                    ohe = (self.output_activation == 'softmax')
                    matches_train += self.count_correct_classifications(y_output,
                                                                        y_target_batch,
                                                                        ohe=ohe)
            epoch_losses_train.append(np.mean(losses_train))

            for step in range(val_steps_per_epoch):
                X_batch, y_target_batch = self.create_batches(X_val, y_val, val_batch_size,
                                                              step)
                _, y_output = self.forward_pass(X_batch)
                y_output = np.clip(y_output,machine_eps, 1.0-machine_eps)
                loss = loss_func(y_output, y_target_batch)
                losses_val.append(loss)
                if compute_accuracy:
                    ohe = (self.output_activation == 'softmax')
                    matches_val += self.count_correct_classifications(y_output,
                                                                      y_target_batch,
                                                                      ohe=ohe)
            epoch_losses_val.append(np.mean(losses_val))

            if display_metrics:
                print("Epoch", epoch)
                print("Training loss", np.mean(losses_train))
                print("Validation loss", np.mean(losses_val))

            if compute_accuracy:
                train_acc = self.compute_accuracy(batch_size, steps_per_epoch,
                                                  matches_train)
                val_acc = self.compute_accuracy(val_batch_size, val_steps_per_epoch,
                                                matches_val)
                if display_metrics:
                    print("Training acc", train_acc)
                    print("Validation acc", val_acc)
                accs_train.append(train_acc)
                accs_val.append(val_acc)

            if epochs_per_save > 0 and epoch % epochs_per_save == 0:
                if save_name is not None:
                    with open(save_name, 'wb') as f:
                        pickle.dump([self.hidden_layer_weights,
                                     self.output_weights,
                                     self.hidden_layer_activations],
                                    f)
                        print('Saved!')

        if visualise_training:
            plt.figure(1)
            plt.plot(np.arange(1, epochs + 1), epoch_losses_train, label='Train')
            plt.plot(np.arange(1, epochs + 1), epoch_losses_val, label='Validation')
            plt.legend()
            plt.show()

            if compute_accuracy:
                plt.figure(2)
                plt.plot(np.arange(1, epochs + 1), accs_train, label='Train')
                plt.plot(np.arange(1, epochs + 1), accs_val, label='Validation')
                plt.legend()
                plt.show()

    def count_correct_classifications(self, y_output, y_target, ohe=False):
        if ohe:
            y_output = np.squeeze(y_output)
            y_target = np.squeeze(y_target)
            output_classes = np.argmax(y_output, axis=-1)
            target_classes = np.argmax(y_target, axis=-1)
            matches = np.sum(output_classes == target_classes)
        else:
            y_output = np.around(y_output)
            matches = np.sum(y_output == y_target)

        return matches

    def compute_accuracy(self, batch_size, steps_per_epoch, total_matches):
        total_samples = batch_size * steps_per_epoch
        return total_matches/float(total_samples)

    def predict(self, X, batch_size):
        steps = math.ceil(X.shape[0]/float(batch_size))
        for step in range(steps):
            X_batch = np.expand_dims(X[batch_size*step:batch_size*(step+1)], axis=-1)
            _, output = self.forward_pass(X_batch)
            if step == 0:
                outputs = np.array(output)
            else:
                outputs = np.concatenate([outputs, output])
        return outputs

    # --------- Functions for Visualisation ---------

    def predict_on_grid(self, xx, yy):
        X_visualise = np.stack([xx, yy], axis=-1)
        X_visualise = np.reshape(X_visualise,
                                 (X_visualise.shape[0] * X_visualise.shape[1], -1))

        visualise_outputs = self.predict(X_visualise, 10)
        visualise_outputs = visualise_outputs.reshape(xx.shape)

        return visualise_outputs

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
