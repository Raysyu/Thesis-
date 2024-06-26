# Authors: Decebal Constantin Mocanu et al.;
# Code associated with SCADS Summer School 2020 tutorial "	Scalable Deep Learning Tutorial"; https://www.scads.de/de/summerschool2020
# This is a pre-alpha free software and was tested in Windows 10 with Python 3.7.6, Numpy 1.17.2, SciPy 1.4.1, Numba 0.48.0

# If you use parts of this code please cite the following article:
# @article{Mocanu2018SET,
#   author  =    {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#   journal =    {Nature Communications},
#   title   =    {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#   year    =    {2018},
#   doi     =    {10.1038/s41467-018-04316-3}
# }

# If you have space please consider citing also these articles

# @phdthesis{Mocanu2017PhDthesis,
#   title     =    "Network computations in artificial intelligence",
#   author    =    "D.C. Mocanu",
#   year      =    "2017",
#   isbn      =    "978-90-386-4305-2",
#   publisher =    "Eindhoven University of Technology",
# }

# @article{Liu2019onemillion,
#   author  =    {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#   journal =    {arXiv:1901.09181},
#   title   =    {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#   year    =    {2019},
# }

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs.
# This SET-MLP implementation was built on top of his MLP code:
# https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from nn_functions import *
from utils.monitor import Monitor
import datetime
import os
import copy
import time
import json
import sys
import numpy as np
from numpy.core.multiarray import ndarray
from numba import njit, prange


import matplotlib.pyplot as plt
import logging

# logging.basicConfig(filename=f'{__file__}.log', level=logging.INFO, format='%(asctime)s %(message)s', filemode='w')
# Alternatively one can log to stdout:
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(message)s', filemode='w')
log = logging.getLogger()
# log.setLevel(logging.INFO)

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask

# def create_sparse_weights(epsilon, n_row, n_col):
#     # Calculate the number of edges to generate
#     n_edges = int(epsilon * (n_row + n_col))
#     limit = np.sqrt(6. / float(n_row))
#     # Generate random degrees for rows and columns
#     row_degrees = np.random.poisson(1, size=n_row)
#     col_degrees = np.random.poisson(1, size=n_col)
#
#     # Make sure the sum of degrees is even
#     if (row_degrees.sum() + col_degrees.sum()) % 2 != 0:
#         row_degrees[0] += 1
#
#     # Create the sparse adjacency matrix
#     adj_matrix = lil_matrix((n_row, n_col))
#
#     # Generate random edges based on degrees
#     for _ in range(n_edges):
#         while True:
#             row = np.random.randint(n_row)
#             col = np.random.randint(n_col)
#             if adj_matrix[row, col] == 0:
#                 adj_matrix[row, col] = np.random.uniform(-limit)
#                 if row + 1 < n_row:
#                     adj_matrix[row + 1, col] = 1
#                 break
#
#     return adj_matrix.tocsr()

def create_sparse_weights(epsilon, n_rows, n_cols):
    # He uniform initialization
    limit = np.sqrt(6. / float(n_rows))

    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])

    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)

    log.info(
        f"Create sparse matrix with {weights.getnnz()} connections and {(weights.getnnz() / (n_rows * n_cols)) * 100} % density level")

    weights = weights.tocsr()
    return weights

def create_sparse_adjacency_matrix(epsilon, n_row, n_col):
    # Calculate the number of edges to generate
    n_edges = int(epsilon * (n_row + n_col))

    # Generate random degrees for rows and columns
    row_degrees = np.random.poisson(1, size=n_row)
    col_degrees = np.random.poisson(1, size=n_col)

    # Make sure the sum of degrees is even
    if (row_degrees.sum() + col_degrees.sum()) % 2 != 0:
        row_degrees[0] += 1

    # Create the sparse adjacency matrix
    adj_matrix = lil_matrix((n_row, n_col), dtype=bool)

    # Generate random edges based on degrees
    while n_edges > 0:
        row = np.random.randint(n_row)
        col = np.random.randint(n_col)
        if adj_matrix[row, col] == 0:
            adj_matrix[row, col] = 1
            n_edges -= 1
            if row + 1 < n_row:
                adj_matrix[row + 1, col] = 1
                n_edges -= 1

    return adj_matrix.tocsr()



def create_sparse_weights_normal_dist(epsilon, n_rows, n_cols):
    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
    weights[mask_weights >= prob] = np.float64(np.random.randn(n_params) / 10)
    log.info(
        f"Create sparse matrix with {weights.getnnz()} connections and {(weights.getnnz() / (n_rows * n_cols)) * 100} % density level")

    weights = weights.tocsr()
    return weights


def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    # TODO(Neil): not sure if we can asume uniqueness here
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return


class SET_MLP:
    def __init__(self, dimensions, activations, epsilon=20, init_network='uniform'):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes


        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------

        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.dropout_rate = 0.  # dropout rate
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed

        self.feature_selection_threshold = 0.3
        self.dimensions = dimensions

        self.save_filename = ""
        self.input_layer_connections = []
        self.weights_evolution = []
        self.monitor = None

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}

        if init_network == 'uniform':
            create_network = create_sparse_weights
        elif init_network == 'normal':
            create_network = create_sparse_weights_normal_dist
        else:
            raise ValueError("Unknown initialization method. Supports uniform and normal distribution")

        for i in range(len(dimensions) - 1):  # create sparse weight matrices
            self.w[i + 1] = create_network(self.epsilon, dimensions[i], dimensions[i + 1])
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
                    # apply dropout
                    a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                    masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = np.float32(1. - self.dropout_rate)

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')

        # compute backpropagation updates
        backpropagation_updates_numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(), np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            if keep_prob != 1:  # dropout for the backpropagation step
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
                delta /= keep_prob
            else:  # normal update
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            backpropagation_updates_numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(), np.mean(delta, axis=0))

        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if index not in self.pdw:
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropout_rate=0., testing=True, save_filename="", monitor=False,
            dropout=False, run_id=-1):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        # Initiate the loss object with the final activation function
        self.loss = loss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.dropout_rate = dropout_rate
        self.save_filename = save_filename

        maximum_accuracy = 0
        metrics = np.zeros((epochs, 4))

        for epoch in range(epochs):
            # Save the entire state of the weights.
            #
            # It can be used for comparing the different topologies of the different networks
            # and can be used for feature selection
            # A deepcopy is required to not overwrite the previous results.
            self.weights_evolution.append(copy.deepcopy(self.w))

            t1 = datetime.datetime.now()
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                end = (j + 1) * batch_size
                z, a, masks = self._feed_forward(x_[k:end], dropout)

                self._back_prop(z, a, masks, y_[k:end])

            t2 = datetime.datetime.now()

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if testing:
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test, batch_size=batch_size)
                accuracy_train, activations_train = self.predict(x, y_true, batch_size=batch_size)

                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[epoch, 0] = loss_train
                metrics[epoch, 1] = loss_test
                metrics[epoch, 2] = accuracy_train
                metrics[epoch, 3] = accuracy_test

                log.info(f"[run_id={run_id}] ----------------")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Training time: {t2 - t1}s")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Loss train: {loss_train}")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Accuracy train: {accuracy_train}")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Testing time: {t4 - t3}s")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Loss test: {loss_test}")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Accuracy test: {accuracy_test}")
                log.info(f"[run_id={run_id}] [eps={self.epsilon}] Maximum accuracy val: {maximum_accuracy}")
                log.info(f"[run_id={run_id}] -- Finished epoch {epoch} --")

            if epoch < epochs - 1:  # do not change connectivity pattern after the last epoch
                self.weights_evolution_fast()  # same behaviour as the one below, but it is much faster.
                # self.weights_evolution_slow() # this implementation is more didactic, but slow.

        return metrics

    def get_threshold_interval(self, i=1, weights=None, threshold=None):
        if weights is None:
            weights = self.w[i]

        values = np.sort(weights.data)
        first_zero_pos = find_first_pos(values, 0)
        last_zero_pos = find_last_pos(values, 0)

        if not threshold:
            threshold = self.zeta

        largest_negative = values[int((1 - threshold) * first_zero_pos)]
        smallest_positive = values[
            int(min(values.shape[0] - 1, last_zero_pos + threshold * (values.shape[0] - last_zero_pos)))]

        return largest_negative, smallest_positive

    # counts the number of non-small (not close to zero) input connections per feature
    def get_core_input_connections(self, weights=None):
        if weights is None:
            weights = self.w[1]

        wcoo = weights.tocoo()
        vals_w = wcoo.data
        rows_w = wcoo.row
        cols_w = wcoo.col

        largest_negative, smallest_positive = self.get_threshold_interval(1, weights=weights)
        # remove the weights (W) closest to zero and modify PD as well
        pruned_indices = (vals_w > smallest_positive) | (vals_w < largest_negative)
        vals_w_new = vals_w[pruned_indices]
        rows_w_new = rows_w[pruned_indices]
        cols_w_new = cols_w[pruned_indices]

        return coo_matrix((vals_w_new, (rows_w_new, cols_w_new)), (self.dimensions[0], self.dimensions[1])).getnnz(
            axis=1)

    def vis_feature_selection(self, feature_selection):
        image_dim = (28, 28)
        f_data = np.reshape(feature_selection, image_dim)

        plt.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
        plt.title("Title")
        plt.show()

    def feature_selection(self, threshold=0.1, weights=None):
        """
        Selects the strongest features based on the number of strong connections of the input neuron

        :param threshold: the percentage of selected features TODO: Not really the percentage, more a mean dev. term
        :param weights: the weights to select from
        :return the strongest features
        """
        feature_strength = self.get_core_input_connections(weights=weights)

        absolute_threshold = (1 - threshold) * np.mean(feature_strength)

        feature_selection = feature_strength > absolute_threshold

        self.vis_feature_selection(feature_selection)

        return feature_selection

    def feature_selection_mean(self, sparsity=0.4, weights=None) -> ndarray:
        # TODO(Neil): explain why we choose only the first layer
        # the main reason is that this first layer will already have
        # most of the important information in it, given that everything
        # gets backpropageted

        if weights is None:
            weights = self.w[1]

        means = np.asarray(np.mean(np.abs(weights), axis=1)).flatten()
        means_sorted = np.sort(means)
        threshold_idx = int(means.size * sparsity)

        n = len(means)
        if threshold_idx == n:
            return np.ones(n, dtype=bool)

        means_threshold = means_sorted[threshold_idx]

        feature_selection = means >= means_threshold

        return feature_selection

    def weights_evolution_slow(self):
        # this represents the core of the SET procedure.
        # It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers - 1):
            largest_negative, smallest_positive = self.get_threshold_interval(i)
            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")

            # remove the weights closest to zero
            keep_connections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if (val < largest_negative) or (val > smallest_positive):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keep_connections += 1
            limit = np.sqrt(6. / float(self.dimensions[i]))

            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keep_connections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.uniform(-limit, limit)
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weights_evolution_fast(self):
        def weights_evolution_fast(self):
            for i in range(1, self.n_layers - 1):
                t_ev_1 = datetime.datetime.now()

                wcoo = self.w[i].tocoo()
                vals_w = wcoo.data
                rows_w = wcoo.row
                cols_w = wcoo.col

                pdcoo = self.pdw[i].tocoo()
                vals_pd = pdcoo.data
                rows_pd = pdcoo.row
                cols_pd = pdcoo.col

                largest_negative, smallest_positive = self.get_threshold_interval(i)

                # Remove the weights (W) closest to zero and modify PD as well
                pruned_indices = (vals_w > smallest_positive) | (vals_w < largest_negative)
                vals_w_new = vals_w[pruned_indices]
                rows_w_new = rows_w[pruned_indices]
                cols_w_new = cols_w[pruned_indices]

                new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
                old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

                new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index, new_w_row_col_index)

                vals_pd_new = vals_pd[new_pd_row_col_index_flag]
                rows_pd_new = rows_pd[new_pd_row_col_index_flag]
                cols_pd_new = cols_pd[new_pd_row_col_index_flag]

                self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)),
                                         (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                if i == 1:
                    self.input_layer_connections.append(coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                                                   (self.dimensions[i - 1], self.dimensions[i])).getnnz(
                        axis=1))
                    np.savez_compressed(self.save_filename + "_input_connections.npz",
                                        inputLayerConnections=self.input_layer_connections)

                # Add new random connections
                keep_connections = np.size(rows_w_new)
                length_random = vals_w.shape[0] - keep_connections
                limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                random_vals = np.random.uniform(-limit, limit, length_random)
                zero_vals = 0 * random_vals

                # Check for (i, j) pruned => (i+1, j) pruned
                pruned_indices_i_plus_1 = np.isin(rows_w_new, rows_w_new + 1) & np.isin(cols_w_new, cols_w_new)

                # Update rows_w_new and cols_w_new based on (i, j) pruned => (i+1, j) pruned
                rows_w_new = rows_w_new[~pruned_indices_i_plus_1]
                cols_w_new = cols_w_new[~pruned_indices_i_plus_1]

                # Check for (i, j) added => (i+1, j) added
                added_indices = np.isin(rows_w_new, rows_w_new - 1) & np.isin(cols_w_new, cols_w_new)

                # Update rows_w_new and cols_w_new based on (i, j) added => (i+1, j) added
                rows_w_new = np.concatenate((rows_w_new, rows_w_new[added_indices] + 1))
                cols_w_new = np.concatenate((cols_w_new, cols_w_new[added_indices]))

                # Update vals_w_new based on pruned_indices_i_plus_1 and added_indices
                vals_w_new = vals_w_new[~pruned_indices_i_plus_1]
                vals_w_new = np.concatenate((vals_w_new, random_vals[~added_indices]))

                self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                       (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                t_ev_2 = datetime.datetime.now()

            # print("Weights evolution time for layer", i, "is", t_ev_2 - t_ev_1)

    def predict(self, x_test, y_test, batch_size=100):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros(y_test.shape)
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations


if __name__ == "__main__":
    sum_training_time = 0
    runs = 1

    print(f"See {__file__}.log for information on epoch performance")

    for i in range(runs):
        # load data
        no_training_samples = 5000  # max 60000 for Fashion MNIST
        no_testing_samples = 1000  # max 10000 for Fshion MNIST
        x_train, y_train, x_test, y_test = load_fashion_mnist_data(no_training_samples, no_testing_samples)

        # set model parameters
        no_hidden_neurons_layer = 3000
        epsilon = 1  # set the sparsity level
        zeta = 0.3  # in [0..1]. It gives the percentage of unimportant connections which are removed and replaced with random ones after every epoch
        no_training_epochs = 10  # 400
        batch_size = 40
        dropout_rate = 0  # 0.2
        learning_rate = 0.05
        momentum = 0.9
        weight_decay = 0.0002

        np.random.seed(i)

        # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        set_mlp = SET_MLP((x_train.shape[1], no_hidden_neurons_layer, no_hidden_neurons_layer, no_hidden_neurons_layer,
                           y_train.shape[1]),
                          (Relu, Relu, Relu, Softmax), epsilon=epsilon)

        start_time = time.time()
        # train SET-MLP
        set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=no_training_epochs,
                    batch_size=batch_size, learning_rate=learning_rate,
                    momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate, testing=True,
                    save_filename="Pretrained_results/set_mlp_" + str(
                        no_training_samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(i), monitor=True)

        step_time = time.time() - start_time
        log.info(f"\nTotal training time: {step_time}")
        sum_training_time += step_time

        # selected_indices = set_mlp.feature_selection()
        selected_indices = set_mlp.feature_selection_mean()

        selected_x_train = x_train[:, selected_indices]
        selected_x_test = x_test[:, selected_indices]

        # test SET-MLP
        accuracy, _ = set_mlp.predict(x_test, y_test, batch_size=100)

        log.info(f"\nAccuracy of the last epoch on the testing data: {accuracy}")
    log.info(f"Average training time over {runs} runs is {sum_training_time / runs} seconds")
