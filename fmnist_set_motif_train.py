import copy
import datetime
import logging
import os
import pickle
from multiprocessing import Pool
import bz2

import matplotlib.pyplot as plt
import numpy as np
import psutil

from set_mlp_motif import SET_MLP, Relu, Softmax, CrossEntropy
from utils.monitor import Monitor
from fmnist_data import load_fashion_mnist_data

from train_utils import sample_weights_and_metrics

FOLDER = "benchmarks_motif_fmnist"


def vis_feature_selection(feature_selection, epoch=0, sparsity=0.5, id=0):
    image_dim = (28, 28)
    f_data = np.reshape(feature_selection, image_dim)

    plt.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
    plt.title(f"epoch: {epoch}, Sparsity: {sparsity}, id: {id}")
    plt.show()


# return the index of where a one is inside an array
def index_one(x):
    return np.where(x == 1)


def single_run_density(run_id, set_params, density_levels, n_training_epochs,
                       fname="", save_compressed=True):
    """
    the density levels are the set epsilon sparsity levels
    """
    print(f"[run={run_id}] Job started")
    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples = 1000  # max 10000 for Fashion MNIST
    n_features = 784  # Fashion MNIST has 28*28=784 pixels as features

    # SET model parameters
    n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
    zeta = set_params['zeta']
    batch_size = set_params['batch_size']
    dropout_rate = set_params['dropout_rate']
    learning_rate = set_params['learning_rate']
    momentum = set_params['momentum']
    weight_decay = set_params['weight_decay']

    sum_training_time = 0

    np.random.seed(run_id)

    x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, run_id)

    if os.path.isfile(fname):
        with open(fname, "rb") as h:
            results = pickle.load(h)
    else:
        results = {'density_levels': density_levels, 'runs': []}

    for epsilon in density_levels:
        logging.info(f"[run_id={run_id}] Starting SET-Sparsity: epsilon={epsilon}")
        set_params['epsilon'] = epsilon
        # create SET-MLP (Multilayer Perceptron w/ adaptive sparse connectivity trained & Sparse Evolutionary Training)

        set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
                           y_train.shape[1]), (Relu, Relu, Relu, Softmax), epsilon=epsilon)

        start_time = datetime.datetime.now()
        # train SET-MLP to find important features
        set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=n_training_epochs,
                                  batch_size=batch_size, learning_rate=learning_rate,
                                  momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate,
                                  testing=True, run_id=run_id,
                                  save_filename="", monitor=False)

        dt = datetime.datetime.now() - start_time

        sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 199]
        print("HELLO")
        print(len(set_mlp.weights_evolution))
        evolved_weights, set_metrics = sample_weights_and_metrics(set_mlp.weights_evolution, set_metrics, sample_epochs)

        run_result = {'run_id': run_id, 'set_params': copy.deepcopy(set_params), 'set_metrics': set_metrics,
                      'evolved_weights': evolved_weights, 'training_time': dt}

        results['runs'].append({'set_sparsity': epsilon, 'run': run_result})

        fname = f"{FOLDER}/set_mlp_density_run_{run_id}.pickle"
        # save preliminary results
        if save_compressed:
            with bz2.BZ2File(f"{fname}.pbz2", "w") as h:
                pickle.dump(results, h)
        else:
            with open(fname, "wb") as h:
                pickle.dump(results, h)


def single_run(run_id, set_params, models, sample_epochs,
               sparseness_levels, n_training_epochs, use_pretrained=False):
    # instead of returning just save everything directly inside here??
    print(f"[run={run_id}] Job started")

    n_models = len(models)
    # TODO(Neil): Maybe make these global?
    # load data
    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples = 1000  # max 10000 for Fashion MNIST
    n_features = 784  # Fashion MNIST has 28*28=784 pixels as features

    # SET model parameters
    n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
    epsilon = set_params['epsilon']
    zeta = set_params['zeta']
    batch_size = set_params['batch_size']
    dropout_rate = set_params['dropout_rate']
    learning_rate = set_params['learning_rate']
    momentum = set_params['momentum']
    weight_decay = set_params['weight_decay']

    sum_training_time = 0

    np.random.seed(run_id)

    x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, run_id)

    # create SET-MLP (Multilayer Perceptron w/ adaptive sparse connectivity trained & Sparse Evolutionary Training)

    set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
                       y_train.shape[1]), (Relu, Relu, Relu, Softmax), epsilon=epsilon)

    start_time = datetime.datetime.now()

    # train SET-MLP to find important features
    set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=n_training_epochs,
                              batch_size=batch_size, learning_rate=learning_rate,
                              momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate,
                              testing=True,
                              save_filename="", monitor=False)
    # save_filename="Pretrained_results/set_mlp_" + str(
    #     n_training_samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(run_id), monitor=True)

    # After every epoch we store all weight layers to do feature selection and topology comparison
    evolved_weights = set_mlp.weights_evolution

    dt = datetime.datetime.now() - start_time

    step_time = datetime.datetime.now() - start_time
    print("\nTotal training time: ", step_time)
    sum_training_time += step_time

    result = {'run_id': run_id, 'set_params': set_params, 'set_metrics':
        set_metrics, 'evolved_weights': evolved_weights, 'training_time':
                  dt}

    with open(f"{FOLDER}/set_mlp_run_{run_id}.pickle", "wb") as h:
        pickle.dump(result, h)

    set_pretrained = None
    # TODO(Neil): hardcoded path

    fname = f"benchmarks/benchmark_22_05_2021_13_59_10/set_mlp_run_{run_id}.pickle"
    if os.path.isfile(fname):
        try:
            with open(fname, "rb") as h:
                set_pretrained = pickle.load(h)
        except EOFError:
            return
    else:
        return

    print(f"-------Finished testing run: {run_id}")


def fmnist_train_set_different_densities_sequential(runs=10, n_training_epochs=100, set_sparsity_levels=None,
                                                    use_logical_cores=True):
    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 6000,
                  'epsilon': 13,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 40, 'dropout_rate': 0, 'learning_rate': 0.05, 'momentum': 0.9,
                  'weight_decay': 0.0002}

    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    for i in range(runs):
        fname = f"{FOLDER}/set_mlp_density_run_{i}.pickle"
        print(f'[run={i}] Starting job')
        single_run_density(i, set_params, set_sparsity_levels, n_training_epochs, fname)
        print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")


runs = 4
n_training_epochs = 100
# set_sparsity_levels = [1, 2, 3, 4, 5, 6, 13, 32, 64, 128, 256] # , 512, 1024]
set_sparsity_levels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]  # , 512, 1024]
# the levels are chosen to have [0.16, 0.5, 1, 2, 5, 10, 20, 40, 80, 100] % density in the first layer
use_logical_cores = False
# FOLDER = "benchmarks/benchmark_02_06_2021_13_02_23"

# fmnist_train_set_differnt_densities(runs, n_training_epochs, set_sparsity_levels, use_logical_cores=use_logical_cores)
fmnist_train_set_different_densities_sequential(runs, n_training_epochs, set_sparsity_levels,
                                                use_logical_cores=use_logical_cores)
