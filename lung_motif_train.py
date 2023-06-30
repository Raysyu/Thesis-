import copy
import datetime
import logging
import os
import pickle
from multiprocessing import Pool

import numpy as np
import psutil

from nn_functions import Relu, Sigmoid, MSE
from set_mlp_motif import SET_MLP

from lung_data import load_lung_data, train_test_split_normalize


FOLDER = "benchmarks_lung_motif"
TEST_SIZE = 1/3


def lung_single_run(X_train_, X_test_, y_train_, y_test_, set_params_, run_id=0):
    n_hidden_neurons_layer = set_params_['n_hidden_neurons_layer']
    epochs = set_params_['epochs']
    epsilon = set_params_['epsilon']
    zeta = set_params_['zeta']
    batch_size = set_params_['batch_size']
    dropout_rate = set_params_['dropout_rate']
    learning_rate = set_params_['learning_rate']
    momentum = set_params_['momentum']
    weight_decay = set_params_['weight_decay']

    start_time = datetime.datetime.now()

    set_mlp = SET_MLP(
        (X_train_.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer, y_train_.shape[1]),
        (Relu, Relu, Relu, Sigmoid), epsilon=epsilon, init_network='normal')

    set_metrics = set_mlp.fit(X_train_, y_train_, X_test_, y_test_, loss=MSE, epochs=epochs, zeta=zeta,
                              batch_size=batch_size,
                              dropout_rate=dropout_rate, learning_rate=learning_rate, momentum=momentum,
                              weight_decay=weight_decay,
                              testing=True, run_id=run_id)

    dt = datetime.datetime.now() - start_time
    evolved_weights = set_mlp.weights_evolution

    run_result = {'run_id': run_id, 'set_params': copy.deepcopy(set_params_), 'set_metrics': set_metrics,
                  'evolved_weights': evolved_weights, 'training_time': dt}

    return run_result


def lung_density_runs(run_id, set_params, density_levels, n_training_epochs, data, fname="", folder=""):
    np.random.seed(run_id)

    X_train, X_test, y_train, y_test = data

    if os.path.isfile(fname):
        with open(fname, "rb") as h:
            results = pickle.load(h)
    else:
        results = {'density_levels': density_levels, 'runs': []}

    for epsilon in density_levels:
        logging.info(f"[run_id={run_id}] Starting SET-Sparsity: epsilon={epsilon}")
        set_params['epsilon'] = epsilon
        set_params['epochs'] = n_training_epochs

        run_result = lung_single_run(X_train, X_test, y_train, y_test, set_params, run_id=run_id)

        results['runs'].append({'set_sparsity': epsilon, 'run': run_result})

        fname = f"{folder}/set_mlp_density_run_{run_id}.pickle"
        # save preliminary results
        with open(fname, "wb") as h:
            pickle.dump(results, h)


def lung_train_set_differnt_densities(runs=10, n_training_epochs=100, set_sparsity_levels=None, use_logical_cores=True,
                                      folder=''):
    set_params = {'n_hidden_neurons_layer': 3000,
                  'epochs': 100,
                  'epsilon': 20,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 2, 'dropout_rate': 0, 'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.0002}

    X, y = load_lung_data()

    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    with Pool(processes=n_cores) as pool:
        futures = []
        for i in range(runs):
            remaining_density_levels = copy.copy(set_sparsity_levels)
            # check if results already exist
            fname = f"{folder}/set_mlp_density_run_{i}.pickle"
            if os.path.isfile(fname):
                with open(fname, "rb") as h:
                    result = pickle.load(h)
                    for el in result['runs']:
                        remaining_density_levels.remove(el['set_sparsity'])

            data = train_test_split_normalize(X, y, test_size=TEST_SIZE, random_state=i)
            futures.append(pool.apply_async(lung_density_runs, (
                i, set_params, remaining_density_levels, n_training_epochs, data, fname, folder)))

        for i, future in enumerate(futures):
            print(f'[run={i}] Starting job')
            future.get()
            print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")


def test():
    run_id = 0

    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 3000,
                  'epochs': 5,  # 100,
                  'epsilon': 20,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 2, 'dropout_rate': 0, 'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.0002}

    X, y = load_lung_data()

    X_train, X_test, y_train, y_test = train_test_split_normalize(X, y, test_size=TEST_SIZE, random_state=run_id)

    feature_selection = lung_single_run(X_train, X_test, y_train, y_test, set_params)

    X_train = X_train[:, feature_selection]
    X_test = X_test[:, feature_selection]


if __name__ == '__main__':

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    sub_folder = "benchmark_lung"
    date_format = "%d_%m_%Y_%H_%M_%S"
    FOLDER = f"{FOLDER}/{sub_folder}_{datetime.datetime.now().strftime(date_format)}"
    os.makedirs(FOLDER)

    runs = 4
    n_training_epochs = 100
    set_sparsity_levels = [1, 2, 3, 4, 5, 6, 13, 32]  # , 512, 1024]

    logical_cores = False

    lung_train_set_differnt_densities(runs, n_training_epochs, set_sparsity_levels, use_logical_cores=logical_cores,
                                      folder=FOLDER)