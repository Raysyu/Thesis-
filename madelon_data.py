import numpy as np

from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = './data/'
PATH = "MADELON/"
FULL_PATH = DATA_PATH + PATH
TEST_SIZE = 1 / 3

def save_madelon_npz() -> (ndarray, ndarray):
    train_data_path = FULL_PATH + 'madelon_train.data'
    train_resp_path = FULL_PATH + 'madelon_train.labels'
    val_data_path = FULL_PATH + 'madelon_valid.data'
    val_resp_path = FULL_PATH + 'madelon_valid.labels'
    X_train = np.loadtxt(train_data_path)
    y_train = np.loadtxt(train_resp_path)
    X_test = np.loadtxt(val_data_path)
    y_test = np.loadtxt(val_resp_path)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    np.savez_compressed(FULL_PATH + "madelon", X=X, y=y.reshape(len(y), 1))

    return X, y


def load_madelon_npz():
    data = np.load(FULL_PATH + "madelon.npz")

    X = data['X']
    y = data['y']
    enc = OneHotEncoder().fit(y)
    y = enc.transform(y).astype('uint8').toarray()

    return X, y


def train_test_split_normalize(X_: ndarray, y_: ndarray, test_size=TEST_SIZE, random_state=42) \
        -> (ndarray, ndarray, ndarray, ndarray):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=test_size, random_state=random_state)

    normalize = StandardScaler()
    normalize.fit(X_train_)
    X_train_ = normalize.transform(X_train_)
    X_test_ = normalize.transform(X_test_)
    return X_train_, X_test_, y_train_, y_test_

