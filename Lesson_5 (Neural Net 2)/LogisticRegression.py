import os
import numpy as np
from numpy.random import default_rng

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and each element should be
                   an integer from 0 to C-1 where C is the number of classes
               - classes : a numpy array with shape (C, ), which contains the
                   unique class labels corresponding to the integers in y
    """

    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "": # handle empty rows in file
            row = line.strip().split(",")
            x.append(list(map(float, row[:-1])))
            y_labels.append(row[-1])

    x = x[:100]
    y_labels = y_labels[:100]

    [classes, y] = np.unique(y_labels, return_inverse=True)

    x = np.array(x)
    y = np.array(y)
    return (x, y, classes)


def split_dataset(x, y, test_proportion, random_generator=default_rng()):
    """ Split dataset into training and test sets, according to the given
        test set proportion.

    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Output label, numpy array with shape (N,)
        test_proprotion (float): the desired proportion of test examples
                                 (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test)
               - x_train (np.ndarray): Training instances shape (N_train, K)
               - x_test (np.ndarray): Test instances shape (N_test, K)
               - y_train (np.ndarray): Training labels, shape (N_train, )
               - y_test (np.ndarray): Test labels, shape (N_test, )
    """

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)


(x, y, classes) = read_dataset("iris.data")
x_train, x_test, y_train, y_test = split_dataset(x, y,
                                                 test_proportion=0.2,
                                                 random_generator=default_rng())
mu = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = (x_train - mu) / sigma
x_test = (x_test - mu) / sigma