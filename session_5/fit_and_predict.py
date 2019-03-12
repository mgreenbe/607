from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np


def fit(X, y, k=None):
    """
    `fit` takes two positional arguments, one-dimensional vectors `X` and `y` of the same length (training data) and one keyword argument `k`, a positive integer model hyperparameter.
    `fit` returns an object meant to be passed to `predict` as its the keyword argument `fit_data`.
    """
    if type(X) != np.ndarray or len(X.shape) != 1:
        raise Exception(
            "First positional argument must be a 1-dimensional ndarray.")
    if type(y) != np.ndarray or len(y.shape) != 1:
        raise Exception(
            "First positional argument must be a 1-dimensional ndarray.")
    if k is None:
        raise Exception("Required keyword argument k missing")
    P = PolynomialFeatures(degree=k)
    X_ = P.fit_transform(X.reshape(-1, 1))
    R = LinearRegression(fit_intercept=False)
    R.fit(X_, y)
    return R.coef_


def predict(X, fit_data=None):
    """
    `predict` takes one positional argument, a one-dimensional vector `X` (test data), and one keyword argument `fit_data`.
    The keyword argument `fit_data` should be passed the output from the `fit` function.
    The function `predict` outputs a vector `y` of predicted $y$-values corresponding to the input vector `X`.
    """
    if type(X) != np.ndarray or len(X.shape) != 1:
        raise Exception(
            "First positional argument must be a 1-dimensional ndarray.")
    if fit_data is None:
        raise Exception("Required keyword argument fit_data missing")
    P = PolynomialFeatures(degree=len(fit_data) - 1)
    X_ = P.fit_transform(X.reshape(-1, 1))
    return X_ @ fit_data
