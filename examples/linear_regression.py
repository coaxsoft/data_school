from sklearn import linear_model
from matplotlib import pyplot as pl

import numpy as np


def f(x):
    return np.e ** (0.4 * x) * np.random.normal(1, 0.2, size=len(x))


X = np.random.normal(0, 1, size=100)
d = f(X)

reg = linear_model.LinearRegression()

if __name__ == '__main__':
    feature_set = [[v] for v in X]
    reg.fit(feature_set, d)

    test_x = np.linspace(-3, 3, 50)
    test_feature_set = [[v] for v in test_x]

    y = reg.predict(test_feature_set)
    pl.scatter(X, d)
    pl.plot(test_x, y, 'r', lw=2)
    pl.show()
