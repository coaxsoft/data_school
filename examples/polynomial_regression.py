"""Polynomial regression example on random data normally distributed"""
from matplotlib import pyplot as pl

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def f(x):
    return np.e ** (0.4 * x) * np.random.normal(1, 0.2, size=len(x))


# Generate dataset
X = np.random.normal(0, 1, size=100)
d = f(X)


# Define the model
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])

if __name__ == '__main__':
    # That should be done for scikit-learn
    # It expects to accept the list of features
    feature_set = [[v] for v in X]

    # Training the model
    model.fit(feature_set, d)

    # Generate test data on which we'll try to predict data by our model
    test_x = np.linspace(-3, 3, 50)
    test_feature_set = [[v] for v in test_x]

    # Trying to see how our model predict data
    y = model.predict(test_feature_set)
    pl.scatter(X, d)
    pl.plot(test_x, y, 'r', lw=2)
    pl.show()
