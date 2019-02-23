import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/animesh-agarwal/Machine-Learning/master/LogisticRegression/data/marks.txt"
    content = requests.get(url).content

    data = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None)

    X = data.iloc[:, :-1]
    y_label = data.iloc[:, -1]

    train_x = np.hstack((np.ones(shape=(len(X), 1)), np.array(X)))
    train_y = np.array(y_label).reshape(len(y_label), 1)

    theta = np.random.uniform(-0.5, 0.5, size=(train_x.shape[1], 1))

    EPOCHS = 10000
    learning_rate = 0.01
    m = len(train_x)
    for i in range(EPOCHS):
        h = train_x.dot(theta)
        a = sigmoid(h)

        d = a - train_y
        grad = train_x.T.dot(d) / m
        theta -= learning_rate * grad

    res_d = train_x.dot(theta)
    print(loss(res_d, train_y))
