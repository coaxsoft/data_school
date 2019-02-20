from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(multi_class='auto', solver='lbfgs')

if __name__ == '__main__':
    model.fit(X, y)
    acc = model.score(X, y)
