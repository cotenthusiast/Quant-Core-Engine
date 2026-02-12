import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def load_data(test_size = 0.25, random_state = 12):
    data = load_breast_cancer()
    x = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify = y
    )
    return x_train, x_test, y_train, y_test

def standardize_fit(x_train):
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-12
    return mu, sd

def standardize_apply(x, mu, sd):
    return (x - mu) / sd