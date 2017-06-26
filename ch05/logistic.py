import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradientDescent(dataSet, labels, alpha = 0.1, cycles = 1000):
    rows, cols = dataSet.shape
    theta = np.zeros((1, cols)).T

    for i in range(cycles):
        h = np.array(sigmoid(dataSet * theta))
        g = dataSet.T * (labels - h) / rows
        theta += alpha * g

    return theta
