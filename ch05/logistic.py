import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradientDescent(dataSet, labels, cycles = 1000, alpha = 0.1):
    rows, cols = dataSet.shape
    theta = np.zeros((1, cols)).T

    for i in range(cycles):
        h = np.array(sigmoid(dataSet * theta))
        g = dataSet.T * (labels - h) / rows
        theta += alpha * g

    return theta

def randGradientDescent(dataSet, labels, cycles = 100):
    rows, cols = dataSet.shape
    theta = np.zeros((1, cols)).T

    for i in range(cycles):
        for j in range(rows):
            alpha = 4 / (1+i+j) + 0.01
            h = np.array(sigmoid(dataSet[j] * theta))
            g = dataSet[j].T * (labels[j] - h)
            theta += alpha * g

    return theta

def classify(dataSet, theta):
    h = np.array(sigmoid(dataSet * theta))
    return h
