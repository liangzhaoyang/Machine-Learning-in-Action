import numpy as np
import logistic
import plot2D

def createDataSet(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    dataSet, labels = [], []
    for line in lines:
        data = line.split()
        dataSet.append([1] + list(map(float, data[:-1])))
        labels.append([int(data[-1])])

    return np.matrix(dataSet), np.array(labels)

def calcResult(dataSet, labels, theta):
    # Calculate correctness
    res = logistic.classify(dataSet, theta)
    correct = 0
    for i, x in enumerate(res):
        if x >= 0.5 and labels[i] == 1:
            correct += 1
        elif x < 0.5 and labels[i] == 0:
            correct += 1
    print('Correctness: %d/%d = %.2f%%' % (correct, len(res), correct/len(res)*100))

    # Plot data
    plot2D.plotData(dataSet, labels, theta)

dataSet, labels = createDataSet('data/tryLogistic/data.txt')

# Normal gradient descent
theta = logistic.gradientDescent(dataSet, labels)
calcResult(dataSet, labels, theta)

# Random gradient descent
theta = logistic.randGradientDescent(dataSet, labels)
calcResult(dataSet, labels, theta)
