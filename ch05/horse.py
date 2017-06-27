import numpy as np
import logistic

def createDataSet(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    dataSet, labels = [], []
    for line in lines:
        data = line.split()
        dataSet.append([1] + list(map(float, data[:-1])))
        labels.append([int(float(data[-1]))])

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

trainingSet, trainingLabels = createDataSet('data/horse/training.txt')
testSet, testLabels = createDataSet('data/horse/test.txt')
theta = logistic.gradientDescent(trainingSet, trainingLabels, 5000, 0.0005)
calcResult(testSet, testLabels, theta)
