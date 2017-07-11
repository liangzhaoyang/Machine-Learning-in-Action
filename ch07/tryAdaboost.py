import numpy as np
import adaboost

def createTrainingSet():
    dataSet = np.array([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    labels = np.array([1, 1, -1, -1, 1])
    return dataSet, labels

def createTestSet():
    dataSet = np.array([[0.0, 0.0], [5.0, 5.0]])
    labels = np.array([-1, 1])
    return dataSet, labels

trainingSet, trainingLabels = createTrainingSet()
stumps = adaboost.train(trainingSet, trainingLabels)
testSet, testLabels = createTestSet()
res = adaboost.classify(testSet, stumps)
print('Correctness: %d/%d' % (np.sum(res == testLabels), len(testLabels)))
