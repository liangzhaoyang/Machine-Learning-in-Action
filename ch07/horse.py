import numpy as np
import adaboost

def createDataSet(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    dataSet, labels = [], []
    for line in lines:
        data = line.split()
        dataSet.append(list(map(float, data[:-1])))
        labels.append(int(float(data[-1])))

    return np.array(dataSet), np.array(labels)

trainingSet, trainingLabels = createDataSet('data/horse/training.txt')
stumps = adaboost.train(trainingSet, trainingLabels)

testSet, testLabels = createDataSet('data/horse/test.txt')
res = adaboost.classify(testSet, stumps)
a, b = np.sum(res == testLabels), len(testLabels)
print('Correctness: %d/%d = %.2f%%' % (a, b, a/b * 100))
