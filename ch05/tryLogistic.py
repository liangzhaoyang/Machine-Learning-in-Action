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

dataSet, labels = createDataSet('data/tryLogistic/data.txt')
theta = logistic.gradientDescent(dataSet, labels)

# Calculate result
res = (logistic.sigmoid(dataSet * theta)).reshape(1, len(labels)).tolist()[0]
correct = 0
for i, x in enumerate(res):
    if x >= 0.5 and labels[i] == 1:
        correct += 1
    elif x < 0.5 and labels[i] == 0:
        correct += 1
print('Correctness: %d/%d = %.2f%%' % (correct, len(res), correct/len(res)*100))

# Plot data
plot2D.plotData(dataSet, labels, theta)
