import pickle
import numpy as np
import kNN

def createDataSet(filename):
    group, labels = [], []
    file = open(filename, 'rb')
    d = pickle.load(file)
    file.close()

    for label in d.keys():
        for data in d[label]:
            group.append(list(map(float, data)))
            labels.append(label)
    return np.array(group), labels

# Create training set
group, labels = createDataSet('data/digits/training.pickle')

# Create and try test set
testGroup, testLabels = createDataSet('data/digits/test.pickle')
correct = 0
for i in range(testGroup.shape[0]):
    res = kNN.classify0(testGroup[i], group, labels, 3)
    if res == testLabels[i]:
        correct += 1

print('Correctness: %.2f%%' % (correct / testGroup.shape[0] * 100))
