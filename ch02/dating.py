import numpy as np
import kNN

def createDataSetFromFile(filename):
    # Read lines
    file = open(filename)
    lines = file.readlines()
    file.close()

    # Change lines into array
    featureCount = len(lines[0].split()) - 1
    group = np.zeros((len(lines), featureCount))
    labels = []

    for i in range(len(lines)):
        lst = lines[i].split()
        group[i] = np.array(lst[:-1])
        labels.append(lst[-1])

    return (group, labels)

# Get training set
group, labels = createDataSetFromFile('data/dating/training.txt')
group = kNN.normalize(group)

# Try on test set
testGroup, testLabels = createDataSetFromFile('data/dating/test.txt')
testGroup = kNN.normalize(testGroup)

correct = 0
for i in range(testGroup.shape[0]):
    res = kNN.classify0(testGroup[i], group, labels, 3)
    if res == 'didntLike':
        res = '1'
    elif res == 'smallDoses':
        res = '2'
    else:
        res = '3'
    if res == testLabels[i]:
        correct += 1

print('Correct rate: %.2f%%' % (correct / testGroup.shape[0] * 100))
