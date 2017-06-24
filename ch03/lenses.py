import decisionTreeID3
import plotTree

def createDataSet(filename):
    file = open(filename)
    lines = file.readlines()
    file.close()

    dataSet = []
    labels = []
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    for line in lines:
        lst = line.split()
        dataSet.append(lst[:4])
        labels.append(' '.join(lst[4:]))

    return dataSet, labels, features

# Create training data
dataSet, labels, features = createDataSet('data/lenses/training.txt')
tree = decisionTreeID3.buildTree(dataSet, labels)
plotTree.createPlot(tree, features)

# Create test data
testData, testLabels, _ = createDataSet('data/lenses/test.txt')
correct = 0
for i in range(len(testData)):
    res = decisionTreeID3.evaluate(tree, testData[i])
    if res == testLabels[i]:
        correct += 1

print('Correctness: %d/%d' % (correct, len(testData)))
