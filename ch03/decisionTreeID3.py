import math
import operator

def getMajority(labels):
    count = {}
    for label in labels:
        if label not in count.keys():
            count[label] = 1
        else:
            count[label] += 1

    return sorted(count.items(), key = operator.itemgetter(1), reverse = True)[0][0]

def calcEntropy(labels):
    count = {}
    for label in labels:
        if label not in count.keys():
            count[label] = 1
        else:
            count[label] += 1

    entropy = 0
    for label in count.keys():
        p = count[label] / len(labels)
        entropy += -p * math.log2(p)

    return entropy

def splitDataSet(id, value, dataSet, labels):
    newDataSet = []
    newLabels = []

    for i in range(len(dataSet)):
        if dataSet[i][id] == value:
            newDataSet.append(dataSet[i][:id] + [None] + dataSet[i][id+1:])
            newLabels.append(labels[i])

    return newDataSet, newLabels

def getBestFeature(dataSet, labels):
    entropy = math.inf
    best = -1

    for i in range(len(dataSet[0])):
        if dataSet[0][i] == None:
            # Feature already used
            continue

        subSet = {}
        for data in dataSet:
            if data[i] not in subSet.keys():
                subSet[data[i]] = splitDataSet(i, data[i], dataSet, labels)

        e = 0
        for k in subSet.keys():
            p = len(subSet[k][0]) / len(dataSet)
            e += p * calcEntropy(subSet[k][1])

        if e < entropy:
            entropy = e
            best = i

    return best

def buildTree(dataSet, labels):
    if labels.count(labels[0]) == len(labels):
        # All labels are the same
        return -1, labels[0]
    elif len(dataSet[0]) == 0:
        return -1, getMajority(labels)

    best = getBestFeature(dataSet, labels)
    subTree = {}
    for data in dataSet:
        if data[best] not in subTree.keys():
            subSet, subLabels = splitDataSet(best, data[best], dataSet, labels)
            subTree[data[best]] = buildTree(subSet, subLabels)

    return best, subTree

def evaluate(tree, data):
    if tree[0] < 0:
        return tree[1]

    for item in tree[1].items():
        if data[tree[0]] == item[0]:
            return evaluate(item[1], data)
