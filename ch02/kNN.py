import numpy as np
import operator

def normalize(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    rangeVals = maxVals - minVals
    dataSize = dataSet.shape[0]
    return (dataSet - np.tile(minVals, (dataSize, 1))) / np.tile(rangeVals, (dataSize, 1))

def classify0(vec, dataSet, labels, k):
    # Calculate distance
    dataSize = dataSet.shape[0]
    disArr = np.tile(vec, (dataSize, 1)) - dataSet
    disArr **= 2
    disArr = disArr.sum(axis = 1)
    disArr **= 0.5

    # Sort and select k distances
    index = disArr.argsort()
    count = {}
    for i in range(k):
        label = labels[index[i]]
        if label not in count.keys():
            count[label] = 1
        else:
            count[label] += 1
    count = sorted(count.items(), key = operator.itemgetter(1), reverse = True)

    return count[0][0]
