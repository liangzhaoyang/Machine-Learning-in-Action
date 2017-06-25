import math
import numpy as np

def calcProbability(x, mean, std):
    if std == 0:
        return 1 if x == mean else 1e-100
    return min(1/(std*math.sqrt(2*math.pi)) * math.exp((x-mean)**2/(-2*std*std)), 1)

def train(dataSet, labels):
    groups = {}
    res = {}

    for i in range(len(dataSet)):
        if labels[i] not in res.keys():
            groups[labels[i]] = []
            res[labels[i]] = {}
        groups[labels[i]].append(dataSet[i])

    for label in res.keys():
        res[label]['p'] = len(groups[label]) / len(dataSet)
        res[label]['mean'] = np.mean(groups[label], axis = 0)
        res[label]['std'] = np.std(groups[label], axis = 0)

    return res

def classify(data, model):
    res = ''
    best = -math.inf

    for k, d in model.items():
        p = math.log(d['p'])
        for i in range(len(data)):
            p += math.log(calcProbability(data[i], d['mean'][i], d['std'][i]))

        if p > best:
            res = k
            best = p

    return res
