import numpy as np

def stumpClassify(stump, dataSet):
    ret = np.ones(len(dataSet)) * -1
    if stump['type'] == '<':
        ret[dataSet[:, stump['col']] < stump['val']] = 1
    else:
        ret[dataSet[:, stump['col']] > stump['val']] = 1

    return ret

def buildStump(dataSet, labels, D, stepCnt = 10):
    bestStump = {}
    minError = np.inf
    bestClassRes = np.array([])

    for i in range(len(dataSet[0])):
        mn, mx = np.min(dataSet[:, i]), np.max(dataSet[:, i])
        step = (mx - mn) / (stepCnt - 1)
        for j in range(stepCnt):
            for k in ['<' , '>']:
                # Check stump's error
                stump = {
                    'col': i,
                    'val': mn + step * j,
                    'type': k
                }
                classRes = stumpClassify(stump, dataSet)
                error = np.sum(D[classRes != labels])

                if error < minError:
                    bestStump = stump
                    minError = error
                    bestClassRes = classRes

    return bestStump, minError, bestClassRes

def train(dataSet, labels, cycles = 50):
    rows, cols = dataSet.shape
    stumps = []
    D = np.array([1/rows for i in range(rows)])
    totError = np.array([0.0 for i in range(rows)])

    for i in range(cycles):
        stump, error, classRes = buildStump(dataSet, labels, D)
        alpha = 0.5 * np.log((1-error) / error)
        stump['alpha'] = alpha
        stumps.append(stump)

        totError += alpha * classRes
        if error >= 0.5 or np.sum(np.sign(totError) == labels) == rows:
            break
        D[classRes != labels] *= np.exp(alpha)
        D[classRes == labels] *= np.exp(-alpha)
        D /= np.sum(D)

    return stumps

def classify(dataSet, stumps):
    res = np.zeros(len(dataSet))
    for stump in stumps:
        res += stumpClassify(stump, dataSet) * stump['alpha']
    return np.sign(res)
