import matplotlib.pyplot as plt

DECISION_NODE = {
    'boxstyle': 'square',
    'fc': '0.8',
}

LEAF_NODE = {
    'boxstyle': 'round4',
    'fc': '0.8',
}

def getTreeInfo(tree):
    if tree[0] < 0:
        return 0, [1]

    dep, count = 0, [1]
    for item in tree[1].items():
        d, c = getTreeInfo(item[1])
        d += 1
        c = [0] + c
        if d > dep:
            dep = d
        if len(c) > len(count):
            count, c = c, count
        for i in range(len(c)):
            count[i] += c[i]

    return dep, count

def plotNode(text, center, parent, nodeType, ax):
    ax.annotate(
        text, xy = parent, xycoords = 'axes fraction',
        xytext = center, textcoords = 'axes fraction',
        va = 'center', ha = 'center', bbox = nodeType,
        arrowprops = {'arrowstyle': '<-'}
    )

def plotTree(tree, features, parent, text, dep, totalDep, count, deltaX, ax):
    deltaX[dep] += 1 / (count[dep] + 1)
    center = (deltaX[dep], 1 - dep / totalDep)
        
    # Plot text
    textPos = ((center[0] + parent[0]) / 2, (center[1] + parent[1]) / 2)
    ax.text(*textPos, text)

    if tree[0] < 0:
        # Leaf node
        plotNode(tree[1], center, parent, LEAF_NODE, ax)
    else:
        # Decision node
        plotNode(features[tree[0]], center, parent, DECISION_NODE, ax)
        for item in tree[1].items():
            plotTree(item[1], features, center, item[0], dep + 1, totalDep, count, deltaX, ax)

def createPlot(tree, features):
    ax = plt.subplot(1, 1, 1)
    dep, count = getTreeInfo(tree)
    plotTree(tree, features, (0.5, 1.0), '', 0, dep, count, [0 for i in range(len(count))], ax)
    plt.show()
