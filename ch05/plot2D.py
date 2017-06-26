import numpy as np
import matplotlib.pyplot as plt

def plotData(dataSet, labels, theta):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x0, y0, x1, y1 = [], [], [], []
    for i in range(len(labels)):
        if labels[i][0] == 0:
            x0.append(dataSet.item(i, 1))
            y0.append(dataSet.item(i, 2))
        else:
            x1.append(dataSet.item(i, 1))
            y1.append(dataSet.item(i, 2))

    ax.scatter(x0, y0, c = 'r', marker = 'x')
    ax.scatter(x1, y1, c = 'g', marker = 'o')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-theta[0] - theta[1]*x) / theta[2]
    ax.plot(x, y)

    plt.show()
