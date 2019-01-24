import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return (1.0)/(1.0+np.exp(-x))
def gradientDescent(line,allPoints,y,alpha):
    m = allPoints.shape[0]
    for i in range(600):
        p = sigmoid(allPoints*line)
        gradient = (allPoints.T * (p-y)) /  (alpha/m)
        line = line - gradient
        w1 = line.item(0)
        w2 = line.item(1)
        b = line.item(2)
        x1 = np.array([allPoints[:,0].min(),allPoints[:,1].max()])
        x2 = -b / w2 + x1 * ( -w1 / w2 )
    ln = plt.plot(x1,x2)

points = 100
bias = np.ones(points)
np.random.seed(0)

topRegion = np.array([np.random.normal(10,2,points),np.random.normal(12,2,points),bias]).T
bottomRegion = np.array([np.random.normal(5,2,points),np.random.normal(6,2,points),bias]).T

y = np.array([np.zeros(points),np.ones(points)]).reshape(points*2,1)

line = np.matrix([np.ones(3)]).T

allPoints = np.vstack((topRegion,bottomRegion))
fig,ax = plt.subplots(figsize=(4,4))
ax.scatter(topRegion[:,0],topRegion[:,1])
ax.scatter(bottomRegion[:,0],bottomRegion[:,1])
gradientDescent(line,allPoints,y,0.006)
plt.show()