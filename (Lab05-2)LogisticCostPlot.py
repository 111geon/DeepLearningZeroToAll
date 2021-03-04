# 1차원의 w에 대해서 b없이 logistic cost function을 plot 해본다.
import numpy as np
import matplotlib.pyplot as plt

x = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
y = [0,0,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1]
xy = list(zip(x, y))

w = np.linspace(-3,3,101)
diff = []
cost = []
for i in w:
    for j in xy:
        hy = np.divide(1., 1. + np.exp(-i*j[0]))
        diff.append(-j[1]*np.log(hy) - (1-j[1])*np.log(1-hy))
    cost.append(sum(diff)/len(diff))
    diff = []

plt.plot(w, cost)
plt.show()
