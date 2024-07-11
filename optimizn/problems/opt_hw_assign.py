import numpy as np
import matplotlib.pyplot as plt

q1 = 0.1
q2 = 0.5
n = 10
xs = np.arange(0, n, 0.1)
ys = []
for x in xs:
    ys.append(q1**x+q2**(n-x))

plt.plot(xs, ys)
plt.show()

x = (n*np.log(q2)+np.log(np.log(q2)/np.log(q1)))/np.log(q1*q2)
x = 2.714
print(x)
print(xs[np.argmin(ys)])

