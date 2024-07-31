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

#########################


def lmb_fn1(x, q=0.3):
    return q**x*np.log(q)


def lmb_fn2(x, q=0.5):
    p = 1-q
    return q**x*(p/q*np.log(q)*x+(p/q+np.log(q)))


def lmb_fn3(x, q=0.5):
    p = 1-q
    term1 = lmb_fn2(x, q)
    term2 = np.log(q)*x**2+(2-np.log(q))*x-1
    term2 = term2*q**x*p**2/2/q**2
    return term1 + term2


ys1 = [lmb_fn3(x, .1) for x in xs]
ys2 = [lmb_fn3(x, .2) for x in xs]
ys3 = [lmb_fn3(x, .3) for x in xs]
ys4 = [lmb_fn3(x, .4) for x in xs]
ys5 = [lmb_fn3(x, .5) for x in xs]
ys6 = [lmb_fn3(x, .6) for x in xs]

plt.plot(xs, ys1, color=[.10, .20, .20])
plt.plot(xs, ys2, color=[.20, .20, .20])
plt.plot(xs, ys3, color=[.30, .20, .20])
plt.plot(xs, ys4, color=[.40, .20, .20])
plt.plot(xs, ys5, color=[.50, .20, .20])
plt.plot(xs, ys6, color="red")

plt.show()

########################

import numpy as np
import matplotlib.pyplot as plt


def x_fn_q(q):
    q1 = 0.3
    q2 = q
    n = 10
    x = (n*np.log(q2)+np.log(np.log(q2)/np.log(q1)))/np.log(q1*q2)
    return x


qs = np.arange(0, 1, 0.01)
xs = [x_fn_q(x) for x in qs]
plt.plot(qs, xs)
plt.show()
