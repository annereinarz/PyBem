from numpy import *
from numpy.linalg import norm


def f(x,y):
   return sin(x*y)*cos(y)

n = 10
A = empty([n,n])
x = empty([n])
y = empty([n])

for i in range(n):
   x[i] = float(i+1)/n
   y[i] = float(i+1)/(n+1)+2

for i in range(n):
   A[i,:] = f(x[i],y[:])

import aca
S = aca.approximate(f,x,y)
print norm(A-S)
