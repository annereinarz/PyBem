from numpy import *
from numpy.linalg import norm

def approximate(f,x,y, eps=10e-14, steps=None):
   assert size(x) == size(y)
   n = size(x)
   if not steps:
      steps = n

   S = zeros([n,n])
   u = empty([n,steps])
   v = empty([n,steps])
   alltheis = range(n)
   allthejs = range(n)

   j = 0

   for k in range(steps):
      # find maximal element
      ii = argmax(abs(f(x[alltheis],y[j])))
      i  = alltheis[ii]
      del alltheis[ii]

      jj = argmax(abs(f(x[i],y[allthejs])))
      j  = allthejs[jj]
      del allthejs[jj]

      #get tilde u, u and v
      sum1 = zeros([n])
      for l in range(k):
         sum1 += v[j,l]*u[:,l]

      tildeu = f(x, y[j]) - sum1
      u[:,k] = tildeu/tildeu[i]
      if abs(tildeu[i]) < 10e-14:
         break

      sum2 = zeros([n])
      for l in range(k):
         sum2 += v[:,l]*u[i,l]

      v[:,k] = f(x[i], y) - sum2

      # calculate improvement
      improvement = mat(u)[:,k] * mat(v)[:,k].T
      if norm(improvement) < eps:
         break

      # update the approximant S
      S += improvement

   return S
