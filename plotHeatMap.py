from numpy import array, linspace, empty
from matplotlib.pyplot import figure

#Plot the solution u at nxn grid points
def plotSpace(u, n):
   s = linspace(-1.0, 1.0, n)
   A = empty([n,n])

   for i in range(n):
      for j in range(n):
         x = array([ s[i],s[j] ])
         A[i,j] = u(x)
   f = figure()
   i = f.gca().imshow(A, interpolation='nearest')
   f.colorbar(i)
   f.show()
   return f

