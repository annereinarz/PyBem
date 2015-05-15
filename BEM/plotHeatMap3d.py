from numpy import array, empty, linspace, meshgrid
from matplotlib.pyplot import figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot(u, n):
   s = linspace(-1.0, 1.0, n)
   Z = empty([n,n])

   for i in range(n):
      for j in range(n):
         x = array([ s[i]
                   , s[j]
                   ])
         Z[i,j] = u(x) or 0

   f = figure()
   X,Y = meshgrid(s,s)
   f.gca(projection='3d').plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=1, cstride=1)
   #f.gca(projection='3d').scatter(X,Y,Z, c=Z, cmap=cm.coolwarm)
   #f.gca(projection='3d').contourf(X,Y,Z, zdir='z',offset=0, cmap=cm.coolwarm)
   f.show()


if __name__ == '__main__':

   from numpy.linalg import norm
   from numpy import cos, pi

   def u(x):
      if norm(x) < 1:
         return cos(pi/2*x[0]) * cos(pi/2*x[1])

   plot(u,20)
