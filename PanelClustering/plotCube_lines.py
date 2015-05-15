def plotCube(ax, cube, **kwargs):
   from itertools import product, combinations
   from operator import eq
   vertices = product(*cube)
   for v1, v2 in combinations(vertices, 2):
      if 2 == map(eq, v1, v2).count(True):
         ax.plot3D(*zip(v1,v2), **kwargs)

if __name__ == '__main__':
   from matplotlib.pyplot import figure
   from mpl_toolkits.mplot3d import Axes3D

   f = figure()
   a = f.gca(projection='3d')
   plotCube(a, [(0,1),(.5,1),(0 ,1 )], color='blue' )
   plotCube(a, [(0,1),(0,.5),(0 ,.5)], color='red'  )
   plotCube(a, [(0,1),(0,.5),(.5,1 )], color='green')
   f.show()
