def plotCube(a, cube, **kwargs):
   from matplotlib.patches import Rectangle
   from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d

   for i,zdir in enumerate("xyz"):
      for z in cube[i]:
         (a1,b1), (a2,b2) = cube[:i] + cube[i+1:]
         r = Rectangle((a1, a2), b1-a1, b2-a2, **kwargs)
         a.add_patch(r)                        # add while it's still a 2d patch
         pathpatch_2d_to_3d(r, z=z, zdir=zdir) # transform to 3d in place


if __name__ == '__main__':
   from matplotlib.pyplot import figure
   from mpl_toolkits.mplot3d import Axes3D

   f = figure()
   a = Axes3D(f)
   #a = f.gca(projection='3d')
   plotCube(a, [(0,1),(.5,1),(0 ,1 )], color='blue',  alpha=0.5)
   plotCube(a, [(0,1),(0,.5),(0 ,.5)], color='red',   alpha=0.5)
   plotCube(a, [(0,1),(0,.5),(.5,1 )], color='green', alpha=0.5)
   f.show()
