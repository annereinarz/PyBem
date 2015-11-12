#Choose the domain
from projection import circle, interval

endT = 4.
time_proj = interval(0,endT)

radius = 1.
space_proj = circle(radius)

#Choose right hand side g(x,t)
from numpy import arctan2, cos

def g(x,t):
    h = arctan2(x[:,1],x[:,0])
    return t**2*cos(h).reshape(-1,1)

#Chose number of degrees of freedom in time and space
N = 3 

#intialise
cnt   = 0
ndof  = []
norm  = []

for sigma in [ 6./5.]:
	for i in range(N):
    		  from numpy import floor
 		  Nx = 2**(i+2)
 		  Nt = 2**(int(floor((i+2)*sigma)))
  		  
		  from basis import Const_basis
		  basis = Const_basis(Nt,Nx)

 		  #Set up Matrix and vector of the right hand side
		  from assembleStiffness import assembleSingleLayer
 		  A = assembleSingleLayer(basis, space_proj, time_proj)
  		  print "matrix set up"
		  from assembleVector import assembleVector
		  B = assembleVector(g, basis, space_proj, time_proj)
  		  print "vector set up"
    
  		  #solve the linear system    
		  from solve import solveMem
 		  sol = solveMem(A,B,Nx,Nt)
  		  print "linear system solved"

  		  #convergence values)
 		  ndof.append(len(sol))
		  from util import dot
 	          norm.append(dot(B,sol))
    		  print cnt
    		  cnt += 1

	print "nshape", ndof
        print "norm", norm

from calcSol import calcSolIndirect
u = calcSolIndirect(space_proj, time_proj, sol, basis)
from plotHeatMap import plotSpace
plotSpace(lambda x: u(x,1), 20)


