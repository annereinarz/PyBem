from projection import *
from basis import Const_basis
from assembleMatrix import assembleSingleLayer
from assembleVector import assembleVector
from calcSol import calcSolIndirect, fundamentalSol
from plotHeatMap import plotSpace, plotTime, plotRadial, plotLinfty, plotConv, plotLinftySpace
from L2proj import boundaryFlux
from exSol import  exSolt2, exSolcos, exSolcost2, exSolfun
from util import dot
from solve import solveMem
from numpy import exp

from matplotlib.pyplot import figure
from numpy import cos, array, arctan2, zeros

def L2norm(bflux,space_proj, t):
    from integrate1d import gauleg
    from numpy import array
    xquad,wquad = gauleg(60)
    X = space_proj(xquad.reshape(-1,1))
    y = array(map(lambda xh,yh: bflux(array([xh,yh]),array(t)), X[:,0], X[:,1])).reshape(-1)
    norm = sum(y**2*wquad) 
    return norm
#Choose the domain
endT = 4.
time_proj = interval(0,endT)
radius = 1.
space_proj = circle(radius)

#Chose g(x,t)
#def g(x,t):
#    return t**4*exp(-2*t)
def g(x,t):
    h = arctan2(x[:,1],x[:,0])
    return t**2*cos(h).reshape(-1,1)

N = 7
sigma = 1
norm = []
ndof = []
#Chose number of degrees of freedom in time and space
for i in range(N):
    from numpy import floor
    Nx = 2**(i+2)
    Nt = 2**(int(floor((i+2)*sigma)))
    basis = Const_basis(Nt,Nx)

    from cProfile import run
    from time import clock
    #Set up Matrix and vector of the right hand side
    start = clock()
    A = assembleSingleLayer(basis, space_proj, time_proj)
    elap1 = (clock()-start)
    
    start = clock()
    print "matrix set up"
    B = assembleVector(g, basis, space_proj, time_proj)
    elap2 = (clock()-start)
    print "vector set up"
    
    start = clock()
    #solve the linear system    
    if A.shape[0] == A.shape[1]:
        from numpy.linalg import solve
        sol = solve(A,B)
    else:
        sol = solveMem(A,B,Nx,Nt)
    bflux = boundaryFlux(sol, basis, space_proj, time_proj)    
    elap3 = (clock()-start)
    print "linear system solved"
    print "time taken: ", elap1+elap2+elap3

    #convergence values)
    norm.append(L2norm(bflux,space_proj,endT))
    ndof.append(len(sol))

print norm, ndof