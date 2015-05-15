from projection import *
from basis import Const_multiscale,Sparse_multiscale,Sparse_multiscaleT
from assembleMatrix import assembleSingleLayer
from assembleVector import assembleVector
from calcSol import calcSolIndirect, fundamentalSol
from plotHeatMap import plotSpace, plotTime, plotRadial, plotLinfty, plotConv, plotLinftySpace
from L2proj import boundaryFlux
from exSol import  exSolt2, exSolcos, exSolcost2, exSolfun
from util import dot
from solve import solveMem

from numpy import cos, array, arctan2, zeros
import os
result_directory = "results/indirect"
if not os.path.exists(result_directory):
    os.makedirs(result_directory)


#Chose the domain
endT = 4.
time_proj = interval(0,endT)
radius = 1.
space_proj = circle(radius)
#a = 0.4
#b = 0.2
#space_proj = ellipse(a,b)
#space_proj = outside_ellipse(a,b)
#space_proj = wibblywobbly()

#Chose g(x,t)
def g(x,t):
    h = arctan2(x[:,1],x[:,0])
    return t**2*cos(h).reshape(-1,1)
g.name = "cos-t2"
#g = lambda x,t: t**2
#g.name = "t-squared"
#g = lambda x,t: fundamentalSol(x,t)
#g.name = "fundamental-sol"

N = 8

cnt  = 0
norm = zeros(N)
ndof = zeros(N)
max  = zeros(N)
#Chose number of degrees of freedom in time and space
for i in range(N):
    Nt = i+1
    basis = Sparse_multiscale(Nt,1)

    from cProfile import run
    from time import clock
    #Set up Matrix and vector of the right hand side
    start = clock()
    A =  assembleSingleLayer(basis, space_proj, time_proj)
    
    print "A", A.shape
    print "matrix set up"
    B = assembleVector(g, basis, space_proj, time_proj)
    print "B", B.shape
    print "vector set up"

    #solve the linear system    
    from numpy.linalg import solve
    sol = solve(A,B)
    print "linear system solved"
    elap = (clock()-start)
    print "time taken: ", elap
    #print sol

    #convergence values)
    norm[cnt] = dot(B,sol)
    print "norm is: ", norm[cnt]
    ndof[cnt] = A.shape[0]

    #bflux = boundaryFlux(sol, basis, space_proj, time_proj)
    #u = calcSolIndirect(space_proj, time_proj, bflux)
    #plotTime(u, exSolt2, endT)
    #max[cnt] = plotLinfty(u, exSolt2, time_proj)
    #max[cnt] = plotLinftySpace(u, exSolcos, space_proj)
    #plotRadial(u,exSolcos,1)
    print cnt
    cnt += 1


    fname = result_directory + "/const {} {} {} i={}".format(g.name, space_proj, time_proj, i)
    with open(fname,"w") as f:
        f.write("{}\n".format(sol))


fname = result_directory + "/const {} {} {} N={}".format(g.name, space_proj, time_proj, N)
with open(fname,"w") as f:
    f.write("{}\n".format(norm))
    f.write("{}\n".format(ndof))
print ndof,norm
plotConv([norm],[ndof],['Gamma = circle, f = t^2'])
#bflux = boundaryFlux(sol, basis, space_proj, time_proj)
#u = calcSolIndirect(space_proj, time_proj, bflux)
#print "Solution", u(array([0.,0.89]),t
#t = 1
#plotSpace(lambda x: u(x,t), 30)
#t = 2
#plotSpace(lambda x: u(x,t), 30)
#t = 3
#plotSpace(lambda x: u(x,t), 30)
#t = 4
#plotSpace(lambda x: u(x,t), 30)
#plotTime(u, exSolcost2, endT)
#plotRadial(u,exSolcos,1)





