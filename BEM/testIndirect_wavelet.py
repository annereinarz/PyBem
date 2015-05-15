from projection import *
from basis import Wavelet_basis
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
import os
result_directory = "results/indirect"
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

def L2norm(f,N):    
        from integrate1d import gauleg
        xquad,wquad = gauleg(100)
        a = 0
        for i in range(N):
            li = i/N; ri = (i+1)/N
            a = a + sqrt(sum((ri-li)*f(li+(ri-li)*xquad).reshape(-1)**2*wquad))
        return a

#Choose the domain
endT = 4.
time_proj = interval(0,endT)
#radius = 1.
#space_proj = circle(radius)
a = 0.8
b = 0.5
space_proj = ellipse(a,b)
#space_proj = outside_ellipse(a,b)
#space_proj = wibblywobbly()

#Chose g(x,t)
def g(x,t):
    h = arctan2(x[:,1],x[:,0])
    return t**2*cos(2*h).reshape(-1,1)
g.name = "cos-t2"

#g = lambda x,t: t**2
#g.name = "t-squared"

#g = lambda x,t: fundamentalSol(x,t)
#g.name = "fundamental-sol"

#def g(x,t):
#    return t**4*exp(-2*t)

N = 5

cnt  = 0
#errors  = []
#errors2 = []
times = []
ndof = []
#max  = zeros(N-1)
norm = zeros(N)
norm2 = zeros(N)
sigma = 1
#Chose number of degrees of freedom in time and space
for i in range(N):
    from numpy import floor
    Nt = 2**(int(floor((i+3)*sigma)))
    basis = Wavelet_basis(Nt,i+3,3)

    from cProfile import run
    from time import clock
    #Set up Matrix and vector of the right hand side
    start = clock()
    from cutoff import *
    A2 = assembleSingleLayer(basis, space_proj, time_proj,cutoff=cutoff)
    A  = assembleSingleLayer(basis, space_proj, time_proj)
    elap1 = (clock()-start)
    
    #plot matrix
    #import matplotlib.pylab as plt
    #from numpy import log
    #fig = plt.figure()
    #ax=fig.add_subplot(1,1,1) #?
    #ax.set_aspect('equal') # set axis equal
    #from solve import createFullMatrix
    #plt.imshow(log(abs(createFullMatrix(A,basis.nx,basis.nt))),interpolation='nearest')
    #plt.spy(createFullMatrix(A,basis.nx,basis.nt),precision=0.0000001, markersize=1)
    #plt.colorbar()
    #plt.show()
    
    #A2 = assembleSingleLayer(basis, space_proj, time_proj,cutoff=cutoff)
    #fig2 = plt.figure()
    #ax=fig2.add_subplot(1,1,1) #?
   # ax.set_aspect('equal') # set axis equal
    #plt.imshow(log(abs(createFullMatrix(A2,basis.nx,basis.nt))),interpolation='nearest')
    #plt.spy(createFullMatrix(A,basis.nx,basis.nt),precision=0.0000001, markersize=1)
    #plt.colorbar()
    #plt.show()
    
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
        sol2 = solve(A2,B)
    else:
        sol = solveMem(A,B,basis.nx,Nt)
        sol2 = solveMem(A2,B,basis.nx,Nt)
    elap3 = (clock()-start)
    print "linear system solved"
    print "time taken: ", elap1+elap2+elap3

    #convergence values)
    norm[cnt] = dot(B,sol)
    ndof.append(len(sol))
    norm2[cnt] = dot(B,sol2)
    
    u = calcSolIndirect(space_proj, time_proj, sol, basis, basis.nx, basis.nt)
    
    times.append([elap1,elap2,elap3])
 
    #plotTime(u, exSolt2, endT)
    #max[cnt] = plotLinfty(u, exSolt2, time_proj)
    #max[cnt] = plotLinftySpace(u, exSolcos, space_proj)
    #plotRadial(u,exSolcost2,1)
    print cnt
    cnt += 1

print "times", times
print "nshape", ndof
print "Norm", norm
print "Norm: compressed", norm2

plotConv([norm,norm2],[array(ndof),array(ndof)],['without compression', 'with compression'])
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





