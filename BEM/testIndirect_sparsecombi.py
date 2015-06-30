from projection import *
from basis import Const_multiscale, Sparse_multiscale, SparseIndices
from calcSol import calcSolIndirect, fundamentalSol
from plotHeatMap import plotSpace, plotTime, plotRadial, plotLinfty, plotConv, plotTimeConv, plotLinftySpace
from L2proj import boundaryFlux
from exSol import  exSolt2, exSolcos, exSolcost2, exSolfun
from util import dot
from solve import solveMem
from numpy.linalg import solve
from basis import Multiscale2Const2d, Const_basis
from numpy import cos, array, arctan2, zeros
from solve import solveMem
from sparseCombi import *

#Chose the domain
endT = 4.
time_proj = interval(0,endT)
#radius = 1.
#space_proj = circle(radius)
a = 1
b = 1
space_proj = ellipse(a,b)
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

from basis import SIndices, findIndices

if __name__ == '__main__':
    from cProfile import run
    from time import clock
    N = 10

    cnt  = 0
    elap = zeros(N)
    elap2 = zeros(N)
    #elap3= zeros(N-1)
    norm = zeros(N)
    norm2= zeros(N)
    #norm3= zeros(N-1)
    ndof = zeros(N)
    #ndofc= zeros(N-1)
    #max  = zeros(N)
    #for i in range(N-1):
    #    start = clock()
    #    solc,Bc = fullTens(i, sigma,space_proj,time_proj,g)
    #    norm3[cnt] = dot(Bc,solc)**0.5
    #    ndofc[cnt] = len(sols)
    #    elap3[i] = (clock()-start)
    #    cnt = cnt+1
    sigma = 1
    cnt = 0
    for i in range(N):    
        #for the time comparison: direct impl. of sparse grids
        #start = clock()
        #sols,Bs = sparseN(i, sigma,space_proj,time_proj,g)
        #elap2[i] = (clock()-start)  
        
        #Combination technique 
        start = clock()
        solSparse, Bsparse = sparseCombi(i,sigma,space_proj,time_proj,g)
        elap[i] = (clock()-start)

        #convergence values)
        norm[cnt] = dot(Bsparse,solSparse)
        #norm2[cnt] = dot(Bs,sols)
        print "norm is: ", norm[cnt], norm2[cnt]
        ndof[cnt] = len(solSparse)

        print cnt
        cnt += 1
    print "norm=",norm
    #print "norm2 = ",norm2
    print "elap =", elap
    #print "elap2 = ",elap2
    #print "elap3 = ",elap3
    print "ndof+=",ndof
    #print "ndofc=",ndofc
    #plotTimeConv([norm,norm2],[elap,elap2],[ndof,ndof],['sparse grid combination technique','direct implementation of sparse grids', 'full tensor product discretisation'])
    plotConv([norm],[ndof],['sparse grid combination technique'])




