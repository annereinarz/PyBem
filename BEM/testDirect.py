from projection import circle, interval, ellipse
from basis import Const_basis, Const_multiscale, Constant1D, Wavelet_basis
from assembleMatrix import assembleDoubleLayer, assembleSingleLayer
from assembleVector import assembleVector
from calcSol import calcSolIndirect, calcSolDirect, nfundamentalSol
from plotHeatMap import plotSpace, plotTime, plotRadial, plotLinfty, plotLinftyBF, plotLinftyBFspace, plotConv
from L2proj import boundaryFlux, L2proj
from integrate import integrate
from exSol import boundaryFluxt2, boundaryFluxcos, boundaryFluxcost2
from util import dot
from assembleKG import assembleKG_efficient, assembleKG_debug


def testDirect(g, time_proj, space_proj, basis_type, N, assembleKG = assembleKG_efficient):
    norm   = []
    ndof   = []
    L2norm = []
    
    for i in range(N):
        print i
        
        basis = basis_type(2 ** (i + 2), 2 ** (i + 2))
        
        # Set up Matrix and vector of the right hand side
        A = assembleSingleLayer(basis, space_proj, time_proj)
        print "matrix set up"
        B = assembleVector(g, basis, space_proj, time_proj).reshape(-1, 1)
        print "vector of the right hand side set up"
        KG = assembleKG(g,B,basis,space_proj, time_proj)
        print "vector for the double layer operator set up"
        # solve the linear system
        if isinstance(basis.baseT, Constant1D):
            from solve import solveMem
            sol = solveMem(A, KG.reshape(-1, 1) + 0.5 * B, basis.nx, basis.nt)
        else:
            sol = solve(A, KG.reshape(-1, 1) + 0.5 * B)
        print "linear system solved"
        
        #Create solution on boundary and interior
        bflux = boundaryFlux(sol, basis, space_proj, time_proj)
        u = calcSolDirect(space_proj, time_proj, g, sol, basis, basis.nx, basis.nt)
        
        L2norm.append([calcL2norm(bflux,space_proj, 0.5), calcL2norm(bflux,space_proj, 1)])
        #optional plotting
        #plotLinftyBF(bflux, boundaryFluxcost2, time_proj)
        norm.append(dot(KG.reshape(-1, 1) + 0.5 * B, sol))
        ndof.append(basis.nx * basis.nt)

    return [L2norm, norm, ndof]


def calcL2norm(bflux,space_proj, t):
    from integrate1d import gauleg
    from numpy import array#, sin, cos, pi
    xquad,wquad = gauleg(60)
    #f = lambda xh: cos(2*pi*xh-3*pi/2)
    X = space_proj(xquad.reshape(-1,1))
    assert X.shape[1] == 2
    y = array(map(lambda xh,yh: bflux(array([xh,yh]),array(t)), X[:,0], X[:,1])).reshape(-1)
    err = abs(sum(y**2*wquad)) 
    return err

#Chose the domain
endT = 4.
time_proj = interval(0,endT)
radius = 1.
space_proj = circle(radius)
#a = 0.4
#b = 1.
#space_proj = ellipse(a,b)


#Chose g(x,t)
def g(x,t):
    from numpy import arctan2, cos
    h = arctan2(x[:,1],x[:,0])
    return t**2*cos(h).reshape(-1,1)
#g = lambda x,t: t**2
#FIXME: Currently only works for pw constants
basis_type = Const_basis

N = 6
from numpy import array
L2norm, norm, ndof = testDirect(g, time_proj, space_proj, basis_type, N)
plotConv([array(norm)],[array(ndof)],['energy norm convergence'])
print "L2norm", L2norm
print "norm", norm
print "ndof", ndof
