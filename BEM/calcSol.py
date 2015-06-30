from numpy import isnan, exp, pi, array, dot, sqrt, sum, fromiter, tile, select
from scipy.special import jn_zeros
from integrate import integrate
from util import norm_
import projection

numQuadPoints = 75
d = 2
             
def fundamentalSol(x,t):
    if all(t>0) == False:
        return 0
    f =  (t>0)*(4*pi*t)**(-d/2.)*exp(-norm_(x)**2/(4.*t))
    return f

def nfundamentalSol(n,x,t):
        return fundamentalSol(x,t)/(2*t)*dot_prod(n,x)

def dot_prod(x,y):
    assert x.shape == y.shape
    return fromiter(map(dot,x,y), dtype=x.dtype).reshape(x.shape[0], 1)

def calcSolIndirect(space_proj, time_proj, sol, base, nx, nt):
    def u(x,t):
        def K0(xsol):
            Itotal = 0
            for alpha in range(nx):
                for m in range(nt):
                   a1 = base.baseX[alpha].support[0](0); a2 = base.baseX[alpha].support[0](1)
                   b1 = base.baseT[m].support[0](0);     b2 = base.baseT[m].support[0](1)
                   def help(y,s):
                       return time_proj.derivative(s)*space_proj.derivative(y)*xsol[m*nx+alpha]*fundamentalSol(x-space_proj(y),t-time_proj(s))

                   if t <= b2 and t> b1:
                       Itotal += integrate(help, base.baseX[alpha].support[0], base.baseT[m].support[0], n = 50, t=base.baseT[m].support[0].inverse(t),nsing=18)
                   else:
                        Itotal += integrate(help, base.baseX[alpha].support[0], base.baseT[m].support[0], n = 50)
            return Itotal      
         
        if space_proj.contains(x):
            return K0(sol)
    return u

#nx,nt - total number of elements in the mesh, needed to interpret bFlux
#base  - basis used in time and space, assumed to be constant for now
def calcSolDirect(space_proj, time_proj, g, xsol, base, nx,nt):
    def u(x,t):
        def K0(xsol):
            Itotal = 0
            for alpha in range(nx):
                for m in range(nt):
                   a1 = base.baseX[alpha].support[0](0); a2 = base.baseX[alpha].support[0](1)
                   b1 = base.baseT[m].support[0](0);     b2 = base.baseT[m].support[0](1)
                   def help(y,s):
                       return time_proj.derivative(s)*space_proj.derivative(y)*xsol[m*nx+alpha]*fundamentalSol(x-space_proj(y),t-time_proj(s))

                   if t <= b2 and t> b1:
                       Itotal += integrate(help, base.baseX[alpha].support[0], base.baseT[m].support[0], n = 50, t=base.baseT[m].support[0].inverse(t),nsing=18)
                   else:
                        Itotal += integrate(help, base.baseX[alpha].support[0], base.baseT[m].support[0], n = 50)
            return Itotal
        def K1(f):
            return integrate(lambda y,s: f(y,s)*nfundamentalSol(space_proj.normal(y),x-y,t-s), space_proj, time_proj, n = 120, t=time_proj.inverse(t)) 

        if space_proj.contains(x):
            return K0(xsol) - K1(g)
    return u


def calcSolDirect2(space_proj, time_proj, g, boundaryFlux):
    def u(x,t):
        def K0(f):
            def help(y,s):
                 return f(y,s)*fundamentalSol(x-y,t-s)
            return integrate(help, space_proj, time_proj, n = numQuadPoints, t=time_proj.inverse(t))

        def K1(f):
            return integrate(lambda y,s: f(y,s)*nfundamentalSol(space_proj.normal(y),x-y,t-s), space_proj, time_proj, n = numQuadPoints, t=time_proj.inverse(t)) 

        if space_proj.contains(x):
            return K0(boundaryFlux) - K1(g)
    return u


from L2proj import approximate,approximateVec, L2proj
from numpy import arctan2,cos
from basis import Wavelet_basis, Const_basis
from plotHeatMap import plotLinftyBFspace, plotLinftySpace, plotRadial, plotTime
from exSol import boundaryFluxcos, exSolcos, exSolt2
if __name__ == '__main__':
    from plotHeatMap import plotSpace
    endT = 4
    time_proj = projection.interval(0,endT)    
    radius = 1.
    space_proj = projection.circle(radius)

    def g(x,t):
        #return t**2  
        return radius*cos(arctan2(x[:,1],x[:,0])).reshape(-1,1)

    def boundaryFlux(x,t):
        #if x.ndim==1:
        #    x = x.reshape(1,x.size)
        #if t.ndim==1:
        #    t = t.reshape(1,t.size)
        h = arctan2(x[:,1],x[:,0])
        return cos(h).reshape(-1,1)
        #alpha = jn_zeros(0,27)
        #return t + 4*sum((1.-exp(-alpha**2*t))/(alpha**4), axis=1).reshape(t.shape)   #cos(h)

    nt = 5
    from basis import Linear_basis
    base = Linear_basis(5,10)
    
    bFlux = approximateVec(boundaryFlux,base,  space_proj, time_proj)
    #bFlux2 = approximate(boundaryFlux,base,  space_proj, time_proj)
    #plotLinftyBFspace(bFlux, boundaryFluxcos, space_proj)
    
    #t = 2
    u = calcSolDirect(space_proj, time_proj,g, bFlux, base, base.nx,base.nt)
    #u = calcSolDirect2(space_proj, time_proj,g, bFlux2)
    #fig = plotSpace(lambda x: u2(x,t), 30)
    #fig.show()
    #fig = plotSpace(lambda x: u(x,t), 30)
    #fig.show()
    plotRadial(u,exSolcos,endT)
    #plotLinftySpace(u, exSolcos, space_proj)
    #plotTime(u,exSolt2,time_proj.end)

