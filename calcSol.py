numQuadPoints = 50
d = 2
             
from numpy import pi, exp
from util  import norm_

#Returns the Greens function of the heat equation at the values x,t
def fundamentalSol(x,t):
    if all(t>0) == False:
        return 0
    f =  (t>0)*(4*pi*t)**(-d/2.)*exp(-norm_(x)**2/(4.*t))
    return f

from util import dot_prod

#returns the normal derivative of the Greens function of the heat
#equation at x,t, where n is the outer normal
def nfundamentalSol(n,x,t):
        return fundamentalSol(x,t)/(2*t)*dot_prod(n,x)


from integrate import integrate

#calculates the solution using the direct formulation,
#given a boundary flux boundaryFlux and a right hand side g
#and the projections in space and time space_proj, time_proj
def calcSolDirect(space_proj, time_proj, g, boundaryFlux, **kwargs):
    def u(x,t):
        def K0(f):
            if 'base' in kwargs:
		base = kwargs['base']
            	Itotal = 0
            	for alpha in range(base.nx):
                     for m in range(base.nt):
                   	a1 = base.baseX[alpha].support[0](0); a2 = base.baseX[alpha].support[0](1)
                   	b1 = base.baseT[m].support[0](0);     b2 = base.baseT[m].support[0](1)
                   	def help(y,s):
                       		return time_proj.derivative(s)*space_proj.derivative(y)*f[m*base.nx+alpha]*fundamentalSol(x-space_proj(y),t-time_proj(s))

                   	if t <= b2 and t> b1:
                       		Itotal += integrate(help, base.baseX[alpha].support[0], base.baseT[m].support[0],n = 50,t=base.baseT[m].support[0].inverse(t))
                   	else:
                	        Itotal += integrate(help, base.baseX[alpha].support[0], base.baseT[m].support[0],n = 50)
            	return Itotal
            else:  #f is a function
		def help(y,s):
                	 return f(y,s)*fundamentalSol(x-y,t-s)
            	return integrate(help, space_proj, time_proj, n = numQuadPoints, t=time_proj.inverse(t))

        def K1(f):
            return integrate(lambda y,s: f(y,s)*nfundamentalSol(space_proj.normal(y),x-y,t-s), space_proj,time_proj, n=numQuadPoints, t=time_proj.inverse(t)) 

        if space_proj.contains(x):
            return K0(boundaryFlux) - K1(g)
    return u


#Calculates the solution using the indirect formulation
def calcSolIndirect(space_proj, time_proj, sol, base):
    nx = base.nx
    nt = base.nt
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


#Test the calculation of the Direct Method using the exact boundary Flux
if __name__ == "__main__":
	from numpy import arctan2, cos, array
	def boundaryFlux(x,t):
    		h = arctan2(x[:,1],x[:,0])
    		return cos(h).reshape(-1,1)
	def g(x,t):
    		h = arctan2(x[:,1],x[:,0])
    		return cos(h).reshape(-1,1)
	from projection import interval, circle
	space_proj = circle(1.)
	time_proj = interval(0.,1.)
	from L2proj import L2proj
	from basis import Const_basis
	base = Const_basis(10,10)
	BFlux = L2proj(boundaryFlux,base,space_proj,time_proj)
	u  = calcSolDirect(space_proj, time_proj, g, BFlux, base = base)
	from plotHeatMap import plotSpace
	plotSpace(lambda x: u(x,1.), 20)



