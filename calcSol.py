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


