# [b, c, varargout] = segconv( type, e, N, varargin )
#  INPUT:
#  type - either 'a' if algebraic convergence is expected, 
#         ie: e(i) = c*N(i)^-b or 'e' if exponential convergence is
#         expected, ie e(i) = c*exp(-b*N^(1/gamma))
#  e    - sequence for which we wish to analyse the convergence of length n
#  N    - second sequence, for example degrees of freedom, also of length n
#  VARARGIN:
#  gamma - if gamma for the exponential convergence is known it can be given
#  OUTPUT:
#  b and c in the above formulas
#  VARARGOUT:
#  gamma parameter gamma in the exponential convergence if not given
#
from numpy import log,ones,exp
def seqconv( type, e, N ):
    if type == 'a':  #algebraic convergence
        # Find b
        b = -log(e[1:]/e[0:-1])/log( N[1:]/N[0:-1])
        # Find c
        c = e[0:-1]/( N[0:-1]**(-b) )
        return (b,c)    
    #Else: Find gamma                       
    def gfun(x): 
        a1 = (N[2:]**x - N[1:-1]**x)
        a2 = (N[1:-1]**x - N[0:-2]**x)
        b1 = (log(e[2:]) - log(e[1:-1]))/(log(e[1:-1]) - log(e[0:-2]))
        return ( a1/a2 - b1 )
    def dgfun(x):
        a = (N[2:]**x*log(N[2:])- N[1:-1]**x*log(N[1:-1]))/(N[1:-1]**x-N[0:-2]**x)
        c = (N[2:]**x-N[1:-1]**x) * (N[1:-1]**x * log(N[1:-1])-N[0:-2]**x*log(N[0:-2])) / ((N[1:-1]**x-N[0:-2]**x)**2)
        return ( a  - c)
    y = newtonit(ones(e[0:-2].shape), gfun, dgfun, 10**(-15), 100)
    from numpy import array,sum
    if sum(abs(array(y))) < 10e-6:
        y = 2
    gamma = array(1./y).reshape(-1)
    #from numpy import array
    #gamma = array([1.]).reshape(1)#1./y
    # Find b
    b = - log(e[1:-1]/e[0:-2] )/(N[1:-1]**(1./gamma) - N[0:-2]**(1./gamma) )
    # Find c
    c = e[0:-2] / exp(-b*N[0:-2]**(1./gamma))
    return (b,c,gamma)

from numpy import isnan
def newtonit(x,fun,dfun,tol,maxit):
    for i in range(maxit):
        y = x - fun(x)/dfun(x)
        diff = abs(y-x)
        if sum(diff) < tol:
            return y
        if any(isnan(y)):
            return 0
        x=y
    return y


if __name__ == '__main__':
    from numpy import exp, array
    N = array([1,2,3,4,5])
    e = exp(-2*N**(0.5))
    #print 'ne', N.shape, e.shape
    b,c,gamma = seqconv('e',e,N)
    print b,c,gamma
    
    
    

