from numpy import exp, sum, cos, arctan2, sqrt
from scipy.special import jn, jn_zeros

def boundaryFluxt2(x,t):
    alpha = jn_zeros(0,47)
    h = 4*sum((1.-exp(-alpha**2*t))/(alpha**4),axis = 1)
    h = h.reshape(t.shape)
    return t - h

def exSolt2(x,t):
    alpha = jn_zeros(0,47)
    #print "x", x
    r = sqrt(x[0]**2+x[1]**2)
    return t**2 - 4*sum(jn(0,alpha*r)/(alpha**3*jn(1,alpha))*(t-(1.-exp(-alpha**2*t))/(alpha**2)))

def exSolcos(x,t):
    h = arctan2(x[1],x[0])
    r = sqrt(x[0]**2+x[1]**2)
    return r*cos(h)

def boundaryFluxcos(x,t):
    h = arctan2(x[1],x[0])
    return cos(h)

def boundaryFluxcost2(x,t):
    t = t.reshape(-1,1)
    alpha = jn_zeros(1,47)
    h = 4*sum((1.-exp(-alpha**2*t))/(alpha**4),axis = 1)
    h = h.reshape(t.shape)
    phi = arctan2(x[1],x[0])
    return (t**2 + t/4. - h)*cos(phi)

def exSolcost2(x,t):
    alpha = jn_zeros(1,47)
    r = sqrt(x[0]**2+x[1]**2)
    h = arctan2(x[1],x[0])
    return (r*t**2 - 4*sum(jn(1,alpha*r)/(alpha**3*jn(2,alpha))*(t-(1.-exp(-alpha**2*t))/(alpha**2))))*cos(h)

def exSolfun(x,t):
    from calcSol import fundamentalSol
    return fundamentalSol(x,t).reshape(-1)