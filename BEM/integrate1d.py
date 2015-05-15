from numpy import empty, finfo, sum, prod, ix_, hsplit, cumsum, hstack, vstack, linspace, ones, zeros, exp, tile, array, floor

import operator
def prod_(factors):
    return reduce(operator.mul, factors, 1)

def isint(a):
    return a-floor(a)==0  # FIXME Does this need an eps?


def r_jaclog(N,a):
    assert ((N>0) & (a>-1))
    abj = r_jacobi01(2*N)
    mom = mm_log(2*N,a)
    return chebyshev(N,mom,abj)[0]

import numpy as np
from math import gamma
def mm_log(N,a):
    assert(a>-1)
    def buildC():
        c = 1
        for n in xrange(1,N+1):
            yield c
            c = .5*n*c/(2*n-1)
    c = np.fromiter(buildC(), float)
    mm = empty(N)
    for n in range(1,N+1):
        if isint(a) and a<n-1:
           p = xrange(n-a-1,n+a+1)
           mm[n-1] = (-1)**(n-1.-a)/prod_(p)
           mm[n-1] = (gamma(a+1))**2*mm[n-1]
        elif n==1:
            mm[0] = 1/(a+1.)**2
        else:
            k = array(range(1,n))
            s = 1./(a+1+k)-1./(a+1.-k)
            p = (a+1.-k)/(a+1.+k)
            mm[n-1] = (1./(a+1.)+np.sum(s))*np.prod(p)/(a+1.)
    mm *= c
    return mm

def chebyshev(N,mom,abm):
    ab = zeros([N,2])
    assert(N>0)
    #print mom.shape[0], ab.shape[1]
    if (N > mom.shape[0]/2.):
        N = mom.shape[0]/2.
    if (N > (abm.shape[0]+1)/2.):
        N=(abm.shape[0]+1.)/2.
    ab[0,0] = abm[0,0] + float(mom[1])/mom[0]
    ab[0,1] = mom[0]
    if (N==1):
        normsq[0] = mom[0]
        return [ab,mom]
    sig = zeros([N+1,2*N])
    sig[1,:]     = mom[0:2*N]
    for n in range(3,N+2):
        for m in range(n-1,2*N-n+3):
            sig[n-1,m-1] = sig[n-2,m] - (ab[n-3,0] - abm[m-1,0])*sig[n-2,m-1] - ab[n-3,1]*sig[n-3,m-1] + abm[m-1,1]*sig[n-2,m-2]
        ab[n-2,0] = abm[n-2,0] + sig[n-1,n-1]/sig[n-1,n-2] - sig[n-2,n-2]/sig[n-2,n-3]
        ab[n-2,1] = sig[n-1,n-2]/sig[n-2,n-3]
    from numpy import diag
    normsq = diag(sig, -1)
    normsq = normsq.T 
    return [ab, normsq]


# GAUSS Gauss quadrature rule.
from numpy import diag, sqrt, argsort
from numpy.linalg import eig
def gauss(N,ab):
    N0 = ab.shape[0]
    assert (N0>=N)  #otherwise imput array is to short
    J = diag(ab[:,0]) + diag(sqrt(ab[1:,1]),1) + diag(sqrt(ab[1:,1]),-1)
    D,V = eig(J) #generate eigenvalues (unsorted)
    I = argsort(D) #sorting the eigenvalues
    D = D[I]
    V = V[:,I]
    return [D, ab[0,1]*(V[0,:].T)**2]

def r_jacobi(N,a,b):
    assert((N>0) & (a>-1) & (b>-1))
    nu = float(b-a)/(a+b+2)
    mu=2**(a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+2)
    if N==1:
        return array([[nu, mu]]) # We don't need B1 here
    nab = 2*array(range(N-1))+2+a+b
    A   = vstack([[[nu]], ((b**2-a**2)*ones([1,N-1])/(nab*(nab+2.))).T])
    n   = array(range(1,N-1),int) # We need the type here for the empty case
    nab = nab[n]
    B1 = 4.*(a+1)*(b+1)/((a+b+2.)**2*(a+b+3.))
    B  = 4.*(n+1+a)*(n+1+b)*(n+1)*(n+1+a+b)/((nab**2)*(nab+1.)*(nab-1.))
    muB = vstack([mu, B1, B.reshape(-1,1)])
    return hstack([A, muB])

def r_jacobi01(N):
    cd = r_jacobi(N,0,0)
    ab = empty([N,2])
    ab[:,0]   = (1+cd[:,0])/2.
    ab[0,1]   = cd[0,1]/2.
    ab[1:,1]  = cd[1:,1]/4.
    return ab

#Routine gives equidistant quadrature points and constant weights
#ONLY FOR DEBUGGING
def equidistant(n):
    x = linspace(1./n,1,n)
    w = 1./n*ones([n])
    return (x,w)

#Gives a composite gauss-legendre type rule for functions with a singularity
#at t, which are zero at values smaller than t
def sing_gauleg(n,t):
    assert t != 0
    x,w = array(cgauleg(n))
    x *= -1
    x += 1
    x *= t
    w *= t
    return x,w

#cgauleg with cutoff
def sing_gauleg2(n,t):
    if t == 1.:
        x,w = array(cgauleg(n))
        x *= -1
        x += 1
        w = w
        return x,w
    x,w = sing_gauleg(n,1)
    x *= t
    w *= t
    x1,w1 = array(cgauleg(n))
    x1 *= (1-t)
    x1 += t
    w1 *= (1-t)
    from numpy import hstack
    x = hstack([x,x1])
    w = hstack([w,w1])
    #print x.shape, x1.shape
    return x,w

#Composite Gauss-Jacobi points and weights in the interval [0,1]
from memoize import memoize
@memoize
def cgauleg(n):
    from numpy import sqrt
    sigma = (sqrt(2)-1)**2    # parameter of geometric subdivision
    #b = 1.0       # slope parameter
    #delta = 1.0   # Gevrey parameter
    #m = ceil(b*n^(1/delta));
    x = empty([(n*(n+3))/2-1])
    w = empty([(n*(n+3))/2-1])

    xl = sigma
    xr = 1.
    cnt = 0
    for j in range(n-1):
        #nj = ceil(n*(1+(1-j)/m)^delta);
        nj = n-j     # Version without variable delta and b. The other version is not stable.
        x1,w1 = gauleg(nj)
        x1 = xl+(xr-xl)*x1
        w1=(xr-xl)*w1
        x[cnt:cnt+nj]= x1[0:nj]
        w[cnt:cnt+nj] = w1[0:nj]
        xr=xl
        xl=xl*sigma
        cnt += nj 
    #nj = ceil(n/(m^delta));
    nj = n
    x1,w1 = array(gauleg(nj))
    x1=xr*x1
    w1=xr*w1
    x[cnt:cnt+nj] = x1[0:nj]
    w[cnt:cnt+nj] = w1[0:nj]
    return (x,w)
   
#Gauss-legendre points and weights in the interval [0,1].  
from numpy import cos, pi, abs
@memoize
def gauleg(n):
    x = empty([n])
    w = empty([n])
    m=int((n+1)/2)
    xm=0.0
    xl=1.0
    for i in range(m):
        z=cos(pi*(i+1-0.25)/(n+0.5))
        while True:
            p1=1.0
            p2=0.0
            for j in range(n):
                p3=p2
                p2=p1
                p1=((2.0*j+1.0)*z*p2-j*p3)/(j+1)
            pp=n*(z*p1-p2)/(z*z-1.0)
            z1=z
            z=z1-p1/pp
            if abs(z-z1) < finfo(float).eps:
                break
            x[i]=xm-xl*z
        x[n-1-i]=xm+xl*z
        w[i]=2.0*xl/((1.0-z*z)*pp*pp)
        w[n-1-i]=w[i]
    x = (x+1.)/2.
    w = w/2.
    return (x, w)

def GJquad(alpha,beta,N):
    assert N != 0
    if (N == 1):
        return ( (alpha-beta)/(alpha+beta+2), 2)
    x = zeros(N)
    w = zeros(N)
    # Compute J
    h = array(range(1,N))
    h1 = 2.*array(range(N))+alpha+beta
    if abs(alpha + beta) < 10**-14:  #ensure you do not divide by zero
        help = h1
        help[0] = 1
    else:
        help = h1  
    J = diag(-(alpha**2-beta**2)/(h1+2.)/help)
    J += diag(2./(h1[0:N-1]+2)*sqrt(h*(h+alpha+beta)*(h+alpha)*(h+beta)/(h1[0:N-1]+1)/(h1[0:N-1]+3)),  1)
    J += diag(2./(h1[0:N-1]+2)*sqrt(h*(h+alpha+beta)*(h+alpha)*(h+beta)/(h1[0:N-1]+1)/(h1[0:N-1]+3)), -1)
    if (alpha+beta < 10**-14):
         J[0,0]=0.
    # Compute quadrature nodes and weights 
    D,V = eig(J)
    I = argsort(D) #sorting the eigenvalues
    D = D[I]
    V = V[:,I]
    x = D
    w = V[0,:]**2*2**(alpha+beta+1.)/(alpha+beta+1.)*gamma(alpha+1.)*gamma(beta+1.)/gamma(alpha+beta+1.)
    return (x,w)