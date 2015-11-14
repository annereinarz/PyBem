from integrate1d import gauleg, sing_gauleg
from numpy import prod, sum

#Integrates the function f over the domains specified
#by projs, the optional arguments specify number of quadrature points
#and where the singularity is located, if empty it is assumed f is non-singular
def integrate(f, *projs, **kwargs):
    n = kwargs['n']   #number of quadrature points in regular directions

    def integrate_on_0_1(f, (Xs,W)):
        return sum(f(*Xs).reshape(-1,1)*W)

    if 'flag' in kwargs:
	flag = kwargs['flag']
        if flag == 'SingNone':
           return integrate_on_0_1(f, SingNone(n))
        if flag == 'SingDiag':
           return integrate_on_0_1(f, SingDiag((n,n)))
        elif flag == 'SingLeftupper':
            return integrate_on_0_1(f, SingLeftupper((n,n)))
        else:    # Last case is a2 == b1:
            return integrate_on_0_1(f, SingRightbottom((n,n)))	

    if 'nsing' in kwargs:
        nsing = kwargs['nsing']
    else:
       nsing = 18     #if number of points in singular direction is not specified use 18
    def g(*ths):
        return ( f (*(p(th) for p,th in zip(projs,ths)))
               * prod([p.derivative(th) for p,th in zip(projs,ths)], axis=0)
               )
    dims = tuple(p.dim for p in projs) 
    if 't' in kwargs:
        if 'x' in kwargs:
            return integrate_on_0_1(g, SingXT((nsing,dims,kwargs['x'],kwargs['t'])))
        return integrate_on_0_1(g, SingT((nsing, dims, kwargs['t'])))
    return integrate_on_0_1(g, Reg((n, dims)))

def SingXT((n,dims,x,t)):
    return createQuadRule(dims, sing_gauleg(n,t=x, flag = 1), sing_gauleg(n,t=t, flag = 2))

def SingT((n,dims,t)):
    return createQuadRule(dims, gauleg(n), sing_gauleg(n,t=t, flag = 2))

def Reg((n,dims)):
    return createQuadRule(dims, gauleg(n), gauleg(n))

from memoize import memoize
from numpy import hstack

@memoize
def SingNone(n):
    assert n>0, 'need more than zero quadrature points'
    X,W = tensor(gauleg(n), gauleg(n+1))
    Xs = (X[:,0].reshape(-1,1), X[:,1].reshape(-1,1))
    return (Xs,W.reshape(-1,1))

@memoize
def SingDiag((n_gl, n_cgl)):
    assert n_gl>0 and n_cgl>0, 'need more than zero quadrature points'
    X,W = tensor(gauleg(n_gl), sing_gauleg(n_cgl))
    W = W * (1-X[:,1])
    X[:,0] = X[:,0]*(1-X[:,1])
    X[:,1] = X[:,1] + X[:,0]
    X0 = hstack([ X[:,0], X[:,1] ])
    X1 = hstack([ X[:,1], X[:,0] ])
    Wn = hstack([ W,      W      ])
    return ((X0.reshape(-1,1),X1.reshape(-1,1)), Wn.reshape(-1,1))

@memoize
def SingRightbottom((n_gl, n_cgl)):
    assert n_gl>0 and n_cgl>0, 'need more than zero quadrature points'
    X,W = tensor(gauleg(n_gl), sing_gauleg(n_cgl))
    W = W * X[:,1]
    X[:,0] = X[:,0]*X[:,1]
    X0 = hstack([ 1-X[:,0], 1-X[:,1] ])
    X1 = hstack([   X[:,1],   X[:,0] ])
    Wn = hstack([ W,      W      ])
    return ((X0.reshape(-1,1),X1.reshape(-1,1)), Wn.reshape(-1,1))

@memoize
def SingLeftupper((n_gl, n_cgl)):
    assert n_gl>0 and n_cgl>0, 'need more than zero quadrature points'
    X,W = tensor(gauleg(n_gl), sing_gauleg(n_cgl))
    W = W * X[:,1]
    X[:,0] = X[:,0]*X[:,1]
    X0 = hstack([   X[:,0],   X[:,1] ])
    X1 = hstack([ 1-X[:,1], 1-X[:,0] ])
    Wn = hstack([ W,      W      ])
    return ((X0.reshape(-1,1),X1.reshape(-1,1)), Wn.reshape(-1,1))

from numpy import cumsum, hsplit

#The tensor product of several one-dimensional quadrature rules
def createQuadRule(dims, r1, r2):
    d = sum(dims)
    X,W = r1
    for i in range(d-1):
       X,W = tensor((X,W), r2)
    Xs = hsplit(X, cumsum(dims[:-1]))
    return (Xs, W.reshape(-1,1))

from numpy import empty, tile, ix_

#The tensor product of two one dimensional quadrature rules
def tensor((X1,W1), (X2,W2)):
    d = 2
    X = empty([len(X1),len(X2), d])
    for i, a in enumerate(ix_(X1,X2)):
        X[...,i] = a
    X = X.reshape(-1, d)
    W = tile(1.0, [len(W1),len(W2)])
    for a in ix_(W1,W2):
        W[...] *= a
    W = W.reshape(1, -1)  # Need this format to avoid tranposing in all the other routines
    return X,W


#Routine plots the quadrature points,
#used for debugging purposes
from matplotlib.pyplot import figure
def plot(x,w):
    f = figure()
    if isinstance(x, tuple) and len(x) == 2:
        f.gca().plot(x[0],x[1],'o')
    elif len(x.shape) == 1:
        f.gca().vlines(x,[0],w)
    elif x.shape[1] == 2:
        f.gca().plot(x[:,0],x[:,1],'o')
    else:
        raise RuntimeError("Can't plot quadrature points with shape {}.".format(x.shape))
    f.show()
