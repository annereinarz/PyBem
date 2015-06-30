from numpy import floor, select, array, zeros, repeat
from functools import partial
from integrate import integrate
from projection import interval, circle
from basis import *

from numpy import zeros, matrix

def BilinearformM(bi,bj, px,pt):
    bXalpha, bTm = bi
    bXbeta,  bTn = bj
    support1  = bXalpha.support
    support2  = bXbeta.support
    supportT1 = bTm.support
    supportT2 = bTn.support
    m = 0
    for s1 in support1:
        for s2 in support2:
            for sT1 in supportT1:
               for sT2 in supportT2:
                  assert s1.intersect(s2) == s2.intersect(s1)
                  assert sT1.intersect(sT2) == sT2.intersect(sT1)
                  intersection  = s1.intersect(s2)
                  intersectionT = sT1.intersect(sT2)
                  if(intersection and intersection.length and intersectionT and intersectionT.length):
                     bi = lambda x,t: bXalpha(x) * bTm(t)
                     bj = lambda x,t: bXbeta(x)  * bTn(t)
                     m += integrate(lambda x,t: bi(x,t)*bj(x,t) *px.derivative(x)*pt.derivative(t), intersection,intersectionT, n = 3)
    return m

def assembleMass(base, px, pt):
    if isinstance(base, Sparse_multiscale):
      N = base.nx
      A = matrix(zeros( (N,N) ))
      print "A", N
      for i in range(N):
         for j in range(N):
            A[i,j] = BilinearformM( (base.baseX[i],base.baseT[i]),(base.baseX[j], base.baseT[j]), px,pt)
      return A
    Nx   = base.nx;
    Nt   = base.nt
    A    = matrix(zeros((Nt*Nx,Nt*Nx)))
    for m in range(Nt):
        for n in range(Nt):
            for alpha in range(base.nx):
                for beta in range(base.nx):
                    bi = (base.baseX[alpha], base.baseT[m])
                    bj = (base.baseX[beta],  base.baseT[n])
                    A[m*Nx + alpha, n*Nx + beta] = BilinearformM(bi,bj,px,pt)
    return A

from numpy.linalg import solve
from numpy import min,max
from assembleVector import assembleVector

def L2proj(B,base,px,pt):
    M = assembleMass(base, px, pt)   #assemble mass matrix
    return solve(M,B).reshape(-1)    

def approximate(f, base, px, pt):
    M = assembleMass(base, px, pt)
    B = assembleVector(f, base, px, pt)
    x = solve(M,B).reshape(-1)
    return boundaryFlux(x, base, px,pt)

def approximateVec(f, base, px, pt):
    M = assembleMass(base, px, pt)
    B = assembleVector(f, base, px, pt)
    return solve(M,B).reshape(-1)

from basis import *
def boundaryFlux(sol,basis, px, pt):
   #assert basis.nx*basis.nt == len(sol)
   if isinstance(basis.baseX, Constant1D) and isinstance(basis.baseT, Constant1D):    
       def f(x,t):
           #print "___|", sol.shape
           xh = px.inverse(x).reshape(-1,1)
           #xh = x
           th = pt.inverse(t).reshape(-1,1)
           return sum( float(sol[a+basis.nx*n]) * basis.baseX[a](xh) * basis.baseT[n](th) for a in range(basis.nx) for n in range(basis.nt) )
       return f
   if isinstance(basis, Sparse_multiscale) or isinstance(basis, Linear1D):
      def fh(x,t):
         #if len(x.shape) == 1:
         #   x = x.reshape(-1,2)
         #assert len(x.shape) == 2, "Input vector x must have two dimensions."
         a = 0
         for xk in px.inverse(x):
            for tk in pt.inverse(t):
               tk = tk.reshape(-1.1)
               xk = xk.reshape(-1,1)
               alpha = basis.element_indexX(xk)
               m     = basis.element_indexT(tk)
               from numpy import intersect1d
               i     = intersect1d(alpha, m).reshape(-1)
               def f(k):
                  return sol[i[k]]*basis.baseX[i[k]](xk)*basis.baseT[i[k]](tk)
               a += sum( f(k) for k in range(len(i)) )
         return a
      return fh
   def fh(x,t):
        #if len(x.shape) == 1:
        #    x = x.reshape(-1,2)
        #assert len(x.shape) == 2, "Input vector x must have two dimensions."
        a = zeros(len(x))
        cnt = 0
        for (xk,tk) in zip(px.inverse(x),pt.inverse(t)):
               xk = xk.reshape(-1,1)
               tk = tk.reshape(-1,1)
               alpha = basis.element_indexX(xk).reshape(1,-1)
               #print alpha
               m     = basis.element_indexT(tk).reshape(1,-1)
               #print m
               def f(i,j):
                  if not alpha[0][i] < basis.baseX.n:
                      print xk
                  coeff = sol[alpha[0][i]+ m[0][j]*basis.nx]
                  b1    = basis.baseX[alpha[0][i]](xk)
                  b2    = basis.baseT[m[0][j]](tk)
                  return coeff *b1*b2
               a[cnt] = sum( f(i, j) for i in range(alpha.shape[1]) for j in range(m.shape[1]) )
               cnt = cnt + 1
        return a
   return fh



if __name__ == '__main__':
    from numpy import cos, linspace, zeros, repeat, arctan2, vstack, array
    from matplotlib.pyplot import figure
    from mpl_toolkits.mplot3d import Axes3D

    from projection import circle

    proj = circle(1.0)
    def g(x,t):
        h = arctan2(x[:,1],x[:,0])
        return (cos(h).reshape(-1,1))

    from basis import Sparse_multiscale, Const_multiscale, Const_basis, Wavelet_basis
    #b = Sparse_multiscale(4)
    from numpy import zeros
    N = 4
    ndof1 = zeros(N)
    err1 = zeros(N)
    ndof2 = zeros(N)
    err2 = zeros(N)
    for i in range(N):
        b1 = Linear_basis(2**4,2**(i+2))
        b2 = Const_basis(2**4,2**(i+2))
        print b1.nx, b2.nx
        c1 = approximate(g, b1, proj, interval(0,1))    
        c2 = approximate(g, b2, proj, interval(0,1))
        t = 0.5
        X = proj(linspace(0,1,123).reshape(-1,1))
        T = repeat(t,123).reshape(-1,1)
        ndof1[i] = b1.nx*b1.nt
        err1[i]  = max(abs(g(X,T).reshape(-1) - vstack([c1(x.reshape(1,2), array(t).reshape(1,1)) for x in X]).reshape(-1)))
        ndof2[i] = b2.nx*b2.nt
        err2[i]  = max(abs(g(X,T).reshape(-1) - vstack([c2(x.reshape(1,2), array(t).reshape(1,1)) for x in X]).reshape(-1)))
        #f = figure()
        #a = f.gca(projection='3d')
        #a.plot3D(X[:,0], X[:,1], zeros(X[:,0].shape))
        #a.plot3D(X[:,0], X[:,1], g(X,T).reshape(-1))
        #a.plot3D(X[:,0], X[:,1], vstack([c(x.reshape(1,2), array(t).reshape(1,1)) for x in X]).reshape(-1))
        #f.show()
    f = figure()
    f.gca().loglog(ndof1, err1, 'x-')
    f.show()
    f = figure()
    f.gca().loglog(ndof2, err2, 'o-')
    f.show()

