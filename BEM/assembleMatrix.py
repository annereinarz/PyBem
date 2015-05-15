from integrate import *
from numpy import matrix, zeros, array
from timekernel import *
from basis import Constant1D, Multiscale1D, Wavelet1D, Sparse_multiscale

#Routines which specify which kernel should be used
def assembleSingleLayer(*args,**kwargs):
    flag = "singleLayer"
    return assembleMatrix(flag, *args,**kwargs)

def assembleDoubleLayer(*args,**kwargs):
    flag = "doubleLayer"
    return assembleMatrix(flag, *args,**kwargs)   
   
   
   
#Assembles a BE matrix using the given kernel
def assembleMatrix(flag, basis, space_proj, time_proj, **kwargs):
   #Get size of the matrix
   Nx = basis.nx
   Nt = basis.nt

   #Currently sparse multiscale uses a slightly different structure, needs to be changed
   if isinstance(basis, Sparse_multiscale):
      A = zeros([Nt,Nt])
      for i in xrange(Nt):
         for j in xrange(Nt):
            A[i,j] = BilinearformA(flag, basis, i,j,space_proj, time_proj, m=i,n=j)
      return A

   if isinstance( basis.baseT, Constant1D):
       #For constants in time certain symmetries in the matrix can be used to save memory
       A = zeros([Nx*Nt,Nx])
   else:
       A = matrix(zeros((Nt*Nx,Nt*Nx)))

   #run over the indices of all the basis functions
   for m in xrange(Nt):
       for alpha in xrange(basis.nx):
           for beta in xrange(alpha,basis.nx): 
               
               #If wavelet compression is needed
               if "cutoff" in kwargs and kwargs['cutoff'](basis, alpha, beta, space_proj,1.5,1.5):
                   continue # don't calculate this entry

               if isinstance( basis.baseT, Constant1D): 
                   A[m*Nx+alpha, beta]  = BilinearformA(flag, basis, alpha,beta, space_proj,time_proj, l=m)
                   A[m*Nx+beta , alpha] = A[m*Nx+alpha,beta]
               else:
                   for n in xrange(Nt):
                       A[m*Nx+alpha, n*Nx+beta] = BilinearformA(flag, basis, alpha,beta, space_proj,time_proj, m=m, n=n)
                       A[m*Nx+beta, n*Nx+alpha] = A[m*Nx+alpha, n*Nx+beta]
   return A

from numpy import fromiter,dot
def dot_prod(x,y):
    assert x.shape == y.shape
    return fromiter(map(dot,x,y), dtype=x.dtype).reshape(x.shape[0], 1)

def BilinearformA(flag, basis, alpha,beta, space_proj,time_proj,**kwargs):
    numx = 12
    endT = float(time_proj.length)
    basis1 = basis.baseX[alpha]
    intervals1 = basis.baseX[alpha].support
    basis2 = basis.baseX[beta]
    intervals2 = basis.baseX[beta].support

    #function to calculate the matrix entries
    from memoize import memoize
    def matrixEntries(s):
            a1,a2,b1,b2 = s
            fk2 = lambda x,y:  ( space_proj.derivative(a1+(a2-a1)*x) * space_proj.derivative(b1+(b2-b1)*y) )*(a2-a1)*(b2-b1)
            # Pick integrations routine(s)
            if flag == 'singleLayer':
                f =  lambda x,y: fk2(x,y)*timeQuad(space_proj(a1+(a2-a1)*x) - space_proj(b1+(b2-b1)*y), endT, basis, **kwargs).reshape(-1,1)
                def split_and_integrate(sing, smooth, flag):
                    f1 = lambda x,y: fk2(x,y) * gmmsing(-space_proj(a1+(a2-a1)*x) + space_proj(b1+(b2-b1)*y),x,y,endT, basis, **kwargs).reshape(-1,1)
                    f2 = lambda x,y: fk2(x,y) * gmmsmooth(-space_proj(a1+(a2-a1)*x)+space_proj(b1+(b2-b1)*y),x,y,endT, basis, flag=flag, **kwargs)
                    return integrate_on_0_1(f1, sing) + integrate_on_0_1(f2, smooth)
            elif flag == 'doubleLayer':
                def f(x, y):
                    xhat = space_proj(a1 + (a2 - a1) * x)
                    yhat = space_proj(b1 + (b2 - b1) * y)
                    return fk2(x, y) * dot_prod(space_proj.normal(yhat), xhat - yhat) * timeQuadN(xhat - yhat, endT, basis, **kwargs).reshape(-1, 1)
            else:
                raise "unknown flag"
            if a2 < b1 or b2 < a1:
                return integrate_on_0_1(f, SingNone(numx))
            if a1 == b1:
                return integrate_on_0_1(f, SingDiag(numx,numx))
                #return split_and_integrate(SingLogDiag(numx), SingNone(numx), diag)
            elif a1 == b2:
                #print "something went wrong" may happen after all for the multiscale case
                return integrate_on_0_1(f, SingLeftupper(numx,numx))
                #return split_and_integrate(SingLogLeftupper(numx), SingNone(numx), lu)
            else: # Last case is a2 == b1:
                A1 = integrate_on_0_1(f, SingRightbottom(numx+3,numx+3))
                #A2 = split_and_integrate(SingLogRightbottom(numx), SingNone(numx), rb)
                return A1
    if isinstance(basis.baseX, Wavelet1D):
        matrixEntries = memoize(matrixEntries)
    a = 0
    for i1 in intervals1:
        for i2 in intervals2:
              # Overlapping intervals need to be split in order to be able to integrate over them.
              #from projection import splitIntervals
              splittedI1 = i1.splitAt(i2.endpoints())
              splittedI2 = i2.splitAt(i1.endpoints())
              for s1 in splittedI1:
                  for s2 in splittedI2:
                     a1  = s1(0); a2 = s1(1)
                     b1  = s2(0); b2 = s2(1)
                     #only for constant basis functions
                     x   = array([(a1+a2)/2]); y = array([(b1+b2)/2])
                     fk1 = basis1(x)*basis2(y)
                     a  += fk1*matrixEntries( (a1,a2,b1,b2) )
    return a

#from matplotlib import pyplot
def plotMatrix(A):
   from matplotlib.pyplot import figure, colorbar
   from numpy import log,abs
   #f = figure()
   import matplotlib.pylab as pl
   import scipy.sparse as sps
   pl.spy(A,markersize=1)
   pl.show()
   #g = f.gca().imshow(log(abs(A)), interpolation='nearest')
   #f.colorbar(g)
   #f.show()
