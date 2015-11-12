from integrate import integrate 
from numpy import matrix, zeros, array

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

   #For constants in time certain symmetries in the matrix can be used to save memory
   A = zeros([Nx*Nt,Nx])

   #run over the indices of all the basis functions
   for m in xrange(Nt):
       for alpha in xrange(basis.nx):
           for beta in xrange(alpha,basis.nx): 
               A[m*Nx+alpha, beta]  = BilinearformA(flag, basis, alpha,beta, space_proj,time_proj, l=m)
               A[m*Nx+beta , alpha] = A[m*Nx+alpha,beta]
   return A

from util import dot_prod
from timekernel import timeQuad, timeQuadN
def BilinearformA(flag, basis, alpha,beta, space_proj,time_proj,**kwargs):
    numx = 12
    endT = float(time_proj.length)

    basis1 = basis.baseX[alpha]
    intervals1 = basis.baseX[alpha].support
    basis2 = basis.baseX[beta]
    intervals2 = basis.baseX[beta].support

    #function to calculate the matrix entries
    def matrixEntries(s):
            a1,a2,b1,b2 = s
            fk2 = lambda x,y:  ( space_proj.derivative(a1+(a2-a1)*x) * space_proj.derivative(b1+(b2-b1)*y) )*(a2-a1)*(b2-b1)
            # Pick integrations routine(s)
            if flag == 'singleLayer':
                def f(x, y):
                    xhat = space_proj(a1 + (a2 - a1) * x)
                    yhat = space_proj(b1 + (b2 - b1) * y)		      
		    return fk2(x,y)*timeQuad(xhat - yhat, endT, basis, **kwargs).reshape(-1,1)
            elif flag == 'doubleLayer':
                def f(x, y):
                    xhat = space_proj(a1 + (a2 - a1) * x)
                    yhat = space_proj(b1 + (b2 - b1) * y)
                    return fk2(x, y) * dot_prod(space_proj.normal(yhat), xhat - yhat) * timeQuadN(xhat - yhat, endT, basis, **kwargs).reshape(-1, 1)
            else:
                raise "unknown flag"
            if a2 < b1 or b2 < a1:
                return integrate(f,[], n = numx, flag = 'SingNone')
            if a1 == b1:
                return integrate(f, [], n = numx, flag = 'SingDiag')
            elif a1 == b2:
                return integrate(f, [], n = numx, flag = 'SingLeftupper')
            else:    # Last case is a2 == b1:
                return integrate(f, [], n = numx+3, flag = 'SingRightbottom')
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

#To see matrix structure, plot the matrix
def plotMatrix(A):
   import matplotlib.pylab as pl
   pl.spy(A,markersize=1)
   pl.show()


#If run as main, show an example matrix
if __name__ == '__main__':
	from basis import Const_basis
	b = Const_basis(4,4)
	from projection import circle, interval
	space_proj  = circle(1.)
	time_proj   = interval(0,1)  
	A = assembleSingleLayer(b, space_proj, time_proj)
	plotMatrix(A)
