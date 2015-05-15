from integrate import integrate
from projection import interval
from scipy.special import expn
#from projection import *
from numpy import pi, array, zeros, exp, prod

#assembling the vector
def assembleVector(g, basis, space_proj, time_proj):
    from basis import Sparse_multiscale
    if isinstance(basis, Sparse_multiscale):
      N = basis.nx
      numx = 25
      B = array(zeros(N))
      for i in range(N):
         intervalsT = basis.baseT[i].support
         intervalsX = basis.baseX[i].support
         for s1 in intervalsX:
            for s2 in intervalsT:
               f = vectorEntries(basis.baseX[i], s1, basis.baseT[i], s2, g, space_proj, time_proj)
               B[i] += integrate(f, interval(0,1), interval(0,1), n=numx)
      return B


    Nx = basis.nx
    Nt = basis.nt
    numx = 25
    B = array(zeros(Nt*Nx))
    for l in range(Nt):
          #print "finished block ", l
          intervals2 = basis.baseT[l].support
          #run over the indices of all the basis functions
          for alpha in range(basis.nx):
               intervals1 = basis.baseX[alpha].support
               #print intervals1
               for s1 in intervals1:
                  for s2 in intervals2:
                     #print "s1,s2", s1, s2
                     f = vectorEntries(basis.baseX[alpha],s1, basis.baseT[l],s2, g, space_proj, time_proj)
                     B[l*Nx+alpha] += integrate(f, interval(0,1), interval(0,1), n=numx)
    return B



#takes two basis functions and a kernel which is singular at x=y, the corresponding projection and number of quadrature points
#todo: should take arbitrary dimensions for x,y
def vectorEntries(basis1, support1, basis2, support2, func, proj1, proj2):
    a1 = support1(0); a2 = support1(1)
    b1 = support2(0); b2 = support2(1)

    def f(x,t):
        x = a1+(a2-a1)*x
        t = b1+(b2-b1)*t
        #print basis1(x), basis2(t)
        return prod(
                [ (basis1(x)*basis2(t)).reshape(-1)
                , func(proj1(x), proj2(t)).reshape(-1)
                , (a2-a1)
                , (b2-b1)
                , proj1.derivative(x).reshape(-1)
                , proj2.derivative(t).reshape(-1)
                ], axis=0).reshape(-1,1)

    return f
