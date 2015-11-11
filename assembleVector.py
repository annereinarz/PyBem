from integrate import integrate
from numpy import array, zeros, prod

#assembling the vector
def assembleVector(g, basis, space_proj, time_proj):
    Nx = basis.nx
    Nt = basis.nt
    numx = 25
    B = array(zeros(Nt*Nx))
    for l in range(Nt):
          intervals2 = basis.baseT[l].support
          for alpha in range(basis.nx):
               intervals1 = basis.baseX[alpha].support
               for s1 in intervals1:
                  for s2 in intervals2:
                     f = vectorEntries(basis.baseX[alpha],s1, basis.baseT[l],s2, g, space_proj, time_proj)
		     from projection import interval
                     B[l*Nx+alpha] += integrate(f, interval(0,1), interval(0,1), n=numx)
    return B



#takes two basis functions and a kernel which is singular at x=y, the corresponding projection and number of quadrature points
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
