from integrate import integrate

# Calculates the L^2 scalarproduct between the basis functions bi and bj,
# where px,pt are the projections in space and time respectively
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
                     m += integrate(lambda x,t: bi(x,t)*bj(x,t) *px.derivative(x)*pt.derivative(t), intersection,intersectionT, n = 13)
    return m

from numpy import zeros, matrix

#Sets up the mass matrix for the basis base,
# where px,pt are the projections in space and time respectively
def assembleMass(base, px, pt):
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

