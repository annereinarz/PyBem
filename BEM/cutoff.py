from numpy import min,max, hstack, array
from util import norm_

#base  ---- the basis used
#alpha, beta ---- basis functions
#px    ---- projection used in space
#a,delta -- parameters of the compression, the same parameters are used here for both compressions
def cutoff(base,alpha,beta, space_proj, a,delta):
    ############################
    # FIRST MATRIX COMPRESSION #
    ############################

    #Calculate Cut-off parameter
    j  = base.baseX[alpha].j        #j for first basis function
    j1 = base.baseX[beta].j         #j for second basis function
    if j == 0 or j1 == 0:          #case of constant functions
        return False
    jmax = base.jmax
    Bjj = a*max([2**(-j),2**(-j1),2**((jmax*(2.*delta+1.)-j*(delta+4.)-j1*(delta+4.))/7.)]);

    #Calculate supports of psi_j,k and psi_j1,k1
    alpha1 = space_proj(array([[base.baseX[alpha].support[0].start]]))
    alpha2 = space_proj(array([[base.baseX[alpha].support[-1].end]]))
    beta1  = space_proj(array([[base.baseX[beta].support[0].start]]))
    beta2  = space_proj(array([[base.baseX[beta].support[-1].end]]))
    #Calculate the circles which enclose the supports
    m1 = 0.5*(alpha1-alpha2)+alpha2
    r1 = 0.5*norm_(alpha1-alpha2)
    m2 = 0.5*(beta1-beta2)+beta2
    r2 = 0.5*norm_(beta1-beta2)

    #Calculate distance
    dist = max([0, norm_(m1-m2)-r1-r2])
    #Compare
    if dist > Bjj:
        return True

    #############################
    # SECOND MATRIX COMPRESSION #
    #############################

    #Calculate Cut-off parameter
    BjjS = a*max([2.**(-j),2.**(-j1),2.**((jmax*(2.*delta+1.)-3.*max([j,j1])-(j+j1)*(delta+1.))/4.)])

    #Calculate singular support of the first basis function
    alphaEndpoints = [ s.end for s in base.baseX[alpha].support[0:-1] ]
    #Calculate singular support of the second basis function
    betaEndpoints  = [ s.end for s in base.baseX[beta ].support[0:-1] ]
    minSoFar = float('inf')
    for s in alphaEndpoints:
        m = norm_(m2 - space_proj(array([[s]]))) - r2
        if m < minSoFar:
            minSoFar = m
    for s in betaEndpoints:
        m = norm_(m1 - space_proj(array([[s]]))) -r1
        if m < minSoFar:
            minSoFar = m
    dist = max([0, minSoFar])
    if dist > BjjS:
        return True

    return False
