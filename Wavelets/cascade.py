from collections import defaultdict

def cascade(h, phi_0, j_max):
    start = min(phi_0.keys())
    end   = max(phi_0.keys())
    il = (end-start)/2         # must be integer
    phi_j = phi_0
    h     = defaultdict(int, h)
    for j in range(1,j_max+1):
        phi_jm1 = dict(phi_j)  # copy previous
        phi_j   = {}           # start fresh
        ind_1 = il * 2**j
        ind_2 = il * 2**(j-1) - 1
        for m in range(-ind_1, ind_1+1):
            for l in range(-ind_2,ind_2+1):
               phi_j[2**(-j)*m] = sum( [ h[m-2*l] * phi_jm1[2**(-(j-1))*l] for l in range(-ind_2,ind_2+1) ] )
    return phi_j

def approxHat(j_max):
    h     = {-1:0.5, 0:1, 1:0.5}
    phi_0 = {-1:0,   0:1, 1:0  }
    return cascade(h,phi_0, j_max)

def approxBioHat(j_max):
    h     = {-2:-1./8., -1:1./4., 0:3./4., 1:1./4., 2:-1./8.}
    phi_0 = {-2:0,      -1:0,     0:1,     1:0,     2:0     }
    return cascade(h,phi_0, j_max)

if __name__ == '__main__':
    from numpy import arange, zeros
    n = 8
    phi = approxBioHat(n)
    h = 2**(-n)
    j = arange(-2,2+h,h)
    phiarr = zeros(j.shape[0])
    for i in range(j.shape[0]):
        phiarr[i] = phi[j[i]]
    from matplotlib.pyplot import figure
    f = figure()
    f.gca().plot(j,phiarr,'b-')
    f.show()
