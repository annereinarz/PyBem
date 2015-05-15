from numpy import zeros
from BEM.integrate import GJTria
from basis import basis

# Assemble the vector of the right hand side
# g - function of the rhs
# T - triangulation
def assembleVector(g, m):
    B = zeros(m.n)
    # run over all inner nodes
    for t in m.triangles:
        f1 = t.f1
        f2 = t.f2
        for localNum, n in enumerate(t.nodes):
            globalNum = m.numbering.get(n)
            if globalNum is not None: # only for inner nodes, since Dirichlet boundary is assumed
                a = GJTria(lambda x, y: g(f1(x,y),f2(x,y)) * basis(localNum, x, y) * t.det, 30)
                B[globalNum] += a
    return B

if __name__=='__main__':
    from numpy import sin,pi
    from mesh import meshCircle
    m = meshCircle(2)
    m.plot()
    g = lambda x,y: sin(pi*x)*sin(pi*y)
    B = assembleVector(g,m)
    print B
