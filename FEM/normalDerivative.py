def findBoundaryTriangle(m, n1,n2):
        for t in m.triangles:
            if n1 in t.nodes and n2 in t.nodes:
                return t
        raise('blarg!')
        
#Given a solution in the domain and the mesh
#normalDerivative(u,m) calculates the boundary
#flux and returns it as coefficient vector
def normalDerivative(coeff,basis,m):
    from numpy import sqrt, arctan2
    #Loop over all boundary nodes
    delu = []
    for (n1,n2) in zip(m.boundaryNodes, m.boundaryNodes[1:]+[m.boundaryNodes[0]]):
        t = findBoundaryTriangle(m,n1,n2)
        x = n1.x + 0.5*(n2.x-n1.x)
        y = n1.y + 0.5*(n2.y-n1.y)
        nx = x/sqrt(x**2+y**2)
        ny = y/sqrt(x**2+y**2)
        h = .0001
        while not t.contains(x-3*h*nx, y-3*h*ny):
            h = h/2.
        def funu(x,y):
            number = m.numbering
            result = 0
            #find triangles in which x,y are
            for T in m.findTriangles(x,y):
                #print T
                i = number.get(T.nodes[0])
                j = number.get(T.nodes[1])
                k = number.get(T.nodes[2])
                xhat = T.f1inv(x,y)
                yhat = T.f2inv(x,y)
                if not i is None: result += coeff[i]*basis(0,xhat,yhat)
                if not j is None: result += coeff[j]*basis(1,xhat,yhat)
                if not k is None: result += coeff[k]*basis(2,xhat,yhat)
            #print x,y, m.findTriangles(x,y), result
            if result == 0:
                from plot import plotMesh
                f = plotMesh(m)
                f.gca().plot(x,y,'rx')
                f.show()
            return result
        phi = arctan2(n2.y,n2.x)
        #fourth order finite difference scheme
        delu += [(phi, (11*funu(x,y) - 18*funu(x-h*nx, y-h*ny) + 9*funu(x-2*h*nx, y-2*h*ny) - 2*funu(x-3*h*nx, y-3*h*ny))/(6*h))]
    r_n = sqrt(m.boundaryNodes[0].x**2+m.boundaryNodes[0].y**2)
    def nD(x,y):
        r = sqrt(x**2+y**2)
        #assert abs(r - r_n) < 10e-10
        phi = arctan2(y,x)
        delu.sort()
        for phi_n, du in delu:
            if phi <= phi_n:
                return du
        _,du_0 = delu[0]
        return du_0
    return nD
