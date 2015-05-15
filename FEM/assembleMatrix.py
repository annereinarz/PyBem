from numpy import mat, zeros, max
from BEM.integrate import GJTria
from basis import basis, basis_nabla

#assemble the mass matrix
# m  -  input mesh
def assembleMass(m):
    A= mat(zeros([m.n, m.n]))
    # run over all inner nodes, ie over all basis functions
    for triangle in m.triangles:
        for cnt in range(3):
            n1 = triangle.nodes[cnt]
            n2 = triangle.nodes[(cnt+1)%3]
            i = m.numbering.get(n1)
            j = m.numbering.get(n2)
            if i is None:
                continue
            A[i,i] += GJTria(lambda x,y: basis(cnt,x,y)*basis(cnt,x,y)*triangle.det, 3)/2
            if j is None:
                continue
            A[i,j] += GJTria(lambda x,y: basis(cnt, x,y)*basis((cnt+1)%3, x,y)*triangle.det, 3)
    A = A + A.T
    return A

def assembleStiffness(m):
    #print m.n
    A= mat(zeros([m.n, m.n]))
    # run over all inner nodes, ie over all basis functions
    number = { n:i for i,n in enumerate(m.innerNodes) }
    for triangle in m.triangles:
        #print triangle.P
        for cnt in range(3):
            n1 = triangle.nodes[cnt]
            n2 = triangle.nodes[(cnt+1)%3]
            i = m.numbering.get(n1)
            j = m.numbering.get(n2)
            if i is None:
                continue
            A[i,i] += GJTria(lambda x,y: dot_prod(triangle.P,basis_nabla(cnt),basis_nabla(cnt)) * triangle.det,3)/2.
            if j is None:
                continue            
            A[i,j] += GJTria(lambda x,y: dot_prod(triangle.P,basis_nabla(cnt),basis_nabla((cnt+1)%3)) * triangle.det,3)
    A = A + A.T
    return A

def dot_prod(A,v1,v2):
    return (v1[0]*A[0,0]+v1[1]*A[0,1])*(v2[0]*A[0,0]+v2[1]*A[0,1]) + (v1[0]*A[1,0]+v1[1]*A[1,1])*(v2[0]*A[1,0]+v2[1]*A[1,1])
