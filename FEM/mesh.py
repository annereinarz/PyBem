from numpy import matrix, array, zeros
from numpy.linalg import inv
#Nodes
class node(object):
    def __init__(self, x):
        self.x = x[0]
        self.y = x[1]
    def vector(self):
        return array([self.x,self.y])
    def __repr__(self):
        return "({},{})".format(self.x,self.y)      
        
#place N points along a straight line from a to b
def meshInterval(a,b,N):
    step = (b.vector() - a.vector())/float(N)
    list = [a]
    for i in range(1,N):
        list.append(node(a.vector()+i*step))
    list.append(b)
    return list

from numpy import sqrt,sin,cos, arctan2, pi
#Place N points along a curved line from a to b
#Assumes that the center of the circle is at (0,0)
def meshArc(a,b,N):
    r = sqrt(a.vector()[0]**2+a.vector()[1]**2)  #get radius
    assert r == sqrt(b.vector()[0]**2+b.vector()[1]**2) #ensure the two points lie on the same circle
    theta_a = arctan2(a.vector()[1],a.vector()[0])
    theta_b = arctan2(b.vector()[1],b.vector()[0])
    list = [a]
    if theta_a > theta_b:
        theta_b = theta_b + 2*pi
    for i in range(1,N):
        theta_h  = theta_a + (theta_b - theta_a)/N*i
        h = array([r*cos(theta_h), r*sin(theta_h)])
        list.append(node(h))
    list.append(b)
    return list

#Given the 4 edge nodes of a quadrilateral places nodes around the boundary and returns them as 4 lists
def quadrilateralBoundary(a,b,c,d,N):
    return [meshInterval(a,b,N), meshInterval(b,c,N), meshInterval(c,d,N), meshInterval(d,a,N)]

#class for a line between two points
class line(object):
    def __init__(self, start, end):
        self.start = start
        self.end   = end
        
    def vector(self):
        return self.end-self.start 
    
    def length(self):
        v = self.vector()
        return sqrt(v[0]**2+v[1]**2)

#place the inner nodes given the boundary nodes
def innerNodesQuadrilateral(ab,bc,cd,da):
    N = len(ab)-1
    ad = list(reversed(da))
    dc = list(reversed(cd))
    innerNodes = []
    for i, (p_ab, p_dc) in enumerate(zip(ab[1:-1], dc[1:-1])):
        line1 = line(p_ab.vector(), p_dc.vector())
        step1 = line1.vector() / N
        row = []
        for j, (p_ad, p_bc) in enumerate(zip(ad[1:-1], bc[1:-1])):
            #line2 = line(p_ad.vector(), p_bc.vector())
            #step2 = line2.vector() / N
            x = line1.start + (j+1)*step1
            #x = intersect(line1,line2)
            row.append(node(x))
        innerNodes.append(row)
    return innerNodes

#Merge the boundary and inner nodes into one list, so that
#they are ordered from left to right
def mergeBoundaryAndInnerNodes(boundary, inner):
    boundary2 = list(reversed(boundary[2]))
    boundary3 = list(reversed(boundary[3]))
    return [boundary3] + [ [l]+i+[r] for l,i,r in zip(boundary[0][1:-1],inner,boundary2[1:-1]) ] + [boundary[1]]

#Check wether a ray p intersects the segment between a and b
def rayIntersectsSegment(p, a,b):
    "Does the horizontal ray originating in p intersect the line segment from a to b?"
    if b[1]-a[1] == 0:  #lines are parallel
        return False
    lam = (p[1]-a[1]) / (b[1]-a[1])
    if not 0. <= lam <= 1.:  #ray misses segment
        return False
    sig = a[0] + lam*(b[0]-a[0]) - p[0]
    return sig >= 0  #side of the ray

import numpy as np
#Class for triangles
class triangle(object):
    def __init__(self, a,b,c):
        A = zeros([2,2])
        A[0,0] = (b.vector()[0]-a.vector()[0])
        A[1,0] = (c.vector()[0]-a.vector()[0])
        A[0,1] = (b.vector()[1]-a.vector()[1])
        A[1,1] = (c.vector()[1]-a.vector()[1])
        self.A = matrix(A)
        #self.detA = np.linalg.det(A)
        self.det = abs(np.linalg.det(A))
        self.nodes = (a,b,c) 
        self.P = inv(self.A)#inv(self.A)*inv(self.A).T*self.det

    def f1(self, x, y):
        return self.nodes[0].vector()[0] + self.A[0,0]*x+self.A[1,0]*y
    
    def f2(self, x, y):
        return self.nodes[0].vector()[1] + self.A[0,1]*x+self.A[1,1]*y
    
    def f1inv(self, xhat, yhat):
        a = self.nodes[0]
        invA = inv(self.A)
        return invA[0,0]*(xhat - a.vector()[0])  + invA[1,0]*(yhat - a.vector()[1])
    
    def f2inv(self, xhat, yhat):
        a = self.nodes[0]
        invA = inv(self.A)
        return invA[0,1]*(xhat - a.vector()[0]) + invA[1,1]*(yhat - a.vector()[1])

    def contains(self, x,y):
        p = array([x,y])
        cnt  = 0
        for i in range(3):
            a = self.nodes[i].vector()
            b = self.nodes[(i+1)%3].vector()
            #Check whether the horizontal ray originating in p intersects the line segment from a to b
            if b[1]-a[1] == 0:  #lines are parallel
                continue
            lam = (p[1]-a[1]) / (b[1]-a[1])
            if not 0. <= lam <= 1.:  #ray misses segment
                continue
            sig = a[0] + lam*(b[0]-a[0]) - p[0]
            if abs(sig) < 10e-10:
                # The point lies on the boundary of the triangle
                return True
            if sig >= 0: # Positive side of the ray
                cnt += 1 
        if cnt%2 == 1:
            return True
        return False
    center_x = property(lambda self: np.average([ n.x for n in self.nodes]) )
    center_y = property(lambda self: np.average([ n.y for n in self.nodes]) )
            
    def __repr__(self):
        return "triangle({},{},{})".format(*self.nodes)

#Given the nodes, create a mesh of triangle
#assumes the nodes are ordered from left to right and row by row
def meshQuadrilateral(nodes):
    triangles = []
    for row1, row2 in zip(nodes, nodes[1:]):
        for (a,b),(c,d) in zip(zip(row1,row1[1:]), zip(row2,row2[1:])):
            triangles.append(triangle(a,b,c))
            triangles.append(triangle(b,c,d))
    return triangles

def meshQuadrilateral4(nodes):
    triangles = []
    nod = []
    for row1, row2 in zip(nodes, nodes[1:]):
        for (a,b),(c,d) in zip(zip(row1,row1[1:]), zip(row2,row2[1:])):
            #create middle node
            n = node([0.25*(b.vector()[0]+c.vector()[0]+a.vector()[0]+d.vector()[0]), 
                      0.25*(b.vector()[1]+a.vector()[1]+c.vector()[1]+d.vector()[1])])
            nod.append(n)
            triangles.append(triangle(n,b,a))
            triangles.append(triangle(a,c,n))
            triangles.append(triangle(n,d,c))
            triangles.append(triangle(n,b,d))
    return (triangles, nod)

#returns the corners of a rectangle of size centered at (0,0)
def cornerRectangle(size):
    a = node(array([-size,-size]))
    b = node(array([size,-size]))
    c = node(array([size,size]))
    d = node(array([-size,size]))
    return (a,b,c,d)

class mesh(object):
    def __init__(self,l,r,Triangles,nodesInner,nodes):
        self.triangles  = list(Triangles)
        self.boundaryNodes = nodes
        self.innerNodes = nodesInner
        self.n = len(nodesInner)
        self.numbering = { n:i for i,n in enumerate(list(self.innerNodes)) }
        self.l = l
        self.r = r

    def findTriangles(self, x,y):
        #return filter(lambda t: t.contains(x,y), self.triangles)
        for t in self.triangles:
            if t.contains(x,y):
                return [t]
        return []


def join(lists):
    "Join a list of lists into one list."
    return [ x for l in lists for x in l ]

def meshSquare(N, R = 0.25):
    a = node([0.,0.])
    b = node([R, 0.])
    c = node([R, R ])
    d = node([0.,R ])
    boundary = quadrilateralBoundary(a,b,c,d,N)
    inner = innerNodesQuadrilateral(*boundary)
    square = mergeBoundaryAndInnerNodes(boundary,inner)
    triangles,newNodes = meshQuadrilateral4(square)
    m = mesh(0,R, triangles, join(inner)+newNodes, join(boundary))
    return m

def meshSquareBounded(N, R = 0.25):
    a = node([0.,0.])
    b = node([R, 0.])
    c = node([R, R ])
    d = node([0.,R ])
    boundary = quadrilateralBoundary(a,b,c,d,N)
    inner = innerNodesQuadrilateral(*boundary)
    square = mergeBoundaryAndInnerNodes(boundary,inner)
    triangles,newNodes = meshQuadrilateral4(square)
    m = mesh(0,R, triangles, list(set(join(inner)+newNodes)) + list(set(join(boundary))), [])
    return m


def meshCircleBounded(N,R = 0.25):
    return meshCircle(N,R,"bound")
#Creates a shape-regular mesh of triangles on the circle of radius R
#R is assumed to be 0.25 if not given
def meshCircle(N,R = 0.25, var = 0):
    #middle square
    a,b,c,d = cornerRectangle(R/(sqrt(2)+2))
    A,B,C,D = cornerRectangle(1/sqrt(2)*R)
    
    boundary = quadrilateralBoundary(a,b,c,d,N)
    inner = innerNodesQuadrilateral(*boundary)
    square = mergeBoundaryAndInnerNodes(boundary,inner)
    T,nMiddle = meshQuadrilateral4(square)
    #create arcs
    
    ab,bc,cd,da = boundary
    ba = list(reversed(ab))
    cb = list(reversed(bc))
    dc = list(reversed(cd))
    ad = list(reversed(da))
    
    aA = meshInterval(a,A, N)
    Aa = list(reversed(aA))
    bB = meshInterval(b,B, N)
    Bb = list(reversed(bB))
    cC = meshInterval(c,C, N)
    Cc = list(reversed(cC))
    dD = meshInterval(d,D, N)
    Dd = list(reversed(dD))
    
    AB = meshArc(A,B, N)
    BC = meshArc(B,C, N)
    CD = meshArc(C,D, N)
    DA = meshArc(D,A, N)
    
    boundaryBottom = [AB,Bb,ba,aA]
    innerBottom = innerNodesQuadrilateral(*boundaryBottom)
    squareBottom = mergeBoundaryAndInnerNodes(boundaryBottom,innerBottom)
    Tbottom,nBottom = meshQuadrilateral4(squareBottom)
    
    boundaryRight  = [BC,Cc,cb,bB]
    innerRight = innerNodesQuadrilateral(*boundaryRight)
    squareRight = mergeBoundaryAndInnerNodes(boundaryRight,innerRight)
    Tright,nRight = meshQuadrilateral4(squareRight)
    
    boundaryTop    = [CD,Dd,dc,cC]
    innerTop = innerNodesQuadrilateral(*boundaryTop)
    squareTop = mergeBoundaryAndInnerNodes(boundaryTop,innerTop)
    Ttop,nTop = meshQuadrilateral4(squareTop)
    
    boundaryLeft   = [DA,Aa,ad,dD]
    innerLeft = innerNodesQuadrilateral(*boundaryLeft)
    squareLeft = mergeBoundaryAndInnerNodes(boundaryLeft,innerLeft)
    Tleft,nLeft = meshQuadrilateral4(squareLeft)
    
    ins = list(set(nMiddle+nTop+nBottom+nRight+nLeft+join(square+inner+innerLeft+innerRight+innerBottom+innerTop)+Aa+Bb+Cc+Dd)-set([A,B,C,D]))
    ins = [ n for _,__,n in sorted((n.x,n.y,n) for n in ins) ] 
    bns = AB+BC+CD+DA
    bns.remove(A)
    bns.remove(B)
    bns.remove(C)
    bns.remove(D)
    if var:
        ins += bns
    return mesh(-R,R,T+Tbottom+Tright+Ttop+Tleft, ins, bns)
