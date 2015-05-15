from assembleVector import assembleVector
from assembleMatrix import assembleMass
from numpy.linalg import solve

def L2projection(g,m):
    B = assembleVector(g,m)
    A = assembleMass(m)
    return solve(A,B)


if __name__=='__main__':
    from numpy import sin,pi,ones
    from mesh import meshCircle, meshSquare, meshSquareBounded
    from plot import plotSol, plotMesh
    m = meshCircle(4,1,"bound")
    #plotMesh(m)
    g = lambda x,y: 4-x**2-y**2   #sin((sqrt(x**2+y**2)+0.25)*2*pi)
    #g = lambda x,y: x**2 + y**2 +2  #non-zero boundary
    #g = lambda x,y: sin(pi*x)*sin(pi*y)
    x = L2projection(g,m)
    fig = plotSol(x, m)
    plotMesh(m,fig,zs=0, zdir='z').show()
    
    