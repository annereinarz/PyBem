from meshG import createClusterTree
from covering import minimalCovering

def hypercube(d):
    return [(0.,1.)]*d

d = 2
t = createClusterTree(hypercube(d), levels=5)

print minimalCovering(lambda c: True , t) # -> one cluster
print minimalCovering(lambda c: False, t) # -> all panels

from numpy.linalg import norm
from numpy import empty

def admissableWRT(eta, x, cluster):
    center = empty([d])
    radius = empty([d])
    for i in range(d):
        (a,b) = cluster.getBounds(i)
        center[i] = (a+b)/2.
        radius[i] = (b-a)/2.
    radius = norm(radius)
    return radius <= eta * norm(x - center)

from numpy import array
from functools import partial

adm = partial(admissableWRT, 0.5, array([0.0]*d))


def plotMinimalCovering(admissable, t):
    assert d < 3
    
    from matplotlib.pyplot import figure, Rectangle
    from clustering import panel
    
    f = figure()
    a0,b0 = t.getBounds(0)
    a1,b1 = t.getBounds(1) if d==2 else (0,1)
    a = f.add_axes((a0,a1,b0,b1))
    
    for cluster in minimalCovering(admissable,t):
        a0,b0 = cluster.getBounds(0)
        a1,b1 = cluster.getBounds(1) if d==2 else (0.4,0.6)
        color = 'green' if cluster.isPanel() else 'red'
        a.add_patch(Rectangle((a0,a1),b0-a0,b1-a1, facecolor=color)) 
    
    f.show()

    
plotMinimalCovering(adm, t)
