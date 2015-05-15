from meshG import createClusterTree
from covering import minimalCovering
from matrixPartitioning import transform

def hypercube(d):
    return [(0.,1.)]*d

d = 1
t = createClusterTree(hypercube(d), levels=5)
tt = transform((t,t))

from numpy.linalg import norm
from numpy import empty

def admissablePair(eta, clusterPair):
    t1,t2 = clusterPair.t1, clusterPair.t2
    center1 = empty([d])
    center2 = empty([d])
    radius2 = empty([d])
    for i in range(d):
        (a1,b1) = t1.getBounds(i)
        (a2,b2) = t2.getBounds(i)
        center1[i] = (a1+b1)/2.
        center2[i] = (a2+b2)/2.
        radius2[i] = (b2-a2)/2.
    radius2 = norm(radius2)
    return radius2 <= eta * norm(center2 - center1)

from numpy import array
from functools import partial

adm = partial(admissablePair, 0.4)

def plotMinimalCovering(admissable, t):
    assert d == 1
    
    from matplotlib.pyplot import figure, Rectangle
    from clustering import panel
    
    f = figure()
    a0,b0 = t.t1.getBounds(0)
    a1,b1 = t.t2.getBounds(0)
    a = f.add_axes((a0,a1,b0,b1))
    
    for cluster in minimalCovering(admissable,t):
        a0,b0 = cluster.t1.getBounds(0)
        a1,b1 = cluster.t2.getBounds(0)
        color = 'green' if cluster.isPanel() else 'red'
        a.add_patch(Rectangle((a0,a1),b0-a0,b1-a1, facecolor=color)) 
    
    f.show()
    

plotMinimalCovering(adm, tt)
