def bisect(cube,axis=0):
    (a,b) = cube[axis]
    m = (a+b)/2
    return ( cube[0:axis]+[(a,m)]+cube[axis+1:]
           , cube[0:axis]+[(m,b)]+cube[axis+1:]
           )

from clustering import panel, cluster


def createClusterTree(cube, levels):
    d = len(cube)
    def recurse(cube, levels, axis):
        if levels == 1:
            return panel(cube)
        i1, i2 = bisect(cube, axis)
        c1 = recurse(i1, levels=levels-1, axis=(axis+1)%d)
        c2 = recurse(i2, levels=levels-1, axis=(axis+1)%d)
        return cluster([c1,c2], cube)
    return recurse(cube, levels, 0)

        