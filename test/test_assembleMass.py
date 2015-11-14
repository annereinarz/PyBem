from assembleMass import assembleMass


def test():
    from projection import circle, interval
    px = circle(1.)
    pt = interval(0,1)
    from basis import Const_basis
    b = Const_basis(3,3)
    A = assembleMass(b, px,pt)
    from numpy import diag, repeat
    from numpy.linalg import norm
    expectedA = diag(repeat([0.6981317], 3*3))
    assert norm(A - expectedA) < 10e-7
