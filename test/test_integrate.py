from integrate import Reg


def test_higher_dimensional_quadrature():
   from numpy import cos, sin
   g = lambda x,t: t**2*cos(x)
   exsol = sin(1)/3.
   (Xs,W) =  Reg((10,[1,1]))
   assert abs(exsol - sum(g(*Xs)*W)) < 10e-7
