from integrate1d import sing_gauleg, gauleg
from numpy import abs, sin, cos, sqrt


def test_singularity_at_0_5():
    f1 = lambda x: abs(x-0.5)**(-0.5)  # function with a singularity at 0.5
    ex1 = 2*sqrt(2)
    (x,w) = sing_gauleg(18, t=0.5, flag=1)	
    assert abs(ex1 - sum(f1(x)*w)) < 10e-7

def test_singularity_at_0():
    f2 = lambda x: x**(-0.5)        # function with a singularity at 0
    ex2 = 2
    (x,w) = sing_gauleg(18)	
    assert abs(ex2 - sum(f2(x)*w)) < 10e-7	

def test_no_singularity():
    f3 = lambda x: sin(x)*cos(x)    #function without singularity 
    ex3 = sin(1)**2/2.
    (x,w) = gauleg(4)	
    assert abs(ex3 - sum(f3(x)*w)) < 10e-7
