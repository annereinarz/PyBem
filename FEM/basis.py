def basis(num,x,y):
    if num == 0:
        return 1-x-y
    if num == 2:
        return y
    if num == 1:
        return x
    raise 'invalid local node index'

from numpy import array
def basis_nabla(num):
    if num == 0:
        return array([-1,-1])
    if num == 2:
        return array([0,1])
    if num == 1:
        return array([1,0])
    raise 'invalid local node index'