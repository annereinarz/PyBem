import numpy as np

def mat_vec(A, x):
    "Multiply compressed block matrix A with vector x."
    h, w = A.shape
    if h == w and h == x.size:
        return A*x
    numberOfBlocks = h / w
    blockSize      = w
    assert x.size == h
    y = np.zeros([h])
    for n in range(numberOfBlocks):
        for k in range(n+1):
            y[n*blockSize:(n+1)*blockSize] += np.dot(A[k*blockSize:(k+1)*blockSize,:], x[(n-k)*blockSize:((n-k)+1)*blockSize])
    return y

def liftToVector(elementwiseFunction):
    """
    Lift an elementwise function with multiple parameters to vectors.
    
    Does not handle keyword arguments. 
    """
    from numpy import hstack, apply_along_axis
    def liftedFunction(x,t):
        return apply_along_axis(lambda xt: elementwiseFunction(xt[0:2],xt[2:3]), 1, hstack([x,t])) 
    return liftedFunction


if __name__ == '__main__':
     A = np.array([[1,2],[1,2],[1,2],[1,2]])
     x = np.array([1,1,1,1])
     print mat_vec(A,x)