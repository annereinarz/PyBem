from numpy import sqrt, sum

#returns the two norm of z
def norm_(z):
	z = z.reshape(-1,2)
	assert z.shape[1] == 2
	return sqrt(sum(z**2, axis=1)).reshape(z.shape[0], 1)

#checks if the dimension of x is d
def hasDimension(x,d):
   return len(x.shape) == 2 and x.shape[1] == d

#concatenates two lists into a new list
def concat(iter):
    return list(x for xs in iter for x in xs)

#calculates the dot product between two 1xn vectors
def dot(a,b):
    assert len(a) == len(b)
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum

#calculates the dot product between the rows of x and y leaving the columns,
#this can return a vector
import numpy as np
def dot_prod(x,y):
    assert x.shape == y.shape
    return np.fromiter(map(np.dot,x,y), dtype=x.dtype).reshape(x.shape[0], 1)
