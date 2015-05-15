from numpy import sqrt, sum

def norm_(z):
	z = z.reshape(-1,2)
	assert z.shape[1] == 2
	return sqrt(sum(z**2, axis=1)).reshape(z.shape[0], 1)

def hasDimension(x,d):
   return len(x.shape) == 2 and x.shape[1] == d


def concat(iter):
    "Concatenate a lists of lists together into one list."
    return list(x for xs in iter for x in xs)
   
def dot(a,b):
    assert len(a) == len(b)
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum 
