from numpy import zeros,sqrt
from functools import partial
from projection import interval

class base:
   """
   Abstract base class for bases.

   Subclasses have to initialize the 'base' field and
   are strongly encouraged to override 'element_index'
   with a more efficient implementation.
   """

   def __getitem__(self,i):
        return self.base[i]

   def element_index(self,x):
      #print "Warning: Using default element_index implementation! (SLOW!)"
      indices = []
      for (i,f) in enumerate(self.base):
        if any(s.contains(x) for s in f.support):
           indices.append(i)
      return array(indices, dtype=int)


#Linear spline basis
def LinearSpline(x):
    a = zeros(x.shape)
    for i in range(x.shape[0]):
        if x[i] <= 0 and x[i] >= -1:
            a[i] = 1+x[i]
        elif x[i] >= 0 and x[i] <= 1:
            a[i] = 1-x[i]
        else:
            a[i] =  0
    return a

class Linear1D(base):
   def __init__(self, n):
        self.n = n+1
        self.base = []
        h = 1./float(self.n-1)
        for i in xrange(self.n):
           b         = partial(lambda i,t: LinearSpline(-i+t/h), i)
           if i>0 and i<self.n-1:
               b.support = [ interval(float(i-1)/n, float(i)/n), interval(float(i)/n, float(i+1)/n) ]
           elif i==0:
                b.support = [ interval(float(i)/n, float(i+1)/n) ]
           else:
                b.support = [ interval(float(i-1)/n, float(i)/n) ]
           self.base.append(b)

class LinearWavelet1D(base):
   def __init__(self, n):
        self.n = n+1
        self.base = []
        h = 1./float(self.n-1)
        for i in xrange(self.n):
           #TODO: So far only generators not actual wavelets
           #modified left functions, since we only have an initial condition, constraints on the right are unnecessary
           if i == 0:
               b         = partial(lambda i,t: LinearSpline(-i+t/h)+LinearSpline(-i-1+t/h), i)
           #interior functions
           else:
               b         = partial(lambda i,t: sqrt(2)*(-1./8*LinearSpline(-i+2+t/h)+1./4*LinearSpline(-i+1+t/h)+3./4*LinearSpline(-i+t/h)
                                   +1./4*LinearSpline(-i-1+t/h)-1./8*LinearSpline(-i-2+t/h)), i)
               
           #supports
           if i>0 and i<self.n-1:
               b.support = [ interval(float(i-1)/n, float(i)/n), interval(float(i)/n, float(i+1)/n) ]
           elif i==0:
                b.support = [ interval(float(i)/n, float(i+1)/n) ]
           else:
                b.support = [ interval(float(i-1)/n, float(i)/n) ]
           self.base.append(b)

if __name__ == '__main__':
    from matplotlib.pyplot import figure
    from numpy import linspace, array
    a = array(linspace(0,1,101))
    n = 10
    base = LinearWavelet1D(n)
    f = figure()
    for i in range(3,5):
        f.gca().plot(a,base[i](a))
    f.show()