from numpy import array, select, hstack
from functools import partial
from projection import interval

d = 2

#   Abstract base class for bases.
#   Subclasses have to initialize the 'base' field and
#   are strongly encouraged to override 'element_index'
#   with a more efficient implementation.
class base:
   def __getitem__(self,i):
        return self.base[i]
   def element_index(self,x):
      # "Warning: Using default element_index implementation is SLOW!
      indices = []
      for (i,f) in enumerate(self.base):
        if any(s.contains(x) for s in f.support):
           indices.append(i)
      return array(indices, dtype=int)

#piecewise constant basis in one dimension
class Constant1D(base):
   def __init__(self, n):
        self.n = n
        self.base = []
        for i in xrange(n):
           b         = partial(lambda i,t: i == self.element_index(t), i)
           b.support = [ interval(float(i)/n, float(i+1)/n) ]
           self.base.append(b)
   def element_index(self,t):
       return select([t<1.0], [array(t*self.n,int)], self.n-1)

def Const_basis(nt,nx):
   #pw constants in time and space
   if d == 2:
       return TARDIS(Constant1D(nx), Constant1D(nt))
   if d == 3:
       return TARDIS(TARDIS(Constant1D(nx),Constant1D(nx)), Constant1D(nt))


#Combines space and time bases
class TARDIS:
   def __init__(self, space, time):
      self.baseX = space
      self.baseT = time
   nx = property(lambda self: self.baseX.n)
   nt = property(lambda self: self.baseT.n)
   jmax = property(lambda self: self.baseX.jmax)
   jmin = property(lambda self: self.baseX.jmin)
   def element_indexX(self,x):
       return self.baseX.element_index(x)
   def element_indexT(self,t):
       return self.baseT.element_index(t)


