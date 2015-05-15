from numpy import sum


class CoordinateVector:

   def __init__(self, coeff, base, projs):
      self.coeff = coeff
      self.base  = base
      self.projs = projs

   def __call__(self, *args):
      args_ = [ p.inverse(a).reshape(-1,1) for p,a in zip(self.projs, args) ]
      # Find the indices of all elements containing x,t
      i = self.base.element_indices(*args_)
      # All summands are 0 except at these elements
      return sum( self.coeff[i] * self.base[i](*args_), axis=1 ).reshape(-1,1)
