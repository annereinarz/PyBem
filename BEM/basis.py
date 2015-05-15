from numpy import array, select, hstack
from functools import partial
from projection import interval

d = 2


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


class MultiscaleComplement(base):

    def __init__(self, l):
        self.l = l
        self.n = 2**(l-1) if l > 0 else 1

        self.base = []
        #get all the basis functions on the level l
        for j in range(0,2**l,2):
               b         = partial(lambda l,j,x: phi(x,j,2**l), l,j)
               b.support = [ interval(float(j)/(2**l), float(j+1)/(2**l)) ]
               self.base.append(b)

    def element_index(self,x):
       indices = []
       ind = select([x<1.0], [array(x*2**self.l, int)], 2**self.l-1)
       if ind%2 == 0:
          indices.append(ind/2)
       return array(indices, dtype=int)



class Multiscale1D(base):

    def __init__(self, l):
        self.l = l
        self.n = 2**l

        self.base = []
        for i in range(l+1):
            #get all the basis functions on each level up to l
            for j in range(0,2**i,2):
               b         = partial(lambda i,j,x: phi(x,j,2**i), i,j)
               b.support = [ interval(float(j)/(2**i), float(j+1)/(2**i)) ]
               self.base.append(b)

    def element_index(self,x):
       indices = [0]
       for i in range(1,self.l+1):
          ind = select([x<1.0], [array(x*2**i, int)], 2**i-1)
          if ind%2 == 0:
             indices.append(ind/2 + 2**(i-1))
       return array(indices, dtype=int)


from numpy import ones,zeros, all
def psi13(x):
   #x = x.reshape(-1,1)
   if x[0] < -1:
       assert all(x[1:] < -1) 
       return 0
   elif x[0] < 0:
       assert all(-1 <= x[1:]) and all(x[1:] < 0)
       return -1/8.
   elif x[0] < 0.5:
       assert all(0 <= x[1:]) and all(x[1:] < 0.5)
       return 1.
   elif  x[0] < 1:
       assert all(0.5 <= x[1:]) and all(x[1:] < 1)
       return -1.
   elif x[0] < 2:
       assert all(1 <= x[1:]) and all(x[1:] < 2)
       return 1/8.
   else:
       assert all(2 <= x[1:])
       return 0.


def wavelet13(k,j,x):
    return 2**(j/2.)*psi13(2.**j*x-k)

def phi(x,i,N):
    x = x.reshape(-1,1)
    N = float(N)
    from numpy import logical_and
    return logical_and(i/float(N) <= x, x <= (i+1)/float(N))

class Wavelet1D(base):

   def __init__(self, jmax, jmin):
        assert jmin <= jmax
        assert jmin >= 3  # to avoid overlap
        # Basis corresponds to a mesh of Interval [0,1] divided into n equal slices both in x and in y
        self.n = sum(2**j for j in range(jmin,jmax+1)) + 2**jmin
        self.jmin = jmin
        self.jmax = jmax
        self.base = []

        for i in xrange(2**jmin):
            b         = partial(lambda i,x: phi(x,i,2**jmin), i)
            b.support = [ interval(float(i)/(2**jmin), float(i+1)/(2**jmin)) ]
            b.j       = 0
            self.base.append(b)

        for j in xrange(jmin,jmax+1):
           for k in xrange(2**j):
               f = partial(lambda j,k,x: wavelet13(k,j,x), j, k)
               s = interval(2**float(-j) * (k-1), 2**float(-j) * (2+k)).splitAt([2**float(-j) * (k+a) for a in [0.,0.5,1.]])
               if k == 0:
                   # left-most function, extends outside the interval and needs to be wrapped around
                   f = partial(lambda f,x: f(select([x>0.5],[x-1],x)),f)
                   s = [s[0].shiftedBy(1)] + s[1:]
               elif k == 2**j-1:
                   # right-most function, extends outside the interval and needs to be wrapped around
                   f = partial(lambda f,x: f(select([x<0.5],[x+1],x)),f)
                   s = s[:-1] + [s[-1].shiftedBy(-1)]
               b         = f
               b.support = s
               b.j       = j
               self.base.append(b)

   def element_index(self,x):
       assert len(x.shape) == 2 and x.shape[1] == 1, "Expected shape is (_,1). Actual shape was {}.".format(x.shape)
       offset = 2**self.jmin
       idx = select([x < 1.0], [array(x * 2**self.jmin, int)], 2**self.jmin-1)
       l = [idx]
       for j in range(self.jmin, self.jmax+1):
           idx = select([x < 1.0], [array(x * 2**j, int)], 2**j-1)
           l.append( offset + select([idx != 0     ], [ idx - 1 ], 2**j-1) )
           l.append( offset +                           idx                )
           l.append( offset + select([idx != 2**j-1], [ idx + 1 ], 0     ) )
           offset += 2**j
       return hstack(l)


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


class CombinedBase(base):

   def __init__(self):
        self.n = 0
        self.children = []
        self.base = []

   def combine(self, b):
       self.n += b.n
       self.children.append(b)
       self.base += b.base

   def element_index(self,x):
       offset = 0
       arrays = []
       for c in self.children:
          e_i = c.element_index(x)
          if e_i.size > 0:                # Since hstack can't deal with having some arrays be empty (and therefore of the wrong shape)
             arrays.append(offset + e_i)
          offset += c.n
       return hstack(arrays)


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

def SparseIndices(nmax,sigma):
    Indices = []
    for it in range(nmax+1):
        for ix in range(nmax+1):
            if(it/sigma + ix*sigma <= nmax):
               Indices.append([ix,it])
    return Indices

def SparseIndicesT(nmax,T):
    Indices = []
    for it in range(nmax+1):
        for ix in range(nmax+1):
            if(it/2+ ix - T*max(it/2,ix) <= (1-T)*nmax):
               Indices.append([ix,it])
    return Indices

#Used for combination technique
def SIndices(level,l,sigma):
    I = []
    #find (it,ix) such that it/sigma+ix*sigma=level-l
    for it in range(level+1):
        for ix in range(level+1):
            if it/sigma+ix*sigma == level-l:
               I.append([it,ix])
    return I

#TODO: Fix, can be done more efficiently
def findIndices(basis, it,ix):
    I = []
    for m in range(2**it):
        for alpha in range(2**ix):
            #find corresponding i and add to list
            for i in range(basis.nx):
               if basis.baseX[i].alpha == alpha and basis.baseT[i].m == m:
                    I.append(i)
                    break
    assert len(I) == 2**it*2**ix
    return I

class Sparse_multiscale(TARDIS):
   def __init__(self, l,sigma):
      bx = base()
      bx.base = []
      bt = base()
      bt.base = []
      for (it,ix) in SparseIndices(l,sigma):
         m = 2**(it-1) if it > 0 else 0
         for ft in MultiscaleComplement(it):
               alpha = 2**(ix-1) if ix > 0 else 0
               for fx in MultiscaleComplement(ix):
                  fx.alpha = alpha
                  ft.m     = m
                  bx.base.append(fx)
                  bt.base.append(ft)
                  alpha += 1
               m+=1
      bx.n = len(bx.base)
      bt.n = len(bt.base)
      TARDIS.__init__(self, space=bx, time=bt)
      
class Sparse_multiscaleT(TARDIS):
   def __init__(self, l,T):
      bx = base()
      bx.base = []
      bt = base()
      bt.base = []
      for (it,ix) in SparseIndicesT(l,T):
         m = 2**(it-1) if it > 0 else 0
         for ft in MultiscaleComplement(it):
               alpha = 2**(ix-1) if ix > 0 else 0
               for fx in MultiscaleComplement(ix):
                  fx.alpha = alpha
                  ft.m     = m
                  bx.base.append(fx)
                  bt.base.append(ft)
                  alpha += 1
               m+=1
      bx.n = len(bx.base)
      bt.n = len(bt.base)
      TARDIS.__init__(self, space=bx, time=bt)


def Const_multiscale(lt, lx):
   #full multiscale, pw constants in time and space 
   return TARDIS(Multiscale1D(lx),Multiscale1D(lt))


def Wavelet_basis(nt,jmax,jmin):
   #constant wavelets in space pw constants in time
   return TARDIS(Wavelet1D(jmax,jmin), Constant1D(nt))

def Const_basis(nt,nx):
   #pw constants in time and space
   return TARDIS(Constant1D(nx), Constant1D(nt))

from numpy import log2,linspace
from numpy import matrix
def Multiscale2Const2d(multiscale,const):
    assert multiscale.nx*multiscale.nt == const.nx*const.nt
    A = zeros([multiscale.nx*multiscale.nt, const.nx*const.nt])
    lx = int(log2(multiscale.nx))
    lt = int(log2(multiscale.nt))
    
    for currentlevelx in range(lx+1):
        for indexonlevelx in range(2**max(0,currentlevelx-1)):
            for currentlevelt in range(lt+1):
                for indexonlevelt in range(2**max(0,currentlevelt-1)):
                    if currentlevelx == 0:
                        kmalpha = indexonlevelx
                    else:
                        kmalpha = 2**(currentlevelx-1) + indexonlevelx
                    if currentlevelt == 0:
                        kmm = indexonlevelt
                    else:
                        kmm     = 2**(currentlevelt-1) + indexonlevelt
                    km = kmalpha + const.nx*kmm

                    for ix in range(2**(lx-currentlevelx)):
                        for it in range(2**(lt-currentlevelt)):
                            kcalpha = 2**(lx-currentlevelx+1)*indexonlevelx + ix
                            kcm     = 2**(lt-currentlevelt+1)*indexonlevelt + it
                            kc = kcalpha + const.nx*kcm
                            A[kc,km] = 1
    return A

def Multiscale2Const(multiscale, const):
    assert multiscale.n == const.n
    A = zeros([multiscale.n, const.n])
    l = int(log2(multiscale.n))
    for currentlevel in range(l+1):
        for indexonlevel in range(2**max(0,currentlevel-1)):
            if currentlevel == 0:
                km = indexonlevel
            else:
                km = 2**(currentlevel-1) + indexonlevel
            for i in range(2**(l-currentlevel)):
                kc = 2**(l-currentlevel+1)*indexonlevel + i
                A[kc,km] = 1
    return A




##############
#            #
# Test Files #
#            #
##############
from numpy import tile,ones
from numpy.random import rand
def testMultiscale2Const1d():
    baseMultiscale = Multiscale1D(4)
    baseConst      = Constant1D(2**4)
    
    coeff = rand(baseConst.n,1)
    A = matrix(Multiscale2Const(baseMultiscale, baseConst))
    
    def f(coeff, base, x):
        print coeff.shape, x.shape
        h = 0
        for i in range(base.n):
            h += coeff[i]*base[i](x)
        print "h",h
        return h
            
    x = linspace(0,1,100)
    f1 = zeros(100)
    f2 = zeros(100)
    for i in range(100):
        f1[i] = f(A*coeff, baseConst, x[i])
        f2[i] = f(coeff, baseMultiscale, x[i])
            
    from matplotlib.pyplot import figure
    
    f = figure()
    f.gca().plot(x,f1,'x-')
    f.gca().plot(x,f2,'r--')
    f.show()


def testMultiscale2Const(lt,lx):
    baseMultiscale = Const_multiscale(lt,lx)
    baseConst      = Const_basis(2**lt,2**lx)
    from numpy import cos, arctan2
    def g(x,t):
        h = arctan2(x[:,1],x[:,0])
        return cos(h).reshape(-1,1)
    from assembleVector import assembleVector
    from projection import circle
    circ = circle(1)
    coeff = assembleVector(g,baseConst, circ,interval(0,1))
    print "coeff shape ",coeff.shape
    A = matrix(Multiscale2Const2d(baseMultiscale, baseConst))
    #print (A*coeff).shape
    
    def f(coeff, base, x,t):
        #print coeff.shape, x.shape
        h = 0
        for alpha in range(base.nx):
            for m in range(base.nt):
                h += coeff[alpha+m*base.nx]*base.baseX[alpha](x)*base.baseT[m](t)
        #print "h",h
        return h
            
    x = linspace(0,1,100)
    t = 0.2
    f1 = zeros(100)
    f2 = zeros(100)
    from numpy.linalg import solve
    help = solve(A,coeff)
    for i in range(100):
        f1[i] = f(coeff, baseConst, array(x[i]),array(t))
        f2[i] = f(help, baseMultiscale, array(x[i]),array(t))
            
    from matplotlib.pyplot import figure
    
    f = figure()
    f.gca().plot(x,f1,'x-')
    f.gca().plot(x,f2,'r--')
    f.show()




