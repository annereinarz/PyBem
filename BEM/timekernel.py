from scipy.special import expn
from numpy import prod, pi, exp, array, zeros
from util import hasDimension, concat

diag = 0
lu = 1
rb = -1

#analytically calculated time integration for the single-layer operator
def timeQuad(z, endT, basis, **kwargs):
   if 'l' in kwargs:
      l = kwargs['l']
      ht = endT*basis.baseT[l].support[0].length
      return timeQuadhelp(z,ht,l)
   else:
      m = kwargs['m']
      h1 = endT*basis.baseT[m].support[0].length
      istart1 = endT*basis.baseT[m].support[0].start
      n = kwargs['n']
      h2 = endT*basis.baseT[n].support[0].length
      istart2 = endT*basis.baseT[n].support[0].start
      a = int(max(h1,h2)/min(h1,h2))
      h = zeros([z.shape[0]])
      for  i in range(a):
          if h2 > h1:
              l = int(istart1/h1) - (int(istart2/h1)+i)
          else:
              l = int(istart1/h2)+i - int(istart2/h2)
          if l >= 0:
              h += timeQuadhelp(z,min(h1,h2),l)
      return h
  
def timeQuadN(z, endT, basis, **kwargs):
    if 'l' in kwargs:
        l = kwargs['l']
        ht = endT*basis.baseT[l].support[0].length
        return timeQuadNhelp(z,ht,l)
    else:
        print 'not implemented yet for the double layer operator'
     
def timeQuadNhelp(z,ht,l):
   if not hasDimension(z,2):
      raise NotImplementedError("timeQuadN not implemented for dimension other then 2. Input had shape {}.".format(z.shape))
   a1 = (z[:,0]**2+z[:,1]**2)/(ht*4)
   if l == 0:
        return 1./(8*pi) * (-expn(1,a1)+exp(-a1)/a1)
   elif l == 1:
       a2 = a1/2.
       return 1./(8*pi) * (exp(-a2)/a2 - expn(1,a2) - 2*exp(-a1)/a1 + 2*expn(1,a1))
   elif l<0:
       return 0
   alm1 = a1/(l-1);
   alp1 = a1/(l+1);
   al   = a1/l;
   return 1./(8*pi)* ( 2*expn(1,al)- 2*exp(-al)/al - expn(1,alm1)  + exp(-alm1)/alm1 - expn(1,alp1)  + exp(-alp1)/alp1 )
     

def timeQuadhelp(z,ht,l):
   if not hasDimension(z,2):
      raise NotImplementedError("timeQuad not implemented for dimension other then 2. Input had shape {}.".format(z.shape))
   a1 = (z[:,0]**2+z[:,1]**2)/(ht*4)
   if l == 0:
      return ht/(4*pi) * (expn(1,a1)*(1+a1)-exp(-a1))
   if l == 1:
      return ht/(4*pi) * ((a1+2.)*expn(1,a1/2.) - 2.*(a1+1.)*expn(1,a1) - 2.*exp(-a1/2.) + 2.*exp(-a1))
   if l < 0:
      return 0
   return ht/(4*pi) * ((a1+l-1)*expn(1,a1/(l-1)) + (a1+l+1)*expn(1,a1/(l+1)) - 2*(a1+l)*expn(1,a1/l)- (l-1)*exp(-a1/(l-1)) - (l+1)*exp(-a1/(l+1)) + 2*l*exp(-a1/l) )

def gmmsing(z,x,y,endT, basis, **kwargs):
    if 'l' in kwargs:
       l = kwargs['l']
       ht =  endT*basis.baseT[l].support[0].length
       a = gmmsinghelp(z,x,y,ht,l)
       return a 
    else:
      m = kwargs['m']
      h1 = endT*basis.baseT[m].support[0].length
      istart1 = endT*basis.baseT[m].support[0].start
      n = kwargs['n']
      h2 = endT*basis.baseT[n].support[0].length
      istart2 = endT*basis.baseT[n].support[0].start
      a = int(max(h1,h2)/min(h1,h2))
      h = zeros([z.shape[0]])       
      for  i in range(a):
          if h2 > h1:
              l = int(istart1/h1) - (int(istart2/h1)+i)
          else:
              l = int(istart1/h2)+i - int(istart2/h2)
          if l >= 0:
              h += gmmsinghelp(z,x,y,min(h1,h2),l)
      return h

def gmmsinghelp(z,x,y,ht,l):
    a1 = (z[:,0]**2+z[:,1]**2)/(ht*4)
    if (l==0):
        return ht/(4*pi)*2*(1+a1)
    elif (l==1):
        return ht/(4*pi)*(-4*(1+a1)+2*(2+a1))
    else:
        return ht/(4*pi)*(2*(l-1+a1)+2*(l+1+a1) -4*(l+a1))

def gmmsmooth(z,x,y,endT, basis,flag, **kwargs):
    if 'l' in kwargs:
       l = kwargs['l']
       ht =  endT*basis.baseT[l].support[0].length
       return gmmsmoothhelp(z,x,y,ht,flag,l)
    else:
      m = kwargs['m']
      h1 = endT*basis.baseT[m].support[0].length
      istart1 = endT*basis.baseT[m].support[0].start
      n = kwargs['n']
      h2 = endT*basis.baseT[n].support[0].length
      istart2 = endT*basis.baseT[n].support[0].start
      a = int(max(h1,h2)/min(h1,h2))
      h = zeros([z.shape[0],1]) 
      for  i in range(a):
          if h2 > h1:
              l = int(istart1/h1) - (int(istart2/h1)+i)
          else:
              l = int(istart1/h2)+i - int(istart2/h2)
          if l >= 0:
              h += gmmsmoothhelp(z,x,y,min(h1,h2),flag,l)
      return h
def gmmsmoothhelp(z,x,y,ht,flag, l):
    a1 = ((z[:,0]**2+z[:,1]**2)/(ht*4)).reshape(-1,1)
    if l > 1:
        return ht/(4*pi)*(integrandsmooth(l-1,a1,x,y,ht,flag)+integrandsmooth(l+1,a1,x,y,ht,flag) -2*integrandsmooth(l,a1,x,y,ht,flag))
    elif l == 0:
        return ht/(4*pi)*integrandsmooth(1,a1,x,y,ht,flag)
    else: # Last case is l == 1:
        return ht/(4*pi)*(-2*integrandsmooth(1,a1,x,y,ht,flag)+integrandsmooth(2,a1,x,y,ht,flag))

from numpy import log
from math import factorial
def integrandsmooth(l,a1,x,y,ht,flag):
    al = a1/l
    p = 0
    for k in range(1,20):   #todo figure out optimal number for this range
        p = p + (-al)**k/(k*factorial(k));
    gam =  0.5772156649015328606   #The Euler-Mascheroni constant
    return ((l+a1)*(-gam -log(al/(x-y+flag)**2) - p) - l*exp(-al))




