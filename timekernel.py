from scipy.special import expn
from numpy import prod, pi, exp, array, zeros
from util import hasDimension, concat


#analytically calculated time integration for the single-layer operator
def timeQuad(z, endT, basis, **kwargs):
   assert 'l' in kwargs
   l = kwargs['l']
   ht = endT*basis.baseT[l].support[0].length
   return timeQuadhelp(z,ht,l)
 

#analytically calculated time integration for the double-layer operator
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






