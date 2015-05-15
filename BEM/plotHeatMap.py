from numpy import array, empty, linspace, arctan2, arange, zeros, ones, hstack
from matplotlib.pyplot import figure



def plotSpace(u, n):
   s = linspace(-1.0, 1.0, n)
   A = empty([n,n])

   for i in range(n):
      for j in range(n):
         x = array([ s[i],s[j] ])
         A[i,j] = u(x)
         #print "finished step: ",i,j
   f = figure()
   i = f.gca().imshow(A, interpolation='nearest')#,vmin=3,vmax=5)
   f.colorbar(i)
   #f.show()
   return f

def plotLinftyBF(bflux, exSol, time_proj):
    t = (arange(0,1,0.01)).reshape(-1,1)
    x = array([1.,0.]).reshape(2,1)
    f = figure()
    yex = array(map(lambda h: exSol(x,h),time_proj(t))).reshape(-1,1)
    yh  = array(map(lambda h: bflux(x,h),time_proj(t))).reshape(-1,1)
    y = abs(yh - yex)
    f.gca().plot(time_proj(t),y , 'b-')
    f.gca().set_title('pointwise error of the boundary flux')
    f.gca().set_xlabel('t')
    f.gca().set_ylabel('error')
    f.show()
    f = figure()    
    f.gca().plot(time_proj(t),yex , 'b-')
    f.gca().plot(time_proj(t),yh , 'r-')
    f.gca().set_title('boundary flux')
    f.gca().set_xlabel('t')
    f.gca().set_ylabel('boundary flux')
    f.show()    

    return max(array(y))

def calcPointvalues(bflux, exSol, space_proj):
    x = arange(0,1,0.05).reshape(-1,1)
    xh = space_proj(x)
    t = array(1.).reshape(1,1)
    y1 = map(lambda h: exSol(h,t), xh)
    y1 = array(y1).reshape(-1,1)
    y2 = map(lambda h: bflux(array(h),t), xh)
    y2 = array(y2).reshape(-1,1)
    y = abs(y1 - y2)
    return [x,y,y1,y2]
    
def plotLinftyBFspace(bflux,exSol,space_proj):
    x,y,y1,y2 = calcPointvalues(bflux,exSol,space_proj)
    f = figure()
    f.gca().plot(x,y , 'b-')
    f.gca().set_title('pointwise error of the boundary flux')
    f.gca().set_xlabel('x')
    f.gca().set_ylabel('error')
    f.show()
    
    f = figure()    
    f.gca().plot(x,y1 , 'b-')
    f.gca().plot(x,y2 , 'r-')
    f.gca().set_title('boundary flux')
    f.gca().set_xlabel('x')
    f.gca().set_ylabel('boundary flux')
    f.show()    


def plotLinfty(u, exSol, time_proj):
    #plot time at a constant space step
    point = [0.,0.]
    f = figure()
    ax = f.gca()
    endT = time_proj.length
    t = arange(0.02,endT,0.05)
    #print t

    #print 'u in plot', u(array([0.,0.]),1.)
    y = map(lambda h: u(array(point),h), t)
    yex = map(lambda h: exSol(array(point),h),t)
    yex = array(yex).reshape(-1)
    yerr = abs(yex - array(y))
    ax.plot(t,yerr,'b-')
    ax.set_title('pointwise error at {}'.format(point))
    ax.set_xlabel('time')
    ax.set_ylabel('error')
    f.show()
    from numpy import max
    return max(yerr)

def plotLinftySpace(u, exSol, space_proj):
    #plot time at a constant space step
    f = figure()
    ax = f.gca()
    x = arange(-0.90,0.95,0.05).reshape(-1,1)
    xh = hstack([x, zeros(x.shape)])
    y = map(lambda h: u(h, 1.), xh)
    #print array(y).shape
    yex = map(lambda h: exSol(h, 1.),xh)
    #print array(yex).shape
    yerr = abs(array(yex) - array(y))
    ax.plot(x,yerr,'b-')
    ax.set_title('pointwise error at t = 1.')
    ax.set_xlabel('x')
    ax.set_ylabel('error')
    f.show()
    from numpy import max
    return max(yerr)

from sequenceConv import seqconv
from numpy import hstack,exp
def plotConv(norms, ndofs, legs):
    f = figure()
    for norm, ndof, leg in zip(norms,ndofs, legs):
        N = norm.size
        #print norm.shape, ndof.shape
        error =  abs((norm[N-1]*ones(norm[0:N-1].shape)-norm[0:N-1])/norm[N-1])
        print error
        #b,c,gamma = seqconv('e', error, ndof[:-1])
        f.gca().loglog(ndof[0:-1], error, 'x-', label = leg)
        #from numpy import exp
        #f.gca().loglog(ndof, hstack([error,c[-1]*exp(-b[-1]*ndof[-1]**(1./gamma[-1]))]), 'x-', label = leg)
        #print hstack([error,c[-1]*exp(-b[-1]*ndof[-1]**(1./gamma[-1]))])
    f.gca().set_title('convergence')
    f.gca().set_xlabel('ndof')
    f.gca().set_ylabel('error')
    f.gca().legend(loc='lower left')
    f.show()
    f.savefig('results/conv.eps')

def plotTimeConv(norms, times, ndofs, legs):
    f = figure()
    for norm, time, ndof, leg in zip(norms,times,ndofs,legs):
        N = norm.size
        assert norm.size==time.size
        error =  abs((norm[N-1]*ones(norm[0:N-1].shape)-norm[0:N-1])/norm[N-1])
        print "error", error
        #print error, ndof
        b,c,gamma = seqconv('e', error, ndof[:-1])
        from numpy import exp
        print "error", error, c[-1]*exp(-b[-1]*ndof[-1]**(1./gamma[-1]))
        #f.gca().loglog(time[0:-1], error, 'x-', label = leg)
        f.gca().loglog(time, hstack([error,c[-1]*exp(-b[-1]*ndof[-1]**(1./gamma[-1]))]), 'x-', label = leg)
    f.gca().set_title('time taken vs. error')
    f.gca().set_xlabel('time taken in s')
    f.gca().set_ylabel('error')
    f.gca().legend(loc='lower left')
    f.show()
    f.savefig('results/time.eps')

def plotTime(u,exSol,endT):
    #plot time at a constant space step
    f = figure()
    ax = f.gca()
    t = arange(0.02,endT,0.01)
    y = map(lambda h: u(array([0.2,0.4]),h), t)
    ax.plot(t,array(y),'b-')
    yex = map(lambda h: exSol(array([0.2,0.4]),h),t)
    yex = array(yex).reshape(-1)
    ax.plot(t,array(yex),'k--')
    f.show()

def plotRadial(u,exSolt2, n):
    "Plot radial cut of u in space, showing n time-steps."
    f = figure()
    ax = f.gca()
    x = arange(0.,1.,0.01)
    for t in range(1,n+1):
        y = map(lambda x: u(array([x,0]),t), x)
        yex = map(lambda x: exSolt2(array([x,0.]),t), x)
        ax.plot(x, y, label="t={}".format(t))
        ax.plot(x, yex, 'k--', label="exact Sol, t={}".format(t))
    ax.set_xlabel('r')
    ax.set_ylabel('u(r,t)')
    ax.set_title('radial cut of the solution at different time steps')
    ax.legend(loc='upper left')
    f.show()

if __name__ == '__main__':

   from numpy.linalg import norm
   from numpy import cos, pi, sqrt, exp, sum
   from scipy.special import jn, jn_zeros
   t=1
   def u(x):
      if norm(x) < 0.98:
         r =  sqrt(x[0]**2+x[1]**2)
         alpha = jn_zeros(0,47)
         return t**2 - 4*sum(jn(0,alpha*r)/(alpha**3*jn(1,alpha))*(t-(1.-exp(-alpha**2*t))/(alpha**2)))
   plotSpace(u,20)
