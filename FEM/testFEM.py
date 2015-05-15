from assembleVector import assembleVector
from numpy.linalg import solve
from assembleMatrix import assembleStiffness, assembleMass
from L2proj import L2projection
from numpy import sin,pi,ones, array,zeros,cos,exp,arctan2
from mesh import meshCircle, meshSquare, meshSquareBounded, meshCircleBounded
from plot import plotSol, plotMesh
from matplotlib.pyplot import figure
from normalDerivative import normalDerivative
from matplotlib.pylab import figure
from basis import basis
from scipy.special import jn, jn_zeros

    #right hand side, in all these tests we use zero initial values        
f1 = lambda x,y,t: -2*t
g1 = lambda x,y,t: t**2
def bFlux1(x,y,t):
    alpha = jn_zeros(0,47)
    return t - 4*sum((1.-exp(-alpha**2*t))/(alpha**4))*ones(x.shape)
def f2(x,y,t):
        phi = arctan2(y,x)
        r = sqrt(x**2+y**2)
        return cos(phi)*3
def g2(x,y,t):
        phi = arctan2(y,x)
        r = sqrt(x**2+y**2)
        return r**2*cos(phi)
def bFlux2(x,y,t):
    return cos(arctan2(y,x))
def exsol3(x,y,t):
        from scipy.special import jn, jn_zeros
        alpha = jn_zeros(1,47)
        r = sqrt(x**2+y**2)
        h = arctan2(y,x)
        return (r*t**2 - 4*sum(jn(1,alpha*r)/(alpha**3*jn(2,alpha))*(t-(1.-exp(-alpha**2*t))/(alpha**2))))*cos(h)
def f3(x,y,t):
        phi = arctan2(y,x)
        r = sqrt(x**2+y**2)
        return -(2*t*r**2+t**2*(-3))*cos(phi)
def bFlux3(x,y,t):
        alpha = jn_zeros(1,47)
        h = 4*sum((1.-exp(-alpha**2*t))/(alpha**4))
        phi = arctan2(y,x)
        return (t**2 + t/4. - h)*cos(phi)
def g3(x,y,t):
        phi = arctan2(y,x)
        r = sqrt(x**2+y**2)
        return t**2*r**2*cos(phi)
def f4(x,y,t):
        return -x*t*2
def g4(x,y,t):
        return x*t**2



def solveSystem(g,m):
    B = assembleVector(g,m)
    #print B
    A = assembleStiffness(m)
    #print A
    x = solve(A,B)
    return (energynorm(x,B), x)

def energynorm(x,B):
    return sum(x*B)

def CrankNicolson(g,v, m, ht, endT):
    Nt  = int(float(endT)/ht)
    sol = zeros([m.n, Nt+1])
    sol[:,0] = L2projection(v,m)
    A   = assembleStiffness(m)
    M   = assembleMass(m)
    for k in range(Nt):
        B = assembleVector(lambda x,y: g(x,y,k*ht),m)
        sol[:,k+1] = solve(M + 0.5*ht*A,  M*sol[:,k].reshape(-1,1) - 0.5*ht*A*sol[:,k].reshape(-1,1) + ht*B.reshape(-1,1)).reshape(-1)
    return sol

def implicitEuler(g,v, m, ht, endT):
    Nt  = int(float(endT)/ht)
    sol = zeros([m.n, Nt+1])
    sol[:,0] = L2projection(v,m)
    A   = assembleStiffness(m)
    M   = assembleMass(m)
    for k in range(Nt):
        B = assembleVector(lambda x,y: g(x,y,k*ht),m)
        sol[:,k+1] = solve(M + ht*A,  M*sol[:,k].reshape(-1,1) + ht*B.reshape(-1,1)).reshape(-1)
    return sol

#gets a coefficient vector and the mesh, returns the L^2 norm
def l2norm(x,m):
    norm = 0
    # run over all inner nodes
    for t in m.triangles:
        f1 = t.f1
        f2 = t.f2
        for localNum, n in enumerate(t.nodes):
            globalNum = m.numbering.get(n)
            if globalNum is not None: # only for inner nodes, since Dirichlet boundary is assumed
                a = GJTria(lambda x, y: (x[globalNum] * basis(localNum, x, y))**2 * t.det, 20)
                norm += a
    return sqrt(norm)
    
def testHelmholtz():
    k = 8
    f = lambda x,y: 4+k*(0.25**2-x**2-y**2)
    m = meshCircle(4)
    M = assembleMass(m)
    A = assembleStiffness(m)
    B = assembleVector(f,m)
    x = solve(A+k*M, B)
    fig = plotSol(x,m)

def testCircle():
    def f(x,y):
        return 4.
        #rs = x**2+y**2
        #return (2*pi)**2*rs*cos(pi*rs)
    def u(x,y):
        rs = x**2+y**2
        return 0.25-rs
        #return cos(pi*rs)
    m = meshCircle(4,0.5)
    #plot the l2-projection of the solution
    u1 = L2projection(u, m)
    fig = plotSol(u1,m)
    #plotMesh(m,fig,zs=0, zdir='z').show()

    #solve laplace problem and plot solution
    norm, x = solveSystem(f,m)
    fig2 = plotSol(x,m)
    #plotMesh(m,fig2,zs=0, zdir='z').show()

   
from numpy import sqrt    
def testEuler():
    #the domain Omega is a circle of radius R
    R = 0.5
    m = meshCircle(4,R)
    
    #f gives the right hand side
    f1 = lambda x,y,t: 4
    f2 = lambda x,y,t: pi**2*cos(pi*sqrt(x**2+y**2))+pi/sqrt(x**2+y**2)*sin(pi*sqrt(x**2+y**2))
    def f3(x,y,t):
        r = sqrt(x**2+y**2)
        return 2*t*cos(pi*r)+t**2*(pi/r*sin(pi*r)+pi**2*cos(pi*r)) #only for R = 0.5
    #v gives the initial conditions
    v1 = lambda x,y:  R**2-x**2-y**2
    v2 = lambda x,y: cos(pi*sqrt(x**2+y**2))
    v3 = lambda x,y: 0
    
    #time interval and time step
    endT = 1
    ht = 1./16
    
    #run backward Euler to solve
    x = implicitEuler(f3,v3, m, ht, endT)
    #plot at some time steps to check the answers
    fig = plotSol(x[:,0],m)
    fig = plotSol(x[:,1],m)
    fig = plotSol(x[:,2],m)


def testSquare():
    #f = lambda x,y: 2*y*(1-y) + 2*x*(1-x)
    #u = lambda x,y: x*(1-x)*y*(1-y)
    
    f = lambda x,y: 2*pi**2*u(x,y)
    u = lambda x,y: sin(pi*x)*sin(pi*y)
    
    m = meshSquare(2,1)
    #plot the l2 projection of the solution
    u1 = L2projection(u,m)
    fig = plotSol(u1,m)
    plotMesh(m,fig,zs=0, zdir='z').show()
    norm, x = solveSystem(f,m)
    fig2  = plotSol(x,m)
    plotMesh(m,fig2,zs=0, zdir='z').show()
    
def testConv():
    #rhs for the square
    f = lambda x,y: 1
    steps = 5
    norm = zeros(steps)
    ndof = zeros(steps)
    error = zeros(steps)
    for i in range(1,steps+1):
        m = meshCircle(2**i,1)
        #plotMesh(m)
        norm[i-1], x = solveSystem(f,m)
        ndof[i-1] = m.n
    ex = norm[-1]
    error = sqrt(abs(norm[:-1]-norm[-1])/abs(ex))
    f = figure()
    f.gca().loglog(ndof[:-1], error, 'rx-')
    f.show()
    
    #fig = plotSol(x, m)
    #plotMesh(m,fig,zs=0, zdir='z').show()

def testNonZeroBoundary():
    def g(x,y):
        return x**2+y**2
    def laplace_g(x,y):
        return 4
    m = meshCircle(4,1)
    #solve for utilde
    norm,xtilde = solveSystem(laplace_g,m)
    #find u
    mfull = meshCircle(4,1,"bound")
    B = L2projection(g,mfull)
    x = B
    x[0:m.n] += xtilde
    fig = plotSol(x,mfull)
    

def testEulerNonZeroBoundary():
    #the domain Omega is a circle of radius R
    R = 1
    #set time step and interval length
    endT = 1.
    N = 6
    errors = []
    times = []
    ndof = []
    for it in  range(1,N):
        print it
        ht = 1./2**(2*it)
        m  = meshCircle(2**it,R)  #with zero boundary conditions
        mB = meshCircle(2**it,R,"bound") #without zero boundary conditions
        def EulerMeth():
            #run backward Euler on the modified problem
            xtilde = implicitEuler(f4,lambda x,y: 0, m,ht,endT)
            #calculate and plot the actual solution
            x = zeros([mB.n,int(endT/ht+1)])
            for k in range(int(endT/ht+1)):    
                G = L2projection(lambda x,y: g4(x,y,k*ht), mB)
                x[0:m.n,k] = xtilde[:,k]
                x[:,k]     = x[:,k] + G
            return x
        import time
        start = time.clock()
        x = EulerMeth()
        elapsed = time.clock() - start
        times.append(elapsed)
        ndof.append(x.shape)
        #print elapsed
        #fig = plotSol(x[:,endT/ht],mB)
        def f(coeff,x,y):
            h = 0
            number = mB.numbering
            result = None
            #find triangles in which x,y are
            #print m.findTriangles(x,y)
            for T in mB.findTriangles(x,y):
                result = 0
                i = number.get(T.nodes[0])
                j = number.get(T.nodes[1])
                k = number.get(T.nodes[2])
                xhat = T.f1inv(x,y)
                yhat = T.f2inv(x,y)
                from basis import basis
                if not i is None: result += coeff[i]*basis(0,xhat,yhat)
                if not j is None: result += coeff[j]*basis(1,xhat,yhat)
                if not k is None: result += coeff[k]*basis(2,xhat,yhat)
            #if abs(x) < 0.1 and abs(y) < 0.1 and result == 0:
            #        print x,y,result
            return result
        k = endT/ht
        k2 = endT/2./ht
        errors.append( [abs(f(x[:,k], 0.1,0)  - exsol3(0.1,0,endT))    / abs(exsol3(0.1,0,endT)),
                        abs(f(x[:,k], 0.5,0)  - exsol3(0.5,0,endT))    / abs(exsol3(0.5,0,endT)),
                        abs(f(x[:,k], 0.7,0)  - exsol3(0.7,0,endT))    / abs(exsol3(0.7,0,endT)),
                        abs(f(x[:,k2], 0.1,0) - exsol3(0.1,0,endT/2.)) / abs(exsol3(0.1,0,endT/2.)),
                        abs(f(x[:,k2], 0.5,0) - exsol3(0.5,0,endT/2.)) / abs(exsol3(0.5,0,endT/2.)),
                        abs(f(x[:,k2], 0.7,0) - exsol3(0.7,0,endT/2.)) / abs(exsol3(0.7,0,endT/2.))] )
    
    # plot times versus errors
    print "times", times
    print "error", errors
    print "nshape", ndof
    fig = figure()
    a = fig.gca()
    for e in zip(*errors): # transpose
        a.semilogy(times,e,"x-")
    a.legend(["tk = 1, x=(0,0)","tk = 1, x=(.5,0)","tk = 1, x = (0.7,0)","tk = .5, x=(0,0)","tk = .5, x=(.5,0)","tk = .5, x = (0.7,0)"])
    fig.show()
    

def testboundaryFlux():
    R = 1.
    #set time step and interval length
    endT = 1.
    N = 5
    bfluxH = lambda x,t: bFlux3(sin(2*pi*x-pi),cos(2*pi*x-pi),t)
    
    errL2  = zeros([N])
    errL22 = zeros([N])
    ndof = zeros([N])
    
    for i in range(N):
        ht = 1./(2**(2*i+2))
        print i+1
    
        m  = meshCircle(2**(i+1),R)  #with zero boundary conditions
        mB = meshCircle(2**(i+1),R,"bound") #without zero boundary conditions
        #plotMesh(mB)
        
        #run backward Euler on the modified problem
        xtilde = CrankNicolson(f4,lambda x,y: 0, m,ht,endT)
        #calculate and plot the actual solution
        x = zeros([mB.n,int(endT/ht+1)])
        for k in range(int(endT/ht+1)):    
            G = L2projection(lambda x,y: g4(x,y,k*ht), mB)
            x[0:m.n,k] = xtilde[:,k]
            x[:,k]     = x[:,k] + G
        from BEM.integrate1d import gauleg
        xquad,wquad = gauleg(100)
        
        l = int(.5/ht)
        delu = normalDerivative(x[:,l],basis,mB)
        N = len(m.boundaryNodes)
        errL2[i] = 0
        for i_n in range(N):
            xquadh = float(i_n)/N + xquad/N
            wquadh = wquad/N
            y = array(map(delu,sin(2*pi*xquadh-pi),cos(2*pi*xquadh-pi)))
            errL2[i] += abs(sum(y**2*wquadh))
        ndof[i]  = endT/ht*mB.n

        l = int(1./ht)
        delu = normalDerivative(x[:,l],basis,mB)
        errL22[i] = 0
        for i_n in range(N):
            xquadh = float(i_n)/N + xquad/N
            wquadh = wquad/N
            y = array(map(delu,sin(2*pi*xquadh-pi),cos(2*pi*xquadh-pi)))
            errL22[i] += abs(sum(y**2*wquadh))

        #Plotting
        from numpy import linspace
        xn = linspace(0,1,200)
        l = int(1./ht)
        delu = normalDerivative(x[:,l],basis,mB)
        y = map(delu,sin(2*pi*xn-pi),cos(2*pi*xn-pi))
        f = figure()
        #print xn.shape, bfluxH(xn,l*ht).shape
        #f.gca().plot(xn,bfluxH(xn,l*ht),'r-')
        #f.gca().plot(xn,y,'b-')
        #f.show()
        print errL2[i], errL22[i]
    print "errL2", errL2,errL22
    print ndof
    #fig = figure()
    #fig.gca().loglog(ndof,errL2,'.-')  
    #fig.gca().loglog(ndof,errL22,'.-')    
    #fig.show()

    
if __name__=='__main__':
    #testCircle()
    #testSquare()
    #testConv()
    #testNonZeroBoundary()
    #testEulerNonZeroBoundary()
    #testHelmholtz()
    testboundaryFlux()
    
