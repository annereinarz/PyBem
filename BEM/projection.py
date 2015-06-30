from numpy import ones, hstack, linspace, pi, cos, sin, hstack, arctan2,sqrt
from util import norm_


class domain (object):
    pass


def unique(seq, idfun=None):
   "Return the unique items of seq in the order they occured in first."
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

class interval (domain):
    "An interval from start to end."

    dim = 1

    def __init__(self, start, end):
        assert start <= end, "Interval must have its start before its end. "
        self.start = start
        self.end = end
        self.length = end - start

    def __call__(self, theta):
        return self.start + self.length * theta

    def derivative(self, theta):
        return self.length*ones(theta.shape)

    def inverse(self, t):
        return (t - self.start) / float(self.length)

    def __repr__(self):
        return "interval({},{})".format(self.start, self.end)

    def __eq__(self, other):
        if not isinstance(other, interval):
            return False
        return ( abs(self.start - other.start) < 0.01 
                               and
                 abs(self.end   - other.end)   < 0.01 )

    def __ne__(self, other):
        return not (self == other)

    def contains(self, point):
        return self.start <= point and point <= self.end

    def intersect(self, other):
        if self.start <= other.start and other.end <= self.end:
            return other
        if other.start <= self.start and self.end <= other.end:
            return self
        if self.start <= other.start <= self.end:
            return interval(other.start, self.end)
        if self.start <= other.end <= self.end:
            return  interval(self.start, other.end)
        
    def shiftedBy(self, delta):
        return interval(self.start + delta, self.end + delta)

    def splitAt(self, points):
        "Split the interval at points. The list of points must be sorted."
        #Only split at points that lie inside the interval
        points = filter(lambda p: self.start < p < self.end, points)

        ab = unique([self.start] + points + [self.end])
        return [ interval(a,b) for a,b in zip(ab[:-1], ab[1:]) ]
    
    def endpoints(self):
        return [ self.start, self.end ]

# splits the intervals i1,i2 seperately such that the resulting smaller
# intervals have the same sizes and do not overlap
# intervals are assumed to have lengths in powers of 2
def splitIntervals(i1, i2):
    l1 = i1.splitAt(i2.endpoints())
    l2 = i2.splitAt(i1.endpoints())
    list  = l1 + l2
    s = min(l.length for l in list)  #size of the small intervals
    n1 = int(i1.length/s)
    n2 = int(i2.length/s)
    assert i1.length %s == 0 and i2.length %s == 0
    points1 = [i1.start + k*s for k in range(1,n1)]
    points2 = [i2.start + k*s for k in range(1,n2)]
    l1 = i1.splitAt(points1)
    l2 = i2.splitAt(points2)
    return l1, l2

def angle(x):
    x = x.reshape(-1,2)
    return arctan2(x[:,1],x[:,0])

def polar(x):
    return (norm_(x), angle(x))

class circle (domain):
    "A circle."
    dim = 1

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, theta):
        z = pi*(2*theta-1)
        assert z.shape[1] == 1
        return self.radius * hstack([cos(z), sin(z)])

    def derivative(self, theta):
        return 2*pi*self.radius * ones(theta.shape)

    def inverse(self, x):
        return angle(x) / (2*pi) + .5

    def contains(self, x):
        return norm_(x) < self.radius-0.2

    def normal(self, z):
        return z / self.radius

    def __repr__(self):
        return "circle({})".format(self.radius)

class ellipse (domain):
    "An ellipse, where a, b are the lengths of the axes."

    dim = 1

    def __init__(self, a,b):
        self.a = a
        self.b = b

    def __call__(self, theta):
        #the angle between the x-axis and the major axis of the ellipse is zero
        z = 2*pi*theta
        assert z.shape[1] == 1
        return hstack( [self.a*cos(z),
                        self.b*sin(z)] )

    def derivative(self, theta):
        return 2*pi*sqrt((self.a*sin(2*pi*theta))**2+(self.b*cos(2*pi*theta))**2)

    def inverse(self, x):
        return arctan2(self.a*x[:,1],self.b*x[:,0]) / (2*pi) 

    def contains(self, x):
        x = x.reshape(-1,2)
        return sqrt((1./self.a*x[:,0])**2+(1./self.b*x[:,1])**2) < 1 - 0.05

    def normal(self, z):
        n = [self.b/self.a*z[:,0], self.a/self.b*z[:,1]]
        return n/norm_(n)

    def __repr__(self):
        return "ellipse({},{})".format(self.a, self.b)

class outside_ellipse (domain):
    "The outside of an ellipse, where a, b are the lengths of the axes."

    dim = 1

    def __init__(self, a,b):
        self.a = a
        self.b = b

    def __call__(self, theta):
        #the angle between the x-axis and the major axis of the ellipse is zero
        z = 2*pi*theta
        assert z.shape[1] == 1
        return hstack( [self.a*cos(z),
                        self.b*sin(z)] )

    def derivative(self, theta):
        return 2*pi*sqrt((self.a*sin(2*pi*theta))**2+(self.b*cos(2*pi*theta))**2)

    def inverse(self, x):
        return arctan2(self.a*x[:,1],self.b*x[:,0]) / (2*pi) 

    def contains(self, x):
        x = x.reshape(-1,2)
        return sqrt((1./self.a*x[:,0])**2+(1./self.b*x[:,1])**2) > 1 + 0.05

    def normal(self, z):  #outer normal points into the ellipse
        n = [self.b/self.a*z[:,0], self.a/self.b*z[:,1]]
        return -n/norm_(n)

    def __repr__(self):
        return "out-ellipse({},{})".format(self.a, self.b)


class torus (domain):
    "A torus in 3d of inner radius R and outer radius r."
    dim = 2

    def __init__(self, r, R):
        self.r = r
        self.R = R

    def __call__(self, theta):
        z1 = pi*(2*theta-1)
        z2 = pi*(2*phi-1)
        assert z1.shape[1] == 1 and z2.shape[1] == 1
        return [(self.R+self.r*cos(z1))*cos(z2), (self.R+self.r*cos(z1))*sin(z2),self.r*sin(z1)]

    def __repr__(self):
        return "torus{})".format(self.R, self.r)

class starshaped (domain):
    "Abstract base class for star-shaped domains."

    def contains(self, x):
        if not hasattr(self, 'somePointsOnTheBoundary'):
           print "Initializing contains-check"
           self.somePointsOnTheBoundary = [ self(theta.reshape(1,1)) for theta in linspace(0.0,1.0,500) ]

        rx,ax = polar(x) 

        minSoFar = float('inf')
        for r,a in map(polar, self.somePointsOnTheBoundary):
            m = abs(a-ax)
            if m < minSoFar:
                minSoFar = m
                matchingRadius = r-0.05

        #print "r=",matchingRadius
        return rx < matchingRadius 


class wibblywobbly (starshaped):
    "A certain star-shaped domain."

    dim = 1

    def __call__(self, theta):
        z = 2*pi*theta
        assert z.shape[1] == 1
        a = 4+cos(3*z)+cos(z)
        return 1./7.*hstack( [a*cos(z),
                              a*sin(z)] )

    def derivative(self, theta):
        z = 2*pi*theta
        return 2.*pi/7.*sqrt((4+cos(3*z)+cos(z))**2+(3*sin(3*z)+sin(z))**2)

    def inverse(self, x):
        return arctan2(x[:,1],x[:,0]) / (2*pi) 

    def normal(self, z):
        raise NotImplementedError()

    def __repr__(self):
        return "wibblywobbly()"



def plot(domain, n=50):
    from numpy import array, empty, linspace
    from matplotlib.pyplot import figure 

    def u(x):
       return domain.contains(x)

    s = linspace(-1.0, 1.0, n)
    A = empty([n,n])

    for i in range(n):
       for j in range(n):
          x = array([ (s[i],s[j]) ])
          A[i,j] = u(x)

    f = figure()
    i = f.gca().imshow(A, interpolation='nearest')
    f.show()

if __name__ == '__main__':
    pass
    #plot(circle(0.8))
    #plot(ellipse(0.8,0.4))
    #plot(wibblywobbly())
    #domain = type('starcircle', (starshaped, circle), {})(0.8)
    #plot(ellipse(0.8,0.4))
    #domain = type('starellipse', (starshaped, ellipse), {})(0.8,0.4)
    #plot(domain)
    #plot(wibblywobbly())
