from assembleMass import assembleMass
from assembleVector import assembleVector
from numpy.linalg import solve


# Calculate the L2 projection of a function g onto the space spanned by the basis bas
def L2proj(g,base,px,pt):
    B = assembleVector(g, base, px,pt)
    M = assembleMass(base, px, pt)   #assemble mass matrix
    return solve(M,B).reshape(-1)    



if __name__ == '__main__':
	g = lambda x,t: x[:,1]+x[:,0]
	from projection import circle, interval
	px = circle(1.)
	pt = interval(0,1)
	from basis import Const_basis
	b = Const_basis(3,3)
	print L2proj(g,b,px,pt)
