from solve import solveMem
from basis import findIndices, Multiscale2Const2d, Const_basis, Const_multiscale, Sparse_multiscale, SIndices
from numpy import zeros
from assembleMatrix import assembleSingleLayer
from assembleVector import assembleVector
from numpy.linalg import solve


def sparseCombi(i,sigma,space_proj,time_proj,g):        
    bs = Sparse_multiscale(i+1,sigma)      
    #Combination technique 
    solSparse = zeros(bs.nx)
    Bsparse   = zeros(bs.nx)
    #Set up Matrix and vector of the right hand side
    for l in range(2):
            for (it,ix) in SIndices(i+1,l,sigma):
                basisM = Const_multiscale(it,ix)
                basis  = Const_basis(2**it,2**ix)
                A  =  assembleSingleLayer(basis, space_proj, time_proj)
                B  = assembleVector(g, basis, space_proj, time_proj)
                B2 = assembleVector(g,basisM,space_proj,time_proj)
                #u_it,ix
                sol = solveMem(A,B,basis.nx,basis.nt)
                #sol = solve(A,B)
                baseTrafoA = Multiscale2Const2d(Const_multiscale(it,ix),basis)
                I = findIndices(bs,it,ix)
                from numpy import matrix
                helpsol = solve(baseTrafoA, sol)
                #helpB   = solve(baseTrafoA, B)
                solSparse[I] += (-1)**l*helpsol
                Bsparse[I]   += (-1)**l*B2
    return solSparse, Bsparse


def fullTens(i, space_proj, time_proj, g):
        bc = Const_basis(2**(i+1),2**(2*(i+1)))
        Ac =  assembleSingleLayer(bc, space_proj, time_proj)
        Bc = assembleVector(g, bc, space_proj, time_proj) 
        solc = solveMem(Ac,Bc,bc.nx,bc.nt)
        return solc, Bc
    
def sparseN(i, sigma, space_proj, time_proj, g):
        bs = Sparse_multiscale(i+1,sigma)
        As =  assembleSingleLayer(bs, space_proj, time_proj)
        Bs = assembleVector(g, bs, space_proj, time_proj) 
        sols = solve(As,Bs)
        return sols, Bs
        







