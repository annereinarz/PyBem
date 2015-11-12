from numpy.linalg import solve
from numpy import dot, zeros

def solveMem(A,B,Nx,Nt):
    B = B.reshape(-1)
    x = zeros(B.shape)
    for i in range(Nt):
        sumAx = zeros([Nx])
        for k in range(i):
            sumAx += dot(A[(i-k)*Nx:(i-k+1)*Nx,:],x[k*Nx:(k+1)*Nx])
        x[i*Nx:(i+1)*Nx] = solve(A[0:Nx,:], B[i*Nx:(i+1)*Nx]-sumAx)
    return x

def createFullMatrix(A, Nx,Nt):
    Afull = zeros([Nx*Nt,Nx*Nt])
    for m in range(Nt): #run over all diagonals
        for n in range(Nt-m): #number of entries on diagonal
            Afull[(n+m)*Nx:(n+m+1)*Nx, n*Nx:(n+1)*Nx] = A[m*Nx:(m+1)*Nx,:]
    return Afull