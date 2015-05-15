from matvec import mat_vec, liftToVector
from assembleVector import assembleVector
from assembleMatrix import assembleDoubleLayer
from L2proj import L2proj


def assembleKG_debug(g, B, basis, space_proj, time_proj):
        #Inefficient but easily debuggable version
        def K1(f):
            return liftToVector(lambda x, t: integrate(lambda y, s: f(y, s)
                                                       * nfundamentalSol(space_proj.normal(y), x - y, t - s),
                                                       space_proj, time_proj, n = 30, t=time_proj.inverse(t), x=space_proj.inverse(x)))
        KG = assembleVector(K1(g), basis, space_proj, time_proj).reshape(-1, 1)
        return KG

def assembleKG_efficient(g, B, basis, space_proj, time_proj):        
        K = assembleDoubleLayer(basis, space_proj, time_proj)
        G_approx = L2proj(B, basis, space_proj, time_proj)
        KG = mat_vec(K, G_approx) #matrix-vector multiplication
        return KG