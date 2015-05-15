import os
result_directory = "results/indirect"
if not os.path.exists(result_directory):
    os.makedirs(result_directory)


def output(basis, gname, space_proj, time_proj, N, ndof, norm):
    fname = result_directory + "/{} {} {} {} N={}".format(basis, gname, space_proj, time_proj, N)
    with open(fname,"w") as f:
        f.write("ndof  norm\n")
        for a,b in zip(ndof, norm):
            f.write("{} {}\n".format(a,b))
            
def outputSol(basis, gname, space_proj, time_proj, i, sol,B):
    fname = result_directory + "/{} {} {} {} i={}".format(basis, gname, space_proj, time_proj, i)
    with open(fname,"w") as f:
        f.write("{}\n".format(sol))
        f.write("{}\n".format(B))