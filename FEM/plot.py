import numpy as np
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from basis import basis

def plotRoutine(fn):
    """
    Wrap this around each plot routine. It makes passing the figure optional (creating a new one)
    and always returns the used figure.
    """
    def wrappedFn(*args,**kwargs):
        fig = None
        args = list(args)
        if "fig" in kwargs:
            fig = kwargs.get("fig")
            del kwargs["fig"]
        import inspect
        params = inspect.getargspec(fn)
        if not fig and len(args) >= len(params.args)-len(params.defaults or []):
            fig = args.pop()
        if not fig:
            from matplotlib.pyplot import figure
            fig = figure()
        #print fig, map(type, args), kwargs.keys()
        fn(*args, fig=fig, **kwargs) 
        return fig
    return wrappedFn
        
@plotRoutine
def plotTriangles(list, fig, marker='bx-', **kwargs):
    for i,t in enumerate(list):  #run over all triangles and plot them, with their index
        fig.gca().plot( [n.x for n in t.nodes]+[t.nodes[0].x], [n.y for n in t.nodes]+[t.nodes[0].y], marker,**kwargs )
        if 'zs' not in kwargs:
            fig.gca().text( t.center_x, t.center_y, str(i), color=marker[0], **kwargs )
        
@plotRoutine
def plotNodes(lists, fig, markers = ['co','bs','rx'], **kwargs):
    if type(lists) is set:
        lists =  [ lists ]
    cnt = 0
    for l in lists:
        for node in l:  #run over all nodes in list
            print node
            fig.gca().plot( node.x, node.y, markers[cnt % len(markers)], **kwargs )
        cnt = cnt + 1

@plotRoutine
def plotSol(coeff,m,fig):
    def f(x,y):
        h = 0
        number = m.numbering
        result = None
        #find triangles in which x,y are
        #print m.findTriangles(x,y)
        for T in m.findTriangles(x,y):
            result = 0
            i = number.get(T.nodes[0])
            j = number.get(T.nodes[1])
            k = number.get(T.nodes[2])
            xhat = T.f1inv(x,y)
            yhat = T.f2inv(x,y)
            if not i is None: result += coeff[i]*basis(0,xhat,yhat)
            if not j is None: result += coeff[j]*basis(1,xhat,yhat)
            if not k is None: result += coeff[k]*basis(2,xhat,yhat)
        #if abs(x) < 0.1 and abs(y) < 0.1 and result == 0:
        #        print x,y,result
        return result
    #def f1(x,y):
    #    return (x**2+y**2-0.25**2)/-4        
    ax = fig.gca(projection='3d')
    n = 50
    X,Y = np.meshgrid(np.linspace(m.l,m.r,n), np.linspace(m.l,m.r,n))
    #Z = map(f1,X.reshape(-1),Y.reshape(-1))
    Z2 = map(f,X.reshape(-1),Y.reshape(-1))
    #Zex = map(uex,X.reshape(-1),Y.reshape(-1))
    #Zex = np.array(Zex, dtype=float).reshape(X.ashape)
    #Z = np.array(Z, dtype=float).reshape(X.shape)
    Z2 = np.array(Z2, dtype=float).reshape(X.shape)
    from matplotlib import cm
    stride = 1
    #print "f(0,0) = ", f(0,0)
    #ax.plot_surface(X,Y,Z,rstride=stride,cstride=stride)
    Z2min = np.nanmin(Z2)
    Z2 = np.nan_to_num(Z2)
    ax.plot_surface(X,Y,Z2,rstride=stride,cstride=stride,vmin=Z2min, cmap = cm.coolwarm)
    #ax.plot_wireframe(X,Y,Zex)

@plotRoutine
def plotMesh(m,fig,**kwargs):
    plotTriangles(m.triangles, fig, marker="go-", **kwargs)
    #plotNodes(m.innerNodes, fig, markers=['ro'])
    if 'zs' not in kwargs:
        for node, num in m.numbering.items(): 
            fig.gca().plot( node.x, node.y, 'ro', **kwargs )
            fig.gca().text( node.x, node.y, str(num), color='r' )