#! /bin/python

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


class Grid2D:
    def __init__(self, domains, k=200):
        self.k = k
        self.x1,self.x2  = np.meshgrid(*[np.linspace(*domain,k) for domain in domains])
    @property
    def design_matrix(self): return np.c_[self.x1.ravel(),self.x2.ravel()]
    @property
    def grid_shape(self): return self.x1.shape
    def apply_design_matrix(self,fn):
        y = fn(self.design_matrix)
        return y.reshape(self.grid_shape)
    

def plot_class_pdf(domains, data, k=100,bw_method=None,**kwargs):
    '''Plots a kernel density estimate of the given data
    @param domains: a pair of pairs, each of which describes the limits of the two axes.
        example: [[-1,0],[-1,1]]
    @param k: number of samples per axis
    @param bw_method: controls the KDE bandwidth.
    @param kwargs: additional arguments to the contour function.
    '''
    from matplotlib import cm
    grid = Grid2D(domains, k=k)
    y_tok = np.unique(data.y)
    num_cls = len(y_tok)
    ax = plt.gca()
    cmap = cm.get_cmap('jet')(np.linspace(0,1,num_cls))
    for i,y in enumerate(y_tok):
        kde = sp.stats.gaussian_kde(data.X[data.y==y,:].T,bw_method=bw_method)
        f = grid.apply_design_matrix(lambda x:kde(x.T))
        hnd_c = ax.contour(grid.x1,grid.x2,f,linestyles='-', colors=[cmap[i,:]],**kwargs)
        ax.clabel(hnd_c, inline=1, fontsize=10)

def show_boundary_2D(domains, fn_predict, k=100, levels=[0.5], featmap=lambda x:x, **kwargs):
    '''Shows the decision boundary of a model.
    @param domains: a pair of pairs, each of which describes the limits of the two axes.
        example: [[-1,0],[-1,1]]
    @param fn: a classifier with signature fn_predict(Z) where
            Z = featmap(X) is a pre-processed vector of 2-dimensional features,
            i.e., X has a shape of n by 2
    @param k: number of samples per axis
    @param kwargs: additional arguments to the contour function.
    '''
    x1,x2  = np.meshgrid(*[np.linspace(*domain,k) for domain in domains])
    X = np.c_[x1.ravel(),x2.ravel()]
    y = fn_predict(featmap(X)).reshape(x1.shape)
    plt.contour(x1,x2,y,levels=levels,**kwargs)


def randn_sigma(n,mu,S,rs=np.random,ortho=True,xmap=lambda x:x):
    L = np.linalg.cholesky(S)*np.sqrt(n)
    k = S.shape[0]
    X = xmap(rs.randn(n,k))
    if ortho:
        X = np.linalg.qr(X)[0]
    else:
        X = X
    return X@L.T + np.r_[mu][None,:]

def split_data(x,y,tst_ratio=0.1,seed=0):
    from types import SimpleNamespace
    n = len(x)
    n_tst = int(np.ceil(tst_ratio*n))
    rs = np.random.RandomState(seed)
    p = rs.permutation(n)
    return SimpleNamespace(x_tst=x[p[:n_tst]],y_tst=y[p[:n_tst]],
                            x_trn=x[p[n_tst:]],y_trn=y[p[n_tst:]])


def load_data(name,p):
    from types import SimpleNamespace
    Xy = np.loadtxt(f'data/{name}.csv', delimiter=",")
    return SimpleNamespace(X=Xy[:,0:p],y=Xy[:,p])

def load_data(name,p):
    from types import SimpleNamespace
    Xy = np.loadtxt(f'data/{name}.csv', delimiter=",")
    return SimpleNamespace(X=Xy[:,0:p],y=Xy[:,p])


class _Symtoep:
    '''Convenience trick to generate synmmetric Toeplitz matrices
    (to serve as covaraiance matrices)
    Usage:
    > symtoep[[1,2,3],[4,5]]
    array([[1, 4, 0],
           [4, 2, 5],
           [0, 5, 3]])
    '''
    def __getitem__(self, diags):
        if not isinstance(diags, tuple):
            diags=diags,
        from functools import reduce
        A = reduce(np.add, (np.diag(np.r_[diag],i) for i,diag in enumerate(diags[1:],1)),0)
        offdiag = 0 if np.isscalar(A) else A + A.T
        return np.diag(diags[0]) + offdiag
symtoep = _Symtoep()

def safeshape(X,idx=slice(None,None)):
    try:
        return f'{X.shape[idx]}'
    except:
        return '??'