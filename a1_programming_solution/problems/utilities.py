import numpy as np

from types import SimpleNamespace



def split_data(x,y,tst_ratio=0.1,seed=0):
    from types import SimpleNamespace
    n = len(x)
    n_tst = int(np.ceil(tst_ratio*n))
    rs = np.random.RandomState(seed)
    p = rs.permutation(n)
    return SimpleNamespace(x_tst=x[p[:n_tst]],y_tst=y[p[:n_tst]],
                            x_trn=x[p[n_tst:]],y_trn=y[p[n_tst:]])



def split_data_around_point(x,y,tst_ratio=0.1,x_0=None):
    from types import SimpleNamespace
    n = len(x)
    n_tst = int(np.ceil(tst_ratio*n))
    if x_0 is None:
        x_0 = (np.min(x)+np.max(x))/2
    p = np.argsort(np.abs(x-x_0))
    #return (x[p[:n_tst]],y[p[:n_tst]]), (x[p[n_tst:]],y[p[n_tst:]])
    return SimpleNamespace(x_tst=x[p[:n_tst]],y_tst=y[p[:n_tst]],
                            x_trn=x[p[n_tst:]],y_trn=y[p[n_tst:]])
