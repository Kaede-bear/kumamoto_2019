import numpy as np
from numpy.fft import fftn
import sporco.cnvrep as cr
import sporco.linalg as sl

def convert_to_Df(D, S, cri):
    Dr = np.asarray(D.reshape(cri.shpD), dtype=S.dtype)
    Df = sl.rfftn(Dr, cri.Nv, cri.axisN)
    return Df

def convert_to_D(Df, dsz, cri):
    D = sl.irfftn(Df, cri.Nv, cri.axisN)
    D = cr.bcrop(D, dsz, cri.dimN).squeeze()
    return D

def convert_to_Sf(S, cri):
    Sr = np.asarray(S.reshape(cri.shpS), dtype=S.dtype)
    Sf = sl.rfftn(Sr, None, cri.axisN)
    return Sf

def convert_to_S(Sf, cri):
    S = sl.irfftn(Sf, cri.Nv, cri.axisN).squeeze()
    return S

def convert_to_Xf(X, S, cri):
    Xr = np.asarray(X.reshape(cri.shpX), dtype=S.dtype)
    Xf = sl.rfftn(Xr, cri.Nv, cri.axisN)
    return Xf

def convert_to_X(Xf, cri):
    X = sl.irfftn(Xf, cri.Nv, cri.axisN).squeeze()
    return X

