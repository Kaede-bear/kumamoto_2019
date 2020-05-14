import numpy as np
import Convert as con

import sporco.cnvrep as cr
import sporco.linalg as sl
import sporco.prox as sp
from tqdm import tqdm

#係数の初期値設定
def the_minimum_l2_norm_solution(D, S, cri):
    Df = con.convert_to_Df(D, S, cri)
    Sf = con.convert_to_Sf(S, cri)
    Xf = np.conj(Df) / sl.inner(Df, np.conj(Df)) * Sf
    X = con.convert_to_X(Xf, cri)
    return X  

#パラメータsigmaの設定
def setting_sigma(X):
    sigma = []
    sigma.append(4*np.amax(np.abs(X)))
    for i in range(1, 15):
        sigma.append(0.9*sigma[-1])
    #print(sigma)
    return sigma

def nabla_x_create(A,sigma):
    nabla_x = A * np.exp(-(A*A)/(2*sigma*sigma))
    return nabla_x

def projection_to_solution_space_by_L1(D, X, S, Y, U, parameter_rho_coef, parameter_gamma, iteration, thr, cri, dsz):
    Df = con.convert_to_Df(D, S, cri)
    Xf = con.convert_to_Xf(X, S, cri)

    for i in range(iteration):

        YSUf = con.convert_to_Sf(Y+S-U, cri)
        b = (1 / parameter_rho_coef) * Xf + np.conj(Df) * YSUf
        Xf = sl.solvedbi_sm(Df, (1 / parameter_rho_coef), b)
        X = con.convert_to_X(Xf, cri)
        #X = np.where(X <= 0, 0, X)

        Xf = con.convert_to_Xf(X, S, cri)
        DfXf = np.sum(Df * Xf, axis=cri.axisM)
        DX = con.convert_to_S(DfXf, cri)
        Y = sp.prox_l1(DX - S + U, (parameter_gamma / parameter_rho_coef))

        U = U + DX - Y - S

    return X, Y, U

def projection_to_solution_space_by_L2(D, X, S, parameter_gamma, iteration, thr, cri, dsz):
    Df = con.convert_to_Df(D, S, cri)
    Xf = con.convert_to_Xf(X, S, cri)

    for i in range(iteration):

        Sf = con.convert_to_Sf(S, cri)
        b = (1 / parameter_gamma) * Xf + np.conj(Df) * Sf
        Xf = sl.solvedbi_sm(Df, (1 / parameter_gamma), b)
        X = con.convert_to_X(Xf, cri)

        Xf = con.convert_to_Xf(X, S, cri)

    return X

def dictionary_learning_by_L1(X, S, dsz, G1, H1, G0, H0, parameter_rho_dic, iteration, thr, cri):
    Xf = con.convert_to_Xf(X, S, cri)

    bar_D = tqdm(total=iteration, desc = 'D', leave = False)
    for i in range(iteration):
        
        D, G0, H0, XD = dictionary_learning_by_L1_Dstep(Xf, S, G1, H1, G0, H0, parameter_rho_dic, i, cri, dsz)
        
        Dr = np.asarray(D.reshape(cri.shpD), dtype=S.dtype)
        H1r = np.asarray(H1.reshape(cri.shpD), dtype=S.dtype)
        Pcn = cr.getPcn(dsz, cri.Nv, cri.dimN, cri.dimCd)
        G1r = Pcn(Dr+H1r)
        G1 = cr.bcrop(G1r, dsz, cri.dimN).squeeze()
        
        H1 = H1 + D - G1

        Est = sp.norm_l1(XD - S)
        if(i == 0):
            pre_Est = 1.1 * Est
        
        if((pre_Est - Est) / pre_Est <= thr):
            bar_D.update(iteration - i)
            break
        
        pre_Est = Est

        bar_D.update(1)
    bar_D.close()
    return D, G0, G1, H0, H1

def dictionary_learning_by_L1_Dstep(Xf, S, G1, H1, G0, H0, parameter_rho_dic, iteration, cri, dsz):
    GH = con.convert_to_Df(G1 - H1, S, cri)
    GSH = con.convert_to_Sf(G0 + S - H0, cri)
    b = GH + sl.inner(np.conj(Xf), GSH, cri.axisK)
    Df = sl.solvemdbi_ism(Xf, 1, b, cri.axisM, cri.axisK)
    D = con.convert_to_D(Df, dsz, cri)
    
    XfDf = np.sum(Xf * Df, axis=cri.axisM)
    XD = con.convert_to_S(XfDf, cri)
    G0 = sp.prox_l1(XD - S + H0, (1 / parameter_rho_dic))
    
    H0 = H0 + XD - G0 - S

    return D, G0, H0, XD

def dictionary_learning_by_L2(X, S, dsz, G, H, parameter_rho_dic, iteration, thr, cri):
    Xf = con.convert_to_Xf(X, S, cri)

    bar_D = tqdm(total=iteration, desc = 'D', leave = False)
    for i in range(iteration):
        
        Gf = con.convert_to_Df(G, S, cri)
        Hf = con.convert_to_Df(H, S, cri)
        GH = Gf - Hf
        Sf = con.convert_to_Sf(S, cri)
        b = parameter_rho_dic * GH + sl.inner(np.conj(Xf), Sf, cri.axisK)
        Df = sl.solvemdbi_ism(Xf, parameter_rho_dic, b, cri.axisM, cri.axisK)
        D = con.convert_to_D(Df, dsz, cri)

        XfDf = np.sum(Xf * Df, axis=cri.axisM)
        XD = con.convert_to_S(XfDf, cri)
        
        Dr = np.asarray(D.reshape(cri.shpD), dtype=S.dtype)
        Hr = np.asarray(H.reshape(cri.shpD), dtype=S.dtype)
        Pcn = cr.getPcn(dsz, cri.Nv, cri.dimN, cri.dimCd)
        Gr = Pcn(Dr+Hr)
        G = cr.bcrop(Gr, dsz, cri.dimN).squeeze()
        
        H = H + D - G

        Est = sp.norm_2l2(XD - S)
        if(i == 0):
            pre_Est = 1.1 * Est
        
        if((pre_Est - Est) / pre_Est <= thr):
            bar_D.update(iteration - i)
            break
        
        pre_Est = Est

        bar_D.update(1)
    bar_D.close()
    return D, G, H

def coef_dic_update_L1_L0(parameter_rho_coef, parameter_rho_dic, parameter_mu, parameter_gamma, S0, D, itr, thr, lay):

    S0 = 2 * S0 
    S0 -= np.mean(S0, axis=(0, 1))
    S = S0

    cri = cr.CSC_ConvRepIndexing(D, S)
    dsz = D.shape

    X = the_minimum_l2_norm_solution(D, S, cri)
    parameter_sigma = setting_sigma(X)

    Xstep_iteration = itr[0]
    Dstep_iteration = itr[1]
    total_iteration = itr[2]


    bar_Lay = tqdm(total=len(parameter_sigma), desc = lay, leave = False)
    for l in parameter_sigma:
        bar_L = tqdm(total = total_iteration, desc = 'L', leave = False)
        for L in range(total_iteration):

            if((l == parameter_sigma[0]) & (L == 0)):
                G1 = D
                H1 = np.zeros(D.shape)
                H0 = np.zeros(S.shape)
                Xf = con.convert_to_Xf(X, S, cri)
                Df = con.convert_to_Df(D, S, cri)
                XfDf = np.sum(Xf * Df, axis=cri.axisM)
                XD = con.convert_to_S(XfDf, cri)
                G0 = XD - S
            D, G0, G1, H0, H1 = dictionary_learning_by_L1(X, S, dsz, G1, H1, G0, H0, parameter_rho_dic, Dstep_iteration, thr, cri)

            nabla_x = nabla_x_create(X, l)
            X = X - parameter_mu * nabla_x

            if((l == parameter_sigma[0]) & (L == 0)):
                Xf = con.convert_to_Xf(X, S, cri)
                Df = con.convert_to_Df(D, S, cri)
                XfDf = np.sum(Xf * Df, axis=cri.axisM)
                XD = con.convert_to_S(XfDf, cri)
                Y = XD - S
                U = np.zeros(S.shape)
            X, Y, U = projection_to_solution_space_by_L1(D, X, S, Y, U, parameter_rho_coef, parameter_gamma, Xstep_iteration, thr, cri, dsz)

            if(np.sum(X != 0) == 0):
                bar_L.update(total_iteration-L)
                break
            bar_L.update(1)
        bar_L.close()


        if(np.sum(X != 0) == 0):
            bar_Lay.update(len(parameter_sigma)-l)
            break

        bar_Lay.update(1)
    #X = np.where((-parameter_sigma[-1] < X) & (X < parameter_sigma[-1]), 0, X) #?
    bar_Lay.close()
    
    l0_norm = np.sum(X != 0)
    print("[" + lay + "] L0 norm: %d " % l0_norm)

    return D, X, l0_norm

def coef_dic_update_L2_L0(parameter_gamma, parameter_rho_dic, parameter_mu, S0, D, itr, thr, lay):

    S0 = 2 * S0 
    S0 -= np.mean(S0, axis=(0, 1))
    S = S0

    cri = cr.CSC_ConvRepIndexing(D, S)
    dsz = D.shape

    X = the_minimum_l2_norm_solution(D, S, cri)
    parameter_sigma = setting_sigma(X)

    Xstep_iteration = itr[0]
    Dstep_iteration = itr[1]
    total_iteration = itr[2]


    bar_Lay = tqdm(total=len(parameter_sigma), desc = lay, leave = False)
    for l in parameter_sigma:
        bar_L = tqdm(total = total_iteration, desc = 'L', leave = False)
        for L in range(total_iteration):
                   
            nabla_x = nabla_x_create(X, l)
            X = X - parameter_mu * nabla_x

            X = projection_to_solution_space_by_L2(D, X, S, parameter_gamma, Xstep_iteration, thr, cri, dsz)

            if(np.sum(X != 0) == 0):
                bar_L.update(total_iteration-L)
                break

            if((l == parameter_sigma[0]) & (L == 0)):

                G = D
                H = np.zeros(D.shape)
            D, G, H = dictionary_learning_by_L2(X, S, dsz, G, H, parameter_rho_dic, Dstep_iteration, thr, cri)
            bar_L.update(1)
        bar_L.close()

        if(np.sum(X != 0) == 0):
            bar_Lay.update(len(parameter_sigma)-l)
            break

        bar_Lay.update(1)
    bar_Lay.close()

    #X = np.where((-0.5*parameter_sigma[-1] < X) & (X < parameter_sigma[-1]*0.5), 0, X)
    
    l0_norm = np.sum(X != 0)
    print("[" + lay + "] L0 norm: %d " % l0_norm)

    return D, X, l0_norm

def feature_extraction_L1_L0(parameter_rho_coef, parameter_gamma, parameter_mu, S0, D, itr, thr, lay):

    S0 = 2 * S0 
    S0 -= np.mean(S0, axis=(0, 1))
    S = S0

    cri = cr.CSC_ConvRepIndexing(D, S)
    dsz = D.shape

    X = the_minimum_l2_norm_solution(D, S, cri)

    parameter_sigma = setting_sigma(X)

    Df = con.convert_to_Df(D, S, cri)

    Xstep_iteration = itr[0]
    total_iteration = itr[2]

    bar_Lay = tqdm(total=len(parameter_sigma), desc = lay, leave = False)
    for k in parameter_sigma:

        bar_L = tqdm(total = total_iteration, desc = 'L', leave = False)
        for L in range(total_iteration):

            nabla_x = nabla_x_create(X, k)
            X = X - parameter_mu * nabla_x

            if((k == parameter_sigma[0]) & (L == 0)):
                Xf = con.convert_to_Xf(X, S, cri)
                Df = con.convert_to_Df(D, S, cri)
                XfDf = np.sum(Xf * Df, axis=cri.axisM)
                XD = con.convert_to_S(XfDf, cri)
                Y = XD - S
                U = np.zeros(S.shape)
            X, Y, U = projection_to_solution_space_by_L1(D, X, S, Y, U, parameter_rho_coef, parameter_gamma, Xstep_iteration, thr, cri, dsz)

            if(np.sum(X != 0) == 0):
                bar_L.update(total_iteration-L)
                break
            bar_L.update(1)
        bar_L.close()

        if(np.sum(X != 0) == 0):
            bar_Lay.update(len(parameter_sigma)-k)
            break
        
        bar_Lay.update(1)
    #X = np.where((-parameter_sigma[-1] < X) & (X < parameter_sigma[-1]), 0, X)
    bar_Lay.close()

    l0_norm = np.sum(X != 0)
    print("[" + lay + "] L0 norm: %d " % l0_norm)
    return X, l0_norm

def feature_extraction_L2_L0(parameter_gamma, parameter_mu, S0, D, itr, thr, lay):

    S0 = 2 * S0 
    S0 -= np.mean(S0, axis=(0, 1))
    S = S0

    cri = cr.CSC_ConvRepIndexing(D, S)
    dsz = D.shape

    X = the_minimum_l2_norm_solution(D, S, cri)

    parameter_sigma = setting_sigma(X)

    Xstep_iteration = itr[0]
    total_iteration = itr[2]

    bar_Lay = tqdm(total=len(parameter_sigma), desc = lay, leave = False)
    for k in parameter_sigma:

        bar_L = tqdm(total = total_iteration, desc = 'L', leave = False)
        for L in range(total_iteration):

            nabla_x = nabla_x_create(X, k)
            X = X - parameter_mu * nabla_x

            X = projection_to_solution_space_by_L2(D, X, S, parameter_gamma, Xstep_iteration, thr, cri, dsz)
            if(np.sum(X != 0) == 0):
                bar_L.update(total_iteration-L)
                break
            bar_L.update(1)
        bar_L.close()

        if(np.sum(X != 0) == 0):
            bar_Lay.update(len(parameter_sigma)-k)
            break
        
        bar_Lay.update(1)
    bar_Lay.close()

    #X = np.where((-parameter_sigma[-1] < X) & (X < parameter_sigma[-1]), 0, X)
    l0_norm = np.sum(X != 0)
    print("[" + lay + "] L0 norm: %d " % l0_norm)
    return X, l0_norm