'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-10-31 15:48:58
LastEditTime: 2023-12-15 17:55:54
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/inv_inversion/MC_Compliance-dev/forward_funcs/calcNCompMat.py
'''


import numpy as np
import matplotlib.pyplot as plt

def argdtray(wd, h):
    hh = np.sqrt(np.abs(h))
    th = wd * hh
    if th >= 1.5e-14:
        if h <= 0:
            c = np.cos(th)
            s = -np.sin(th) / hh
        else:
            if th >= 88:
                print(f'TH = {th}, the freqs are possibly too high')
            d = np.exp(th)
            c = 0.5 * (d + 1 / d)
            s = -0.5 * (d - 1 / d) / hh
    else:
        c = 1
        s = -wd
    return c, s

def raydep(p, om, d, ro, vp2, vs2):
    mu = ro * vs2
    n = len(d)
    ist = n
    ysav = 0
    psq = p * p
    r2 = 2 * mu[ist - 1] * p
    RoW = np.sqrt(psq - 1 / vp2[ist - 1])
    SoW = np.sqrt(psq - 1 / vs2[ist - 1])
    ym = np.zeros((ist, 5))
    i = ist
    y = np.zeros(5)
    y[2] = RoW
    y[3] = -SoW
    y[0] = (RoW * SoW - psq) / ro[i - 1]
    y[1] = r2 * y[0] + p
    y[4] = ro[i - 1] - r2 * (p + y[1])
    ym[i - 1, :] = y
    
    while i > 1:
        i -= 1
        ha = psq - 1 / vp2[i - 1]
        ca, sa = argdtray(om * d[i - 1], ha)
        hb = psq - 1 / vs2[i - 1]
        cb, sb = argdtray(om * d[i - 1], hb)
        hbs = hb * sb
        has = ha * sa
        r1 = 1 / ro[i - 1]
        r2 = 2 * mu[i - 1] * p
        b1 = r2 * y[0] - y[1]
        g3 = (y[4] + r2 * (y[1] - b1)) * r1
        g1 = b1 + p * g3
        g2 = ro[i - 1] * y[0] - p * (g1 + b1)
        e1 = cb * g2 - hbs * y[2]
        e2 = -sb * g2 + cb * y[2]
        e3 = cb * y[3] + hbs * g3
        e4 = sb * y[3] + cb * g3
        y[2] = ca * e2 - has * e4
        y[3] = sa * e1 + ca * e3
        g3 = ca * e4 - sa * e2
        b1 = g1 - p * g3
        y[0] = (ca * e1 + has * e3 + p * (g1 + b1)) * r1
        y[1] = r2 * y[0] - b1
        y[4] = ro[i - 1] * g3 - r2 * (y[1] - b1)
        ym[i - 1, :] = y
    
    de = y[4] / np.sqrt(y[0] ** 2 + y[1] ** 2)
    ynorm = 1 / y[2]
    y[0:4] = [0, -ynorm, 0, 0]
    x = np.zeros((ist, 4))
    
    for i in range(ist):
        x[i, 0] = -ym[i, 1] * y[0] - ym[i, 2] * y[1] + ym[i, 0] * y[3]
        x[i, 1] = -ym[i, 3] * y[0] + ym[i, 1] * y[1] - ym[i, 0] * y[2]
        x[i, 2] = -ym[i, 4] * y[1] - ym[i, 1] * y[2] - ym[i, 3] * y[3]
        x[i, 3] = ym[i, 4] * y[0] - ym[i, 2] * y[2] + ym[i, 1] * y[3]
        
        ls = i
        if i >= 1:
            sum_val = np.abs(x[i, 0] + i * x[i, 1])
            pbsq = 1 / vs2[i]
            if sum_val < 1e-4:
                break
        
        ha = psq - 1 / vp2[i]
        ca, sa = argdtray(om * d[i], ha)
        hb = psq - 1 / vs2[i]
        cb, sb = argdtray(om * d[i], hb)
        hbs = hb * sb
        has = ha * sa
        r2 = 2 * p * mu[i]
        e2 = r2 * y[1] - y[2]
        e3 = ro[i] * y[1] - p * e2
        e4 = r2 * y[0] - y[3]
        e1 = ro[i] * y[0] - p * e4
        e6 = ca * e2 - sa * e1
        e8 = cb * e4 - sb * e3
        y[0] = (ca * e1 - has * e2 + p * e8) / ro[i]
        y[1] = (cb * e3 - hbs * e4 + p * e6) / ro[i]
        y[2] = r2 * y[1] - e6
        y[3] = r2 * y[0] - e8
    
    if x[0, 2] == 0:
        raise ValueError('vertical surface stress = 0 in DETRAY')
    
    ist = ls
    v = x[:, 0]
    u = x[:, 1]
    zz = x[:, 2]
    zx = x[:, 3]
    
    return v, u, zz, zx

def dtanh(x):
    """
    More stable hyperbolic tangent of x
    """
    a = np.exp(x * (x <= 50))  # Make sure nothing blows up to NaN
    one = np.ones_like(x)
    y = (np.abs(x) > 50) * (np.abs(x) / x) + (np.abs(x) <= 50) * ((a - one / a) / (a + one / a))
    return y

def gravd(W, h):
    """
    Gravity wave wavenumber determination
    k = rad/meter
    k = gravd(w, h)
    w = angular frequency (can be a vector), h = water depth (meters)
    """
    if len(W) < 1:
        raise ValueError('You must specify at least one frequency')
    
    G = 9.79329
    N = len(W)
    W2 = W * W
    kDEEP = W2 / G
    kSHAL = W / np.sqrt(G * h)
    erDEEP = np.ones_like(W) - G * kDEEP * dtanh(kDEEP * h) / W2
    
    one = np.ones_like(W)
    d = one
    done = np.zeros_like(W, dtype=bool)
    nd = np.where(done == 0)
    k1 = kDEEP
    k2 = kSHAL
    e1 = erDEEP
    ktemp = np.zeros_like(W)
    e2 = np.zeros_like(W)
    
    while True:
        e2[nd] = one[nd] - G * k2[nd] * dtanh(k2[nd] * h) / W2[nd]
        d = e2 * e2
        done = d < 1e-20
        if np.all(done):
            K = k2
            return K
        nd = np.where(done == 0)
        ktemp[nd] = k1[nd] - e1[nd] * (k2[nd] - k1[nd]) / (e2[nd] - e1[nd])
        k1[nd] = k2[nd]
        k2[nd] = ktemp[nd]
        e1[nd] = e2[nd]  # Reset values


def calc_ncomp_mat(depth, freq, model):
    """
    Calculate normalized compliance of a layered earth model.
    
    Parameters:
        depth (float): The water depth (meters) at the site.
        freq (np.array): The frequencies (Hz) to calculate compliance at.
        model (np.array): A 1-D earth model, with layer thicknesses (meters) in column 0,
                         layer densities (g/cc) in column 1,
                         layer compress vels (km/s) in column 2,
                         layer shear vels (km/s) in column 3.
    
    Returns:
        ncomp (np.array): Normalized vertical compliance.
    """

    # Extract model parameters
    thick = model[:, 0]
    rho = model[:, 1] * 1000
    vpsq = 1000000 * model[:, 2] ** 2
    vssq = 1000000 * model[:, 3] ** 2
    
    # Calculate omega
    omega = 2 * np.pi * freq
    
    # Call gravd function (assuming it has been converted to Python as well)
    k = gravd(omega, depth)
    
    # Calculate p
    p = k / omega
    
    # Initialize ncomp array
    ncomp = np.empty(len(p))
    
    # Loop through each p and omega
    for i in range(len(p)):
        # Call raydep function (assuming it has been converted to Python as well)
        v, u, sigzz, sigzx = raydep(p[i], omega[i], thick, rho, vpsq, vssq)
        ncomp[i] = -k[i] * v[0] / (omega[i] * sigzz[0])

    return ncomp


if __name__=="__main__":
    # Define inpModel as a 4x4 matrix
    inpModel = np.array([[1000, 2.500, 4.000, 2.500],
                         [2000, 3.000, 7.000, 4.000],
                         [400, 2.500, 4.000, 1.000],
                         [1000, 3.000, 7.000, 4.000]])
    
    # Define freqs as an array ranging from 0.003 to 0.02 with an interval of 0.001
    freqs = np.arange(0.003, 0.02 + 0.001, 0.001)
    
    # Define wdepth
    wdepth = 3000  # meters
    print(calc_ncomp_mat(wdepth, freqs, inpModel))
    
    # # Call the calcNCompMat function (this function should be defined beforehand)
    # ncomp = calc_ncomp_mat(wdepth, freqs, inpModel)  # Uncomment this line once the function is defined

    # # Plot ncomp against freqs
    # plt.plot(freqs, ncomp, label="Python")  # Uncomment this line once the function is defined and ncomp is calculated
    # plt.xlabel('Frequency')
    # plt.ylabel('ncomp')

    # from scipy.io import loadmat
    # test_mat = "./test.mat"
    # dat = loadmat(test_mat)
    # plt.plot(dat["freqs"][0], dat["ncomp"][0], label="matlab")
    # print(ncomp - dat["ncomp"][0])
    # plt.legend()
    # plt.show()


