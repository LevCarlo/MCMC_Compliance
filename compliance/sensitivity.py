'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-04-19 16:00:31
LastEditTime: 2024-04-19 22:10:13
LastEditors: Mengjie Zheng
Description: Calculate the sensitivity of compliance to the different seismic parameters
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/compliance/sensitivity.py
'''
import numpy as np
from forward_funcs.ncomp_fortran import ncomp_fortran
from utils import resample
import matplotlib.pyplot as plt


def compliance_sensitivity_kernel(model, freq, water_depth, params, dz=None, dp=0.025):
    """
    Calculate the sensitivity of compliance to the different seismic parameters
    Parameters
    ----------
    model : numpy.ndarray
        The seismic model of the Earth structure, thickness, density, Vp, Vs
    freq : numpy.ndarray
        The frequency of the infragravity waves
    water_depth : float
        The water depth 
    params : str
        The seismic parameter to calculate the sensitivity
    dz : float, optional, by default None
        The thickness of the layer for upsample model, in meters
    dp : float, optional
        Parameter increment (%) for numerical partial derivatives, by default 2.5%
    
    Returns
    -------
    numpy.ndarray
        The sensitivity of compliance to the seismic parameter
    """

    key_dict = {"thickness": 0, "density": 1, "Vp": 2, "Vs": 3}
    ipar = key_dict[params]

    # Resample the model if dz is not None
    if dz is not None:
        dzi, parametersi = resample(model[:, 0], model[:, 1:], dz)
        model = np.column_stack((dzi, parametersi))
    
    c0 = ncomp_fortran(water_depth, freq, model) # reference compliance
    
    kernel = np.zeros_like(model[:, 0], dtype=np.float64) 

    cumulative_thickness = model[:, 0].cumsum() # cumulative thickness

    for i in range(model.shape[0]):
        orig_value = model[i, ipar]
        model[i, ipar] = orig_value * (1 + dp)
        c_plus = ncomp_fortran(water_depth, freq, model)

        model[i, ipar] = orig_value * (1 - dp)
        c_minus = ncomp_fortran(water_depth, freq, model)

        model[i, ipar] = orig_value # reset the value

        # Using central relative difference
        kernel[i] = (c_plus[0] - c_minus[0]) / (2 * c0[0] * dp)
    
    return cumulative_thickness, kernel



if __name__ == "__main__":
    # model = np.array([[1000, 2.000, 4.000, 1.600],
    #                   [1000, 2.500, 5.000, 2.000],
    #                   [1000, 2.500, 6.000, 2.800],
    #                   [1000, 2.500, 7.000, 3.200],
    #                   [1000, 2.500, 7.000, 3.200],
    #                   [1000, 2.500, 7.000, 3.200],])
    model = np.array([160, 2.5, 6.0, 3.5] * 55).reshape(-1, 4)
    # z = [0.000000, 0.015457, 0.030914, 0.046371, 0.061828, 0.077285, 0.092742, 0.108199,
    #       0.123656, 0.139113, 0.154570, 0.154570, 0.357815, 0.561060, 0.764305, 0.967550,
    #       1.170795, 1.374040, 1.577285, 1.780530, 1.983775, 2.187020, 2.390265, 2.593510,
    #       2.796755, 3.000000]
    # vs = [0.279630, 0.281428, 0.283226, 0.285024, 0.286822, 0.288620, 0.290418, 0.292216,
    #        0.294014, 0.295812, 0.297610, 4.372850, 4.390541, 4.405654, 4.418437, 4.429141,
    #        4.438013, 4.445303, 4.451260, 4.456133, 4.460170, 4.463621, 4.466734, 4.469759,
    #        4.472945, 4.476540]
    # thick = np.diff(z)
    # vs = np.convolve(vs, np.ones(2) / 2, mode='valid')
    # vp = vs * 1.78
    # rho = np.ones_like(vs) * 2.5
    # model = np.column_stack((thick * 1000, rho, vp, vs))
    water_depth = 2000
    dz = 50
    dp = 0.025
    params = "Vs"

    freq = np.linspace(0.004, 0.016, 10)
    # freq = [0.008]

    fig, ax = plt.subplots()
    for i, f in enumerate(freq):
        depth, kernel = compliance_sensitivity_kernel(model, np.array([f]), water_depth, params, dz, dp)
        ax.plot(kernel[:-1], depth[:-1], label="%.4f" % f)
    
    ax.axvline(0, color='black', lw=0.5, ls='--')
    # ax.set_ylim([0, 500])
    ax.invert_yaxis()

    ax.legend()

    # depth, kernel = compliance_sensitivity_kernel(model, freq, water_depth, params, dz, dp)
    # print(depth, kernel)
    # fig, ax = plt.subplots()
    # ax.plot(kernel[:-1], depth[:-1])
    # # ax.plot(np.abs(kernel[:-1]), depth[:-1])
    # ax.set_ylim([0, 2800])
    # ax.axvline(0, color='black', lw=0.5, ls='--')
    # ax.invert_yaxis()
    fig.savefig("./sensitivity.png")

