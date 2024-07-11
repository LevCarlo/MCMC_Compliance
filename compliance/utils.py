'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-04-19 19:31:09
LastEditTime: 2024-05-05 11:01:19
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/compliance/utils.py
'''
import numpy as np

def resample(thickness, parameters, dz):
    """
    Resample the seismic parameters
    Refer to https://github.com/keurfonluu/disba/blob/master/disba/_helpers.py

    Parameters
    ----------
    thickness : numpy.ndarray
        Layer thickness, in km
    parameters : numpy.ndarray
        Seismic parameters to resample
    dz : float
        Maximum thickness of the layer, in m

    Returns
    -------
    numpy.ndarray
        Resampled thickness
    numpy.ndarray
        Resampled seismic parameters

    """

    thickness = np.asarray(thickness)
    parameters = np.asarray(parameters)
    sizes = np.where(thickness > dz, np.ceil(thickness / dz), 1).astype(int)

    dzi = np.repeat(thickness / sizes, sizes)

    if parameters.ndim == 1:
        parametersi = np.repeat(parameters, sizes)
    else:
        parametersi = np.repeat(parameters, sizes, axis=0)
    
    return dzi, parametersi
    
if __name__=="__main__":
    thickness = np.array([1, 2, 3, 4, 5])
    parameters = np.array([2, 4, 6, 8, 10])
    dz = 0.5
    dzi, parametersi = resample(thickness, parameters, dz)
    print(dzi)
    print(parametersi)
    print(dzi.shape)
    print(parametersi.shape)
    print(dzi.sum())

