'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-04-19 16:00:31
LastEditTime: 2024-04-20 11:17:22
LastEditors: Mengjie Zheng
Description: Calculate the sensitivity of compliance to the different seismic parameters
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/compliance/sensitivity.py
'''
import numpy as np
from forward_funcs.ncomp_fortran import ncomp_fortran
from utils import resample
import matplotlib.pyplot as plt

class ComplianceSensitivity():
    def __init__(self, model, freq, water_depth, params, dz=None, dp=0.025):
        self.model = model
        self.freq = np.asarray([freq])
        self.water_depth = water_depth
        self.params = params
        self.dz = dz
        self.dp = dp

    def compliance_sensitivity_kernel(self):
        """
        Calculate the sensitivity of compliance to the different seismic parameters
        Parameters
        ----------
        model : numpy.ndarray
            The seismic model of the Earth structure, thickness (in meters), density, Vp, Vs
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
        ipar = key_dict[self.params]

        # Resample the model if dz is not None
        if self.dz is not None:
            dzi, parametersi = resample(self.model[:, 0], self.model[:, 1:], self.dz)
            self.model = np.column_stack((dzi, parametersi))
        
        c0 = ncomp_fortran(self.water_depth, self.freq, self.model) # reference compliance
        
        kernel = np.zeros_like(self.model[:, 0], dtype=np.float64) 

        cumulative_thickness = self.model[:, 0].cumsum() # cumulative thickness

        for i in range(self.model.shape[0]):
            orig_value = self.model[i, ipar]
            self.model[i, ipar] = orig_value * (1 + self.dp)
            c_plus = ncomp_fortran(self.water_depth, self.freq, self.model)

            self.model[i, ipar] = orig_value * (1 - self.dp)
            c_minus = ncomp_fortran(self.water_depth, self.freq, self.model)

            self.model[i, ipar] = orig_value # reset the value

            # Using central relative difference
            kernel[i] = (c_plus[0] - c_minus[0]) / (2 * c0[0] * self.dp)
        
        return cumulative_thickness, kernel

if __name__ == "__main__":
    model = np.array([160, 2.5, 6.0, 3.5] * 55).reshape(-1, 4)
    water_depth = 2000
    dz = 50
    dp = 0.025
    params = "Vs"
    freq = np.linspace(0.004, 0.016, 10)

    fig, ax = plt.subplots()
    for f in freq:
        cs = ComplianceSensitivity(model, f, water_depth, params, dz, dp)
        x, y = cs.compliance_sensitivity_kernel()
        ax.plot(y[:-1], x[:-1], label=f"{f:.3f} Hz")
    ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Depth(m)")
    fig.savefig("./sensitivity.png")


    


