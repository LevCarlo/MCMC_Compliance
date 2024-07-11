'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-07-23 09:59:39
LastEditTime: 2024-06-10 10:18:02
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/model.py
'''

import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Type, List
from copy import deepcopy

class BsplineBase:
    """
    BsplineBase is a class for B-spline basis function.
    """
    def __init__(self, z, n, deg=None, alpha=2., eps=np.finfo(float).eps) -> None:
        self.n = len(z)
        self.nBasis = n
        self.deg = deg if deg is not None else 3 + (n >= 4)
        self.alpha = alpha
        self.eps = eps

        if self.nBasis == 1:
            self.basis = np.ones((self.n, 1))
            return
        
        if self.nBasis == 2:
            self.basis = np.ones((2, self.n))
            self.basis[0, :] = np.linspace(1, 0, self.n)
            self.basis[1, :] = np.linspace(0, 1, self.n)
            return
        
        x = np.zeros(n + self.deg)
        x[:self.deg - 1] = -eps * np.ones(self.deg - 1)
        x[self.deg - 1] = 0.
        x[self.deg:n] = np.power(alpha, range(n - self.deg)) * (alpha - 1) / (
                np.power(alpha, n - self.deg + 1) - 1
        )
        x[n] = 1.
        x[n + 1:] = (1 + eps) * np.ones(self.deg - 1)
        x = z[0] + x * (z[-1] - z[0])

        bs0 = np.zeros((len(z), len(x) - 1))
        bs1 = bs0.copy()

        for i in range(bs0.shape[1]):
            bs0[(z >= x[i]) * (z < x[i + 1]), i] = 1

        for irun in range(self.deg - 1):
            for i in range(bs0.shape[1] - irun - 1):
                bs1[:, i] = 0
                if x[i + irun + 1] - x[i] != 0:
                    bs1[:, i] += bs0[:, i] * (z - x[i]) / (x[i + irun + 1] - x[i])
                if x[i + 1 + irun + 1] - x[i + 1] != 0:
                    bs1[:, i] += bs0[:, i + 1] * (x[i + 1 + irun + 1] - z) / (
                            x[i + 1 + irun + 1] - x[i + 1]
                    )
            bs0 = bs1.copy()

        bs = bs1[:, :n].copy()
        self.basis = bs.T

    def __mul__(self, coef):
        if self.nBasis == 1:
            coef = np.array([coef])
        return np.dot(coef, self.basis)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure();plt.plot(np.linspace(0,1,self.n),self.basis.T)
        plt.savefig("bspline.png")

class Params:
    def __init__(self, definition, values, inversion, perturb_type=None, perturb_values=None):
        self.definition = definition 
        self.values = values
        self.inversion = inversion
        self.perturb_type = perturb_type
        self.perturb_values = perturb_values
    
    def to_dict(self):
        return vars(self)
    
    def get(self, attribute):
        return getattr(self, attribute)
    
    def update_values(self, new_values):
        self.values = new_values
        
class GLayer:
    def __init__(self, layer_index, thickness: Type[Params], delta, vs, vp, rho):
        self.layer_index = layer_index
        self.thickness = thickness
        self.delta = delta
        self.vs = vs
        self.vp = vp
        self.rho = rho
    
    def to_dict(self):
        layer_dict = {}
        for key, value in vars(self).items():
            if hasattr(value, 'to_dict'):  # Check if the attribute has a to_dict method
                layer_dict[key] = value.to_dict()
            else:
                layer_dict[key] = value
        return layer_dict

    def discrete(self):
        n = np.maximum(10, int(self.thickness.get("values") / self.delta))
        depths = np.linspace(0, self.thickness.get("values"), n + 1)

        seis_params = {"vs": self.vs, "vp": self.vp, "rho": self.rho}
        seis_values = {"vs": None, "vp": None, "rho": None}

        for key, param in seis_params.items():
            if isinstance(param, Params):
                layer_type = param.get("definition")
                if layer_type == "Constant":
                    seis_values[key] = np.ones(n + 1) * param.get("values")
                elif layer_type == "Gradient":
                    seis_values[key] = np.linspace(param.get("values")[0], param.get("values")[1], n + 1)
                elif layer_type == "Bspline":
                    bs = BsplineBase(depths, len(param.get("values")))
                    seis_values[key] = bs * param.get("values")
            else:
                seis_values[key] = param
        
        return depths, seis_values
    
    def create_model(self):
        depths, seis_values = self.discrete()
        if isinstance(self.vs, Params):
            vsi = seis_values["vs"]
        if isinstance(self.vp, Params):
            vpi = seis_values["vp"]
        else:
            vpi = VpEstimate(z=depths, vs=vsi).estimate(seis_values["vp"])
        if isinstance(self.rho, Params):
            rhoi = seis_values["rho"]
        else:
            rhoi = RhoEstimate(z=depths, vs=vsi, vp=vpi).estimate(seis_values["rho"])
        return depths, vsi, vpi, rhoi
    def update(self, **kwargs):
        for key, value in kwargs.items():
            attribute = getattr(self, key, None)
            if isinstance(attribute, Params):
                attribute.update_values(value)
           
class Layer:
    def __init__(self, layer_index, thickness: Type[Params], delta, layer_type=None):
        self.layer_index = layer_index
        self.thickness = thickness
        self.delta = delta
        self.layer_type = layer_type
    
    def to_dict(self):
        layer_dict = vars(self)
        layer_dict["thickness"] = self.thickness.to_dict()
        return layer_dict
        
class VsLayer(Layer):
    def __init__(self, layer_index, thickness, delta, vs: Type[Params], 
                 vp_derived_method=None, rho_derived_method=None):
        super().__init__(layer_index, thickness, delta, layer_type="Vs")
        self.vs = vs
        self.vp_derived_method = vp_derived_method
        self.rho_derived_method = rho_derived_method
    
    def to_dict(self):
        vs_layer_dict = super().to_dict()
        vs_layer_dict["vs"] = self.vs.to_dict()
        return vs_layer_dict
    
    def discrete(self):
        n = np.maximum(10, int(self.thickness.get("values") / self.delta))
        depths = np.linspace(0, self.thickness.get("values"), n + 1)

        layer_type = self.vs.get("definition")
        if layer_type == "Constant":
            return depths, np.ones(n + 1) * self.vs.get("values")
        elif layer_type == "Gradient":
            return depths, np.linspace(self.vs.get("values")[0], self.vs.get("values")[1], n + 1)
        elif layer_type == "Bspline":
            bs = BsplineBase(depths, len(self.vs.get("values")))
            return depths, bs * self.vs.get("values")
    
    def create_model(self):
        depths, vsi = self.discrete()
        vp = VpEstimate(z=depths, vs=vsi).estimate(self.vp_derived_method)
        rho = RhoEstimate(z=depths, vs=vsi, vp=vp).estimate(self.rho_derived_method)
        return depths, vsi, vp, rho
    
    def update(self, new_thickness, new_vs):
        self.thickness.update_values(new_thickness)
        self.vs.update_values(new_vs)
    
class VpEstimate:
    def __init__(self, z=None, vs=None, rho=None, method: str=None):
        self.z = z
        self.vs = vs
        self.rho = rho 
        self.method = method
    
    def estimate(self, keyword):
        if keyword == "Brocher2005":
            self.method = keyword
            return self._brocher()
        elif keyword == "Castegna1995":
            self.method = keyword
            return self._castegna()
        elif keyword == "Christensen&Shaw1970":
            self.method = keyword
            return self._christensen_shaw()
        elif type(keyword) == int or type(keyword) == float:
            self.method = "Constant"
            return self._constant(keyword)
        else:
            raise ValueError("Unknown keyword")
    
    def _brocher(self):
        conditions = np.logical_and(0 <= self.vs, self.vs <= 4.5)
        return np.where(
            conditions,
            0.9409 + 2.0947 * self.vs - 0.8206 * self.vs ** 2 + 0.2683 * self.vs ** 3 - 0.0251 * self.vs ** 4,
            np.nan
        )

    def _castegna(self):
        return 1.16 * self.vs + 1.36

    def _christensen_shaw(self):
        return 1.511 + 1.304 * self.z - 0.741 * self.z ** 2 + 0.257 * self.z ** 3
    
    def _constant(self, value):
        return self.vs * value
    
class RhoEstimate:
    def __init__(self, z=None, vs=None, vp=None, method: str=None):
        self.z = z
        self.vs = vs
        self.vp = vp 
        self.method = method

    def estimate(self, keyword):
        if keyword == "Brocher2005":
            self.method = keyword
            return self._brocher()
        elif keyword == "Ludwig1979":
            self.method = keyword
            return self._ludwig()
        elif keyword == "Gardner1974":
            self.method = keyword
            return self._gardner()
        elif keyword == "Christensen&Mooney1995":
            self.method = keyword
            return self._christensen_mooney()
        elif keyword == "Godfrey1997":
            self.method = keyword
            return self._godfrey()
        elif keyword == "Feng1986":
            self.method = keyword
            return self._feng()
        elif keyword == "Willoughby&Edwards2000":
            self.method = keyword
            return self._willoughby_edwards()
        elif keyword == "Hamilton1979":
            self.method = keyword
            return self._hamilton()
        elif type(keyword) == int or type(keyword) == float:
            self.method = "Constant"
            return self._constant(keyword)
    
    def _brocher(self):
        return 1.22679 + 1.53201 * self.vs - 0.83668 * self.vs ** 2 + 0.20673 * self.vs ** 3 - 0.01656 * self.vs ** 4

    def _ludwig(self):
        conditions = np.logical_and(1.5 <= self.vp, self.vp <= 8.5)
        return np.where(
            conditions,
            1.6612 * self.vp - 0.4721 * self.vp ** 2 + 0.0671 * self.vp ** 3 - 0.0043 * self.vp ** 4 +
            0.000106 * self.vp ** 5,
            np.nan
        )

    def _gardner(self):
        conditions = np.logical_and(1.5 <= self.vp, self.vp <= 6.1)
        return np.where(
            conditions,
            1.74 * self.vp ** 0.25,
            np.nan
        )

    def _christensen_mooney(self):
        conditions = np.logical_and(5.5 <= self.vp, self.vp <= 7.5)
        return np.where(
            conditions,
            0.541 + 0.3601 * self.vp,
            np.nan
        )

    def _godfrey(self):
        conditions = np.logical_and(5.9 <= self.vp, self.vp <= 7.1)
        return np.where(
            conditions,
            2.4372 + 0.0761 * self.vp,
            np.nan
        )

    def _feng(self):
        conditions = [
            np.logical_and(self.vp <= 6, np.isnan(self.vp) == False),
            np.logical_and(self.vp > 6, self.vp < 7.5),
            np.logical_and(self.vp >= 7.5, np.isnan(self.vp) == False)
        ]
        choices = [
            2.78 + 0.56 * (self.vp - 6.0),
            3.07 + 0.29 * (self.vp - 7.0),
            3.22 + 0.20 * (self.vp - 7.5)
        ]
        return np.select(conditions, choices, default=np.nan)

    def _willoughby_edwards(self):
        return 1.7 + (0.2 / 300) * self.z

    def _hamilton(self):
        return 1.85 + 0.165 * self.vp
    
    def _constant(self, value):
        return np.ones_like(self.vp) * value    

class Model:  
    def __init__(self, layer_num, layers: List, total_thickness=None):
        self.layer_num = layer_num
        self.layers = layers
        self.total_thickness = total_thickness

        if layer_num != len(layers):
            raise ValueError("The number of layers is not equal to layer_num")

    def adjust_last_layer_thickness(self):
        totoal_other_layers_thickness = sum([layer.thickness.get("values") for layer in self.layers[:-1]])
        last_layer_thickness = self.total_thickness - totoal_other_layers_thickness
        if last_layer_thickness < 0:
            raise ValueError("The total thickness is smaller than the sum of other layers")
        self.layers[-1].thickness.update_values(last_layer_thickness)
        
    def combine_layers(self, boundary_flag=False):
        combined_z = []
        combined_vs = []
        combined_vp = []
        combined_rho = []
        accumulated_depth = 0

        for i, layer in enumerate(self.layers[:self.layer_num]):
            if boundary_flag:
                start_index = 0
            else:
                start_index = 0 if i == 0 else 1

            z, vs, vp, rho = layer.create_model()
            z_values = z[start_index:] + accumulated_depth
            vs_values = vs[start_index:]
            vp_values = vp[start_index:]
            rho_values = rho[start_index:]

            combined_z.extend(z_values)
            combined_vs.extend(vs_values)
            combined_vp.extend(vp_values)
            combined_rho.extend(rho_values)

            accumulated_depth += layer.thickness.get("values")
        
        
        # For simplicity, we assume that there is no negative gradient in Vp and density
        combined_vp = np.maximum.accumulate(combined_vp)
        combined_rho = np.maximum.accumulate(combined_rho)

        return np.array(combined_z), np.array(combined_vs), combined_vp, combined_rho
    
    def to_layer_model(self):
        z, vs, vp, rho = self.combine_layers()
        layer_thickness = np.diff(z)
        kernel = np.ones(2) / 2
        new_vs = np.convolve(vs, kernel, mode='valid')
        new_vp = np.convolve(vp, kernel, mode='valid')
        new_rho = np.convolve(rho, kernel, mode='valid')

        return np.column_stack((layer_thickness, new_vs, new_vp, new_rho))
    
    def plot(self, save_flag=False, save_path=None):
        z, vs, vp, rho = self.combine_layers(boundary_flag=True)
        fig, ax = plt.subplots(1, 3, figsize=(6, 4) ,sharey=True)
        ax[0].plot(vs, z, '-o', c='k')
        ax[0].set_xlabel('Vs (km/s)')
        ax[0].set_ylabel('Depth (km)')
        ax[0].invert_yaxis()

        ax[1].plot(vp, z, 'k')
        ax[1].set_xlabel('Vp (km/s)')
        ax[1].invert_yaxis()

        ax[2].plot(rho, z, 'k')
        ax[2].set_xlabel(r'Density $(g/cm^3)$')
        ax[2].invert_yaxis()

        plt.tight_layout()
        if save_flag:
            plt.close()
            fig.savefig(save_path, dpi=300)

    def from_yaml(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        model_config = config['Model']
        layers_config = model_config['Layers']
        layers = []
        for layer_data in layers_config:
            properties = ["vs", "vp", "rho", "thickness"]
            properties_dict = {}
            for prop in properties:
                if isinstance(layer_data[prop], dict):
                    properties_dict[prop] = Params(**layer_data[prop])
                else:
                    properties_dict[prop] = layer_data[prop]
            layer = GLayer(
                layer_index=layer_data['layer_index'],
                thickness=properties_dict['thickness'],
                delta=layer_data['delta'],
                vs=properties_dict['vs'],
                vp=properties_dict['vp'],
                rho=properties_dict['rho']
            )
            layers.append(layer)

        return Model(layer_num=model_config['layer_num'], layers=layers, total_thickness=model_config['total_thickness'])

    def to_yaml(self, file_path):
        model_config = {
            'Model': {
                'layer_num': self.layer_num,
                'Layers': [layer.to_dict() for layer in self.layers],
                'total_thickness': self.total_thickness
            }
        }
        with open(file_path, 'w') as file:
            yaml.dump(model_config, file, default_flow_style=False, sort_keys=False)

    def clone(self):
        return deepcopy(self)   

    def to_dict(self):
        return {
            'layer_num': self.layer_num,
            'layers': [layer.to_dict() for layer in self.layers],
            'total_thickness': self.total_thickness
        }


if __name__ == '__main__':
    thickness0 = Params(definition="Constant", values=0.79, inversion=False)
    vs0 = Params(definition="Gradient", values=[0.5, 2.0], inversion=True, 
                 perturb_type="Percent", perturb_values=[0.1, 0.1])
    layer0 = GLayer(layer_index=0, thickness=thickness0, delta=0.1, vp=2.0, vs=vs0, rho=1.70)
    thickness1 = Params(definition="Constant", values=None, inversion=False)
    vs1 = Params(definition="Bspline", values=[2.5, 2.0, 3.0, 3.5], inversion=True, 
                 perturb_type="Percent", perturb_values=[0.1, 0.1, 0.1, 0.1])
    layer1 = GLayer(layer_index=1, thickness=thickness1, delta=0.1, vp=2.0, vs=vs1, rho=1.70)
    model = Model(layer_num=2, layers=[layer0, layer1], total_thickness=5.0)
    z, vs, vp, rho = model.combine_layers(boundary_flag=True)
    print(z)