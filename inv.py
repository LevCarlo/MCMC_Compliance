'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-10-09 10:19:01
LastEditTime: 2024-06-20 11:38:08
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/inv.py
Refer to: https://www.pymc.io/projects/examples/en/latest/samplers/MLDA_gravity_surveying.html

'''
import os
import numpy as np
import yaml
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from model import Model, Params
from compliance import ncomp_fortran, cal_Ps_delay



class ComplyLogLike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, model, wdepth, comply_data, comply_forward=ncomp_fortran,
                 inverse=True, regularize=False, alpha=None):
        """
        Parameters
        ----------
        model: Model object
        wdepth: water depth | float
        comply_data: freqs, ncomp, ncomp_err | np.ndarray
        comply_forward: compliace forward function
        inverse: =False, without data constraint, check the prior condition
        regularize: default=False, without regularization
        alpha: regularization parameter
        """
        self.model = model
        self.wdepth = wdepth
        self.comply_data = comply_data
        self.comply_forward = comply_forward
        self.inverse = inverse
        self.regularize = regularize
        self.alpha = alpha

    def perform(self, node, inputs, outputs):
        theta, = inputs
        inpara = np.copy(theta)

        try:
            new_model = self.update(inpara)
        except:
            outputs[0][0] = np.array(-np.inf)
            return
        
        if not self.check_prior(new_model):
            outputs[0][0] = np.array(-np.inf)
            return
        
        if not self.inverse:
            outputs[0][0] = np.array(0.).astype(np.float64)
            return
        
        # The output of to_layer_model method is:
        # thickness (km), vs (km/s), vp (km/s), rho (g/cm^3)
        # The input of ncomp_fortran function is:
        # thickness (m), rho (g/cm^3), vp (km/s), vs (km/s)
        layered_model = new_model.to_layer_model()
        comply_model  = layered_model.copy()
        comply_model[:, 0] *= 1000
        comply_model[:, [1, 3]] = comply_model[:, [3, 1]]
        freqs, ncomp, ncomp_err = self.comply_data[:, 0], self.comply_data[:, 1], self.comply_data[:, 2]
        try:
            ncomp_pred = self.comply_forward(self.wdepth, freqs, comply_model)
        except:
            outputs[0][0] = np.array(-np.inf)
            return
        if np.any(np.isnan(ncomp_pred)) or np.all(ncomp_pred == 0):
            outputs[0][0] = np.array(-np.inf)
            return
        
        chiSqr = np.sum(((ncomp - ncomp_pred) / ncomp_err) ** 2)
        if self.regularize:
            vs = layered_model[:, 1]
            second_derivative = np.diff(np.diff(vs))
            second_derivative = np.pad(second_derivative, (1, 1), 'edge')
            reg = np.sum(second_derivative ** 2)
            loglike = -0.5 * (chiSqr + self.alpha * reg)
        else:
            loglike = -0.5 * chiSqr
        
        outputs[0][0] = np.array(loglike)
    
    def update(self, inpara):
        new_model = self.model.clone()
        param_index = 0
        for i, layer in enumerate(new_model.layers):
            velocity = layer.vs.get("values")
            new_velocity = inpara[param_index:param_index + len(velocity)]
            new_thickness = inpara[param_index + len(velocity)]
            layer.update(new_thickness, new_velocity)
            param_index += len(velocity) + 1
        return new_model
    
    def check_prior(self, model):
        # The vs for gradient layer must be increasing
        for i, layer in enumerate(model.layers):
            key = layer.vs.get("definition")
            if key == "Gradient":
                velocity = layer.vs.get("values")
                if velocity[0] > velocity[1]:
                    return False
            else:
                continue
        
        try:
            z, vs, vp, rho = model.combine_layers(boundary_flag=True)
        except:
            return False
        
        if np.any(z < 0) or np.any(np.isnan(z)) or \
           np.any(vs < 0) or np.any(np.isnan(vs)) or \
           np.any(vp < 0) or np.any(np.isnan(vp)) or \
           np.any(rho < 0) or np.any(np.isnan(rho)):
            return False
        
        _, inverse_indices, counts = np.unique(z, return_inverse=True, return_counts=True)
        duplicate_indices = np.where(counts[inverse_indices] > 1)[0]
        
        if len(duplicate_indices) > 0:
            boundary_vs = vs[duplicate_indices]
            diffs = boundary_vs[::2] - boundary_vs[1::2]
            if np.any(diffs > 0):
                return False
        
        return True

class Comply_Ps_LogLike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, model, wdepth,
                 comply_data, p2s_data,
                 comply_forward=ncomp_fortran,
                 p2s_forward=cal_Ps_delay,
                 weight=[1.0, 10.0],
                 inverse=True, regularize=False, alpha=None):
        
        self.model = model
        self.wdepth = wdepth
        self.comply_data = comply_data
        self.p2s_data = p2s_data
        self.comply_forward = comply_forward
        self.p2s_forward = p2s_forward
        self.weight = weight
        self.inverse = inverse
        self.regularize = regularize
        self.alpha = alpha
    
    def perform(self, node, inputs, outputs):
        theta, = inputs
        inpara = np.copy(theta)

        try:
            new_model = self.update(inpara)
        except:
            outputs[0][0] = np.array(-np.inf)
            return
        
        if not self.check_prior(new_model):
            outputs[0][0] = np.array(-np.inf)
            return
        
        if not self.inverse:
            outputs[0][0] = np.array(0.).astype(np.float64)
            return

        layered_model = new_model.to_layer_model()

        # Compliace
        if self.comply_data is not None:
            comply_model  = layered_model.copy()
            comply_model[:, 0] *= 1000
            comply_model[:, [1, 3]] = comply_model[:, [3, 1]]
            freqs, ncomp, ncomp_err = self.comply_data[:, 0], self.comply_data[:, 1], self.comply_data[:, 2]
            try:
                ncomp_pred = self.comply_forward(self.wdepth, freqs, comply_model)
            except:
                outputs[0][0] = np.array(-np.inf)
                return
            if np.any(np.isnan(ncomp_pred)) or np.all(ncomp_pred == 0):
                outputs[0][0] = np.array(-np.inf)
                return
            
            chiSqr_comply = np.sum(((ncomp - ncomp_pred) / ncomp_err) ** 2)
            # print(chiSqr_comply)
            w0 = self.weight[0]
        else:
            chiSqr_comply = 0.0
            w0 = 0.0
        
        # Ps delay
        if self.p2s_data is not None:
            sediment_layer = new_model.layers[0]
            sediment_thickness = sediment_layer.thickness.get("values")

            _, vs, vp, _ = sediment_layer.create_model()
            vsi, vpi = np.average(vs), np.average(vp)
            try:
                p2s_pred = self.p2s_forward(vsi, vpi, sediment_thickness)
            except:
                outputs[0][0] = np.array(-np.inf)
                return
            if np.isnan(p2s_pred):
                outputs[0][0] = np.array(-np.inf)
                return
            
            p2s, p2s_err = self.p2s_data[0], self.p2s_data[1]
            chiSqr_p2s = np.sum(((p2s - p2s_pred) / p2s_err) ** 2)
            # if chiSqr_p2s > 10:
            #     chiSqr_p2s = np.sqrt(10 * chiSqr_p2s)
            w1 = self.weight[1]
        else:
            chiSqr_p2s = 0.0
            w1 = 0.0
        
        chiSqr = chiSqr_comply * w0 + chiSqr_p2s * w1
        if self.regularize:
            vs = layered_model[:, 1]
            second_derivative = np.diff(np.diff(vs))
            second_derivative = np.pad(second_derivative, (1, 1), 'edge')
            reg = np.sum(second_derivative ** 2)
            loglike = -0.5 * (chiSqr + self.alpha * reg)
        else:
            loglike = -0.5 * chiSqr
        
        outputs[0][0] = np.array(loglike)
    
    def update(self, inpara):
        new_model = self.model.clone()
        param_index = 0
        for i, layer in enumerate(new_model.layers):
            new_params = {}
            for attr_name in ["vs", "vp", "rho"]:
                param = getattr(layer, attr_name)
                if isinstance(param, Params) and param.inversion:
                    length = len(param.values)
                    new_params[attr_name] = inpara[param_index:param_index + length]
                    param_index += length
            new_params["thickness"] = inpara[param_index]
            param_index += 1
            layer.update(**new_params)
        new_model.adjust_last_layer_thickness()
        return new_model
    
    def check_prior(self, model):
        # Sediment layer
        sediment_layer = model.layers[0]
        _, vs, vp, _ = sediment_layer.create_model()
        if np.any(vp > 4.0):
            return False
        vp2vs = vp / vs
        if np.any(vp2vs < 1.70) or np.any(vp2vs > 15.0):
            return False
        if sediment_layer.vs.definition == "Gradient":
            vs = sediment_layer.vs.values
            if vs[0] > vs[1]:
                return False
        if isinstance(sediment_layer.vp, Params):
            if sediment_layer.vp.definition == "Gradient":
                vp = sediment_layer.vp.values
                if vp[0] > vp[1]:
                    return False
        
        # crust layer
        crust_layer = model.layers[1]
        _, vs, _, _ = crust_layer.create_model()
        vs_grad = np.diff(vs)
        if np.any(vs_grad < 0):
            return False

        try:
            z, vs, vp, rho = model.combine_layers(boundary_flag=True)
        except:
            return False


        if np.any(z < 0) or np.any(np.isnan(z)) or \
           np.any(vs <= 0) or np.any(np.isnan(vs)) or \
           np.any(vp <= 0) or np.any(np.isnan(vp)) or \
           np.any(rho <= 0) or np.any(np.isnan(rho)):
            return False
        
        # Check the boundary condition, the vs and vp must be increasing
        _, inverse_indices, counts = np.unique(z, return_inverse=True, return_counts=True)
        duplicate_indices = np.where(counts[inverse_indices] > 1)[0]
        
        if len(duplicate_indices) > 0:
            boundary_vs = vs[duplicate_indices]
            boundary_vp = vp[duplicate_indices]
            vs_diffs = boundary_vs[::2] - boundary_vs[1::2]
            vp_diffs = boundary_vp[::2] - boundary_vp[1::2]
            if np.any(vs_diffs > 0) or np.any(vp_diffs > 0):
                return False
        
        return True
class InversionBase:
    def __init__(self, model=None, likelihood=None, hyperparam=None):
        self.model = model
        self.likelihood = likelihood
        self.hyperparam = hyperparam
    
    @classmethod
    def from_yaml(cls, config_filepath, model=None, likelihood=None):
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)
            hparams = config["Inversion"]
        return cls(hyperparam=hparams, model=model, likelihood=likelihood)

class InversionMC(InversionBase):
    def __init__(self, model=None, likelihood=None, hyperparam=None):
        super().__init__(model, likelihood, hyperparam)
    
    def create_MCmodel(self, method="Uniform"):
        model = pm.Model()
        with model:
            variables = []
            for i, layer in enumerate(self.model.layers):
                for j, param in enumerate([layer.vs, layer.vp, layer.rho]):
                    if isinstance(param, Params):
                        inversion = param.get("inversion")
                        values = param.get("values")
                        perturb_type = param.get("perturb_type")
                        perturb_values = param.get("perturb_values")
                        if not inversion:
                            variables.append(pt.shape_padleft(pt.constant(values[0])))
                        else:
                            for k, (mu, perturb) in enumerate(zip(values, perturb_values)):
                                if perturb_type == "Absolute":
                                    sigma = perturb
                                elif perturb_type == "Percent":
                                    sigma = mu * perturb / 100
                                else:
                                    raise ValueError("Unknown perturb type")
                                
                                if method == "Normal":
                                    variables.append(pm.Normal(f'layer_{i}_param_{j}_{k}', mu=mu, sigma=sigma, shape=(1,)))
                                elif method == "Uniform":
                                    variables.append(pm.Uniform(f'layer_{i}_param_{j}_{k}', 
                                                                lower=np.maximum(0, mu - sigma),
                                                                upper=mu + sigma, shape=(1,)))
                                else:
                                    raise ValueError("Unknown method")
                    else:
                        continue
                
                thick_inversion = layer.thickness.get("inversion")
                thick_values = layer.thickness.get("values")
                thick_perturb_type = layer.thickness.get("perturb_type")
                thick_perturb_values = layer.thickness.get("perturb_values")

                mu = thick_values
                if not thick_inversion:
                    variables.append(pt.shape_padleft(pt.constant(mu)))
                else:
                    if thick_perturb_type == "Absolute":
                        sigma = thick_perturb_values
                    elif thick_perturb_type == "Percent":
                        sigma = mu * thick_perturb_values / 100
                    else:
                        raise ValueError("Unknown perturb type")

                    if method == "Normal":
                        variables.append(pm.Normal(f'layer_{i}_thickness', mu=mu, sigma=sigma, shape=(1,)))
                    elif method == "Uniform":
                        variables.append(pm.Uniform(f'layer_{i}_thickness', 
                                                    lower=np.maximum(0, mu - sigma),
                                                    upper=mu + sigma, shape=(1,)))
                    else:
                        raise ValueError("Unknown method")
            
            theta = pm.math.concatenate(variables, axis=0)
            pm.Potential('loglikelihood', self.likelihood(theta))
        return model

    def perform(self):
        MCmodel = self.create_MCmodel(method=self.hyperparam["prior_method"])
        with MCmodel:
            if self.hyperparam["sampling_method"] == "Metropolis":
                trace = pm.sample(draws=self.hyperparam["ndraws"],
                                  step=pm.Metropolis(tune_interval=self.hyperparam["tune_interval"]),
                                  tune=self.hyperparam["nburn"],
                                  cores=self.hyperparam["cores"],
                                  chains=self.hyperparam["chains"],
                                  discard_tuned_samples=True)
            elif self.hyperparam["sampling_method"] == "NUTS":
                trace = pm.sample(draws=self.hyperparam["ndraws"],
                                  tune=self.hyperparam["nburn"],
                                  cores=self.hyperparam["cores"],
                                  chains=self.hyperparam["chains"],
                                  discard_tuned_samples=True)

        return trace
    
