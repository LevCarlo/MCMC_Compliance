'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-07-25 09:57:48
LastEditTime: 2023-08-03 16:10:10
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/Inversion/comply_inv_master/inv.py
'''
import os
import yaml
from model import Model
import pytensor.tensor as at
import numpy as np
import pymc as pm
from calcNCompMat import calc_ncomp_mat
import arviz as az
from typing import Type

class ComplyLogLikelihood(at.Op):
    itypes = [at.dvector]
    otypes = [at.dscalar]

    def __init__(self, model: Type[Model], data, wdepth, forward_func=calc_ncomp_mat):
        """
        Initialize the class.
        model: Model object
        data: compliance data. freq, ncomp, ncomp_err
        wdepth: water depth, in meters (positive)
        loglike: compliance forward function
        """
        self.model = model
        self.data = data
        self.wdepth = wdepth
        self.forward_func = forward_func
    
    def perform(self, node, inputs, outputs):
        theta, = inputs
        inpara = np.copy(theta)

        if np.any(inpara <= 0):
            print("Negative parameters!", inpara)
            outputs[0][0] = np.array(-np.inf)
            return
        
        if not self.check_incresing(inpara):
            outputs[0][0] = np.array(-np.inf)
            return
        
        new_model = self.update_model(inpara)
        layered_model = new_model.to_layer_model()
        # thickness (km), vs (km/s), vp (km/s), rho (g/cm^3)
        if np.any(layered_model < 0) or np.any(np.isnan(layered_model)):
            outputs[0][0] = np.array(-np.inf)
            return

        model_comply = layered_model.copy()
        model_comply[:, 0] *= 1000
        model_comply[:, 1], model_comply[:, 3] = model_comply[:, 3].copy(), model_comply[:, 1].copy()
        freqs, ncomp, ncomp_err = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        ncomp_pred = self.forward_func(self.wdepth, freqs, model_comply)
        if np.any(np.isnan(ncomp_pred)) or np.all(ncomp_pred == 0):
            outputs[0][0] = np.array(-np.inf)
            return
        
        chiSqr = np.sum(((ncomp - ncomp_pred) / ncomp_err) ** 2)
        misfit = np.sqrt(chiSqr / len(ncomp))
        chiSqr = chiSqr if chiSqr < 1e2 else np.sqrt(chiSqr * 1e2) 
        loglike = -0.5 * chiSqr

        
        
        outputs[0][0] = np.array(loglike)

    def update_model(self, inpara):
        model_copy = self.model.clone()
        param_index = 0
        for i, layer in enumerate(model_copy.layers):
            vel_values = layer.vs.get("values")

            new_vel_values = inpara[param_index:param_index + len(vel_values)]
            new_thick_values = inpara[param_index + len(vel_values)]
            layer.update(new_thick_values, new_vel_values)
            param_index += len(vel_values) + 1

        return model_copy

    def check_incresing(self, inpara):
        param_index = 0
        prev_end_value = None
        for i, layer in enumerate(self.model.layers):
            vel_values = layer.vs.get("values")
            new_vel_values = inpara[param_index:param_index + len(vel_values)]

            # Gradient must be positive
            definition_key = layer.vs.get("definition")
            if definition_key == "Gradient":
                if new_vel_values[0] > new_vel_values[1]:
                    return False
                
            if prev_end_value is not None and prev_end_value > new_vel_values[0]:
                return False
        
            prev_end_value = new_vel_values[-1]
            param_index += len(vel_values) + 1
        
        return True
        
        
    
class InversionBase:
    def __init__(self, model: Type[Model]= None, likelihood=None, hyperparameters=None):
        self.model = model
        self.likelihood = likelihood
        self.hyperparameters = hyperparameters
    
    @classmethod
    def from_yaml(cls, file_path, model: Type[Model]=None, likelihood=None):
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            hparams = config["Inversion"]
        return cls(hyperparameters=hparams, model=model, likelihood=likelihood)

class ComplyInversion(InversionBase):
    def __init__(self, model: Type[Model], likelihood, hyperparameters):
        super().__init__(model, likelihood, hyperparameters)
    
    def create_MCmodel(self, method="Uniform"):
        model = pm.Model()
        with model:
            variables = []
            for i, layer in enumerate(self.model.layers):
                vel_flag   = layer.vs.get("inversion")
                vel_values = layer.vs.get("values")
                vel_perturb_type = layer.vs.get("perturb_type")
                vel_perturb = layer.vs.get("perturb_params")

                thick_flag = layer.thickness.get("inversion")
                thick_value = layer.thickness.get("values")
                thick_perturb_type = layer.thickness.get("perturb_type")
                thick_perturb = layer.thickness.get("perturb_params")

                for j, (mu, perturb) in enumerate(zip(vel_values, vel_perturb)):
                    if vel_perturb_type=="Absolute":
                        sigma = perturb
                    elif vel_perturb_type=="Percent":
                        sigma = mu * perturb / 100
                    else:
                        raise ValueError("Unknown perturbation type!")
                    
                    if method == "Normal":
                        variables.append(pm.Normal(f'layer_{i}_param_{j}', mu=mu, sigma=sigma, shape=(1,)))
                    elif method == "Uniform":
                        variables.append(pm.Uniform(f'layer_{i}_param_{j}', 
                                                    lower=np.maximum(mu - sigma, 1e-3), 
                                                    # Avoid negative values, > 1m/s
                                                    upper=np.minimum(mu + sigma, 4.0), 
                                                    # Avoid too large values, < 4km/s
                                                    shape=(1,)))

                mu = thick_value 
                if thick_flag:
                    if thick_perturb_type=="Absolute":
                        sigma = thick_perturb
                    elif thick_perturb_type=="Percent":
                        sigma = mu * thick_perturb / 100
                    else:
                        raise ValueError("Unknown perturbation type!")
                    
                    if method == "Normal":
                        variables.append(pm.Normal(f'layer_{i}_thickness', mu=mu, sigma=sigma, shape=(1,)))
                    elif method == "Uniform":
                        variables.append(pm.Uniform(f'layer_{i}_thickness', 
                                                    # The minimum thickness is required to be larger than 1 km
                                                    lower=np.maximum(mu - sigma, 1), 
                                                    upper=mu + sigma, 
                                                    shape=(1,)))
                else:
                    variables.append(at.shape_padleft(at.constant(mu)))
                
            theta = pm.math.concatenate(variables, axis=0)
            pm.Potential('loglikelihood', self.likelihood(theta))
        
        return model

    def perform(self):
        MCmodel = self.create_MCmodel(method=self.hyperparameters["prior_method"])
        with MCmodel:
            trace = pm.sample(draws=self.hyperparameters["ndraws"],
                              step=pm.Metropolis(tune_interval=self.hyperparameters["tune_interval"]),
                              tune=self.hyperparameters["nburn"],
                              cores=self.hyperparameters["cores"],
                              chains=self.hyperparameters["chains"],
                              discard_tuned_samples=True)
            
        return trace

if __name__ == "__main__":
    pass
    