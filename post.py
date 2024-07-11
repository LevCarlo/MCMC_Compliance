'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-06-10 10:29:45
LastEditTime: 2024-06-20 11:47:42
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/post.py
'''
import sys
sys.path.append('/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance')
import numpy as np
from compliance import ncomp_fortran, cal_Ps_delay
from model import Model, Params
import arviz as az
import xarray as xr
from tqdm import tqdm
import h5py

class ComplyMisfit:
    def __init__(self, model, data_obs, wdepth):
        self.model = model
        self.data_obs = data_obs
        self.wdepth = wdepth
    
    def _cal_misfit(self):
        freqs, ncomp_obs, ncomp_err = self.data_obs[:, 0], self.data_obs[:, 1], self.data_obs[:, 2]
        layer_model = self.model.to_layer_model() # z(in meter), vs, vp, rho
        layer_model[:, [1, 3]] = layer_model[:, [3, 1]]
        layer_model[:, 0] *= 1000
        ncomp_pred = ncomp_fortran(self.wdepth, freqs, layer_model)
        # chiSqr = np.sum((ncomp_pred - ncomp_obs)**2 / ncomp_err**2)
        chiSqr = np.sum(((ncomp_pred - ncomp_obs) / ncomp_err)**2)
        misfit = np.sqrt(chiSqr / len(ncomp_obs))
        return ncomp_pred, chiSqr, misfit

class PsMisfit:
    def __init__(self, model, data_obs):
        self.model = model
        self.data_obs = data_obs
    
    def _cal_misfit(self):
        Ps_obs, Ps_err = self.data_obs[0], self.data_obs[1]
        sediment_layer = self.model.layers[0]
        sediment_thickness = sediment_layer.thickness.values
        _, vs, vp, _ = sediment_layer.create_model()
        vs_ave, vp_ave = np.mean(vs), np.mean(vp)
        Ps_pred = cal_Ps_delay(vs_ave, vp_ave, sediment_thickness)
        chiSqr = np.sum((Ps_pred - Ps_obs)**2 / Ps_err**2)
        misfit = np.sqrt(chiSqr)
        return Ps_pred, chiSqr, misfit

class Posterior:
    def __init__(self, trace, model, posterior_stats):
        self.trace = trace
        self.model = model
        self.posterior_stats = posterior_stats
    
    def update_model(self, inpara):
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
    
    def _evaluate(self):
        """
        Evaluate the misfit of the posterior samples,
        and return the indices of the enhanced samples.
        """
        comply_misfit = self.posterior_stats["comply_misfit"].squeeze()
        comply_chiSqr = self.posterior_stats["comply_chiSqr"].squeeze()
        p2s_misfit = self.posterior_stats["p2s_misfit"].squeeze()
        p2s_chiSqr = self.posterior_stats["p2s_chiSqr"].squeeze()


        # Only compliance available
        if np.all(np.isnan(p2s_misfit.values.flatten())):
            comply_misfit_min = comply_misfit.min().values
            if comply_misfit_min >= 0.5:
                x_crit = 2 * comply_misfit_min
            else:
                x_crit = comply_misfit_min + 0.5
            
            index_accept = np.where(comply_misfit.values <= x_crit)
            joint_misfit = comply_misfit
            Ps_min_index = np.nan
        
        else:
            comply_misfit_norm = (comply_misfit - comply_misfit.min()) / (comply_misfit.max() - comply_misfit.min())
            p2s_misfit_norm = (p2s_misfit - p2s_misfit.min()) / (p2s_misfit.max() - p2s_misfit.min())
            joint_misfit = (comply_misfit_norm + p2s_misfit_norm) / 2

            joint_misfit_min = joint_misfit.min().values
            x_crit = joint_misfit_min + 0.5
            index_accept = np.where(joint_misfit.values <= x_crit)

        samples_accept_dict = {
            "chain_index_accept": index_accept[0],
            "draw_index_accept": index_accept[1],
            "joint_misfit_accept": joint_misfit.values[index_accept],
            "comply_misfit_accept": comply_misfit.values[index_accept],
            "p2s_misfit_accept": p2s_misfit.values[index_accept],
            "comply_chiSqr_accept": comply_chiSqr.values[index_accept],
            "p2s_chiSqr_accept": p2s_chiSqr.values[index_accept]
        }

        return samples_accept_dict

    def _estimate_direct(self):
        """
        Average the model parameters of the enhanced samples
        """
        samples_accept_dict = self._evaluate()
        index_accept = (samples_accept_dict["chain_index_accept"], samples_accept_dict["draw_index_accept"])
        joint_misfit_accept = samples_accept_dict["joint_misfit_accept"]
        f = lambda x: np.where(x <= 1, 1 / x, np.exp(1 - x))
        weight = f(joint_misfit_accept)
        weight /= weight.sum()

        # Calculate the averaged model parameters (not velocity profile itself)
        params_accept = [self.trace.posterior[var].values.squeeze()[index_accept] for var in self.trace.posterior.data_vars]
        params_accept = np.array(params_accept)
        params_ave = np.average(params_accept, axis=1, weights=weight)
        params_ave_dict = {}
        for var_name, value in zip(self.trace.posterior.data_vars, params_ave):
            params_ave_dict[var_name] = value
    
        model_paras = {}
        for i, layer in enumerate(self.model.layers):
            for attr_name, param_idx in zip(["vs", "vp", "rho"], [0, 1, 2]):
                param = getattr(layer, attr_name)
                if isinstance(param, Params):
                    for j, value in enumerate(param.get("values")):
                        model_paras[f"layer_{i}_param_{param_idx}_{j}"] = value
            
            model_paras[f"layer_{i}_thickness"] = layer.thickness.get("values")
        
        for key, value in zip(self.trace.posterior.data_vars, params_ave):
            model_paras[key] = value
        
        model_ave = self.update_model(list(model_paras.values()))
        z, vs, vp, rho = model_ave.combine_layers(boundary_flag=True)

        params_ave_dict["velocity"] = np.vstack([z, vs, vp, rho])

        return params_ave_dict


        # model_ave_data = {"Params": params_ave_dict, "Velocity": np.vstack([z, vs, vp, rho])}

        # # obtain the thickness of layer 1 (sediment layer) 
        # # ===================================================================
        # vs_dataarray = xr.DataArray(
        #     data=np.zeros((len(index_accept[0]), len(index_accept[1]), len(z))),
        #     coords={'chains': range(len(index_accept[0])), 'draws': range(len(index_accept[1])), 'depths': z},
        #     dims=['chains', 'draws', 'depths'],
        #     name='vs'
        # )
        # vp_dataarray = xr.DataArray(
        #     data=np.zeros((len(index_accept[0]), len(index_accept[1]), len(z))),
        #     coords={'chains': range(len(index_accept[0])), 'draws': range(len(index_accept[1])), 'depths': z},
        #     dims=['chains', 'draws', 'depths'],
        #     name='vp'
        # )
        # rho_dataarray = xr.DataArray(
        #     data=np.zeros((len(index_accept[0]), len(index_accept[1]), len(z))),
        #     coords={'chains': range(len(index_accept[0])), 'draws': range(len(index_accept[1])), 'depths': z},
        #     dims=['chains', 'draws', 'depths'],
        #     name='rho'
        # )

        # with tqdm(total=len(index_accept[0]), desc="Interpolating velocity profile") as pbar:
        #     for i in range(len(index_accept[0])):
        #         ichain, idraw = index_accept[0][i], index_accept[1][i]
        #         params = [self.trace.posterior[var].values.squeeze()[ichain, idraw] for var in self.trace.posterior.data_vars]
        #         for key, value in zip(self.trace.posterior.data_vars, params):
        #             model_paras[key] = value
            
        #         model_new = self.update_model(list(model_paras.values()))
        #         zz, vs, vp, rho = model_new.combine_layers(boundary_flag=True)
        #         vs_new = np.interp(z, zz, vs)
        #         vp_new = np.interp(z, zz, vp)
        #         rho_new = np.interp(z, zz, rho)
        #         vs_dataarray.loc[ichain, idraw, :] = vs_new
        #         vp_dataarray.loc[ichain, idraw, :] = vp_new
        #         rho_dataarray.loc[ichain, idraw, :] = rho_new
        #         pbar.update()
        

        # ds = xr.Dataset(
        #     {
        #         'vs': vs_dataarray,
        #         'vp': vp_dataarray,
        #         'rho': rho_dataarray
        #     }
        # )

        # return model_ave_data, ds

        # Determine the misfit-averaged thickness of layer 1 (sediment layer)
        # ===================================================================
        # target_name = [var_name for var_name in list(self.trace.posterior.data_vars) if "thickness" in var_name]
        # if len(target_name) < 0:
        #     raise ValueError("Failed to estimate sediment thickness")
        # target_name = target_name[0]
        # thickness_accept = self.trace.posterior[target_name].values.squeeze()[index_accept]
        # joint_misfit_accept = joint_misfit.values[index_accept]
        # f = lambda x: np.where(x <= 1, 1 / x, np.exp(1 - x))
        # weight = f(joint_misfit_accept)
        # weight /= weight.sum()
        # thickness_ave = np.average(thickness_accept, weights=weight)
        # ===================================================================
        # model_paras = {}
        # for i, layer in enumerate(self.model.layers):
        #     for attr_name, param_idx in zip(["vs", "vp", "rho"], [0, 1, 2]):
        #         param = getattr(layer, attr_name)
        #         if isinstance(param, Params):
        #             for j, value in enumerate(param.get("values")):
        #                 model_paras[f"layer_{i}_param_{param_idx}_{j}"] = value
            
        #     model_paras[f"layer_{i}_thickness"] = layer.thickness.get("values")
        
        # model_paras[target_name] = thickness_ave
        # model_refer = self.update_model(list(model_paras.values()))
        # z_refer, _, _, _ = model_refer.combine_layers(boundary_flag=True)
    
        # return thickness_ave, z_refer

    def _estimate_indirect(self):
        """
        Avearge the converted velocity profile of the enhanced samples
        """
        samples_accept_dict = self._evaluate()
        index_accept = (samples_accept_dict["chain_index_accept"], samples_accept_dict["draw_index_accept"])
        joint_misfit_accept = samples_accept_dict["joint_misfit_accept"]
        f = lambda x: np.where(x <= 1, 1 / x, np.exp(1 - x))
        weight = f(joint_misfit_accept)
        weight /= weight.sum()

        # Determine the misfit-averaged thickness of layer 1 (sediment layer)
        # ===================================================================
        target_name = [var_name for var_name in list(self.trace.posterior.data_vars) if "thickness" in var_name]
        if len(target_name) < 0:
            raise ValueError("Failed to estimate sediment thickness")
        target_name = target_name[0]
        thickness_accept = self.trace.posterior[target_name].values.squeeze()[index_accept]
        thickness_ave = np.average(thickness_accept, weights=weight)
        # ===================================================================
        model_paras = {}
        for i, layer in enumerate(self.model.layers):
            for attr_name, param_idx in zip(["vs", "vp", "rho"], [0, 1, 2]):
                param = getattr(layer, attr_name)
                if isinstance(param, Params):
                    for j, value in enumerate(param.get("values")):
                        model_paras[f"layer_{i}_param_{param_idx}_{j}"] = value
            
            model_paras[f"layer_{i}_thickness"] = layer.thickness.get("values")
        
        model_paras[target_name] = thickness_ave
        model_refer = self.update_model(list(model_paras.values()))
        z_refer, _, _, _ = model_refer.combine_layers(boundary_flag=True)

        vs_dataarray = xr.DataArray(
            data=np.zeros((len(index_accept[0]), len(index_accept[1]), len(z_refer))),
            coords={'chains': range(len(index_accept[0])), 'draws': range(len(index_accept[1])), 'depths': z_refer},
            dims=['chains', 'draws', 'depths'],
            name='vs'
        )
        vp_dataarray = xr.DataArray(
            data=np.zeros((len(index_accept[0]), len(index_accept[1]), len(z_refer))),
            coords={'chains': range(len(index_accept[0])), 'draws': range(len(index_accept[1])), 'depths': z_refer},
            dims=['chains', 'draws', 'depths'],
            name='vp'
        )
        rho_dataarray = xr.DataArray(
            data=np.zeros((len(index_accept[0]), len(index_accept[1]), len(z_refer))),
            coords={'chains': range(len(index_accept[0])), 'draws': range(len(index_accept[1])), 'depths': z_refer},
            dims=['chains', 'draws', 'depths'],
            name='rho'
        )

        with tqdm(total=len(index_accept[0]), desc="Interpolating velocity profile") as pbar:
            for i in range(len(index_accept[0])):
                ichain, idraw = index_accept[0][i], index_accept[1][i]
                params = [self.trace.posterior[var].values.squeeze()[ichain, idraw] for var in self.trace.posterior.data_vars]
                for key, value in zip(self.trace.posterior.data_vars, params):
                    model_paras[key] = value
            
                model_new = self.update_model(list(model_paras.values()))
                zz, vs, vp, rho = model_new.combine_layers(boundary_flag=True)
                vs_new = np.interp(z_refer, zz, vs)
                vp_new = np.interp(z_refer, zz, vp)
                rho_new = np.interp(z_refer, zz, rho)
                vs_dataarray.loc[ichain, idraw, :] = vs_new
                vp_dataarray.loc[ichain, idraw, :] = vp_new
                rho_dataarray.loc[ichain, idraw, :] = rho_new
                pbar.update()
        
        dataarrays = [vs_dataarray, vp_dataarray, rho_dataarray]
        keys = ["vs", "vp", "rho"]
        stats_dict = {}
        for dataarray, key in zip(dataarrays, keys):
            max_values = dataarray.max(dim=["chains", "draws"])
            min_values = dataarray.min(dim=["chains", "draws"])
            median_values = dataarray.median(dim=["chains", "draws"])
            quantile_25 = dataarray.quantile(0.25, dim=["chains", "draws"])
            quantile_75 = dataarray.quantile(0.75, dim=["chains", "draws"])
            mean_pure_values = dataarray.mean(dim=["chains", "draws"])
            std_values = dataarray.std(dim=["chains", "draws"])
            mean_misfit_everaged_values = np.average(dataarray, axis=(0, 1), weights=weight)
            mean_misfit_everaged_dataarray = xr.DataArray(mean_misfit_everaged_values, 
                                                          dims=["depths"], 
                                                          coords={"depths": dataarray.coords["depths"]})

            stats_ds = xr.Dataset(
                {
                    "max": max_values,
                    "min": min_values,
                    "median": median_values,
                    "quantile_25": quantile_25,
                    "quantile_75": quantile_75,
                    "mean_pure": mean_pure_values,
                    "std": std_values,
                    "mean_misfit_averaged": mean_misfit_everaged_dataarray
                }
            )

            stats_dataarray = stats_ds.to_array(dim="stats")
            stats_dataarray.name = key
            stats_dict[key] = stats_dataarray
        
        return stats_dict

class SimpleModel:
    """
    The object for extracting key model properties from the trace,
    in order to plot prior and posterior distributions.
    """        
    def __init__(self, trace, model):
        self.trace = trace
        self.model = model
    
    def update_model(self, inpara):
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
    
    def _extract(self):
        chains = self.trace.posterior['chain'].values
        draws = self.trace.posterior['draw'].values

        model_paras = {}
        for i, layer in enumerate(self.model.layers):
            for attr_name, param_idx in zip(["vs", "vp", "rho"], [0, 1, 2]):
                param = getattr(layer, attr_name)
                if isinstance(param, Params):
                    for j, value in enumerate(param.get("values")):
                        model_paras[f"layer_{i}_param_{param_idx}_{j}"] = value
            
            model_paras[f"layer_{i}_thickness"] = layer.thickness.get("values")
        

        # sediment
        sediment_thickness_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), 1)),
            coords={"chain": chains, "draw": draws, "sediment_thickness": [0]},
            dims=["chain", "draw", "sediment_thickness"],
            name="sediment_thickness"
        )
        sediment_average_vs_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), 1)),
            coords={"chain": chains, "draw": draws, "sediment_average_vs": [0]},
            dims=["chain", "draw", "sediment_average_vs"],
            name="sediment_average_vs"
        )
        sediment_vp2vs_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), 1)),
            coords={"chain": chains, "draw": draws, "sediment_vp2vs": [0]},
            dims=["chain", "draw", "sediment_vp2vs"],
            name="sediment_vp2vs"
        )

        # crust
        vs_at_top_crust_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), 1)),
            coords={"chain": chains, "draw": draws, "vs_at_top_crust": [0]},
            dims=["chain", "draw", "vs_at_top_crust"],
            name="vs_at_top_crust"
        )
        crust_average_vs_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), 1)),
            coords={"chain": chains, "draw": draws, "crust_average_vs": [0]},
            dims=["chain", "draw", "crust_average_vs"],
            name="crust_average_vs"
        )

        # model samples
        z_new = np.arange(0, self.model.total_thickness + 1.0, 1.0)
        vs_samples_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), len(z_new))),
            coords={"chain": chains, "draw": draws, "depth": z_new},
            dims=["chain", "draw", "depth"],
            name="vs_samples"
        )
        vp_samples_dataarray = xr.DataArray(
            data = np.zeros((len(chains), len(draws), len(z_new))),
            coords={"chain": chains, "draw": draws, "depth": z_new},
            dims=["chain", "draw", "depth"],
            name="vp_samples"
        )

        with tqdm(total=len(chains) * len(draws), desc="Extracting model samples") as pbar:
            for ichain in chains:
                for idraw in draws:
                    sample = self.trace.posterior.sel(chain=ichain, draw=idraw)
                    for var in sample.data_vars:
                        model_paras[var] = sample[var].values.item()
                    new_model = self.update_model(list(model_paras.values()))

                    # sediment
                    sediment_layer = new_model.layers[0]
                    sediment_thickness = sediment_layer.thickness.values
                    _, vs, vp, _ = sediment_layer.create_model()
                    vp2vs = np.average(vp) / np.average(vs)
                    sediment_average_vs = np.average(vs)
                    z, vs, vp, _ = new_model.combine_layers(boundary_flag=True)
                    vs_new = np.interp(z_new, z, vs)
                    vp_new = np.interp(z_new, z, vp)

                    # crust
                    crust_layer = new_model.layers[1]
                    _, vs, vp, _ = crust_layer.create_model()
                    vs_at_top_crust = vs[0]
                    crust_average_vs = np.average(vs)

                    sediment_thickness_dataarray.loc[ichain, idraw, :] = sediment_thickness
                    sediment_average_vs_dataarray.loc[ichain, idraw, :] = sediment_average_vs
                    sediment_vp2vs_dataarray.loc[ichain, idraw, :] = vp2vs
                    vs_at_top_crust_dataarray.loc[ichain, idraw, :] = vs_at_top_crust
                    crust_average_vs_dataarray.loc[ichain, idraw, :] = crust_average_vs
                    vs_samples_dataarray.loc[ichain, idraw, :] = vs_new
                    vp_samples_dataarray.loc[ichain, idraw, :] = vp_new
                    pbar.update()
        
        model_data = xr.Dataset(
            {
                "sediment_thickness": sediment_thickness_dataarray,
                "sediment_average_vs": sediment_average_vs_dataarray,
                "sediment_vp2vs": sediment_vp2vs_dataarray,
                "vs_at_top_crust": vs_at_top_crust_dataarray,
                "crust_average_vs": crust_average_vs_dataarray,
                "vs_samples": vs_samples_dataarray,
                "vp_samples": vp_samples_dataarray
            }
        )

        return model_data



        
        

if __name__ == "__main__":
    config_file = "/Volumes/Tect32TB/Mengjie/Alaska.Data/COMPL_INV/II/DATA/XO.LA21/config.yml"
    trace_file = "/Volumes/Tect32TB/Mengjie/Alaska.Data/COMPL_INV/II/DATA/XO.LA21/trace_prior.nc"
    posterior_stats_file = "/Volumes/Tect32TB/Mengjie/Alaska.Data/COMPL_INV/II/DATA/XO.LA21/posterior_raw.nc"
    model = Model.from_yaml(config_file)
    trace = az.from_netcdf(trace_file)
    posterior_stats = xr.load_dataset(posterior_stats_file)
    posterior = Posterior(trace, model, posterior_stats)
    total_thickness, var_names = posterior._estimate_direct()




        