'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-10-09 12:31:24
LastEditTime: 2024-04-20 11:50:48
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/scripts/run.py
'''
import sys
sys.path.append('/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance')

import os
import glob
import numpy as np
import re
from model import Model, Params
from inv import Comply_Ps_LogLike, InversionMC
from tqdm import tqdm
from compliance import ncomp_fortran, cal_Ps_delay
import xarray as xr
import arviz as az
import time
import yaml

def update(model, inpara):
    model_copy = model.clone()
    param_index = 0
    for i, layer in enumerate(model_copy.layers):
        new_params = {}
        for attr_name in ["vs", "vp", "rho"]:
            param = getattr(layer, attr_name)
            if isinstance(param, Params):
                length = len(param.values)
                new_params[attr_name] = inpara[param_index:param_index+length]
                param_index += length
        new_params["thickness"] = inpara[param_index]
        param_index += 1
        layer.update(**new_params)
    return model_copy

def evaluate_compliance_misfit(model, comply_data, wdepth):
    freqs, ncomp, ncomp_err = comply_data[:, 0], comply_data[:, 1], comply_data[:, 2]
    layer_model = model.to_layer_model()
    layer_model[:, [1, 3]] = layer_model[:, [3, 1]]
    layer_model[:, 0] *= 1000
    ncomp_pred = ncomp_fortran(wdepth, freqs, layer_model)
    chiSqr = np.sum(((ncomp - ncomp_pred) / ncomp_err) ** 2)
    misfit = np.sqrt(chiSqr / len(ncomp))
    return ncomp_pred, chiSqr, misfit

def evaluate_p2s_misfit(model, p2s_data):
    p2s, p2s_err = p2s_data[0], p2s_data[1]
    sediment_layer = model.layers[0]
    sediment_thickness = sediment_layer.thickness.get("values")

    _, vs, vp, _ = sediment_layer.create_model()
    vsi, vpi = np.average(vs), np.average(vp)
    p2s_pred = cal_Ps_delay(vsi, vpi, sediment_thickness)
    chiSqr = ((p2s - p2s_pred) / p2s_err) ** 2
    if chiSqr > 10:
        chiSqr = np.sqrt(10 * chiSqr)
    misfit = np.sqrt(chiSqr)
    return p2s_pred, chiSqr, misfit

def process(trace, model, comply_data, p2s_data, wdepth):
    num_chains = len(trace.posterior['chain'])
    num_draws = len(trace.posterior['draw'])
    freqs = comply_data[:, 0]

    comply_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, len(freqs))),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'freqs': freqs},
        dims=['chain', 'draw', 'freqs'],
        name='comply'
    )

    comply_misfit_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'comply_misfit_dim': [0]},
        dims=['chain', 'draw', 'comply_misfit_dim'],
        name='comply_misfit'
    )

    comply_chiSqr_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'comply_chiSqr_dim': [0]},
        dims=['chain', 'draw', 'comply_chiSqr_dim'],
        name='comply_chiSqr'
    )

    p2s_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'p2s_dim': [0]},
        dims=['chain', 'draw', 'p2s_dim'],
        name='p2s'
    )

    p2s_misfit_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'p2s_misfit_dim': [0]},
        dims=['chain', 'draw', 'p2s_misfit_dim'],
        name='p2s_misfit'
    )

    p2s_chiSqr_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'p2s_chiSqr_dim': [0]},
        dims=['chain', 'draw', 'p2s_chiSqr_dim'],
        name='p2s_chiSqr'
    )
    joint_chiSqr_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'joint_chiSqr_dim': [0]},
        dims=['chain', 'draw', 'joint_chiSqr_dim'],
        name='joint_chiSqr'
    )

    vp2vs_sediment_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': range(num_chains), 'draw': range(num_draws), 'vp2vs_sediment_dim': [0]},
        dims=['chain', 'draw', 'vp2vs_sediment_dim'],
        name='vp2vs_sediment'
    )




    model_paras = {}
    for i, layer in enumerate(model.layers):
        for attr_name, param_idx in zip(["vs", "vp", "rho"], [0, 1, 2]):
            param = getattr(layer, attr_name)
            if isinstance(param, Params):
                for j, value in enumerate(param.get("values")):
                    model_paras[f"layer_{i}_param_{param_idx}_{j}"] = value
        
        model_paras[f"layer_{i}_thickness"] = layer.thickness.get("values")

    total_draws = num_chains * num_draws
    with tqdm(total=total_draws, desc="Processing") as pbar:
        for chain in range(num_chains):
            for draw in range(num_draws):
                for var in trace.posterior.data_vars:
                    model_paras[var] = trace.posterior[var].sel(chain=chain, draw=draw).values.item()
                
                new_model = update(model, list(model_paras.values()))
                sediment_layer = new_model.layers[0]
                _, vs, vp, _ = sediment_layer.create_model()
                vsi, vpi = np.average(vs), np.average(vp)
                vp2vs_sediment_dataarray.loc[{'chain': chain, 'draw': draw}] = vpi / vsi

                ncomp_pred, ncomp_chiSqr, comply_misfit = evaluate_compliance_misfit(new_model, comply_data, wdepth)
                if p2s_data is None:
                    p2s_pred, p2s_chiSqr, p2s_misfit = None, None, None
                else:
                    p2s_pred, p2s_chiSqr, p2s_misfit = evaluate_p2s_misfit(new_model, p2s_data)
                # vs_dataarray.loc[{'chain': chain, 'draw': draw}] = vs
                comply_dataarray.loc[{'chain': chain, 'draw': draw}] = ncomp_pred
                comply_misfit_dataarray.loc[{'chain': chain, 'draw': draw}] = comply_misfit
                comply_chiSqr_dataarray.loc[{'chain': chain, 'draw': draw}] = ncomp_chiSqr
                p2s_dataarray.loc[{'chain': chain, 'draw': draw}] = p2s_pred
                p2s_misfit_dataarray.loc[{'chain': chain, 'draw': draw}] = p2s_misfit
                p2s_chiSqr_dataarray.loc[{'chain': chain, 'draw': draw}] = p2s_chiSqr
                if p2s_chiSqr is None:
                    joint_chiSqr_dataarray.loc[{'chain': chain, 'draw': draw}] = ncomp_chiSqr
                else:
                    joint_chiSqr_dataarray.loc[{'chain': chain, 'draw': draw}] = ncomp_chiSqr + 10 * p2s_chiSqr
                pbar.update(1)
    return comply_dataarray, comply_misfit_dataarray, comply_chiSqr_dataarray, \
              p2s_dataarray, p2s_misfit_dataarray, p2s_chiSqr_dataarray, joint_chiSqr_dataarray, \
                    vp2vs_sediment_dataarray


def main(inverse_flag=True):
    datadir = "/Volumes/Tect32TB/Mengjie/Alaska.Data"
    invdir  = os.path.join(datadir, "COMPL_INV", "II", "DATA")

    net = "XO"
    sta = sys.argv[1]
    # sta = "WS75"

    # Load data
    comply_data_filepath = glob.glob(os.path.join(invdir, f"{net}.{sta}", "*m.dat"))
    comply_data = np.loadtxt(comply_data_filepath[0])
    p2s_data_filepath = glob.glob(os.path.join(invdir, f"{net}.{sta}", "*P2S.dat"))
    if len(p2s_data_filepath) <= 0:
        p2s_data = None
    else:
        p2s_data = np.loadtxt(p2s_data_filepath[0])
    
    wdepth = float(re.findall(r'(\d+)m', os.path.split(comply_data_filepath[0])[-1])[0])
    print(f"Sta {net}.{sta}, Water depth: {wdepth} m")

    config_filepath = os.path.join(invdir, f"{net}.{sta}", "config.yml")

    # Load model
    model = Model.from_yaml(config_filepath)
    # Load inv config
    inv = InversionMC.from_yaml(config_filepath)
    # Load weights
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)
    joint_hparams = config["Joint_Inversion"]


    if inverse_flag:
        likelihood = Comply_Ps_LogLike(model=model, wdepth=wdepth,
                                       comply_data=comply_data, p2s_data=p2s_data,
                                       weight=joint_hparams["weight"],
                                       inverse=True)
    else:
        likelihood = Comply_Ps_LogLike(model=model, wdepth=wdepth,
                                       comply_data=comply_data, p2s_data=p2s_data, 
                                       weight=joint_hparams["weight"],
                                       inverse=False)
    
    inv.likelihood = likelihood
    inv.model = model

    # Run inversion
    trace = inv.perform()
    # trace = az.from_netcdf(os.path.join(invdir, f"{net}.{sta}", "trace.nc"))

    if inverse_flag:
        trace.to_netcdf(os.path.join(invdir, f"{net}.{sta}", "trace.nc"))
    else:
        trace.to_netcdf(os.path.join(invdir, f"{net}.{sta}", "trace_prior.nc"))

    # Process trace
    if inverse_flag:
        comply_dataarray, comply_misfit_dataarray, comply_chiSqr_dataarray, p2s_dataarray, \
            p2s_misfit_dataarray, p2s_chiSqr_dataarray, joint_chiSqr_dataarray, vp2vs_sediment_dataarray = \
                process(trace, model, comply_data, p2s_data, wdepth)
        ds = xr.Dataset({
            'comply': comply_dataarray,
            'comply_misfit': comply_misfit_dataarray,
            'comply_chiSqr': comply_chiSqr_dataarray,
            'p2s': p2s_dataarray,
            'p2s_misfit': p2s_misfit_dataarray,
            'p2s_chiSqr': p2s_chiSqr_dataarray,
            'joint_chiSqr': joint_chiSqr_dataarray,
            'vp2vs_sediment': vp2vs_sediment_dataarray}
            )
        
        ds.to_netcdf(os.path.join(invdir, f"{net}.{sta}", "posterior_raw.nc"))

        # Posterior sampling
        # normalized_comply_misfit = (comply_misfit_dataarray - np.min(comply_misfit_dataarray)) / \
        #     (np.max(comply_misfit_dataarray) - np.min(comply_misfit_dataarray))
        # normalized_p2s_misfit = (p2s_misfit_dataarray - np.min(p2s_misfit_dataarray)) / \
        #     (np.max(p2s_misfit_dataarray) - np.min(p2s_misfit_dataarray))
        # if np.all(np.isnan(p2s_misfit_dataarray.values.flatten())):
        #     joint_misfit_data = normalized_comply_misfit

        # else:
        #     joint_misfit_data = (normalized_comply_misfit + normalized_p2s_misfit) / 2
        
        # joint_misfit = joint_misfit_data.values.squeeze()   
        # sorted_idx = np.argsort(joint_misfit, axis=None)
        # # number_elements = int(0.5 * len(sorted_idx))
        # selected_idx = sorted_idx[:]
        # multi_dim_idx = np.unravel_index(selected_idx, joint_misfit.shape)

        # f = open(os.path.join(invdir, f"{net}.{sta}", "selected_chain_draw.joint.dat"), "w")
        # g = open(os.path.join(invdir, f"{net}.{sta}", "selected_params.joint.dat"), "w")
        # f.write("# chain draw joint_misfit comply_misfit p2s_misfit comply_chiSqr p2s_chiSqr\n")
        # header = [var for var in trace.posterior.data_vars]
        # header = "# chain draw " + " ".join(header) + "\n"
        # g.write(header)
        # total_iterations = len(multi_dim_idx[0])
        # for i in tqdm(range(total_iterations)):
        #     ichain = multi_dim_idx[0][i]
        #     idraw = multi_dim_idx[1][i]
        #     jmisfit = joint_misfit[ichain, idraw]
        #     cmisfit = comply_misfit_dataarray[ichain, idraw, 0].values
        #     cchiSqr = comply_chiSqr_dataarray[ichain, idraw, 0].values
        #     pmisfit = p2s_misfit_dataarray[ichain, idraw, 0].values
        #     pchiSqr = p2s_chiSqr_dataarray[ichain, idraw, 0].values
        #     f.write("%d %d %.3f %.3f %.3f %.3f %.6f\n" %
        #             (ichain, idraw, jmisfit, cmisfit, pmisfit, cchiSqr, pchiSqr))
            
        #     content = [trace.posterior.sel(chain=ichain, draw=idraw)[var].values[0] for var in trace.posterior.data_vars]
        #     content = " ".join([str(ichain), str(idraw)] + ["%.6f" % x for x in content]) + "\n"
        #     g.write(content)
            
        # f.close()
        # g.close()

if __name__ == "__main__":
    start_time = time.time()
    main(inverse_flag=False)
    main(inverse_flag=True)
    elapsed_time = (time.time() - start_time) / 3600
    print("Elapsed time: %.2f hours" % elapsed_time)
