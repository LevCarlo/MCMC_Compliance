'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-01-23 10:41:10
LastEditTime: 2024-01-30 15:14:50
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/inv_inversion/MC_Compliance-dev/post_sampling.py
'''
import os
import sys
sys.path.insert(0, "/Users/mengjie/Projects/Alaska.Proj/inv_inversion/MC_Compliance-dev")
from model import Model, VsLayer, Params
import xarray as xr
import arviz as az
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import glob
import shutil

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
    model_copy.adjust_last_layer_thickness()
    return model_copy


datadir = "/Volumes/Tect32TB/Mengjie/Alaska.Data/COMPL_INV/II"
invdir = os.path.join(datadir, "DATA")
resultdir = os.path.join(datadir, "Results", "20240124")
if not os.path.exists(resultdir):
    os.makedirs(resultdir)


# sta = "XO.LD36"
sta = sys.argv[1]
sta = "XO." + sta 
if not os.path.exists(os.path.join(resultdir, sta)):
    os.makedirs(os.path.join(resultdir, sta))

trace_file = os.path.join(invdir, sta, "trace.nc")
posterior_raw_file = os.path.join(invdir, sta, "posterior_raw.nc")
config_file = os.path.join(invdir, sta, "config.yml")
prior_file = os.path.join(invdir, sta, "trace_prior.nc")
comply_data_file = glob.glob(os.path.join(invdir, sta, "*m.dat"))[0]
p2s_data_file = glob.glob(os.path.join(invdir, sta, "*P2S.dat"))

shutil.copyfile(trace_file, os.path.join(resultdir, sta, "trace.nc"))
shutil.copyfile(posterior_raw_file, os.path.join(resultdir, sta, "posterior_raw.nc"))
shutil.copyfile(config_file, os.path.join(resultdir, sta, "config.yml"))
shutil.copyfile(prior_file, os.path.join(resultdir, sta, "trace_prior.nc"))
shutil.copyfile(comply_data_file, os.path.join(resultdir, sta, os.path.basename(comply_data_file)))
if len(p2s_data_file) > 0:
    shutil.copyfile(p2s_data_file[0], os.path.join(resultdir, sta, os.path.basename(p2s_data_file[0])))

trace = az.from_netcdf(trace_file)
posterior_raw = xr.load_dataset(posterior_raw_file)
model = Model.from_yaml(config_file)

comply_misfit = posterior_raw["comply_misfit"]
comply_chiSqr = posterior_raw["comply_chiSqr"]
comply_misfit = comply_misfit.squeeze()
comply_chiSqr = comply_chiSqr.squeeze()
p2s_misfit = posterior_raw["p2s_misfit"]
p2s_chiSqr = posterior_raw["p2s_chiSqr"]
p2s_misfit = p2s_misfit.squeeze()
p2s_chiSqr = p2s_chiSqr.squeeze()
vp2vs_sediment = posterior_raw["vp2vs_sediment"]
vp2vs_sediment = vp2vs_sediment.squeeze()


if np.all(np.isnan(p2s_misfit.values.flatten())): # Only compliance available
    comply_misfit_min = comply_misfit.min().values
    comply_min_index = np.argmin(comply_misfit.values)
    if comply_misfit_min >= 0.5:
        x_crit = 2 * comply_misfit_min
    else:
        x_crit = comply_misfit_min + 0.5
    
    index = np.where(comply_misfit.values <= x_crit)
    joint_misfit = comply_misfit
    p2s_min_index = np.nan
else:
    norm_comply_misfit = (comply_misfit - comply_misfit.min()) / (comply_misfit.max() - comply_misfit.min())
    norm_p2s_misfit = (p2s_misfit - p2s_misfit.min()) / (p2s_misfit.max() - p2s_misfit.min())
    joint_misfit = (norm_comply_misfit + norm_p2s_misfit) / 2

    joint_misfit_min = joint_misfit.min().values
    x_crit = joint_misfit_min + 0.5
    index = np.where(joint_misfit.values <= x_crit)

    joint_min_index = np.where(joint_misfit.values == joint_misfit_min)
    comply_min_index = np.argmin(comply_misfit.values)
    p2s_min_index = np.argmin(p2s_misfit.values)
    
post_samples = np.vstack((index[0], index[1],
                          joint_misfit.values[index], 
                          comply_misfit.values[index], 
                          p2s_misfit.values[index],
                          comply_chiSqr.values[index],
                          p2s_chiSqr.values[index]))

post_samples_file = os.path.join(resultdir, sta, "post_samples.dat")
np.savetxt(post_samples_file, post_samples.T, fmt="%d %d %.6f %.6f %.6f %.6f %.6f",
           header="chain draw joint_misfit comply_misfit p2s_misfit comply_chiSqr p2s_chiSqr")

post_vp2vs_sediment = vp2vs_sediment.values[index]
post_vp2vs_sediment_file = os.path.join(resultdir, sta, "post_vp2vs_sediment.dat")
np.savetxt(post_vp2vs_sediment_file, post_vp2vs_sediment, fmt="%.4f")

selected_params = [trace.posterior[var].values.squeeze()[index] for var in trace.posterior.data_vars]
selected_params = np.array(selected_params)
post_params = np.vstack((index[0], index[1], *selected_params))
selected_models_file = os.path.join(resultdir, sta, "post_params.dat")
np.savetxt(selected_models_file, post_params.T, fmt="%d %d " + " ".join(["%.5f"] * len(selected_params)),
           header="chain draw " + " ".join(trace.posterior.data_vars))

# Calculate averaged model
weight = np.exp(-joint_misfit.values[index])
weight /= weight.sum()
ave_params = np.average(post_params[2:, :], axis=1, weights=weight)
np.savetxt(os.path.join(resultdir, sta, "ave_params.dat"), ave_params,
           fmt="%.6f",
           header=" ".join(trace.posterior.data_vars))
           
model_paras = {}
for i, layer in enumerate(model.layers):
    for attr_name, param_idx in zip(["vs", "vp", "rho"], [0, 1, 2]):
        param = getattr(layer, attr_name)
        if isinstance(param, Params):
            for j, value in enumerate(param.get("values")):
                model_paras[f"layer_{i}_param_{param_idx}_{j}"] = value
        
    model_paras[f"layer_{i}_thickness"] = layer.thickness.get("values")

for var, value in zip(trace.posterior.data_vars, ave_params):
    model_paras[var] = np.round(value, 5)
ave_model = update(model, list(model_paras.values()))
z, vs, _, _ = ave_model.combine_layers(boundary_flag=True)

vs_array = np.zeros((len(index[0]), len(z)))

for i in tqdm(range(len(index[0]))):
    for var, value in zip(trace.posterior.data_vars, selected_params[:, i]):
        model_paras[var] = value
    new_model = update(model, list(model_paras.values()))
    zi, vsi, _, _ = new_model.combine_layers(boundary_flag=True)
    vsi_resample = np.interp(z, zi, vsi)
    vs_array[i, :] = vsi_resample

vs_min, vs_max = vs_array.min(axis=0), vs_array.max(axis=0)
vs_std = vs_array.std(axis=0)

vs_ave = np.vstack((z, vs, vs_min, vs_max, vs_std)).T
unique, counts = np.unique(z, return_counts=True)
duplicates = unique[counts > 1][0]
duplicates_index = np.where(z == duplicates)[0]
vs_ave[duplicates_index, 2:] = np.nan
vs_ave_file = os.path.join(resultdir, sta, "vs_ave.dat")
np.savetxt(vs_ave_file, vs_ave, fmt="%.6f %.6f %.6f %.6f %.6f",
           header="z vs vs_min vs_max vs_std")

with h5py.File(os.path.join(resultdir, sta, "post_samples.h5"), "w") as f:
    f.create_dataset("post_samples", data=post_samples)
    f.create_dataset("post_params", data=post_params)
    f.create_dataset("ave_params", data=ave_params)
    f.create_dataset("vs_array", data=vs_array)
    f.create_dataset("vs_ave", data=vs_ave)
    f.create_dataset("post_vp2vs_sediment", data=post_vp2vs_sediment)




    
