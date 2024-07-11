'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-06-11 12:56:39
LastEditTime: 2024-06-20 11:57:48
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/post_process.py
'''

import os
import sys
sys.path.append('/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance')
import glob
import numpy as np
import shutil
import arviz as az
import xarray as xr
import pickle
import h5py
from model import Model
from post import Posterior, SimpleModel


sta = sys.argv[1]
# sta = "XO.LT02"

# IO
# =============================================================================
datadir = "/Volumes/SeisBig23/mengjie_data/Alaska.Data/COMPLY_INV"
invdir = os.path.join(datadir, "DATA")
resultdir = os.path.join(datadir, "RESULTS", "20240620")
if not os.path.exists(resultdir):
    os.makedirs(resultdir, exist_ok=True)

if not os.path.exists(os.path.join(resultdir, sta)):
    os.makedirs(os.path.join(resultdir, sta), exist_ok=True)

config_file = os.path.join(invdir, sta, "config.yml")
trace_prior_file = os.path.join(invdir, sta, "trace_prior.nc")
trace_posterior_file = os.path.join(invdir, sta, "trace.nc")
posterior_raw_file = os.path.join(invdir, sta, "posterior_raw.nc")
comply_data_file = glob.glob(os.path.join(invdir, sta, "*m.dat"))[0]
p2s_data_file = glob.glob(os.path.join(invdir, sta, "*P2S.dat"))

shutil.copy(trace_prior_file, os.path.join(resultdir, sta, "trace_prior.nc"))
shutil.copy(config_file, os.path.join(resultdir, sta, "config.yml"))
shutil.copy(trace_posterior_file, os.path.join(resultdir, sta, "trace.nc"))
shutil.copy(posterior_raw_file, os.path.join(resultdir, sta, "posterior_raw.nc"))
shutil.copyfile(comply_data_file, os.path.join(resultdir, sta, os.path.basename(comply_data_file)))
if len(p2s_data_file) > 0:
    shutil.copyfile(p2s_data_file[0], os.path.join(resultdir, sta, os.path.basename(p2s_data_file[0])))
# =============================================================================

model = Model.from_yaml(os.path.join(resultdir, sta, "config.yml"))
trace_prior = az.from_netcdf(os.path.join(resultdir, sta, "trace_prior.nc"))
trace_posterior = az.from_netcdf(os.path.join(resultdir, sta, "trace.nc"))
posterior_raw = xr.load_dataset(os.path.join(resultdir, sta, "posterior_raw.nc"))


# Obtain enhanced posterior distribution
print("Obtaining enhanced posterior distribution...")
posterior = Posterior(trace=trace_posterior, model=model, posterior_stats=posterior_raw)
posterior_enhanced = posterior._evaluate()
chain_index_accept = posterior_enhanced["chain_index_accept"]
draw_index_accept = posterior_enhanced["draw_index_accept"]
with h5py.File(os.path.join(resultdir, sta, "posterior_enhanced.h5"), "w") as f:
    for key, value in posterior_enhanced.items():
        f.create_dataset(key, data=value)
    
    for var in trace_posterior.posterior.data_vars:
        f.create_dataset(var, data=trace_posterior.posterior[var].values.squeeze()[chain_index_accept, draw_index_accept])


model0_ave_dict = posterior._estimate_direct()
with open(os.path.join(resultdir, sta, "model0_ave.pkl"), "wb") as f:
    pickle.dump(model0_ave_dict, f)

# model1_ave_dict = posterior._estimate_indirect()
# with open(os.path.join(resultdir, sta, "model1_ave.pkl"), "wb") as f:
#     pickle.dump(model1_ave_dict, f)

# Extract key properties of the model for plotiing prior and posterior distributions
print("Extracting key properties of the prior model ensembles...")
model_prior = SimpleModel(trace=trace_prior, model=model)
model_prior_dataset = model_prior._extract()
model_prior_dataset.to_netcdf(os.path.join(resultdir, sta, "model_prior.nc"))

print("Extracting key properties of the posterior model ensembles...")
model_posterior = SimpleModel(trace=trace_posterior, model=model)
model_posterior_dataset = model_posterior._extract()
model_posterior_dataset.to_netcdf(os.path.join(resultdir, sta, "model_posterior.nc"))

# model1_ave_dict = posterior._estimate_indirect()
# with open(os.path.join(resultdir, sta, "model1_ave.pkl"), "wb") as f:
#     pickle.dump(model1_ave_dict, f)


