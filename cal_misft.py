import os
import sys
sys.path.append('/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance')
import glob
import numpy as np
import re
import yaml
import xarray as xr
from tqdm import tqdm
import time
from model import Model, Params
import arviz as az
from inv import InversionMC, Comply_Ps_LogLike
from post import ComplyMisfit, PsMisfit
from utils import update_model

net = "XO"
sta = sys.argv[1]
# sta = "LA21"

# IO
# =============================================================================
datadir = "/Volumes/SeisBig23/mengjie_data/Alaska.Data"
invdir  = os.path.join(datadir, "COMPLY_INV", "DATA")

comply_data_filepath = glob.glob(os.path.join(invdir, f"{net}.{sta}", "*m.dat"))
p2s_data_filepath = glob.glob(os.path.join(invdir, f"{net}.{sta}", "*P2S.dat"))
config_filepath = os.path.join(invdir, f"{net}.{sta}", "config.yml")
trace_filepath = os.path.join(invdir, f"{net}.{sta}", "trace.nc")

def post_process(trace, model, comply_data, p2s_data, wdepth, weight):
    # num_chains = len(trace.posterior['chain'])
    # num_draws = len(trace.posterior['draw'])
    chains = trace.posterior.chain.values
    draws = trace.posterior.draw.values
    num_chains = len(chains)
    num_draws = len(draws)
    freqs = comply_data[:, 0]

    comply_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, len(freqs))),
        coords={'chain': chains, 'draw': draws, 'freq': freqs},
        dims=['chain', 'draw', 'freq'],
        name='comply'
    )

    comply_misfit_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain':chains, 'draw': draws, 'comply_misfit_dim': [0]},
        dims=['chain', 'draw', 'comply_misfit_dim'],
        name='comply_misfit'
    )

    comply_chiSqr_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': chains, 'draw': draws, 'comply_chiSqr_dim': [0]},
        dims=['chain', 'draw', 'comply_chiSqr_dim'],
        name='comply_chiSqr'
    )

    p2s_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': chains, 'draw': draws, 'p2s_dim': [0]},
        dims=['chain', 'draw', 'p2s_dim'],
        name='p2s'
    )

    p2s_misfit_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': chains, 'draw': draws, 'p2s_misfit_dim': [0]},
        dims=['chain', 'draw', 'p2s_misfit_dim'],
        name='p2s_misfit'
    )

    p2s_chiSqr_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': chains, 'draw': draws, 'p2s_chiSqr_dim': [0]},
        dims=['chain', 'draw', 'p2s_chiSqr_dim'],
        name='p2s_chiSqr'
    )
    joint_chiSqr_dataarray = xr.DataArray(
        data=np.zeros((num_chains, num_draws, 1)),
        coords={'chain': chains, 'draw': draws, 'joint_chiSqr_dim': [0]},
        dims=['chain', 'draw', 'joint_chiSqr_dim'],
        name='joint_chiSqr'
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
    with tqdm(total=total_draws, desc="Post-processing") as pbar:
        for chain_idx, chain in enumerate(chains):
            for draw_idx, draw in enumerate(draws):
                for var in trace.posterior.data_vars:
                    model_paras[var] = trace.posterior[var].sel(chain=chain, draw=draw).values.item()
                
                model_new = update_model(model, list(model_paras.values()))

                comply_misfit = ComplyMisfit(model_new, comply_data, wdepth)
                comply_pred, comply_chiSqr, comply_misfit = comply_misfit._cal_misfit()
                comply_dataarray.loc[chain, draw, :] = comply_pred
                comply_misfit_dataarray.loc[chain, draw, :] = comply_misfit
                comply_chiSqr_dataarray.loc[chain, draw, :] = comply_chiSqr

                ps_misfit = PsMisfit(model_new, p2s_data)
                if p2s_data is None:
                    p2s_pred, p2s_chiSqr, p2s_misfit = None, None, None
                else:
                    p2s_pred, p2s_chiSqr, p2s_misfit = ps_misfit._cal_misfit()
                
                p2s_dataarray.loc[chain, draw, :] = p2s_pred
                p2s_misfit_dataarray.loc[chain, draw, :] = p2s_misfit
                p2s_chiSqr_dataarray.loc[chain, draw, :] = p2s_chiSqr

                w0, w1 = weight[0], weight[1]
                if p2s_chiSqr is None:
                    joint_chiSqr = comply_chiSqr
                else:
                    joint_chiSqr = w0 * comply_chiSqr + w1 * p2s_chiSqr
                joint_chiSqr_dataarray.loc[chain, draw, :] = joint_chiSqr
                pbar.update(1)
        
        ds = xr.Dataset({
            "comply": comply_dataarray,
            "comply_misfit": comply_misfit_dataarray,
            "comply_chiSqr": comply_chiSqr_dataarray,
            "p2s": p2s_dataarray,
            "p2s_misfit": p2s_misfit_dataarray,
            "p2s_chiSqr": p2s_chiSqr_dataarray,
            "joint_chiSqr": joint_chiSqr_dataarray})
        return ds

def main():
    # Load data
    comply_data = np.loadtxt(comply_data_filepath[0])
    if len(p2s_data_filepath) <= 0:
        p2s_data = None
    else:
        p2s_data = np.loadtxt(p2s_data_filepath[0])
    
    wdepth = float(re.findall(r'(\d+)m', os.path.split(comply_data_filepath[0])[-1])[0])
    print("%s %s, water depth: %s m" % (net, sta, wdepth))

    # load model, inversion configuration
    model = Model.from_yaml(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)
    joint_hparams = config["Joint_Inversion"]
    weight = joint_hparams["weight"]

    trace = az.from_netcdf(trace_filepath)

    ds = post_process(trace, model, comply_data, p2s_data, wdepth, weight)
    ds.to_netcdf(os.path.join(invdir, f"{net}.{sta}", "posterior_raw.nc"))

if __name__ == "__main__":
    tbegin = time.time()
    main()
    tend = time.time()
    elapsed = (tend - tbegin) / 3600
    print(f"Elapsed time: {elapsed} hours")