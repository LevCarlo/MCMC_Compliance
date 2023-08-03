'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-07-07 13:56:11
LastEditors: Mengjie Zheng
LastEditTime: 2023-07-26 09:30:53
FilePath: /Projects/Alaska.Proj/Inversion/comply_inv_master/run_station.py
Description: 
'''
import os
import sys
import glob
from model import Model
from inv import ComplyLogLikelihood, ComplyInversion
import numpy as np
import shutil
import re

def main():
    datadir = "/Volumes/Tect32TB/Mengjie/Alaska.Data"
    invdir  = os.path.join(datadir, "COMPL_INV", "II", "DATA")
    # workdir = "/Users/mengjie/Projects/Alaska.Proj/Inversion/comply_inv/inversion"

    # sta = "LA32"
    net = "XO"
    sta = sys.argv[1]

    # Load data
    data_filepath = glob.glob(os.path.join(invdir, f"{net}.{sta}", "*.dat"))
    data = np.loadtxt(data_filepath[0])

    wdepth = float(re.findall(r'(\d+)m', os.path.split(data_filepath[0])[-1])[0])
    print(f"Sta {net}.{sta}, Water depth: {wdepth} m")

    # configfile = os.path.join(invdir, f"{net}.{sta}", "config.new.yml")
    configfile = os.path.join(invdir, f"{net}.{sta}", "config.yml")

    # Load model
    model = Model.from_yaml(configfile)

    # Load inv config
    inv = ComplyInversion.from_yaml(configfile)

    # Build lieklihood function
    loglikelihood = ComplyLogLikelihood(model=model, data=data, wdepth=wdepth)
    inv.likelihood = loglikelihood
    inv.model = model

    # Run inversion
    trace = inv.perform()
    trace.to_netcdf(os.path.join(invdir, f"{net}.{sta}", "trace.nc"))

if __name__ == "__main__":
    main()






