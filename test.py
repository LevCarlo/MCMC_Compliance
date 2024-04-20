'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-04-20 11:32:12
LastEditTime: 2024-04-20 11:49:37
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/scripts/test.py
'''
import sys
sys.path.append('/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance')
from compliance import ncomp_fortran
from inv import Comply_Ps_LogLike, InversionMC
from model import Model, Params

# import sys
# sys.path.append('/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance')

# import os
# import glob
# import numpy as np
# import re
# from model import Model, Params
# from inv import Comply_Ps_LogLike, InversionMC
# from tqdm import tqdm
# from forward_funcs.ncomp_fortran import ncomp_fortran
# from forward_funcs.utils import cal_Ps_delay
# import xarray as xr
# import arviz as az
# import time
# import yaml