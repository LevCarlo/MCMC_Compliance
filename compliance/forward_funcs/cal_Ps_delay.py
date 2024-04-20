'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-04-19 19:30:35
LastEditTime: 2024-04-19 19:30:40
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/compliance/forward_funcs/cal_Ps_delay.py
'''
def cal_Ps_delay(vs, vp, h):
    """
    vs: Average S-wave velocity of the sedimentary layer
    vp: Average P-wave velocity of the sedimentary layer
    h: Thickness of the sedimentary layer
    return: The delay time of the Ps converted phase
    """

    return h * (1 / vs - 1 / vp)