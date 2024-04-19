'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2023-10-09 10:17:01
LastEditTime: 2023-10-09 10:18:04
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/inv_inversion/MC_Compliance/forward_funcs/utils.py
'''


def cal_Ps_delay(vs, vp, h):
    """
    vs: Average S-wave velocity of the sedimentary layer
    vp: Average P-wave velocity of the sedimentary layer
    h: Thickness of the sedimentary layer
    return: The delay time of the Ps converted phase
    """

    return h * (1 / vs - 1 / vp)