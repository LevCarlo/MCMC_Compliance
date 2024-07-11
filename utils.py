'''
Author: Mengjie Zheng
Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
Date: 2024-04-22 13:26:24
LastEditTime: 2024-04-24 12:38:23
LastEditors: Mengjie Zheng
Description: 
FilePath: /Projects/Alaska.Proj/MCMC_Compliance/utils.py
'''
import numpy as np
from numba import jit
from model import Params, Model
import matplotlib.pyplot as plt

def update_model(model, params):
    """
    Update the model with the given parameters.
    Parameters
    ----------
    model : Model class
        The model to be updated.
    params : list
        The parameters to update the model.
    Returns
    -------
    Model class
        The updated model.
    """

    model_copy = model.clone()
    param_index = 0
    for i, layer in enumerate(model_copy.layers):
        new_params = {}
        for attr_name in ["vs", "vp", "rho"]:
            param = getattr(layer, attr_name)
            if isinstance(param, Params):
                length = len(param.values)
                new_params[attr_name] = params[param_index:param_index+length]
                param_index += length
        new_params["thickness"] = params[param_index]
        param_index += 1
        layer.update(**new_params)
    model_copy.adjust_last_layer_thickness()
    return model_copy

# @jit(nopython=True)
def interpolate_segment(x, ys, x_new):
    y_new = np.zeros((len(x_new), ys.shape[1]))
    
    boundary_idx = np.where(x[:-1] == x[1:])[0] + 1
    start_idx = np.concatenate(([0], boundary_idx))
    end_idx = np.concatenate((boundary_idx, [len(x)]))

    boundary_values = x[end_idx - 1]
    segment_idx= np.searchsorted(boundary_values, x_new, side='left')

    for i, (start, end) in enumerate(zip(start_idx, end_idx)):
        segment_x = x[start:end]
        segment_ys = ys[start:end]

        idx_in_segment = segment_idx == i
        nx_in_segment = x_new[idx_in_segment]

        for j in range(ys.shape[1]):
            y_new[idx_in_segment, j] = np.interp(nx_in_segment, segment_x, segment_ys[:, j])
    
    return y_new




if __name__=="__main__":
    x = np.array([0, 1, 1, 2, 3, 3, 4, 5])
    x = np.array(x)
    y = np.random.rand(len(x), 3)
    x_new = np.array([0.5, 1.0, 3.5, 3.1, 3.0, 2.5, 4.5, 5.0])
    # x_new = np.linspace(0, 5, 100)
    # x_new = np.array([5])
    y_new = interpolate_segment(x, y, x_new)
    print(x_new, y_new)
    fig, ax = plt.subplots()
    ax.plot(y[:, 0], x,  '-o')
    ax.plot(y_new[:, 0], x_new, '+')
    ax.plot(np.interp(x_new, x, y[:, 0]), x_new, 'x')
    fig.savefig("test.png")
    