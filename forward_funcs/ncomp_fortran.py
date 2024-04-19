'''
FUNCTION ncomp_fortran.py

Translated from Wayne Crawford's MATLAB code for forward calculating normalized
compliance from synthetic Earth structure  

Original source can be found at (http://www.ipgp.fr/~crawford/Homepage/Software.html).

Stephen Mosher, Feb 2020. 
'''

#################################### IMPORTS ###################################

# The usual.
import numpy as np

# Import helper functions - raydep_ft does the heavy lifting. It's an F95
# file and it has to be compiled using F2PY from NumPy.
# from forward_funcs import gravd, raydep_ft
from . import gravd, raydep_ft
# import gravd, raydep_ft

################################### FUNCTION ###################################

def ncomp_fortran(depth, freqs, model):
  # Compute wavenumber of infragravity waves and slowness vectors.
  ω = 2 * np.pi * freqs
  k = gravd.gravd(ω, depth)
  p = k / ω

  # Compute normalized compliance.
  ncomp = np.zeros(len(ω))
  for i in range(len(p)):
    v, u, sigzz, sigzx = raydep_ft.raydep_ft(p[i], ω[i], model)
    ncomp[i] = -k[i] * v[0] / (ω[i] * sigzz[0])
  return ncomp

#################################### TESTS ####################################
if __name__ == "__main__":
  depth = 1000
  freqs = np.arange(0.003, 0.02 + 0.001, 0.001)
  model = np.array([[1000, 2.500, 4.000, 2.500],
                    [2000, 3.000, 7.000, 4.000],
                    [400, 2.500, 4.000, 1.000],
                    [1000, 3.000, 7.000, 4.000]])
  
  ncomp = ncomp_fortran(depth, freqs, model)
  print(ncomp)