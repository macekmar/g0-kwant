import kwant
import numpy as np

def calc_transmission(syst, engs):
    N = len(syst.leads)
    trans = []

    scalar = False
    if np.isscalar(engs):
        scalar = True
        engs = [engs]

    for e in engs:
        smatrix = kwant.smatrix(syst, e)
        trans.append([[smatrix.transmission(j,i) for i in range(N)] for j in range(N)])
    
    if scalar:
        return trans[0]
    else:
        return np.array(trans)