import numpy as np


def fermi(eng, ef, beta):
    if beta == -1:
        return np.heaviside(ef-eng, 0.5)
    else:
        # Numerical stable calculation of Fermi function
        # x > 0: exp(-x)/(exp(-x) + 1)  |  x < 0: 1/(1 + exp(x))
        e = eng-ef
        m = -(np.sign(e) - 1)/2
        p = -(np.sign(e) + 1)/2
    return np.exp(p*beta*e) / (np.exp(p*beta*e) + np.exp(m*beta*e))
