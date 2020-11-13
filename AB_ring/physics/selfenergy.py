import numpy as np
import scipy as sc
from .fermi import fermi

# Analytical results from paper 1307.6419, Eq. C.3, C.4, and my calculations

## Self energy
def SER(e):
    """Returns Σ(E)^R for a perfect lead with γ = 1.

    Equation C.3 in 1307.6419"""
    e = e/2.0
    return np.piecewise(
            np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
            [   lambda e: (e + np.sqrt(e**2 - 1)),
                lambda e: (e - 1j*np.sqrt(1 - e**2)),
                lambda e: (e - np.sqrt(e**2 - 1))   ])
        
def SEL(e, ef):
    """Returns Σ(E)^< for a perfect lead with γ = 1.

    Equation C.4 in 1307.6419"""
    e = e/2.0
    ef = ef/2.0
    return np.piecewise(
            np.array(e, dtype=np.complex), [(e >= -1.0) & (e <= ef)],
            [   lambda e: (2j*(np.sqrt(1 - e**2))),
                lambda e: 0   ])

def SER_general(e, gamma_att=1.0, gamma_lead=1.0):
    """Returns Σ(E)^R for a perfect lead with general hoppings.

    Equation is:
                    2               ______________
                |γ"|  ⎛  E         / ⎛  E  ⎞2
                ───── ⎜───── ±    /  ⎜─────⎟  − 1
                |γ'|  ⎝2|γ'|    \/   ⎝2|γ'|⎠
    for E < −1 (+) or E > 1 (-) and where γ' is hopping in the leads and 
    γ" is the hopping at the attachemnt. It is similar for −1 < E < 1.
    2|γ"| can be called half-bandwidth. 
    Bandwidth of a conduction band is 4|γ"| (TODO: is it half?)  

    In Kwant we have to take at least 3 sites:

                          γ'  γ'  γ   γ   γ' γ'  
                        ┄┄┄─○───●───◎───●───○─┄┄┄

    where ● are part of the scattering region and ◎ is the quantum dot.
    So, Σ(E)^R is for attachment to ○───● and γ" = γ'.
    If we take only the quantum dot as part of the system, then γ" = γ."""            

    e = e/(2.0*np.abs(gamma_lead))
    return np.abs(gamma_att)**2/np.abs(gamma_lead)*np.piecewise(
            np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
            [   lambda e: (e + np.sqrt(e**2 - 1)),
                lambda e: (e - 1j*np.sqrt(1 - e**2)),
                lambda e: (e - np.sqrt(e**2 - 1))   ])

def SEL_general(e, ef, beta, gamma_att=1.0, gamma_lead=1.0):
    """Returns Σ(E)^< for a perfect lead with general hoppings.

    It is calculated from equation (Eq. 19 and 20 in 1307.6419):

        Σ(E)^< = -f(E)[ Σ(E)^R - Σ(E)^A ],   Σ(E)^A = [Σ(E)^R]^†
    """
    e = e/(2.0*np.abs(gamma_lead))
    ef = ef/(2.0*np.abs(gamma_lead))
    return np.abs(gamma_att)**2/np.abs(gamma_lead)*fermi(e, ef, beta)*np.piecewise(
            np.array(e, dtype=np.complex), [(e >= -1.0) & (e <= 1.0)],
            [   lambda e: (2j*(np.sqrt(1 - e**2))),
                lambda e: 0   ])
