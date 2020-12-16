# # Analytical results for self energies
#
# They are based on paper 1307.6419 (Eq. C.3, C.4) and some my calculations.
# 1307.6419 - Numerical simulations of time resolved quantum electronics, Gaury

import numpy as np
from .fermi import fermi


def SigmaER(eng):
    """Returns Σ(E)^R for a perfect lead with γ = 1.

    Eq. (C.3) in 1307.6419.
    """
    e = eng/2.0
    return np.piecewise(np.array(e, dtype=np.complex),
                        [e < -1.0, (e >= -1.0) & (e <= 1.0)], [
                            lambda e: (e + np.sqrt(e**2 - 1)),
                            lambda e: (e - 1j*np.sqrt(1 - e**2)),
                            lambda e: (e - np.sqrt(e**2 - 1))
    ])


def SigmaEL(eng, ef):
    """Returns Σ(E)^< for a perfect lead with γ = 1.

    Eq. (C.4) in 1307.6419.
    """
    e = eng/2.0
    ef = ef/2.0
    return np.piecewise(np.array(e, dtype=np.complex),
                        [(e >= -1.0) & (e <= ef)], [
                            lambda e: (2j*(np.sqrt(1 - e**2))),
                            lambda e: 0
    ])


def SigmaER_general(eng, gamma_att=1.0, gamma_lead=1.0):
    """Returns Σ(E)^R for a perfect lead with general hoppings.

    The equation is:
                    2               ______________
                |γ"|  ⎛  E         / ⎛  E  ⎞2
                ───── ⎜───── ±    /  ⎜─────⎟  − 1
                |γ'|  ⎝2|γ'|     V   ⎝2|γ'|⎠
    for E < −1 (+) or E > 1 (-) and where γ' is hopping in the leads and
    γ" is the hopping at the attachment. It is similar for −1 < E < 1.
    Half-bandwidth D is 2|γ'|.

    When we study a single quantum dot, we have to take at least 3 sites in
    Kwant:
                          γ'  γ'  γ   γ   γ' γ'
                        ┄┄┄─○───●───◎───●───○─┄┄┄

    where ● are part of the scattering region and ◎ is the quantum dot.
    So, Σ(E)^R is for the attachment between ○───● and γ" = γ'.
    If we take only the quantum dot as part of the system, then γ" = γ.
    """
    e = eng/(2.0*np.abs(gamma_lead))
    return np.abs(gamma_att)**2/np.abs(gamma_lead) * \
            np.piecewise(np.array(e, dtype=np.complex),
                        [e < -1.0, (e >= -1.0) & (e <= 1.0)], [
                            lambda e: (e + np.sqrt(e**2 - 1)),
                            lambda e: (e - 1j*np.sqrt(1 - e**2)),
                            lambda e: (e - np.sqrt(e**2 - 1))
            ])


def SigmaEL_general(eng, ef, beta, gamma_att=1.0, gamma_lead=1.0):
    """Returns Σ(E)^< for a perfect lead with general hoppings.

    It is calculated from Eq. (19) and (20) in 1307.6419:

        Σ(E)^< = -f(E)[ Σ(E)^R - Σ(E)^A ],   Σ(E)^A = [Σ(E)^R]^†
    """
    e = eng/(2.0*np.abs(gamma_lead))
    ef = ef/(2.0*np.abs(gamma_lead))
    return np.abs(gamma_att)**2/np.abs(gamma_lead) * fermi(e, ef, beta) * \
            np.piecewise(
            np.array(e, dtype=np.complex),
            [(e >= -1.0) & (e <= 1.0)], [
                lambda e: (2j*(np.sqrt(1 - e**2))),
                lambda e: 0
            ])
