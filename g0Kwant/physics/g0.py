# # Analytical results for Green functions
#
# They are based on paper 1307.6419 (Eq. C.3, C.4) and some my calculations.
# 1307.6419 - Numerical simulations of time resolved quantum electronics, Gaury

import numpy as np
import scipy as sc
from .fermi import fermi
from .sigma import *

###############################################################################
# # Energy domain

def GER00(e):
    """Returns G_{00}(E)^R for a perfect wire with γ = 1.

    Eq. (C.7) in 1307.6419.
    """
    e = e/2.0
    return np.piecewise(np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
            [   lambda e: -1/2.0 / np.sqrt(e**2-1),
                lambda e: 1/2.0 /( 1j*np.sqrt(1-e**2) ),
                lambda e: 1/2.0 / np.sqrt(e**2-1) ])


def GEL00(e, ef):
    """Returns G_{00}(E)^< for a perfect wire with γ = 1.

    Eq. (C.8) in 1307.6419.
    """
    e = e/2.0
    ef = ef/2.0
    return np.piecewise(np.array(e, dtype=np.complex), [(e >= -1.0) & (e <= ef)],
            [   lambda e: (1j / (np.sqrt(1-e**2))),
                lambda e: 0 ])


def GER01(e):
    """Returns G_{01}(E)^R for a perfect wire with γ = 1.

    Calculated from an inverse of E - H - Σ(E)^R where H is a 2x2 matrix

                              γ   γ    γ   γ   γ
                        ┄┄┄─○───○───◎───●───○─┄┄┄
    Only ◎───● was part of the scattering region.
    """
    e = e/2.0
    # k = np.piecewise(
    #         np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
    #         [   lambda e: (e - np.sqrt(e**2 - 1)),
    #             lambda e: (e + 1j*np.sqrt(1 - e**2)),
    #             lambda e: (e + np.sqrt(e**2 - 1))   ])
    # return  1/(k**2-1)
    k = np.piecewise(
            np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
            [   lambda e: (-np.sqrt(e**2 - 1)),
                lambda e: (+1j*np.sqrt(1 - e**2)),
                lambda e: (+np.sqrt(e**2 - 1))   ])
    return 1.0/2.0*(e - k)/k


def GEL01(e, ef):
    """Returns G_{00}(E)^< for a perfect wire with γ = 1.

    Eq. (C.11) in 1307.6419
    """
    e = e/2.0
    ef = ef/2.0
    return np.piecewise(np.array(e, dtype=np.complex), [(e >= -1.0) & (e <= ef)],
            [   lambda e: (1j*e/ (np.sqrt(1-e**2))),
                lambda e: 0 ])


def GER00_general(e, ed, gamma_dot, gamma_wire=1):
    """Returns G_{00}(E)^R for a quantum dot connected to leads.

    Calculated from an inverse of E - H - Σ(E)^R where H is a 1x1 matrix

                          γ'  γ'  γ   γ   γ'  γ'
                        ┄┄┄─○───○───◎───○───○─┄┄┄
                                    εd
    Only ◎ (=QD) is part of the scattering region.
    Arguments:
        ed         : energy level of the QD εd
        gamma_dot  : hopping between the QD and the leads γ
        gamma_wire : hopping in the leads γ'
    """
    # # Result for ◎───● system:
    # k1 = (e - SigmaER_general(e, gamma_wire, gamma_wire))
    # k2 = (e - ed - SigmaER_general(e, gamma_dot, gamma_wire))
    # return k1/(k1*k2 - gamma_dot**2)
    e = e/(2*np.abs(gamma_wire))
    ed = ed/(2*np.abs(gamma_wire))
    k = np.abs(gamma_dot)**2/np.abs(gamma_wire)**2
    c = np.piecewise(
            np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
            [   lambda e: (+np.sqrt(e**2 - 1)),
                lambda e: (-1j*np.sqrt(1 - e**2)),
                lambda e: (-np.sqrt(e**2 - 1))   ])
    return 1.0/(2.0*np.abs(gamma_wire))/((1-k)*e - ed -k*c)


def GEL00_general(e, ef1, ef2, beta, ed, gamma_dot, gamma_wire):
    """Returns G_{00}(E)^< for a quantum dot connected to wires.

    Calculated from equation
        G_{00}(E)^< = G_{00}(E)^R · Σ(E)^< · [G_{00}(E)^R]^†

    where Σ(E)^< = ∑_m Σ_m(E)^<. """

    # ger = GER00_general(engs, ed, gamma_dot, gamma_wire)
    # sel = 2*SigmaEL_general(engs, ef, gamma_dot, gamma_wire)
    # return ger*sel*ger.conj()
    f1 = fermi(e, ef1, beta)
    f2 = fermi(e, ef2, beta)
    e = e/(2*np.abs(gamma_wire))
    ed = ed/(2*np.abs(gamma_wire))
    k = np.abs(gamma_dot)**2/np.abs(gamma_wire)**2
    sel = np.piecewise(np.array(e, dtype=np.complex), [(e >= -1.0) & (e <= 1)],
            [   lambda e: (np.abs(gamma_dot)**2/np.abs(gamma_wire))*1j*2*np.sqrt(1-e**2),
                lambda e: 0 ])
    ger_gea = 1.0/(4*np.abs(gamma_wire)**2) * 1.0/(2*(1-k)*(e**2-e*ed) + k**2 + ed**2 - e**2)
    return (f1+f2)/2.0*ger_gea*2*sel


def GER01_general(e, ed, gamma_dot, gamma_wire):
    """Returns G_{01}(E)^R for a quantum dot connected to wires.

    Calculated from an inverse of E - H - Σ(E)^R where H is a 2x2 matrix
                          γ'  γ'  γ   γ   γ'  γ'
                        ┄┄┄─○───○───◎───●───○─┄┄┄
                                    εd
    Only ◎───● was part of the scattering region.
    Arguments:
        ed         : εd
        gamma_dot  : γ
        gamma_wire : γ' """
    k1 = (e - SigmaER_general(e, gamma_wire, gamma_wire))
    k2 = (e - ed - SigmaER_general(e, gamma_dot, gamma_wire))
    return gamma_dot/(k1*k2 - gamma_dot**2)

###############################################################################
# # Time domain

def GtR00(ts):
    """Returns G_{00}(t)^R for a perfect wire with γ = 1.

    Eq. (C.9 )in 1307.6419.
    """
    return -1j*sc.special.jv(0,2*ts)*np.heaviside(ts,1)


def GtL00(ts):
    """Returns G_{00}(t)^< for a perfect wire with γ = 1.

    Eq. (C.10) in 1307.6419.
    """
    return 0.5j*sc.special.jv(0,2*ts) - 0.5*sc.special.struve(0,2*ts)


def GtL01(ts):
    """Returns G_{01}(t)^< for a perfect wire with γ = 1.

    Eq. (C.12) in 1307.6419.
    """
    return 0.5*sc.special.jv(1,2*ts) - 0.5j*sc.special.struve(-1,2*ts)
