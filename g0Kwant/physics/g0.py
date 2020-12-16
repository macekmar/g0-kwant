# # Analytical results for Green functions
#
# They are based on paper 1307.6419 (Eq. C.3, C.4) and some my calculations.
# 1307.6419 - Numerical simulations of time resolved quantum electronics, Gaury

import numpy as np
import scipy as sc
from .fermi import fermi
from .sigma import SigmaER_general

###############################################################################
# # Energy domain


def GER00(eng):
    """Returns G_{00}(E)^R for a perfect wire with γ = 1.

    Eq. (C.7) in 1307.6419.
    """
    e = eng/2.0
    return np.piecewise(np.array(e, dtype=np.complex),
                        [e < -1.0, (e >= -1.0) & (e <= 1.0)], [
                        lambda e: -1/2.0 / np.sqrt(e**2-1),
                        lambda e: 1/2.0 / (1j*np.sqrt(1-e**2)),
                        lambda e: 1/2.0 / np.sqrt(e**2-1)
                        ])


def GER00_general(eng, eps_d, gamma, gamma_wire=1):
    """Returns G_{00}(E)^R for a quantum dot connected to leads.

    Calculated from an inverse of E - H - Σ(E)^R where H is a 1x1 matrix

                          γ'  γ'  γ   γ   γ'  γ'
                        ┄┄┄─○───○───◎───○───○─┄┄┄
                                    εd
    Only ◎ (=QD) is part of the scattering region.
    Arguments:
        eps_d : energy level of the QD εd
        gamma : hopping between the QD and the leads γ
        gamma_wire : hopping in the leads γ'
    """
    # # Result for ◎───● system:
    # k1 = (e - SigmaER_general(e, gamma_wire, gamma_wire))
    # k2 = (e - ed - SigmaER_general(e, gamma, gamma_wire))
    # return k1/(k1*k2 - gamma**2)
    e = eng/(2*np.abs(gamma_wire))
    eps_0 = eps_d/(2*np.abs(gamma_wire))
    inv_k2 = np.abs(gamma)**2/np.abs(gamma_wire)**2
    c = np.piecewise(np.array(e, dtype=np.complex),
                     [e < -1.0, (e >= -1.0) & (e <= 1.0)], [
                        lambda e: (+np.sqrt(e**2 - 1)),
                        lambda e: (-1j*np.sqrt(1 - e**2)),
                        lambda e: (-np.sqrt(e**2 - 1))
                    ])
    return 1.0/(2.0*np.abs(gamma_wire))/((1-inv_k2)*e - eps_0 - inv_k2*c)


def GEL00(eng, ef):
    """Returns G_{00}(E)^< for a perfect wire with γ = 1.

    Eq. (C.8) in 1307.6419.
    """
    e = eng/2.0
    ef = ef/2.0
    return np.piecewise(np.array(e, dtype=np.complex),
                        [(e >= -1.0) & (e <= ef)], [
                            lambda e: (1j / (np.sqrt(1-e**2))),
                            lambda e: 0
                        ])


def GEL00_general(eng, eps_d, gamma, ef1, ef2, beta, gamma_wire=1):
    """Returns G_{00}(E)^< for a quantum dot connected to wires.

    Calculated from equation
        G_{00}(E)^< = G_{00}(E)^R · Σ(E)^< · [G_{00}(E)^R]^†

    where Σ(E)^< = ∑_m Σ_m(E)^<. """

    # ger = GER00_general(engs, eps_d, gamma, gamma_wire)
    # sel = 2*SigmaEL_general(engs, ef, gamma, gamma_wire)
    # return ger*sel*ger.conj()
    f1 = fermi(eng, ef1, beta)
    f2 = fermi(eng, ef2, beta)
    e = eng/(2*np.abs(gamma_wire))
    eps_0 = eps_d/(2*np.abs(gamma_wire))
    inv_k2 = np.abs(gamma)**2/np.abs(gamma_wire)**2
    sel = np.piecewise(np.array(e, dtype=np.complex),
                       [(e >= -1.0) & (e <= 1)], [
        lambda e: (np.abs(gamma)**2/np.abs(gamma_wire))*1j*2*np.sqrt(1-e**2),
        lambda e: 0
                      ])
    term = (2*(1-inv_k2) * (e**2-e*eps_0) + inv_k2**2 + eps_0**2 - e**2)
    ger_gea = 1.0/(4*np.abs(gamma_wire)**2) * 1.0/term
    return (f1+f2)/2.0*ger_gea*2*sel


def GER01(eng):
    """Returns G_{01}(E)^R for a perfect wire with γ = 1.

    Calculated from an inverse of E - H - Σ(E)^R where H is a 2x2 matrix

                              γ   γ    γ   γ   γ
                        ┄┄┄─○───○───◎───●───○─┄┄┄
    Only ◎───● was part of the scattering region.
    """
    e = eng/2.0
    # k = np.piecewise(np.array(e, dtype=np.complex),
    #                  [e < -1.0, (e >= -1.0) & (e <= 1.0)], [
    #                   lambda e: (e - np.sqrt(e**2 - 1)),
    #                   lambda e: (e + 1j*np.sqrt(1 - e**2)),
    #                   lambda e: (e + np.sqrt(e**2 - 1))
    #                  ])
    # return  1/(k**2-1)
    k = np.piecewise(np.array(e, dtype=np.complex),
                     [e < -1.0, (e >= -1.0) & (e <= 1.0)], [
                        lambda e: (-np.sqrt(e**2 - 1)),
                        lambda e: (+1j*np.sqrt(1 - e**2)),
                        lambda e: (+np.sqrt(e**2 - 1))
                    ])
    return 1.0/2.0*(e - k)/k


def GER01_general(eng, eps_d, gamma, gamma_wire=1):
    """Returns G_{01}(E)^R for a quantum dot connected to wires.

    Calculated from an inverse of E - H - Σ(E)^R where H is a 2x2 matrix
                          γ'  γ'  γ   γ   γ'  γ'
                        ┄┄┄─○───○───◎───●───○─┄┄┄
                                    εd
    Only ◎───● was part of the scattering region.
    Arguments:
        eps_d : ε_d
        gamma : γ
        gamma_wire : γ' """
    k1 = (eng - SigmaER_general(eng, gamma_wire, gamma_wire))
    k2 = (eng - eps_d - SigmaER_general(eng, gamma, gamma_wire))
    return gamma/(k1*k2 - gamma**2)


def GEL01(eng, ef):
    """Returns G_{00}(E)^< for a perfect wire with γ = 1.

    Eq. (C.11) in 1307.6419
    """
    e = eng/2.0
    ef = ef/2.0
    return np.piecewise(np.array(e, dtype=np.complex),
                        [(e >= -1.0) & (e <= ef)], [
                            lambda e: (1j*e / (np.sqrt(1-e**2))),
                            lambda e: 0
                        ])

###############################################################################
# # Time domain


def GtR00(time):
    """Returns G_{00}(t)^R for a perfect wire with γ = 1.

    Eq. (C.9 )in 1307.6419.
    """
    return -1j*sc.special.jv(0, 2*time)*np.heaviside(time, 1)


def GtL00(time):
    """Returns G_{00}(t)^< for a perfect wire with γ = 1.

    Eq. (C.10) in 1307.6419.
    """
    return 0.5j*sc.special.jv(0, 2*time) - 0.5*sc.special.struve(0, 2*time)


def GtL01(time):
    """Returns G_{01}(t)^< for a perfect wire with γ = 1.

    Eq. (C.12) in 1307.6419.
    """
    return 0.5*sc.special.jv(1, 2*time) - 0.5j*sc.special.struve(-1, 2*time)
