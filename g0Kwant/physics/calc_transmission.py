# # Calculations of transmissions

import kwant
import numpy as np
from .g0 import GER01_general
from .sigma import SigmaER_general
import multiprocessing


def _calc_trans(syst, eng):
    """Calculates transmission matrix for a system `syst` in energy `e`."""
    smatrix = kwant.smatrix(syst, eng)
    N = len(smatrix.lead_info)
    return [[smatrix.transmission(j, i) for i in range(N)] for j in range(N)]


def calc_transmission(syst, eng):
    """Calculates transmission matrix for a system `syst` in energies `eng`."""
    trans = []

    scalar = False
    if np.isscalar(eng):
        scalar = True
        eng = [eng]

    for e in eng:
        trans.append(_calc_trans(syst, e))

    if scalar:
        return trans[0]
    else:
        return np.array(trans)


def calc_transmission_parallel(syst, eng, num_core=4):
    """Calculates transmission matrix for a system `syst` in energies `eng`.

    Parallel version using multiprocessing. Useful in notebooks.
    """
    scalar = False
    if np.isscalar(eng):
        scalar = True
        eng = [eng]

    p = multiprocessing.Pool(num_core)
    trans = p.starmap(_calc_trans, zip(len(eng)*[syst], eng))
    p.close()
    if scalar:
        return trans[0]
    else:
        return np.array(trans)


def transmission_QD_wire(eng, eps_d, gamma, gamma_wire):
    """Calculates transmission for a QD in a 1D wire.

    Equation
        Tₖₗ = Tr[ΓₖGᴿΓₗGᴬ]
    where
        Γₖ = i[ Σₖᴿ - Σₖᴬ ] = -‒2 Im Σₖᴿ
    We start with a 2x2 Hamiltonian, the system consist of a QD and one site in
    the right lead
                          γ'  γ'  γ   γ   γ'  γ'
                        ┄┄┄─○───○─[──◎───●─]─○─┄┄┄
                                    εd
    The result for transmission from right to left is
        Tₗᵣ = Γₗ · Γᵣ · |G₀₁ᴿ|²

    Paramters
    ---------
    gamma : hopping between a QD and the leads

    gamma_wire : hopping in the leads

    eps_d : energy level of the QD
    """
    Gl = -2*SigmaER_general(eng, gamma_att=gamma, gamma_lead=gamma_wire).imag
    Gr = -2*SigmaER_general(eng, gamma_att=gamma_wire,
                            gamma_lead=gamma_wire).imag
    return Gl*Gr*np.abs(GER01_general(eng, eps_d, gamma, gamma_wire))**2


def transmission_QD_wire_peak_pos(eps_d, gamma, gamma_wire):
    """Returns the energies of the transmission T_12 peak and FWHM.

    Calculated from the analytical formula for `transmission_QD_wire`:
                                    1
                T₁₂ =  ─────────────────────────────
                        (k² - 1)²(e₀-e)²/(1-e²) + 1
                k  = γ_wire/γ
                e  = E/(2·γ_wire)
                e₀ = [k²/(k² - 1)]·εd / (2·γ_wire)
    """
    k = gamma_wire/gamma

    # Usually this is the maximum position:
    E0 = k**2/(k**2-1)*eps_d
    # Maximum can also be:
    # E0 = 4*gamma_wire**2/( k**2/(k**2-1)*eps_d)

    # Positions of FWHM
    l = k**2 - 1
    e0 = E0/(2*gamma_wire)
    d = 2*np.sqrt(l**4*e0**2 - (l**2+1)*(l**2*e0**2-1))
    EL = 2*gamma_wire*(2*l**2*e0 - d)/(2*(l**2+1))
    ER = 2*gamma_wire*(2*l**2*e0 + d)/(2*(l**2+1))

    return np.array([EL, E0, ER])
