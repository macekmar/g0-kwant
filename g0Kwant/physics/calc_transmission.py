import kwant
import numpy as np
from .g0 import GER01_general
from .selfenergy import SER_general

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

def transmisson_QD_wire(engs, gamma, gamma_wire, eps_d):
    """Calculates transmission for a QD in a 1D nanowire.

    gamma      - hopping between a QD and the leads
    gamma_wire - hopping in the leads
    eps_d      - energy level of the QD

    It is calculated from:
        Tₖₗ = Tr[ΓₖGᴿΓₗGᴬ]
    where
        Γₖ = i[ Σₖᴿ - Σₖᴬ ] = -‒2 Im Σₖᴿ
    We start with a 2x2 Hamiltonian, the system consist of a QD and one site in
    the right lead
                          γ'  γ'  γ   γ   γ'  γ'
                        ┄┄┄─○───○─[──◎───●─]─○─┄┄┄
                                    εd
    The result for transmisson from right to left is
        Tₗᵣ = Γₗ · Γᵣ · |G₀₁ᴿ|²
    """

    Gl = -2*SER_general(engs, gamma_att=gamma, gamma_lead=gamma_wire).imag
    Gr = -2*SER_general(engs, gamma_att=gamma_wire, gamma_lead=gamma_wire).imag
    return Gl*Gr*np.abs(GER01_general(engs, eps_d, gamma, gamma_wire))**2
