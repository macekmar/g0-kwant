import kwant
import numpy as np

from fermi import fermi

def calc_sigmaER(syst, lead, eng):
    """Calculates Σ_i(E)^R using Kwant."""
    if np.isscalar(eng):
        return syst.leads[lead].selfenergy(eng)
    else:
        return np.array([syst.leads[lead].selfenergy(e) for e in eng])

def calc_sigmaLR(syst, lead, eng, ef, beta):
    """Calcualtes Σ_i(E)^<
    Equation is:
        Σ_i(E)^< = -i Γ_i(E) f_i(E)
    where
        Γ_i(E) = i (Σ_i(E)^R - Σ_i(E)^A)
    """
    sigmaER = calc_sigmaER(syst, lead, eng)
    return -(sigmaER - sigmaER.conj())*fermi(eng, ef, beta)