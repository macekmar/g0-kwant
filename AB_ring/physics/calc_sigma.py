import kwant
import numpy as np

from .fermi import fermi

def calc_sigmaER(syst, lead, eng):
    """Calculates Σ_i(E)^R using Kwant."""
    if np.isscalar(eng):
        return syst.leads[lead].selfenergy(eng)[0,0]
    else:
        return np.array([syst.leads[lead].selfenergy(e)[0,0] for e in eng])

def calc_sigmaEL(syst, lead, eng, ef, beta):
    """Calcualtes Σ_i(E)^<
    Equation is:
        Σ_i(E)^< = -i Γ_i(E) f_i(E)
    where
        Γ_i(E) = i (Σ_i(E)^R - Σ_i(E)^A)
    """
    sigmaER = calc_sigmaER(syst, lead, eng)
    return -(sigmaER - sigmaER.conj())*fermi(eng, ef, beta)

def calc_sigmaEL_from_sigmaER(SigmaER, eng, ef, beta):
    """Calculates Σ_m(E)^< in lead m from Σ_m(E)^R 

    Equation is
        Σ_m(E)^< = i·Γ_m(E)·f(e-e_f),    Γ_m(E) = i·[Σ_m(E)^R - Σ_m(E)^A]
        Σ_m(E)^< =-[Σ_m(E)^R - Σ_m(E)^A]·f(e-e_f)
    where f(e-ef) is Fermi function."""

    return -(SigmaER - SigmaER.conj())*fermi(eng, ef, beta)