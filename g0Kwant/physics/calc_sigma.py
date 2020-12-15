# # Calculation of self energies (Σ, sigma) for a general Kwant `FiniteSystem`

import kwant
import numpy as np

from .fermi import fermi


def calc_sigmaER(syst, lead, eng):
    """Calculates Σ_i(E)^R using Kwant.

    Parameters
    ----------
    syst : Kwant FiniteSystem

    lead : integer
        index of the lead
    eng : list or 1D numpy array
    """
    if np.isscalar(eng):
        return syst.leads[lead].selfenergy(eng)[0, 0]
    else:
        return np.array([syst.leads[lead].selfenergy(e)[0, 0] for e in eng])


def calc_sigmaEL(syst, lead, eng, ef, beta):
    """Calculates Σ_i(E)^< via calc_sigmaER.

    Equation is
        Σ_i(E)^< = -i Γ_i(E) f_i(E)
    where
        Γ_i(E) = i (Σ_i(E)^R - Σ_i(E)^A)
    """
    sigmaER = calc_sigmaER(syst, lead, eng)
    return -(sigmaER - sigmaER.conj()) * fermi(eng, ef, beta)


def calc_sigmaEG(syst, lead, eng, ef, beta):
    """Calculates Σ_i(E)^> via calc_sigmaER.

    Equation is
        Σ_i(E)^> = -i Γ_i(E) (f_i(E) ‒ 1)
    where
        Γ_i(E) = i (Σ_i(E)^R - Σ_i(E)^A)
    """
    sigmaER = calc_sigmaER(syst, lead, eng)
    return -(sigmaER - sigmaER.conj()) * (fermi(eng, ef, beta) - 1)


def calc_sigmaEL_from_sigmaER(SigmaER, eng, ef, beta):
    """Calculates Σ_m(E)^< in lead m from Σ_m(E)^R

    Equation is
        Σ_m(E)^< = i·Γ_m(E)·f(e-e_f),    Γ_m(E) = i·[Σ_m(E)^R - Σ_m(E)^A]
        Σ_m(E)^< =-[Σ_m(E)^R - Σ_m(E)^A]·f(e-e_f)
    where f(e-ef) is Fermi function.

    Parameters
    ----------
    SigmaER : @D or 3D numpy array
        Last two axes are 'system size' × 'system size'.
        For the 3D array, first axis is energy.
    eng : list or 1D numpy array
    ef : float
        Fermi energy of the lead
    beta : float
        inverse temperature
    """
    return -(SigmaER - SigmaER.conj()) * fermi(eng, ef, beta)


def calc_sigmaEG_from_sigmaER(SigmaER, eng, ef, beta):
    """Calculates Σ_m(E)^> in lead m from Σ_m(E)^R

    Equation is
        Σ_m(E)^> = i·Γ_m(E)·(f(e-e_f) ‒ 1),    Γ_m(E) = i·[Σ_m(E)^R - Σ_m(E)^A]
        Σ_m(E)^> =-[Σ_m(E)^R - Σ_m(E)^A]·(f(e-e_f) ‒ 1)
    where f(e-ef) is Fermi function.

    Parameters
    ----------
    SigmaER : @D or 3D numpy array
        Last two axes are 'system size' × 'system size'.
        For the 3D array, first axis is energy.
    eng : list or 1D numpy array
    ef : float
        Fermi energy of the lead
    beta : float
        inverse temperature
    """
    return -(SigmaER - SigmaER.conj()) * (fermi(eng, ef, beta) - 1)
