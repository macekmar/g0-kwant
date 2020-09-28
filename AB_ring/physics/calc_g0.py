import kwant
import numpy as np
import scipy as sc

from .fermi import fermi
from .calc_sigma import *

# # Self energies
# def calc_sigmaER(syst, lead, eng):
#     """Calculates Σ_m(E)^R in lead m using Kwant."""
#     if np.isscalar(eng):
#         return syst.leads[lead].selfenergy(eng)
#     else:
#         return np.array([syst.leads[lead].selfenergy(e)[0,0] for e in eng]) 

# def calc_sigmaEL(SigmaER, eng, ef, beta):
#     """Calculates Σ_m(E)^< in lead m from Σ_m(E)^R 

#     Equation is
#         Σ_m(E)^< = i·Γ_m(E)·f(e-e_f),    Γ_m(E) = i·[Σ_m(E)^R - Σ_m(E)^A]
#         Σ_m(E)^< =-[Σ_m(E)^R - Σ_m(E)^A]·f(e-e_f)
#     where f(e-ef) is Fermi function."""

#     return -(SigmaER - SigmaER.conj())*fermi(eng, ef, beta)

# Green functions
## Energy domain


def calc_GER(syst, engs):
    """Calculates G(E)^R from matrix inversion using Kwant.

    Equation:
        G(E)^R = 1/(E·I - H - ∑_m Σ_m(E)^R)
    Σ_m(E)^R is calculated from `syst`.
    """

    H_mat = syst.hamiltonian_submatrix()
    lead_pos = syst.lead_interfaces
    N,M = H_mat.shape
    # Calculates GER for one energy
    def _calc_GER(e):
        mat = e*np.eye(N,M) - H_mat
        for lead, pos in enumerate(lead_pos):
            mat[pos[0],pos[0]] -= calc_sigmaER(syst, lead, e)
        return np.linalg.inv(mat)

    GER = []
    if np.isscalar(engs):
        return _calc_GER(engs)
    else:
        for e in engs:
            GER.append(_calc_GER(e))
        return np.array(GER)

def calc_GEL(GER, SELs):
    """Calculates matrix G(E)^< from G(E)^R and Σ(E)^<.
    
    Σ(E)^< is a list of tuples:
        (self energies Σ_m(E)^< for lead m, i, j)
    where inidices i,j are such that
        [Σ(E)^<]_ij = Σ_m(E)^<.
    """
    sel_mat = np.zeros(GER.shape, dtype=np.complex)
    for (SEL,i,j) in SELs:
        sel_mat[:,i,j] = SEL[:]
    GEL = np.array([np.dot(GER[i], np.dot(sel_mat[i], GER[i].conj().T)) for i in range(GER.shape[0])])
    return GEL    


## Time domain

def calc_GtL(syst, t, i, j, Emin, Emax, eng_fermis, beta, quad_vec_kwargs={}):
    num_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(num_leads):
            num_channels = wf(lead).shape[0]
            for channel in range(num_channels):
                wf_abs += (-1)**channel * 1j/(2*np.pi) * fermi(e, eng_fermis[lead], beta) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtL = (integral[0][:,0] + 1j*integral[0][:,1])
    return GtL

def calc_GtR(syst, t, i, j, Emin, Emax, quad_vec_kwargs={}):
    num_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(num_leads):
            num_channels = wf(lead).shape[0]
            for channel in range(num_channels):
                wf_abs += 1/(2*np.pi) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtR = -1j * np.heaviside(t,1) * (integral[0][:,0] + 1j*integral[0][:,1])
    return GtR

def calc_GtG_from_GtLR(t, GtL, GtR):
    """Calculates g(t)> from g(t)< and g(t)R.

    Formula is g^> = G^R - g^A + g^< where g^A(t,t') = [g^R(t',t)]^†"""
    return GtR*(t >= 0) - GtR.conj()[::-1]*(t < 0) + GtL
