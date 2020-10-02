import kwant
import numpy as np
import scipy as sc

from .fermi import fermi
from .calc_sigma import *

def calc_GER_inverse_from_SER(engs, ham, SERs):
    """Calculates G(E)^R from matrix inversion using precalculate Σ(E)^R.
    
    This is much faster, since most of the time spent in `calc_GER_inverse`
    is in calling `calc_sigmaER`.
    If Σ(E)^R is not the same as `SER_general` in `selfenergy.py` one could
    first interpolate Σ(E)^R in few points (e.g. 1000) and then call this
    function in many points (e.g. 1e6).

    Hamiltonian does not have yet added Σ(E)^R. In short:
    `ham = syst.hamiltonian_subystem()`

    Since matrix inverse is 𝓞(n³) it get slower with system size. At about
    100x100 it becomes slower than `calc_GELG`.
    We should switch to `calc_GELG_fun`.

    Σ(E)^R is a list of tuples:
            (self energies Σ_m(E)^R for lead m, i, j)
    where inidices i,j are such that
            [Σ(E)^R]_ij = Σ_m(E)^<.
    """

    # (ε·I - H)⁻¹
    mat = np.tensordot(engs, np.eye(*ham.shape, dtype=np.complex), axes=0)
    mat -= ham
    # self energies are missing in H
    for (se, i, j) in SERs:
        mat[:,i,j] -= se
    
    return np.array([np.linalg.inv(m) for m in mat]) 

def calc_GER_inverse(syst, engs):
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

def calc_GELG_inverse(GER, SEs):
    """Calculates matrix G(E)^<,> from G(E)^R and Σ(E)^<,>.

    You have to call it twice with Σ(E)^< and Σ(E)^>.
    
    Σ(E)^<,> is a list of tuples:
        (self energies Σ_m(E)^<,> for lead m, i, j)
    where inidices i,j are such that
        [Σ(E)^<,>]_ij = Σ_m(E)^<.
    
    It calculates it from formula:
        G(E)^<,> = G(E)^R · Σ(E)^<,> · G(E)^A   (Datta, Eq. 8.3.1)
    """
    se_mat = np.zeros(GER.shape, dtype=np.complex)
    for (SE,i,j) in SEs:
        se_mat[:,i,j] = SE[:]
    GEL = np.array([np.dot(GER[i], np.dot(se_mat[i], GER[i].conj().T)) for i in range(GER.shape[0])])
    return GEL

def calc_GELG(syst, omegas, eng_fermis, beta):
    """Calculates matrix G(E)^<,> from wavefunction.
    
    Integrand from equation 22, 1307.6419
    G(E)^> is the same but with (n_F(E) ‒ 1)."""
    nb_leads = len(syst.leads)
    nb_sites = syst.hamiltonian_submatrix().shape[0]
    GEL = np.zeros((len(omegas), nb_sites, nb_sites), dtype=np.complex)
    GEG = np.zeros((len(omegas), nb_sites, nb_sites), dtype=np.complex)
    for iw, w in enumerate(omegas):
        wf = kwant.wave_function(syst, energy=w)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_val = wf(lead)[channel][:,np.newaxis] * np.ones((1,nb_sites))
                GEL[iw,:,:] += (-1)**channel * 1j * fermi(w, eng_fermis[lead], beta) * wf_val * wf_val.T.conj()
                GEG[iw,:,:] += (-1)**channel * 1j * (fermi(w, eng_fermis[lead], beta)-1) * wf_val * wf_val.T.conj()
    return GEL, GEG

def calc_GELG_fun(fun, nb_leads, nb_sites, omegas, eng_fermis, beta):
    """Calculates matrix G(E)^<,> from wavefunction.
    
    `fun` is an interpolation of `kwant.wave_function(syst, energy=w)`

    Integrand from equation 22, 1307.6419
    G(E)^> is the same but with (n_F(E) ‒ 1)."""
    GEL = np.zeros((len(omegas), nb_sites, nb_sites), dtype=np.complex)
    GEG = np.zeros((len(omegas), nb_sites, nb_sites), dtype=np.complex)
    for iw, w in enumerate(omegas):
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf = np.array([fun(w, lead, channel, i) for i in range(nb_sites)])
                wf_val = wf(lead)[channel][:,np.newaxis] * np.ones((1,nb_sites))
                GEL[iw,:,:] += (-1)**channel * 1j * fermi(w, eng_fermis[lead], beta) * wf_val * wf_val.T.conj()
                GEG[iw,:,:] += (-1)**channel * 1j * (fermi(w, eng_fermis[lead], beta)-1) * wf_val * wf_val.T.conj()
    return GEL, GEG
