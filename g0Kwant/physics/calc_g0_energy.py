# # Green functions in energy domain for a general Kwant `FiniteSystem`
#
# Green functions can be calculated in two ways:
#   - by inversion G(E)^R = 1/(E·I - H - ∑_m Σ_m(E)^R) and then
#     calculating G(E)^{<,>}
#   - G^{<,>} from the wave function (How to get G^R, G^A?)
#
# Inversion is only faster for small system sizes (about 100×100) and if we
# have a formula for the self energy, this is `calc_GER_inverse_from_SER`.
# Formula can also be an interpolation.
# Without the precalculated self energies, it is just as slow as the
# calculation with the wave function, this is `calc_GER_inverse`.
#
# The wave function approach is slow because we have to solve a linear system
# for each energy (Why is this slower than calculating inverses?)
# We could do an interpolation, but it seems we need just as many point for the
# interpolation as we are intersted it. In practice it is not worth it.
# These functions are `calc_GELG` and `calc_GELG_fun`.

import kwant
import numpy as np

from .fermi import fermi
from .calc_sigma import calc_SigmaER


def calc_GER_inverse_from_SER(eng, ham, SERs):
    """Calculates G(E)^R from matrix inversion using precalculated Σ(E)^R.

    Equation:
        G(E)^R = 1/(E·I - H - ∑_m Σ_m(E)^R)
    Σ_m(E)^R is supplied by user in `SERs`.

    Parameters
    ----------
    eng : 1D numpy array

    ham : 2D numpy array
        Hamiltonian without the self energies, `syst.hamiltonian_subsystem()`
    SERs : list of tuples (Σ_m(E)^R, i, j),
        Σ_m(E)^R is a self energies for lead m attached in sites i, j
    """

    mat = np.tensordot(eng, np.eye(*ham.shape, dtype=np.complex), axes=0)
    mat -= ham
    for (se, i, j) in SERs:
        mat[:, i, j] -= se
    return np.array([np.linalg.inv(m) for m in mat])


def calc_GER_inverse(syst, eng):
    """Calculates G(E)^R from matrix inversion using Kwant.

    Equation:
        G(E)^R = 1/(E·I - H - ∑_m Σ_m(E)^R)
    Σ_m(E)^R is calculated from `syst`.

    Parameters
    ----------
    syst : Kwant FiniteSystem

    eng : scalar or 1D numpy array
    """
    H_mat = syst.hamiltonian_submatrix()
    lead_pos = syst.lead_interfaces
    N, M = H_mat.shape
    # Calculates GER for one energy

    def _calc_GER(e):
        mat = e*np.eye(N, M) - H_mat
        for lead, pos in enumerate(lead_pos):
            mat[pos[0], pos[0]] -= calc_SigmaER(syst, lead, e)
        return np.linalg.inv(mat)

    GER = []
    if np.isscalar(eng):
        return _calc_GER(eng)
    else:
        for e in eng:
            GER.append(_calc_GER(e))
        return np.array(GER)


def calc_GELG_inverse(GER, SEs):
    """Calculates matrix G(E)^<,> from G(E)^R and Σ(E)^<,>.

    Equation:
        G(E)^<,> = G(E)^R · Σ(E)^<,> · G(E)^A   (Datta, Eq. 8.3.1)
    You have to call it twice with Σ(E)^< and Σ(E)^>.

    Parameters
    ----------
    GER : 3D np.array or a list of 2D np.arrays
        G(E)^R, first axis are energies
    SERs : list of tuples (Σ_m(E)^R, i, j),
        Σ_m(E)^R is a self energy for lead m attached in sites i, j
    """
    se_mat = np.zeros(GER.shape, dtype=np.complex)
    for (SE, i, j) in SEs:
        se_mat[:, i, j] = SE[:]
    GEL = np.array([np.dot(GER[i], np.dot(se_mat[i], GER[i].conj().T))
                    for i in range(GER.shape[0])])
    return GEL


def calc_GELG(syst, eng, ef, beta):
    """Calculates G(E)^<,> from the wave function.

    Integrand from equation 22, 1307.6419 but without the exponential (t=0).
    G(E)^> is the same but with (n_F(E) ‒ 1).

    Parameters
    ----------
    syst : Kwant FiniteSystem

    eng : list or 1D np.array of energies

    ef : list of Fermi energies for the leads

    beta : inverse temperature
    """
    nb_leads = len(syst.leads)
    nb_sites = syst.hamiltonian_submatrix().shape[0]
    GEL = np.zeros((len(eng), nb_sites, nb_sites), dtype=np.complex)
    GEG = np.zeros((len(eng), nb_sites, nb_sites), dtype=np.complex)
    for iw, w in enumerate(eng):
        wf = kwant.wave_function(syst, energy=w)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_val = wf(lead)[channel][:, np.newaxis] * \
                    np.ones((1, nb_sites))
                GEL[iw, :, :] += (-1)**channel * 1j * fermi(w, ef[lead], beta) * wf_val * wf_val.T.conj()
                GEG[iw, :, :] += (-1)**channel * 1j * (fermi(w, ef[lead], beta)-1) * wf_val * wf_val.T.conj()
    return GEL, GEG


def calc_GELG_fun(fun, nb_leads, i, j, k, eng, ef, beta):
    """Calculates G(E)^<,> from the interpolation of the wave function.

    Integrand from equation 22, 1307.6419 but without the exponential (t=0).
    G(E)^> is the same but with (n_F(E) ‒ 1).

    Parameters
    ----------
    fun :
        Interpolation of `kwant.wave_function(syst, energy=eng)`

    k : 1D numpy array

    eng : 1D numpy array
        eng related to k by dispertion relation ε(k)
    nb_leads : number of leads

    i and j : integers

    ef : list of Fermi energies for the leads

    beta : inverse temperature
    """
    assert k.shape == eng.shape
    GEL = np.zeros((len(eng)), dtype=np.complex)
    GEG = np.zeros((len(eng)), dtype=np.complex)
    for lead in range(nb_leads):
        for channel in range(1):
            GEL += (-1)**channel * 1j * fermi(eng, ef[lead], beta) * fun(
                k, lead, channel, i) * fun(k, lead, channel, j).conj()
            GEG += (-1)**channel * 1j * (fermi(eng, ef[lead], beta) - 1) * fun(
                k, lead, channel, i) * fun(k, lead, channel, j).conj()
    return GEL, GEG
