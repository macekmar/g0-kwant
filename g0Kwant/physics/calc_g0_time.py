# # Green functions in time domain
#
# Based on:
# 1307.6419 - Numerical simulations of time-resolved quantum electronics
#
# We are intersted in the integrals such as Eq. (22) in 1307.6419
#              <      ⌠ dE                              †
#     Gᵢⱼ(t,t')  = ∑ₐ ⎮ ── i fₐ(E) Ψₐ,ᵢⱼ(E,t)Ψₐ,ᵢⱼ(E,t')
#                     ⌡ 2π
# For stationary systems, it can be simplified into
#              <      ⌠ dE                         *  -iEt
#        Gᵢⱼ(t)  = ∑ₐ ⎮ ── i fₐ(E) Ψₐ,ᵢⱼ(E)Ψₐ,ᵢⱼ(E)  e                      (1)
#                     ⌡ 2π
# where a goes over all the channels in all the leads and i,j are site indices.
# There is a similar Eq. (25) for Gᵢⱼ(t,t')ᴿ.
#
# Integrals diverge for E close to the band limits. It is better to integrate
# in the k-domain. Dispertion is E(k) = 2|γ|cos(k), where γ is hopping in the
# leads.
# NOTE: E = 0 corresponds to k = π and E = 2|γ| to k = 0, in k domain we
#       integrate from π to 0!
# NOTE: functions `..._intp` and `..._intp_mat` only work for systems with two
#       leads!
#
# We need G(t) evaluated in many points. Instead of integrating (1) for each t
# separately, we integrate a matrix function, where the first axis is the
# integrands at different t and the second axis is for the real and imaginary
# component.
# This is possible, since the costly part is solving the linear system for
# the wave function Ψₐ,ᵢⱼ(E) at the given energy.
# We integrate using `scipy.integrate.quad_vec`
#
# We could interpolate the wave function but in practice integration with the
# interpolation has to evaluate in (many) more points, thus reducing the
# advantage of the interpolation. We could improve the interpolation and reduce
# the number of evaluated points but for this the interpolation takes the
# majority of time.
# It is not worth it.
#
# Both G^< and G^> call wave function. Perhaps we could integrate at the same
# time – integrating a larger matrix.
# I have found that it is faster to split the integration.
#
# This observations gives us the order of parallelization:
#   - split integration of Gᵢⱼ(t)< and Gᵢⱼ(t)<
#   - different i,j indices
#   - split the integration domain a few times (2‒6)
#   - `scipy.integrate.quad_vec` parallelization
#
# We can check the integration using the Eq. (26) in 1307.6419
#             ⌠ dE                        †
#         ∀t: ⎮ ── ∑ₐ Ψₐ,ᵢⱼ(E,t)Ψₐ,ᵢⱼ(E,t) = 𝟙
#             ⌡ 2π
# However, we are missing the oscillatory part exp(iEt). It could be useful to
# control the interpolation of the wave function.
#
# Example with parallelization:
#   import sys
#   import uuid
#
#   def globalize(func):
#       def result(*args, **kwargs):
#           return func(*args, **kwargs)
#       result.__name__ = result.__qualname__ = uuid.uuid4().hex
#       setattr(sys.modules[result.__module__], result.__name__, result)
#       return result
#
#   @globalize
#   def GtL(k):
#       return integrand_GtL(syst, k, times, i, j, [ef1, ef2], beta, 0, gw)
#   res, err = calc_Gt_integral(GtL, 0, np.pi, quad_vec_kwargs={"workers":1})
#
# Without globalize it complains it cannot pickle

import kwant
import numpy as np
import scipy as sc

from .fermi import fermi


def calc_Gt_integral(integrand, k_min, k_max, quad_vec_kwargs={"workers": 4, "full_output": False}):
    """Integrates integrand (some g(t)) on the interval `[k_min, k_max]`."""
    integral = sc.integrate.quad_vec(integrand, a=k_min, b=k_max, **quad_vec_kwargs)
    Gt = integral[0][..., 0] + 1j*integral[0][..., 1]
    return Gt, integral[1:]


def integrand_GtL(syst, k, t, i, j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)<.

    Parameters
    ----------
        syst : Kwant FiniteSystem

        k : wave vector

        t : np.array for which g_ij(t)< is calculated

        i and j : site indices

        ef : list of Fermi energies, one for each lead

        beta : inverse temperature

        eps_i and gamma_wire : parameters for the dispertion relation
    """
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    # Test if we are between fermi levels, else result is 0
    if np.any(np.array([fermi(e, ef, beta) for ef in ef]) > 1e-14):
        wf = kwant.wave_function(syst, energy=e)
        for lead in range(len(syst.leads)):
            for channel in range(1):
                val += (-1)**channel * 1j/(2*np.pi) * \
                        fermi(e, ef[lead], beta) * \
                        wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                        np.exp(-1j*t*e)
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtG(syst, k, t, i, j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)>.

    Parameters
    ----------
        syst : Kwant FiniteSystem

        k : wave vector

        t : np.array for which g_ij(t)> is calculated

        i and j : site indices

        ef : list of Fermi energies, one for each lead

        beta : inverse temperature

        eps_i and gamma_wire : parameters for the dispertion relation
    """
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    # Test if we are between fermi levels, else result is 0
    if np.any(np.array([(fermi(e, ef, beta)-1) for ef in ef]) < -1e-14):
        wf = kwant.wave_function(syst, energy=e)
        for lead in range(len(syst.leads)):
            for channel in range(1):
                val += (-1)**channel * 1j/(2*np.pi) * \
                        (fermi(e, ef[lead], beta)-1) * \
                        wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                        np.exp(-1j*t*e)
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtR(syst, k, t, i, j, eps_i=0, gamma_wire=1):
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    wf = kwant.wave_function(syst, energy=e)
    for lead in range(len(syst.leads)):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                    wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                    np.exp(-1j*t*e)
    val = -1j * np.heaviside(t, 1) * val
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_Gt_control(syst, k, i, j, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (26) in 1307.6419 for |G(t)|².

    Parameters
    ----------
        syst : Kwant FiniteSystem

        k : wave vector

        i : site index

        eps_i and gamma_wire : parameters for the dispertion relation
    """
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = 0
    wf = kwant.wave_function(syst, energy=e)
    for lead in range(len(syst.leads)):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                    wf(lead)[channel][i] * wf(lead)[channel][j].conj()
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)


def calc_GtG_from_GtLR(t, GtL, GtR):
    """Calculates g(t)> from g(t)< and g(t)ᴿ.

    Formula is g^> = Gᴿ - gᴬ + g^< where gᴬ(t,t') = [gᴿ(t',t)]^†"""
    return GtR*(t >= 0) - GtR.conj()[::-1]*(t < 0) + GtL


###############################################################################
# Deprecated
###############################################################################
# # Gt^<

def calc_GtL_E(syst, t, i, j, ef, beta, Emin, Emax, quad_vec_kwargs={}):
    """Calculates g(t)< with integration in the energy domain.

    Deprecated. Integration in k-domain is faster.
    Parameters
    ----------
        syst : Kwant's finalized system

        t : np.array for which g_ij(t)< is calculated

        i and j : indices i,j in g_ij(t)<

        ef : list of Fermi energies, one for each lead

        beta : inverse temperature

        Emin and Emax :
            limits of integration. Integral diverges at the band limits.

        quad_vec_kwargs : arguments for `scipy.integrate.quad_vec`.
    """
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += (-1)**channel * 1j/(2*np.pi) * \
                        fermi(e, ef[lead], beta) * \
                        wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                        np.exp(-1j*t*e)
        return np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtL = (integral[0][:, 0] + 1j*integral[0][:, 1])
    return GtL, integral[1:]


def integrand_GtL_intp(intp, k, t, i, j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)< using interpolation."""
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += (-1)**channel * 1j/(2*np.pi) * fermi(e, ef[lead], beta) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, j).conj() *\
                np.exp(-1j*t*e)
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtL_intp_mat(intp, k, t, idx_i, idx_j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)< using interpolation.

    This function gives the whole matrix [Gᵢⱼ]. Integrating the whole matrix is
    slower then integrating the elements separately.

    idx_i and idx_j are 2D lists of indices. If we are intersted in matrix
                            |G00 G01|
                            |G10 G11|
    we supply [[0,0],[1,1]] and [[0,1], [0,1]].
    """
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0], len(idx_i), len(idx_j)), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            wf1 = np.array([[intp(k, lead, channel, i)
                             for i in row] for row in idx_i])
            wf2 = np.array([[intp(k, lead, channel, i)
                             for i in row] for row in idx_j]).conj()
            val += (-1)**channel * 1j/(2*np.pi) * fermi(e, ef[lead], beta) * \
                wf1 * wf2 * np.exp(-1j*t*e)[:, np.newaxis, np.newaxis]
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)

###############################################################################
# # Gt^<


def calc_GtG_E(syst, t, i, j, ef, beta, Emin, Emax, quad_vec_kwargs={}):
    """Calculates g(t)> with integration in the energy domain.

    Deprecated. Integration in k-domain is faster.
    Parameters
    ----------
        syst : Kwant's finalized system

        t : np.array for which g_ij(t)< is calculated

        i and j : indices i,j in g_ij(t)<

        ef : list of Fermi energies, one for each lead

        beta : inverse temperature

        Emin and Emax :
            limits of integration. Integral diverges at the band limits.

        quad_vec_kwargs : arguments for `scipy.integrate.quad_vec`.
    """
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += (-1)**channel * 1j/(2*np.pi) * \
                        (fermi(e, ef[lead], beta)-1) * \
                        wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                        np.exp(-1j*t*e)

        return np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtG = (integral[0][:, 0] + 1j*integral[0][:, 1])
    return GtG, integral[1:]


def integrand_GtG_intp(intp, k, t, i, j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)> using interpolation."""
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += (-1)**channel * 1j/(2*np.pi) * \
                (fermi(e, ef[lead], beta)-1) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, j).conj() *\
                np.exp(-1j*t*e)
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtG_intp_mat(intp, k, t, idx_i, idx_j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)< using interpolation.

    This function gives the whole matrix [Gᵢⱼ]. Integrating the whole matrix is
    slower then integrating the elements separately.

    idx_i and idx_j are 2D lists of indices. If we are intersted in matrix
                            |G00 G01|
                            |G10 G11|
    we supply [[0,0],[1,1]] and [[0,1], [0,1]].
    """
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0], len(idx_i), len(idx_j)), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            wf1 = np.array([[intp(k, lead, channel, i)
                             for i in row] for row in idx_i])
            wf2 = np.array([[intp(k, lead, channel, i)
                             for i in row] for row in idx_j]).conj()
            val += (-1)**channel * 1j/(2*np.pi) * \
                    (fermi(e, ef[lead], beta)-1) * \
                    wf1 * wf2 * np.exp(-1j*t*e)[:, np.newaxis, np.newaxis]
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)

###############################################################################
# # Gt^R


def calc_GtR_E(syst, t, i, j, Emin, Emax, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += 1 / (2*np.pi) * \
                    wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                    np.exp(-1j*t*e)
        return np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtR = -1j * np.heaviside(t, 1) * (integral[0][:, 0] + 1j*integral[0][:, 1])
    return GtR, integral[1:]


def integrand_GtR_intp(intp, k, t, i, j, eps_i=0, gamma_wire=1):
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, j).conj() *\
                np.exp(-1j*t*e)
    val = -1j * np.heaviside(t, 1) * val
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)

###############################################################################
# # Gt control


def calc_Gt_control_E(syst, i, Emin, Emax, quad_vec_kwargs={}):
    """Calculates |Gt)|² with integration in the energy domain.

    Deprecated. Integration in k-domain is faster.

    Parameters
    ----------
        syst : Kwant's finalized system

        t : np.array for which G_ij(t)< is calculated

        i : index i in G_ii(t)<

        Emin and Emax : limits of integration. Integral diverges at the band limits.

        quad_vec_kwargs : arguments for `scipy.integrate.quad_vec`.
    """
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = 0
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += 1/(2*np.pi) * \
                    wf(lead)[channel][i] * wf(lead)[channel][i].conj()

        return np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(
        integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    val = (integral[0][0] + 1j*integral[0][1])
    return val, integral[1:]


def integrand_Gt_control_intp(intp, k, i, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (26) in 1307.6419 for |G(t)|² using interpolation."""
    e = eps_i + 2*np.abs(gamma_wire)*np.cos(k)
    val = 0
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, i).conj()
    return -2*np.abs(gamma_wire)*np.sin(k)*np.stack((val.real, val.imag), -1)
