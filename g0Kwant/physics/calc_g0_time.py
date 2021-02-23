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
#   - different i,j indices: `nr_idx`
#   - split the integration domain a few times (2‒6): `nr_k`
#   - `scipy.integrate.quad_vec` parallelization: `nr_int`
# NOTE: We do M*(M+1)/2 calculations of different Gᵢⱼ if M is the number of all
#       i or j.
# NOTE: If M*(M+1)/2 is not a nice number compared to world.size it is perhaps
#       better to give partition manually and calculate on
#                   nr_idx·nr_k·nr_int < world.size
#       cores than to have
#                   nr_idx·nr_k·nr_int = world.size
#       and nr_idx small.
# NOTE, TODO: when splitting, we are not aware of the processor topology,
#       nr_int uses OMP parallelization and should be avoided (use nr_int = 1).
#       This can be avoided if we set partition manually and run the job with
#       combined parallelization.
#       The code only checks nr_idx·nr_k <= world.size!!!
#
# We can check the integration using the Eq. (26) in 1307.6419
#             ⌠ dE                        †
#         ∀t: ⎮ ── ∑ₐ Ψₐ,ᵢⱼ(E,t)Ψₐ,ᵢⱼ(E,t) = 𝟙
#             ⌡ 2π
# However, we are missing the oscillatory part exp(iEt). It could be useful to
# control the interpolation of the wave function.
#

from mpi4py import MPI
import sys
import uuid
import kwant
import numpy as np
import scipy as sc

from .fermi import fermi


##############################################################################
# # INTEGRATION

# This is needed for parallelization of quad_vec
# it can only pickled (needed for parallelization) functions which are global
# this makes them global
def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def _check_partition(world, partition, sites):
    if world is None:
        world = MPI.COMM_WORLD
    if partition is None:
        # We calculate only one triangle of the matrix
        # but we calculate G< and G>
        nr_idx_calc = 2*(len(sites)*(len(sites)+1)//2)
        nr_idx = np.gcd(nr_idx_calc, world.size)
        # k is divided at most four times
        nr_k = 4 - np.argmin([world.size//nr_idx % div for div in [4, 3, 2, 1]])
        nr_int = world.size//(nr_idx*nr_k)
    else:
        nr_idx, nr_k, nr_int = partition
    assert nr_idx*nr_k <= world.size, \
        "Number of workers: %d*%d*%d larger than number of processes: %d." % (
            nr_idx, nr_k, nr_int, world.size)
    if world.rank == 0:
        print("Combination: World size: % 2d Nr idx % 2d Nr k: % 2d Nr int: % 2d" % (
            world.size, nr_idx, nr_k, nr_int))
    return world, (nr_idx, nr_k, nr_int)


def calc_GtLG_integrals(syst, times, sites, ef, beta, eps_i=0, gamma_wire=1,
                        world=None, partition=None, epsrel=1e-8):
    """Calculates G_ij(t)< and G_ij(t)> for given times and sites."""
    world, (nr_idx, nr_k, nr_int) = _check_partition(world, partition, sites)

    # Prepare stuff
    idx_j, idx_i = np.meshgrid(sites, sites)
    # Create empty arrays
    GwL = np.zeros((times.shape[0], len(sites), len(sites)), dtype=np.complex)
    GwG = np.zeros((times.shape[0], len(sites), len(sites)), dtype=np.complex)

    # Calculate integration limits
    # 0, ..., nr_idx-1 calculate between 0 and np.pi/nr_k
    # nr_idx, ..., 2*nr_idx calculate between np.pi/nr_k and 2np.pi/nr_k, ...
    # where 0 calculates G<[0,0], 1 calculates G>[0,0], ...
    i_idx = world.rank % nr_idx
    i_k = (world.rank // nr_idx) % nr_k

    a = (i_k + 1) / nr_k * np.pi
    b = i_k / nr_k * np.pi

    # Integrate
    itr = 0
    cache_size = int(2*1024**3)  # 2 GB?
    pts = np.arccos(np.array(ef)/2)
    pts = np.unique(pts)
    for i in range(len(sites)):
        for j in range(i, len(sites)):  # use the fact G^<(t) = G^<(-t)^†
            if world.rank >= nr_idx*nr_k:  # leave free cores for integration
                continue
            if itr == i_idx:
                # print("%d calculates GtL %d %d from %4.3f to %4.3f" % (world.rank, i, j, a/np.pi, b/np.pi))
                @globalize
                def fun_GtL(k):
                    return integrand_GtL(syst, k, times, idx_i[i,j], idx_j[i,j], ef, beta, eps_i, gamma_wire)
                res, _ = integrate(fun_GtL, b, a, quad_vec_kwargs={"cache_size": cache_size, "workers": nr_int, "points": pts, "epsrel": epsrel })
                GwL[:, i, j] -= res
            itr += 1
            itr = itr % nr_idx
            if itr == i_idx:
                # print("%d calculates GtG %d %d from %4.3f to %4.3f" % (world.rank, i, j, a/np.pi, b/np.pi))
                @globalize
                def fun_GtG(k):
                    return integrand_GtG(syst, k, times, idx_i[i,j], idx_j[i,j], ef, beta, eps_i, gamma_wire)
                res, _ = integrate(fun_GtG, b, a, quad_vec_kwargs={"cache_size": cache_size, "workers": nr_int, "points": pts, "epsrel": epsrel })
                GwG[:, i, j] -= res

            itr += 1
            itr = itr % nr_idx

    world.barrier()

    # Reduce results from different processes
    GwL_red = np.zeros((times.shape[0], len(sites), len(sites)), dtype=np.complex)
    GwG_red = np.zeros((times.shape[0], len(sites), len(sites)), dtype=np.complex)
    world.Allreduce(GwL, GwL_red)
    world.Allreduce(GwG, GwG_red)

    # use the fact G^<(t) = G^<(-t)^†
    # moveaxis: transpose on the last two axes
    GwL_red -= (np.moveaxis(np.triu(GwL_red, 1), 1, -1).conj())[::-1]
    GwG_red -= (np.moveaxis(np.triu(GwG_red, 1), 1, -1).conj())[::-1]

    return GwL_red, GwG_red


def integrate(integrand, k_min, k_max, quad_vec_kwargs={"workers": 4, "full_output": False}):
    """Integrates integrand (some g(t)) on the interval `[k_min, k_max]`."""
    integral = sc.integrate.quad_vec(integrand, a=k_min, b=k_max, **quad_vec_kwargs)
    Gt = integral[0][..., 0] + 1j*integral[0][..., 1]
    return Gt, integral[1:]


##############################################################################
# # INTEGRANDS


def integrand_GtL(syst, k,  t, i, j, ef, beta, eps_i=0, gamma_wire=1):
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
    e = eps_i + 2*gamma_wire*np.cos(k)
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
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)


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
    e = eps_i + 2*gamma_wire*np.cos(k)
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
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtR(syst, k, t, i, j, eps_i=0, gamma_wire=1):
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    wf = kwant.wave_function(syst, energy=e)
    for lead in range(len(syst.leads)):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                    wf(lead)[channel][i] * wf(lead)[channel][j].conj() * \
                    np.exp(-1j*t*e)
    val = -1j * np.heaviside(t, 1) * val
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_Gt_control(syst, k, i, j, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (26) in 1307.6419 for |G(t)|².

    Parameters
    ----------
        syst : Kwant FiniteSystem

        k : wave vector

        i : site index

        eps_i and gamma_wire : parameters for the dispertion relation
    """
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = 0
    wf = kwant.wave_function(syst, energy=e)
    for lead in range(len(syst.leads)):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                    wf(lead)[channel][i] * wf(lead)[channel][j].conj()
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)


def calc_GtG_from_GtLR(t, GtL, GtR):
    """Calculates g(t)> from g(t)< and g(t)ᴿ.

    Formula is g^> = Gᴿ - gᴬ + g^< where gᴬ(t,t') = [gᴿ(t',t)]^†"""
    return GtR*(t >= 0) - GtR.conj()[::-1]*(t < 0) + GtL


###############################################################################
# # DEPRECATED

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
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += (-1)**channel * 1j/(2*np.pi) * fermi(e, ef[lead], beta) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, j).conj() *\
                np.exp(-1j*t*e)
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtL_intp_mat(intp, k, t, idx_i, idx_j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)< using interpolation.

    This function gives the whole matrix [Gᵢⱼ]. Integrating the whole matrix is
    slower then integrating the elements separately.

    idx_i and idx_j are 2D lists of indices. If we are intersted in matrix
                            |G00 G01|
                            |G10 G11|
    we supply [[0,0],[1,1]] and [[0,1], [0,1]].
    """
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = np.zeros((t.shape[0], len(idx_i), len(idx_j)), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            wf1 = np.array([[intp(k, lead, channel, i)
                             for i in row] for row in idx_i])
            wf2 = np.array([[intp(k, lead, channel, i)
                             for i in row] for row in idx_j]).conj()
            val += (-1)**channel * 1j/(2*np.pi) * fermi(e, ef[lead], beta) * \
                wf1 * wf2 * np.exp(-1j*t*e)[:, np.newaxis, np.newaxis]
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)

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
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += (-1)**channel * 1j/(2*np.pi) * \
                (fermi(e, ef[lead], beta)-1) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, j).conj() *\
                np.exp(-1j*t*e)
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)


def integrand_GtG_intp_mat(intp, k, t, idx_i, idx_j, ef, beta, eps_i=0, gamma_wire=1):
    """Integrand from Eq. (22) in 1307.6419 for G(t)< using interpolation.

    This function gives the whole matrix [Gᵢⱼ]. Integrating the whole matrix is
    slower then integrating the elements separately.

    idx_i and idx_j are 2D lists of indices. If we are intersted in matrix
                            |G00 G01|
                            |G10 G11|
    we supply [[0,0],[1,1]] and [[0,1], [0,1]].
    """
    e = eps_i + 2*gamma_wire*np.cos(k)
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
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)

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
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, j).conj() *\
                np.exp(-1j*t*e)
    val = -1j * np.heaviside(t, 1) * val
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)

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
    e = eps_i + 2*gamma_wire*np.cos(k)
    val = 0
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * \
                intp(k, lead, channel, i) * intp(k, lead, channel, i).conj()
    return -2*gamma_wire*np.sin(k)*np.stack((val.real, val.imag), -1)
