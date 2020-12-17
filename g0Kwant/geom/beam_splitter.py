# # Beam splitter
#
# We want to have a beam splitter with specified Tᵢⱼ. Beam splitter is defined
# by the vertical hoppings between the wires. We can optimize them for the
# desired Tᵢⱼ. Because of the symmetry we are only intersted T₀ⱼ.
#
# We can describe the transition with 3 parameters:
#   - maximum hopping strength t0,
#   - transition from 0 to t0 and from t0 back to 0,
#   - length of constant section at t0.
#
# Few observations:
#   - transition should have a sigmoid shape (I don't know if this the optimal)
#   - there are minima for several t0, the lowest one is the best, higher t0
#     have higher harmonics in Tᵢⱼ.
#   - sigmoid of shape 4 without the constant section is good enough
#   - T₀₀, T₀₂ will be lower for longer the transition and constant section but
#     they are already small (~1e-3) for sigmoid(4).
#   - t0 of minima get smaller with longer bridge
#
# ## Aharonov-Bohm ring
#   - For the Aharonov-Bohm ring we want T₀₀ = 0, T₀₁ = ½, T₀₂ = 0, T₀₃ = ½.
#     This gives the maximum range (?) for T_ij oscillations of the ring.
#   - We want to put more weight to points near the transmission maximum of the
#     QD in the ring. This maximum can be narrow.
#   - Larger the εd (offset from E = 0) and γ (peak width), larger will be the
#     discrepancy T₀₁ - ½ and T₀₃ - ½.
#

import kwant
import numpy as np
from scipy.optimize import root, minimize_scalar
from ..physics.calc_transmission import calc_transmission_parallel


###############################################################################
# Optimization


def sigmoid_start(N, k=3):
    """ Returns the starting x0 for the `sigmoid`.

    For x = np.linspace(-1, 1, 51) the jump between `sigmoid_x(x)[0]` and
    `sigmoid_x(x)[1]` will be larger than between 0 and `sigmoid_x(x)[0]`.

    For x = np.linspace(-10, 10, 11) we are putting too many points in the
    almost constant tails.

    This function finds the optimal x0 (where np.linspace(-x0, x0, N)) for a
    given parameter k, such that
                    k*sigmoid_x(x)[0] == sigmoid_x(x)[1]
    Technically, it finds the negative x0.

    For large N, we limit x0 to 7 which gives sigmoid_x(-7) ~ 1e-3.
    For large N, k is irrelevant, except if k ~> 1.
    For small N, k > 3 can be beneficial?
    """
    def f(x0): return k*sigmoid_x(x0) - sigmoid_x(x0 - 2*x0/(N-1))
    res = root(f, 0)
    return min(-res.x[0], 7)


def sigmoid_x(x):
    return 1/(1+np.exp(-x))


def sigmoid(N, k=1):
    if N == 0:
        return np.array([]), np.array([])
    if N == 1:
        return np.array([0]), np.array([0.5])
    s0 = sigmoid_start(N, k)
    x = np.linspace(-s0, s0, N)
    return x, sigmoid_x(x)


def t_sigmoid(t0, L_const, L_transition, k=3):
    _, transition = sigmoid(L_transition, k=k)
    t_bridge = t0 * np.concatenate((transition,
                                    np.ones(L_const+2),
                                    transition[::-1]))
    return t_bridge


def beam_transmission(eng, t0, L_const, L_transition, k=3.0, workers=4):
    """Calculates transmissions T_ij for a given beam geometry.

    Parameters
    ----------
    eng : numpy array
        energies in which transmission is calculated
    t0, L_const, L_transition : double, integer, integer
        parameters of the beam geometry
    k : double
        parameter for the sigmoid_start
    workers : integer
        number of processors for calc_transmission_parallel
    """
    syst, lat = beam_splitter(t_sigmoid(t0, L_const, L_transition))
    syst = syst.finalized()
    return calc_transmission_parallel(syst, eng, workers)


def _cost_function(t0, L_const, L_transition, eng, t_opt, weight=1, k=3):
    trans = beam_transmission(eng, t0, L_const, L_transition, k=k)
    diff = trans[:, 0, :] - t_opt
    return np.sum(np.abs(diff)**2*weight)


def _find_first_minimum(L_const, L_transition, eng, t_opt, weight=1, dt=0.025, k=3):
    """Finds first minimum of t0.

    It makes steps dt from t0 = 0 until it crosses a minimum. One should lower
    stepping t0 for longer beam splitter. For t_opt = [0,0.5,0,0.5], cost
    functions starts with a maximum. For t_opt = [0, 0.9, 0, 0.1], it starts
    closer to the first minimum, so one should also decrease t0.

    Parameters
    ----------
    t_opt: 1D numpy array
        Desired transmissions np.array([T00, T01, T02, T03])
    """
    t0 = dt
    prev = 1e99
    while True:
        val = _cost_function(t0, L_const, L_transition, eng, t_opt, weight=weight, k=k)
        if val < prev:
            prev = val
            t0 += dt
        else:
            break
    return [t0-2*dt, t0 - dt, t0]


def find_optimal_t0(L_const, L_transition, eng, t_opt, weight=1, dt=0.025, k=3):
    bracket = _find_first_minimum(L_const, L_transition, eng, t_opt, weight=weight, dt=dt, k=k)

    def cf(t0):
        return _cost_function(t0, L_const, L_transition, eng, t_opt, weight=weight)

    res = minimize_scalar(cf, bracket=bracket, method="brent")
    return res

###############################################################################
# Geometry

def _beam_splitter(syst, lat, gamma_beam, eps_i, gamma_wire, L0=0):
    """Creates a beam splitter starting at position L0 without leads."""

    L = len(gamma_beam)
    # On site potential
    for i in range(L0, L0 + L + 2):
        syst[lat(i, 0)] = eps_i
        syst[lat(i, 1)] = eps_i
    # Hoppings
    for i in range(L0, L0 + L + 1):
        syst[lat(i, 0), lat(i + 1, 0)] = gamma_wire
        syst[lat(i, 1), lat(i + 1, 1)] = gamma_wire
    for il, i in enumerate(range(L0 + 1, L0 + L + 1)):
        syst[lat(i, 0), lat(i, 1)] = gamma_beam[il]


def beam_splitter(gamma_beam, eps_i=0, gamma_wire=1):
    """Creates a beam splitter with leads attached.

    Example where `len(gamma_beam)` = 4
         0   ┄┄┄─○──●───●───●───●───●───●──○─┄┄┄   1 (= Kwant terminals)
                        │   │   │   │
         2   ┄┄┄─○──●───●───●───●───●───●──○─┄┄┄   4

    Parameters
    ----------
    gamma_beam : 1D array, length L
        vertical hopping in the beam
    eps_i :
        on site potential in the leads (● and ⋯)
    gamma_wire :
        horizontal hopping in the leads (●──●, ●──○ and ○‒⋯)
    """

    lat = kwant.lattice.square(1.0)
    syst = kwant.Builder()

    # Make central part
    _beam_splitter(syst, lat, gamma_beam, eps_i, gamma_wire, L0=0)

    # Attach leads
    sym_lead = kwant.TranslationalSymmetry((-1, 0))
    for i in [1, 0]:
        lead = kwant.Builder(sym_lead)
        lead[lat(0, i)] = eps_i
        lead[lat.neighbors()] = gamma_wire

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    return syst, lat
