import numpy as np
from g0Kwant.geom.wire import *
from g0Kwant.physics.g0 import *
from g0Kwant.physics.calc_g0_energy import *
from g0Kwant.physics.calc_sigma import *
from g0Kwant.physics.calc_g0_time import *


def test_g0_energy_consistency():
    L = 3
    Gamma = 0.7
    gamma = np.sqrt(Gamma/2.0)
    gamma_wire = 1.5
    eps_d = 0.3
    bias = 0.2
    ef_left = -bias/2
    ef_right = bias/2
    beta = 100.0

    wire = wire_with_quantum_dot(
        L, eps_d, gamma, eps_i=0, gamma_wire=gamma_wire)
    wire = wire.finalized()

    eng = np.linspace(-2*gamma_wire + 1e-5, 2*gamma_wire - 1e-5, 101)

    # Analytical calculation
    GER00_ana = GER00_general(eng, eps_d, gamma, gamma_wire)
    GEL00_ana = GEL00_general(eng, eps_d, gamma, ef_left, ef_right, beta, gamma_wire)
    GER01_ana = GER01_general(eng, eps_d, gamma, gamma_wire)

    # Using inverse
    GER_inv = calc_GER_inverse(wire, eng)
    assert np.allclose(GER00_ana, GER_inv[:, L, L])
    assert np.allclose(GER01_ana, GER_inv[:, L, L + 1])

    i0 = wire.lead_interfaces[0][0]
    assert i0 == 0
    SEL_0 = calc_SigmaEL(wire, 0, eng, ef_left, beta)
    i1 = wire.lead_interfaces[1][0]
    assert i1 == 2*L
    SEL_1 = calc_SigmaEL(wire, 1, eng, ef_right, beta)
    SELs = [(SEL_0, i0, i0), (SEL_1, i1, i1)]

    GEL_inv = calc_GELG_inverse(GER_inv, SELs)
    assert np.allclose(GEL00_ana, GEL_inv[:, L, L])

    # Using inverse via SigmaER
    i0 = wire.lead_interfaces[0][0]
    SER_0 = calc_SigmaER(wire, 0, eng)
    i1 = wire.lead_interfaces[1][0]
    SER_1 = calc_SigmaER(wire, 1, eng)
    ham = wire.hamiltonian_submatrix()
    SERs = [(SER_0, i0, i0), (SER_1, i1, i1)]

    GER_inv = calc_GER_inverse_from_SER(ham, eng, SERs)
    assert np.allclose(GER00_ana, GER_inv[:, L, L])
    assert np.allclose(GER01_ana, GER_inv[:, L, L + 1])

    # Using wave function
    GEL_wf, GEG_wf = calc_GELG(wire, eng, [ef_left, ef_right], beta)
    assert np.allclose(GEL00_ana, GEL_wf[:, L, L])

    # Test G^> - G^< = G^R - G^A
    assert np.allclose((GEG_wf - GEL_wf).imag, 2*GER_inv.imag)


def test_g0_time_integrand_to_energy():
    L = 3
    Gamma = 0.7
    gamma = np.sqrt(Gamma/2.0)
    gamma_wire = 1.5
    eps_d = 0.3
    bias = 0.2
    ef_left = -bias/2
    ef_right = bias/2
    beta = 100.0

    wire = wire_with_quantum_dot(
        L, eps_d, gamma, eps_i=0, gamma_wire=gamma_wire)
    wire = wire.finalized()

    k = np.linspace(1e-5, np.pi - 1e-5, 101)
    eng = 2*np.abs(gamma_wire)*np.cos(k)
    dedk = -2*np.abs(gamma_wire)*np.sin(k)  # de/dk, used in the integrand

    GEL_wf, GEG_wf = calc_GELG(wire, eng, [ef_left, ef_right], beta)

    GEL_int_ = np.array([integrand_GtL(wire, _k, np.array([0]), L, L + 2, [ef_left, ef_right], beta, 0, gamma_wire) for _k in k])
    GEG_int_ = np.array([integrand_GtG(wire, _k, np.array([0]), L, L + 2, [ef_left, ef_right], beta, 0, gamma_wire) for _k in k])
    GEL_int = (GEL_int_[:, 0, 0] + 1j*GEL_int_[:, 0, 1])*2*np.pi/dedk
    GEG_int = (GEG_int_[:, 0, 0] + 1j*GEG_int_[:, 0, 1])*2*np.pi/dedk
    assert np.allclose(GEL_int, GEL_wf[:, L, L + 2])
    assert np.allclose(GEG_int, GEG_wf[:, L, L + 2])
