import sys
import uuid
import numpy as np
from g0Kwant.geom.wire import *
from g0Kwant.physics.g0 import *
from g0Kwant.physics.calc_g0_time import *


# This is needed for parallelization of quad_vec
# it can only pickle (needed for parallelization) functions which are global
# this makes them global
def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def test_g0_time_consistency():
    L = 3
    gamma = 1.0
    gamma_wire = 1.0
    eps_d = 0.0
    ef_left = 0
    ef_right = 0
    beta = -1

    wire = wire_with_quantum_dot(L, eps_d, gamma, eps_i=0, gamma_wire=gamma_wire)
    wire = wire.finalized()

    times = np.linspace(-5, 5, 101)

    @globalize
    def int_GtR00(k):
        return integrand_GtR(wire, k, times, L, L, 0, gamma_wire)

    @globalize
    def int_GtL00(k):
        return integrand_GtL(wire, k, times, L, L, [ef_left, ef_right], beta, 0, gamma_wire)

    @globalize
    def int_GtL01(k):
        return integrand_GtL(wire, k, times, L, L + 1, [ef_left, ef_right], beta, 0, gamma_wire)

    # Analytical results
    GtR00_ana = GtR00(times)
    GtL00_ana = GtL00(times)
    GtL01_ana = GtL01(times)

    # Using integration
    GtR00_int, err = calc_Gt_integral(int_GtR00, np.pi, 0)
    GtL00_int, err = calc_Gt_integral(int_GtL00, np.pi, 0)
    GtL01_int, err = calc_Gt_integral(int_GtL01, np.pi, 0)

    assert np.allclose(GtR00_int, GtR00_ana)
    assert np.allclose(GtL00_int, GtL00_ana)
    assert np.allclose(GtL01_int, GtL01_ana)


def test_normalization_condition():
    L = 3
    Gamma = 0.7
    gamma = np.sqrt(Gamma/2.0)
    gamma_wire = 1.5
    eps_d = 0.3
    bias = 0.2
    ef_left = -bias/2
    ef_right = bias/2
    beta = 100.0

    wire = wire_with_quantum_dot(L, eps_d, gamma, eps_i=0, gamma_wire=gamma_wire)
    wire = wire.finalized()

    @globalize
    def int_GtControl_00(k):
        return integrand_Gt_control(wire, k, L, L, 0, gamma_wire)
    
    @globalize
    def int_GtControl_01(k):
        return integrand_Gt_control(wire, k, L, L+1, 0, gamma_wire)

    @globalize
    def int_GtControl_LL(k):
        return integrand_Gt_control(wire, k, 2*L, 2*L, 0, gamma_wire)

    C00, err = calc_Gt_integral(int_GtControl_00, np.pi, 0)
    C01, err = calc_Gt_integral(int_GtControl_01, np.pi, 0)
    CLL, err = calc_Gt_integral(int_GtControl_LL, np.pi, 0)

    assert np.allclose(C00, 1)
    assert np.allclose(C01, 0)
    assert np.allclose(CLL, 1)