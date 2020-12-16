import numpy as np
import kwant
from g0Kwant.geom.wire import wire_with_quantum_dot

# In calc_g0_time.py we use hard-coded analytical formulas for dispertion
# relation because we also need its derivative


def test_dispersion():
    L = 1
    eps_d = 0.0
    gamma = 1.0
    eps_i = -0.2
    gamma_wire = 1.0

    wire = wire_with_quantum_dot(L, eps_d, gamma, eps_i=eps_i, gamma_wire=gamma_wire)
    wire = wire.finalized()

    bands = kwant.physics.Bands(wire.leads[0])
    k = np.linspace(-np.pi+1e-5, np.pi-1e-5, 101)
    eng = np.array([bands(_k)[0] for _k in k])
    # TODO: implement this as a function in calc_g0_time
    eng_ana = eps_i + 2*np.abs(gamma_wire)*np.cos(k)

    assert np.allclose(eng, eng_ana)
