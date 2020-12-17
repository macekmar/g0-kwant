import numpy as np
from g0Kwant.physics.g0 import *


def test_g0_analytical_formula_consistency():
    E = np.linspace(-1.99, 1.99, 201)
    eps_d = 0.0
    beta = -1
    assert np.allclose(GER00(E), GER00_general(E, eps_d, 1, 1))
    assert np.allclose(GER01(E), GER01_general(E, eps_d, 1, 1))
    for ef in [-3, -0.7, 1.2]:
        assert np.allclose(GEL00(E, ef),
                           GEL00_general(E, eps_d, 1, ef, ef, beta, 1))
