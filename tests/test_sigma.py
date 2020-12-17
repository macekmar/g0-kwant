import numpy as np
from g0Kwant.physics.sigma import *


def test_conditions():
    E = np.linspace(-1.99, 1.99, 201)
    assert np.allclose(SigmaEL_general(E, -3, -1, 1, 1), np.zeros_like(E))


def test_Sigma_analytical_formula_consistency():
    E = np.linspace(-1.99, 1.99, 201)
    for ef in [-3, -0.7, 1.2]:
        assert np.allclose(SigmaER(E), SigmaER_general(E, 1, 1))
        assert np.allclose(SigmaEL(E, ef), SigmaEL_general(E, ef, -1, 1, 1))
