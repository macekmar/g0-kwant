import numpy as np
from g0Kwant.physics.calc_transmission import *
from g0Kwant.geom.wire import wire_with_quantum_dot


# def test_parallel():
#     eps_d = -0.5
#     Gamma = 0.1
#     half_bandwidth = 2
#     gamma_wire = half_bandwidth/2.0
#     gamma = (half_bandwidth*Gamma/4)**0.5
#     wire = wire_with_quantum_dot(3, eps_d, gamma, gamma_wire=gamma_wire)
#     wire = wire.finalized()

#     eng = np.linspace(-half_bandwidth + 1e-15, half_bandwidth - 1e-15, 41)
#     T1 = calc_transmission(wire, eng)
#     T2 = calc_transmission_parallel(wire, eng)
#     assert np.allclose(T1, T2)


# def test_consistency_T_QD():
#     eps_d = -0.5
#     Gamma = 0.1
#     half_bandwidth = 2
#     gamma_wire = half_bandwidth/2.0
#     gamma = (half_bandwidth*Gamma/4)**0.5
#     wire = wire_with_quantum_dot(3, eps_d, gamma, gamma_wire=gamma_wire)
#     wire = wire.finalized()

#     eng = np.linspace(-half_bandwidth + 1e-15, half_bandwidth - 1e-15, 41)
#     T01_Kwant = calc_transmission(wire, eng)[:, 0, 1]
#     T01_ana = transmission_QD_wire(eng, eps_d, gamma, gamma_wire)
#     assert np.allclose(T01_Kwant, T01_ana)

#     eng = transmission_QD_wire_peak_pos(eps_d, gamma, gamma_wire)
#     T01_Kwant = calc_transmission(wire, eng)[:, 0, 1]
#     T01_ana = transmission_QD_wire(eng, eps_d, gamma, gamma_wire)
#     assert np.allclose(T01_Kwant, np.array([0.5, 1, 0.5]))
#     assert np.allclose(T01_ana, np.array([0.5, 1, 0.5]))