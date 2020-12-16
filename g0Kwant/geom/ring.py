import kwant
import numpy as np


def _beam_splitter(syst, lat, gamma_beam, eps_i, gamma_wire, L0=0):
    """Creates a beam splitter starting at position L0 without leads.

    Example where `len(gamma_beam)` = 4
             ┄┄┄─○──●───●───●───●───●───●──○─┄┄┄
                        │   │   │   │
             ┄┄┄─○──●───●───●───●───●───●──○─┄┄┄

                    |       |           |
                    L0      L0+2        L0+5   <── positions

    gamma_beam : 1D array, length L:
        vertical hopping in the beam
    eps_i :
        on site potential in the leads (● and ⋯)
    gamma_wire :
        horizontal hopping in the leads (●──●, ●──○ and ○‒⋯)
    L0 : starting position of the beam
    """

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
    _beam_splitter(syst, lat, gamma_beam, eps_i, gamma_wire, 0)

    # Attach leads
    sym_lead = kwant.TranslationalSymmetry((-1, 0))
    for i in [1, 0]:
        lead = kwant.Builder(sym_lead)
        lead[lat(0, i)] = eps_i
        lead[lat.neighbors()] = gamma_wire

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    return syst, lat


def ring(L, QD, phase, gamma_beam, eps_i=0, gamma_wire=1):
    """Creates an Aharonov-Bohm ring with beam splitters and mag. field

    Example where `len(gamma_beam)` = 3 and `L` = 3

                                magnetic field
                                      A 
         ┄┄┄─○──●───●───●───●───●───●─┼─●───●───●───●───●──○─┄┄┄
                    │   │   │         ┴     │   │   │
         ┄┄┄─○──●───●───●───●───●───◎───●───●───●───●───●──○─┄┄┄

                |               |   |   |  
               -L_beam-1        0   1   L-1                <── positions

    L : 
        length of the ring
    QD :
        parameters for the quantum dot: (εd, γ) or ((x,y), εd, γ)
    phase :
        parameters for phase: phase or ((x,y), phase)
    gamma_beam : 1D array, length L
        vertical hopping in the beam
    eps_i :
        on site potential in the leads (● and ⋯)
    gamma_wire :
        horizontal hopping in the leads (●──●, ●──○ and ○‒⋯)

    """
    lat = kwant.lattice.square(1.0)
    syst = kwant.Builder()

    # Beam splitters
    L_beam = len(gamma_beam)
    _beam_splitter(syst, lat, gamma_beam, eps_i, gamma_wire, -L_beam - 1)
    _beam_splitter(syst, lat, gamma_beam[::-1], eps_i, gamma_wire, L - 1)

    # Add ring
    for i in range(0, L):
        syst[lat(i, 0)] = eps_i
        syst[lat(i, 1)] = eps_i
    for i in range(0, L):
        syst[lat(i, 0), lat(i+1, 0)] = gamma_wire
        syst[lat(i, 1), lat(i+1, 1)] = gamma_wire

    # Add magnetic field
    if np.isscalar(phase):
        i, j = L//2, 0
        phi = phase
    else:
        assert len(phase) == 2
        i, j = phase[0]
        phi = phase = 1
    syst[lat(i, j), lat(i+1, j)] = gamma_wire*np.exp(1j*phi)

    # Add quantum dot
    assert len(QD) == 2 or len(QD) == 3
    if len(QD) == 2:
        i, j = L//2, 1
        eps_d, gamma = QD[0], QD[1]
    if len(QD) == 3:
        i, j = QD[0]
        eps_d, gamma = QD[1], QD[2]
    syst[lat(i, j)] = eps_d
    syst[lat(i-1, j), lat(i, j)] = gamma
    syst[lat(i, j), lat(i+1, j)] = gamma

    # Attach leads
    sym_lead = kwant.TranslationalSymmetry((-1, 0))
    for i in [1, 0]:
        lead = kwant.Builder(sym_lead)
        lead[lat(0, i)] = eps_i
        lead[lat.neighbors()] = gamma_wire

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    return syst, lat
