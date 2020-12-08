import kwant
import numpy as np

def _beam_splitter(syst, lat, t, U, t_beam, L0=0):
    """Creates a beam splitter starting at position L0 without leads.

    Example where `len(t_beam)` = 4
                ┄┄┄─○───○───○───○───○───○─┄┄┄
                        │   │   │   │
                ┄┄┄─○───○───○───○───○───○─┄┄┄

                    |       |           |
                    L0      L0+2        L0+5   <── positions

    `U` :       on site potential in the leads (○ and ⋯)
    `t` :       horizontal hopping in the leads (○──○ and ○‒⋯)
    `t_beam` :  1D array, length L: vertical hopping in the beam
    `L0` :      starting position of the beam
    """

    L = len(t_beam)
    # On site potential
    for i in range(L0, L0 + L + 2):
        syst[lat(i, 0)] = U
        syst[lat(i, 1)] = U
    # Hoppings
    for i in range(L0, L0 + L + 1):
        syst[lat(i, 0), lat(i + 1, 0)] = t
        syst[lat(i, 1), lat(i + 1, 1)] = t
    for il, i in enumerate(range(L0 + 1, L0 + L + 1)):
        syst[lat(i, 0), lat(i, 1)] = t_beam[il]

def beam_splitter(t, U, t_beam):
    """Creates a beam splitter with leads attached.

        `U` :       on site potential in the leads (○ and ⋯)
        `t` :       horizontal hopping in the leads (○──○ and ○‒⋯)
        `t_beam` :  1D array, length L: vertical hopping in the beam"""

    lat = kwant.lattice.square(1.0)
    syst = kwant.Builder()

    # Make central part
    _beam_splitter(syst, lat, t, U, t_beam, 0)

    # Attach leads
    sym_lead = kwant.TranslationalSymmetry((-1, 0))
    for i in [1, 0]:
        lead = kwant.Builder(sym_lead)
        lead[lat(0, i)] = U
        lead[lat.neighbors()] = t

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    return syst, lat


def ring(t, U, t_beam, L, phase=([0,0],0), QD=([0,1], 0, 1)):
    """Creates an Aharonov-Bohm ring with beam splitters and mag. field

    Example where `len(t_beam)` = 3 and `L` = 3

                                magnetic field
                                      A 
            ┄┄┄─○───○───○───○───○───○─┼─○───○───○───○───○─┄┄┄
                    │   │   │         ┴     │   │   │
            ┄┄┄─○───○───○───○───○───○───○───○───○───○───○─┄┄┄

                |               |   |   |  
               -L_beam-1        0   1   L-1                <── positions

    """
    lat = kwant.lattice.square(1.0)
    syst = kwant.Builder()

    # Beam splitters
    L_beam = len(t_beam)
    _beam_splitter(syst, lat, t, U, t_beam, -L_beam - 1)
    _beam_splitter(syst, lat, t, U, t_beam[::-1], L - 1)

    # Add ring
    for i in range(0, L):
        syst[lat(i, 0)] = U
        syst[lat(i, 1)] = U
    for i in range(0, L):
        syst[lat(i,0), lat(i+1,0)] = t
        syst[lat(i,1), lat(i+1,1)] = t

    # Add magnetic field
    syst[lat(phase[0][0],phase[0][1]), lat(phase[0][0]+1,phase[0][1])] = t*np.exp(1j*phase[1])
    # Add quantum dot
    pos = QD[0]
    eps_d = QD[1]
    gamma = QD[2]
    syst[lat(pos[0], pos[1])] = eps_d
    syst[lat(pos[0]-1, pos[1]), lat(pos[0], pos[1])] = gamma
    syst[lat(pos[0], pos[1]), lat(pos[0]+1, pos[1])] = gamma

    # Attach leads
    sym_lead = kwant.TranslationalSymmetry((-1, 0))
    for i in [1, 0]:
        lead = kwant.Builder(sym_lead)
        lead[lat(0, i)] = U
        lead[lat.neighbors()] = t

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    return syst, lat
