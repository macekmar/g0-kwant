import kwant
import numpy as np
import matplotlib.pyplot as plt
from .beam_splitter import _beam_splitter


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


def plot_ring(syst, eps_d):
    """Plots ring with some additional data."""
    def site_color(site):
        def cmap(e):
            # color limits in +- range
            r = 2*np.abs(eps_d)
            return plt.get_cmap("RdYlBu")((e + r) / (2*r))
        return cmap(syst.hamiltonian(site, site).real)

    def hop_lw(s1, s2):
        return min(0.2, np.abs(syst.hamiltonian(s1, s2))/3)

    def hop_color(s1, s2):
        def cmap(angle):
            # So that there is no jump pi -> -pi for np.angle(-1-+0.01j)
            # TODO: % 2 and then /2???
            angle = (2 + angle/np.pi) % 2
            return plt.get_cmap("RdBu_r")(angle/2)
        return cmap(np.angle(syst.hamiltonian(s1, s2)))

    fig, ax = plt.subplots(figsize=(13, 8))
    kwant.plot(syst,
               site_symbol="o", site_color=site_color, site_edgecolor="k", site_lw=0.05, site_size=0.35,
               hop_lw=hop_lw, hop_color=hop_color,
               ax=ax)
    ax.set_aspect("equal")

    # Print site numbers
    keys = list(syst.id_by_site.keys())
    for site in keys:
        x, y = site.pos
        ax.text(x-0.045, y-0.02, "% 2d" %
                syst.id_by_site[site], horizontalalignment='center', verticalalignment='center', fontsize=9)

    # Print leads' numbers
    for i, leads in enumerate(syst.lead_interfaces):
        x, y = syst.sites[leads[0]].pos  # Assuming 1D wires
        if x < 0:
            ax.text(x - 3, y, "T=%d" % i, fontsize=12,
                    horizontalalignment='right', verticalalignment='center')
        else:
            ax.text(x + 3, y, "T=%d" % i, fontsize=12,
                    horizontalalignment='left', verticalalignment='center')

    return fig, ax
