import kwant

def wire_with_quantum_dot(L, ed, gamma, U=0, t=1):
    """Creates a 1D wire with a quantum dot (QD) at position 0.

    Example for L = 2:
                ┄┄┄─○──●──●──◎──●──●──○─┄┄┄
        ◎   - quantum dot, 
        ●   - lead sites part of the finite system, 
        ○─┄ - leads

    `L` :       >0, length of leads in system
    `ed` :      potential on the quantum dot (◎)
    `gamma` :   hopping between the quantum dot an the leads (●──◎)
    `U` : on site potential in the leads (○ and ┄┄)
    `t` : hopping in the leads (●──●, ●──○ and ○─┄┄)

    Note, in 2D hoppings on site potential has a term -4t. This is not the 
    case here.
    """
    lat = kwant.lattice.chain(1)
    wire = kwant.Builder()

    # Extended leads
    for i in range(-L,L+1):
        wire[lat(i)] = U
        wire[lat.neighbors()] = t
    # Quantum dot
    wire[lat(0)] = ed
    wire[lat(-1),lat(0)] = gamma
    wire[lat(0),lat(1)] = gamma

    # Leads
    sym_lead = kwant.TranslationalSymmetry((-1,))
    lead = kwant.Builder(sym_lead)
    lead[lat(0)] = U
    lead[lat.neighbors()] = t

    wire.attach_lead(lead)
    wire.attach_lead(lead.reversed())

    return wire
