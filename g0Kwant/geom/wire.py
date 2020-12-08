import kwant

def wire_with_quantum_dot(L, eps_d, gamma_dot, eps_i=0, gamma_wire=1):
    """Creates a 1D wire with a quantum dot (QD) at position 0.

    Example for L = 2:
                ┄┄┄─○──●──●──◎──●──●──○─┄┄┄
        ◎   - quantum dot, 
        ●   - lead sites part of the finite system, 
        ○─┄ - leads

    `L`          : >0, length of leads in system
    `eps_d`      : potential on the quantum dot (◎)
    `gamma_dot`  : hopping between the quantum dot an the leads (●──◎)
    `eps_i`      : on site potential in the leads (○ and ┄┄)
    `gamma_wire` : hopping in the leads (●──●, ●──○ and ○─┄┄)

    Note, in 2D hoppings on site potential has a term -4t. This is not the
    case here.
    """
    lat = kwant.lattice.chain(1)
    wire = kwant.Builder()

    # Extended leads
    for i in range(-L,L+1):
        wire[lat(i)] = eps_i
        wire[lat.neighbors()] = gamma_wire
    # Quantum dot
    wire[lat(0)] = eps_d
    wire[lat(-1),lat(0)] = gamma_dot
    wire[lat(0),lat(1)] = gamma_dot

    # Leads
    sym_lead = kwant.TranslationalSymmetry((-1,))
    lead = kwant.Builder(sym_lead)
    lead[lat(0)] = eps_i
    lead[lat.neighbors()] = gamma_wire

    wire.attach_lead(lead)
    wire.attach_lead(lead.reversed())

    return wire
