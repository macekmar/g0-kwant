import numpy as np
import scipy as sc
# Analytical results from paper 1307.6419, Eq. C.3-C.12

## Self energy
def SER(e, Gamma=2):
    return (Gamma/2)*np.piecewise(
            np.array(e, dtype=np.complex), [e < -2.0, (e >= -2.0) & (e <= 2.0)],
           [lambda e: (e/2.0 + np.sqrt((e/2.0)**2 - 1)),
            lambda e: (e/2.0 - 1j*np.sqrt(1 - (e/2.0)**2)),
            lambda e: (e/2.0 - np.sqrt((e/2.0)**2 - 1))
           ])
def SEL(e, ef, Gamma=2):
    return (Gamma/2)*np.piecewise(
            np.array(e, dtype=np.complex), [(e >= -2.0) & (e <= ef)],
           [lambda e: (2j*(np.sqrt(1 - (e/2.0)**2))),
            lambda e: 0
           ])

def SER_general(e, gamma_att=1.0, gamma_lead=1.0):
    """Calculates Σ(E)^R for a perfect semi-infinite lead.

    Equation is:
                    2               ______________
                |γ'|  ⎛  E         ╱ ⎛  E  ⎞2
                ───── ⎜───── ±    ╱  ⎜─────⎟  ‒ 1
                |γ"|  ⎝2|γ"|    ╲╱   ⎝2|γ"|⎠
    for E...


    where γ" is hopping in the leads and γ' is the hopping at the attachemnt.
    
    In Kwant we have to take at least 3 sites:

                          γ'  γ'  γ   γ   γ' γ'  
                        ┄┄┄─○───●───◎───●───○─┄┄┄

    where ● are part of the scattering region and ◎ is the quantum dot.
    So, Σ(E)^R is for attachment to ○───● and γ' = γ".
    """            

    e = e/(2.0*np.abs(gamma_lead))
    return np.abs(gamma_att)**2/np.abs(gamma_lead)*np.piecewise(
            np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
           [lambda e: (e + np.sqrt(e**2 - 1)),
            lambda e: (e - 1j*np.sqrt(1 - e**2)),
            lambda e: (e - np.sqrt(e**2 - 1))
           ])
def SEL_general(e, ef, gamma_att=1.0, gamma_lead=1.0):
    e = e/(2.0*np.abs(gamma_lead))
    ef = ef/(2.0*np.abs(gamma_lead))
    return np.abs(gamma_att)**2/np.abs(gamma_lead)*np.piecewise(
            np.array(e, dtype=np.complex), [(e >= -1.0) & (e <= ef)],
           [lambda e: (2j*(np.sqrt(1 - e**2))),
            lambda e: 0
           ])



## x,x Green functions
### energy domain
def GER00_general(engs, ed, gamma_dot, gamma_wire):
    return 1.0/(engs - ed - 2*SER_general(engs, gamma_dot, gamma_wire))

def GER00(e):
    return np.piecewise(np.array(e, dtype=np.complex), [e < -2.0, (e >= -2.0) & (e <= 2.0)],
           [lambda e: -1/2.0 / np.sqrt((e/2.0)**2-1),
            lambda e: 1/2.0 /( 1j*np.sqrt(1-(e/2.0)**2) ),
            lambda e: 1/2.0 / np.sqrt((e/2.0)**2-1)
           ])

def GEL00_general(engs, ef, ed, gamma_dot, gamma_wire):
    ger = GER00_general(engs, ed, gamma_dot, gamma_wire)
    sel = 2*SEL_general(engs, ef, gamma_dot, gamma_wire)
    return ger*sel*ger.conj()


def GEL00(e, ef):
    return np.piecewise(np.array(e, dtype=np.complex), [(e >= -2.0) & (e <= ef)],
           [lambda e: (1j / (np.sqrt(1-(e/2.0)**2))),
            lambda e: 0
           ])

### time domain
def GtR00(ts):
    return -1j*sc.special.jv(0,2*ts)*np.heaviside(ts,1)
def GtL00(ts):
    return 0.5j*sc.special.jv(0,2*ts) - 0.5*sc.special.struve(0,2*ts)

## x,x+1 Green functions
def GER01(e):
    e = e/2.0
    a = np.piecewise(
            np.array(e, dtype=np.complex), [e < -1.0, (e >= -1.0) & (e <= 1.0)],
           [lambda e: (e - np.sqrt(e**2 - 1)),
            lambda e: (e + 1j*np.sqrt(1 - e**2)),
            lambda e: (e + np.sqrt(e**2 - 1))
           ]) 
    return  1/(a**2-1)

def GEL01(e, ef):
    return np.piecewise(np.array(e, dtype=np.complex), [(e >= -2.0) & (e <= ef)],
           [lambda e: (1j*e/2/ (np.sqrt(1-(e/2.0)**2))),
            lambda e: 0
           ])
def GtL01(ts):
    return 0.5*sc.special.jv(1,2*ts) - 0.5j*sc.special.struve(-1,2*ts)
