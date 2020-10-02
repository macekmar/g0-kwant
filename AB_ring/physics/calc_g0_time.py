import kwant
import numpy as np
import scipy as sc

from .fermi import fermi
from .calc_sigma import *

# Green functions

###############################################################################
## Gt^<

# def _integrand_GtL(e, channel, lead):

def calc_GtL(syst, t, i, j, Emin, Emax, eng_fermis, beta, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += (-1)**channel * 1j/(2*np.pi) * fermi(e, eng_fermis[lead], beta) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtL = (integral[0][:,0] + 1j*integral[0][:,1])
    return GtL

def calc_GtL_k(syst, t, i, j, Kmin, Kmax, eng_fermis, beta, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)
    e0 = syst.leads[0].hamiltonian(0,0)
    gamma = syst.leads[0].hamiltonian(0,1) # What to do if leads are different?
    def integrand(k):
        e = e0 + 2*np.abs(gamma)*np.cos(k)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        if np.any(np.array([fermi(e, ef, beta) for ef in eng_fermis]) > 1e-12):  # Test if we are between fermi levels
            wf = kwant.wave_function(syst, energy=e)
            for lead in range(nb_leads):
                nb_channels = wf(lead).shape[0]
                for channel in range(nb_channels):
                    wf_abs += (-1)**channel * 1j/(2*np.pi) * fermi(e, eng_fermis[lead], beta) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  -2*np.abs(gamma)*np.sin(k)*np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Kmin, b=Kmax, **quad_vec_kwargs)
    GtL = (integral[0][:,0] + 1j*integral[0][:,1])
    return GtL 

def calc_GtL_k_intp(intp, syst, t, i, j, Kmin, Kmax, eng_fermis, beta, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)
    e0 = syst.leads[0].hamiltonian(0,0)
    gamma = syst.leads[0].hamiltonian(0,1) # What to do if leads are different?
    def integrand(k):
        e = e0 + 2*np.abs(gamma)*np.cos(k)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        if np.any(np.array([fermi(e, ef, beta) for ef in eng_fermis]) > 1e-12):  # Test if we are between fermi levels
            for lead in range(nb_leads):
                nb_channels = 1
                for channel in range(nb_channels):
                    wf_abs += (-1)**channel * 1j/(2*np.pi) * fermi(e, eng_fermis[lead], beta) * intp(k, lead, channel, i) * np.conj(intp(k, lead, channel, j) * np.exp(1j*t*e))

        return  -2*np.abs(gamma)*np.sin(k)*np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Kmin, b=Kmax, **quad_vec_kwargs)
    GtL = (integral[0][:,0] + 1j*integral[0][:,1])
    return GtL 

###############################################################################
## Gt^R

def calc_GtR(syst, t, i, j, Emin, Emax, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += 1/(2*np.pi) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtR = -1j * np.heaviside(t,1) * (integral[0][:,0] + 1j*integral[0][:,1])
    return GtR

def calc_GtR_k(syst, t, i, j, Kmin, Kmax, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)
    e0 = syst.leads[0].hamiltonian(0,0)
    gamma = syst.leads[0].hamiltonian(0,1) # What to do if leads are different?
    def integrand(k):
        e = e0 + 2*np.abs(gamma)*np.cos(k)
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += 1/(2*np.pi) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  -2*np.abs(gamma)*np.sin(k)*np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(integrand, a=Kmin, b=Kmax, **quad_vec_kwargs)
    GtR = -1j * np.heaviside(t,1) * (integral[0][:,0] + 1j*integral[0][:,1])
    return GtR

def calc_GtR_k_intp(intp, syst, t, i, j, Kmin, Kmax, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)
    e0 = syst.leads[0].hamiltonian(0,0)
    gamma = syst.leads[0].hamiltonian(0,1) # What to do if leads are different?
    def integrand(k):
        e = e0 + 2*np.abs(gamma)*np.cos(k)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(2):
            for channel in range(1):
                wf_abs += 1/(2*np.pi) * intp(k,lead, channel, i) * np.conj(intp(k,lead, channel, j) * np.exp(1j*t*e))
        return  -2*np.abs(gamma)*np.sin(k)*np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(integrand, a=Kmin, b=Kmax, **quad_vec_kwargs)
    GtR = -1j * np.heaviside(t,1) * (integral[0][:,0] + 1j*integral[0][:,1])
    return GtR

def calc_G_control_intp(intp, syst, i, j, Kmin, Kmax, quad_vec_kwargs={}): # Basically t = 0, i=j
    nb_leads = len(syst.leads)
    e0 = syst.leads[0].hamiltonian(0,0)
    gamma = syst.leads[0].hamiltonian(0,1) # What to do if leads are different?
    def integrand(k):
        e = e0 + 2*np.abs(gamma)*np.cos(k)
        wf_abs = 0
        for lead in range(2):
            for channel in range(1):
                wf_abs += 1/(2*np.pi) * intp(k,lead, channel, i) * np.conj(intp(k,lead, channel, i))
        return  -2*np.abs(gamma)*np.sin(k)*np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(integrand, a=Kmin, b=Kmax, **quad_vec_kwargs)
    return integral[0][0] + 1j*integral[0][1] 

def calc_GtG_from_GtLR(t, GtL, GtR):
    """Calculates g(t)> from g(t)< and g(t)R.

    Formula is g^> = G^R - g^A + g^< where g^A(t,t') = [g^R(t',t)]^†"""
    return GtR*(t >= 0) - GtR.conj()[::-1]*(t < 0) + GtL
