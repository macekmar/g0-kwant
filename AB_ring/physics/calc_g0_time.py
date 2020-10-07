import kwant
import numpy as np
import scipy as sc

from .fermi import fermi
from .calc_sigma import *

# Green functions

def calc_Gt_integral(integrand, k_min, k_max, quad_vec_kwargs={"workers": 4, "full_output": False}):
    integral = sc.integrate.quad_vec(integrand, a=k_min, b=k_max, **quad_vec_kwargs)
    Gt = integral[0][...,0] + 1j*integral[0][...,1] # Works for 2D (G<,>,R) and 1D result (control)
    return Gt, integral[1:]

###############################################################################
## Gt^<

def calc_GtL_E(syst, t, i, j, Emin, Emax, eng_fermis, beta, quad_vec_kwargs={}):
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
    return GtL, integral[1:]

def integrand_GtL(k, syst, i, j, t, eng_fermis, beta, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    if np.any(np.array([fermi(e, ef, beta) for ef in eng_fermis]) > 1e-14):  # Test if we are between fermi levels, else result is 0
        wf = kwant.wave_function(syst, energy=e)
        for lead in range(2):
            for channel in range(1):
                val += (-1)**channel * 1j/(2*np.pi) * fermi(e, eng_fermis[lead], beta) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

def integrand_GtL_intp(k, intp, i, j, t, eng_fermis, beta, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += (-1)**channel * 1j/(2*np.pi) * fermi(e, eng_fermis[lead], beta) * intp(k, lead, channel, i) * np.conj(intp(k, lead, channel, j) * np.exp(1j*t*e))
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

###############################################################################
## Gt^>

def calc_GtG_from_GtLR(t, GtL, GtR):
    """Calculates g(t)> from g(t)< and g(t)R.

    Formula is g^> = G^R - g^A + g^< where g^A(t,t') = [g^R(t',t)]^†"""
    return GtR*(t >= 0) - GtR.conj()[::-1]*(t < 0) + GtL

def calc_GtG_E(syst, t, i, j, Emin, Emax, eng_fermis, beta, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = np.zeros((t.shape[0],), dtype=np.complex)
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += (-1)**channel * 1j/(2*np.pi) * (fermi(e, eng_fermis[lead], beta)-1) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))

        return  np.stack((wf_abs.real, wf_abs.imag), -1)

    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    GtG = (integral[0][:,0] + 1j*integral[0][:,1])
    return GtG, integral[1:]

def integrand_GtG(k, syst, i, j, t, eng_fermis, beta, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    if np.any(np.array([(fermi(e, ef, beta)-1) for ef in eng_fermis]) < -1e-14):  # Test if we are between fermi levels, else result is 0
        wf = kwant.wave_function(syst, energy=e)
        for lead in range(2):
            for channel in range(1):
                val += (-1)**channel * 1j/(2*np.pi) * (fermi(e, eng_fermis[lead], beta)-1) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

def integrand_GtG_intp(k, intp, i, j, t, eng_fermis, beta, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += (-1)**channel * 1j/(2*np.pi) * (fermi(e, eng_fermis[lead], beta)-1) * intp(k, lead, channel, i) * np.conj(intp(k, lead, channel, j) * np.exp(1j*t*e))
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)


###############################################################################
## Gt^R

def calc_GtR_E(syst, t, i, j, Emin, Emax, quad_vec_kwargs={}):
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
    return GtR, integral[1:]

def integrand_GtR(k, syst, i, j, t, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    wf = kwant.wave_function(syst, energy=e)
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][j] * np.exp(1j*t*e))
    val = -1j * np.heaviside(t,1) * val
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

def integrand_GtR_intp(k, intp, i, j, t, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = np.zeros((t.shape[0],), dtype=np.complex)
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * intp(k, lead, channel, i) * np.conj(intp(k, lead, channel, j) * np.exp(1j*t*e))
    val = -1j * np.heaviside(t,1) * val
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

###############################################################################
## Control

def calc_Gt_control_E(syst, i, Emin, Emax, quad_vec_kwargs={}):
    nb_leads = len(syst.leads)

    def integrand(e):
        wf = kwant.wave_function(syst, energy=e)
        wf_abs = 0
        for lead in range(nb_leads):
            nb_channels = wf(lead).shape[0]
            for channel in range(nb_channels):
                wf_abs += 1/(2*np.pi) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][i])

        return  np.stack((wf_abs.real, wf_abs.imag), -1)
    integral = sc.integrate.quad_vec(integrand, a=Emin, b=Emax, **quad_vec_kwargs)
    val = (integral[0][0] + 1j*integral[0][1])
    return val, integral[1:]

def integrand_Gt_control(k, syst, i, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = 0
    wf = kwant.wave_function(syst, energy=e)
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * wf(lead)[channel][i] * np.conj(wf(lead)[channel][i])
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

def integrand_Gt_control_intp(k, intp, i, e0, gamma):
    e = e0 + 2*np.abs(gamma)*np.cos(k)
    val = 0
    for lead in range(2):
        for channel in range(1):
            val += 1/(2*np.pi) * intp(k, lead, channel, i) * np.conj(intp(k, lead, channel, i))
    return  -2*np.abs(gamma)*np.sin(k)*np.stack((val.real, val.imag), -1)

