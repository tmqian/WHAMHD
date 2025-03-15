import sys
import os
import warnings
import numpy as np
from scipy import optimize
import xarray as xr
import matplotlib.pyplot as plt
import spe_parser
import logging
import argparse
import MDSplus as mds
import time

import matplotlib.pylab as pylab
#import matplotlib.cm as cm
N = 100
cmap = pylab.cm.coolwarm(np.linspace(0,1,N))
def get_color_index(V, M=80):
    input_ax = np.linspace(-M,M,N)
    idx = np.argmin(np.abs(V - input_ax))
    return idx


# Physics constants
mC = 1.9944236560726842e-26  # mass of carbon in kg
mHe = 6.646473665811757e-27  # mass of helium in kg
cvel = 2.99792458e8  # light speed in m/s

def joule_to_eV(joule):
    '''return eV value for joule'''
    return joule * 6.24150962915265e+18

# Used ROI (ROI8 is blank)
ROI = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10])
WEIGHT = np.array([1, 1, 1, 1/2, 1, 1, 1, 1/2, 1, 1])
# LOS for the ROI
LOS = (np.array([
        6, 5, 8, 3, 10, 
        1,
        11, 2, 9,
        4, 7
]) - 5.8) * 15

# wavelength calibration results from the experiments taken on 9/12/2024. The files are 
# WHAM2_466nm_roi 00.spe in calib directory.
# For the definitions of alpha, dx0, dx1, w0, w1, see twogauss functions
WL_ALPHA = [
    0.40056876, 0.41836623, 0.4679105 , 0.50796722, 0.54198214,
    0.56667849, 0.53052786, 0.54982814, 0.48585918, 0.36459809, 
    0.37546005
]
WL_DX0 = [
    0.11236539, 0.09642858, 0.08485799, 0.07234005, 0.06704428,
    0.06494156, 0.06467431, 0.06683342, 0.07417026, 0.08781449, 
    0.10058697
]
WL_DX1 = [
    0.15288245, 0.12998626, 0.11704517, 0.10076999, 0.09563926,
    0.09082419, 0.09435086, 0.09331702, 0.10578473, 0.12829384,
    0.13407799
]
WL_W0 = [
    0.02739202, 0.02617437, 0.02309229, 0.02040617, 0.01856676,
    0.01794303, 0.01877269, 0.01845044, 0.02063121, 0.02462845,
    0.02652926    
]
WL_W1 = [
    0.06163808, 0.05824325, 0.06429075, 0.0484748 , 0.05739492,
    0.05008298, 0.05431421, 0.05008244, 0.05055447, 0.06030736,
    0.07379463
]
WL_W0STD = 0.00358  # error in the instrumental width
WL_DX0STD = 0.002869  # error in the instrumental width

######################################################


def squaresum(x, y):
    return np.sqrt(x * x + y * y)

def normal(x):
    return 1 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x ** 2)

def Gauss(x, A, x0, sigma, offset):
    """
    Standard Gaussian function with area A, centroid x0, 
    standard deviation sigma, and offset.
    """
    return normal((x - x0) / sigma) / sigma * A + offset

def twogauss(x, A, x0, w, y0, alpha, dx0, dx1, w0, w1):
    return A * (
        Gauss(x, 1 - alpha, dx0 + x0, np.sqrt(w**2 + w0**2), 0) + 
        Gauss(x, alpha, dx1 + x0, np.sqrt(w**2 + w1**2), 0)
    ) + y0


def fit_CIII(data, roi):
    r''' For CIII line 
    
    Parameters
    ----------
    data1d: 1d xr.DataArray, with wavelength

    Returns
    -------
    xr.Dataset
    '''
    data1d = data.sel(wavelength=slice(464.6, 465.4))
    # calibrations
    instprofile = lambda x, A, x0, w, y0: twogauss(
        x, A, x0, w, y0,
        alpha=WL_ALPHA[roi],
        dx0=WL_DX0[roi], dx1=WL_DX1[roi],
        w0=WL_W0[roi], w1=WL_W1[roi]
    )
    
    def profile(x, bg, dxC, nC, wC, dxCu=0, nCu=0, wCu=1, dxO=0, nO=0, wO=1):
        y = bg
        for wl, gA in [
            (464.7418, 3.63e+08),
            (465.0246, 2.18e+08),
            (465.1473, 7.24e+07)
        ]:
            y = y + instprofile(x, np.abs(nC) * gA, wl + dxC, wC, 0)
        for wl, gA in [
            (464.927084, 1e+08),
        ]:
            y = y + instprofile(x, np.abs(nCu) * gA, wl + dxCu, wCu, 0)
        for wl, gA in [
            (465.08384, 1e8),
        ]:
            y = y + instprofile(x, nO * gA, wl + dxO, wO, 0)
        return y

    p0 = 0.0, 0.06, 1e-6, 0.03, 0.05, 1e-6, 0.02
    try:
        if data1d.max() > 1000:
            popt, pcov = optimize.curve_fit(
                profile, data1d['wavelength'].values, data1d.values,
                p0=p0[:4]
            )
            popt, pcov = optimize.curve_fit(
                profile, data1d['wavelength'].values, data1d.values,
                p0=tuple(popt) + (popt[1] - 0.01, popt[2] / 2) + tuple(popt[3:])
            )
        else:
            popt = np.full(8, np.nan)
            pcov = np.full((8, 8), np.nan)
    except RuntimeError:
        popt = np.full(8, np.nan)
        pcov = np.full((8, 8), np.nan)
    perr = np.sqrt(np.diagonal(pcov))

    result = xr.Dataset({
        'shift': popt[1], 'shift_err': perr[1],
        'intensity': popt[2], 'intensity_err': perr[2],
        'width': popt[3], 'width_err': perr[3],
        #'shiftCu': popt[4], 'shiftCu_err': perr[4],
        #'intensityCu': popt[5], 'intensityCu_err': perr[5],
        #'widthCu': popt[6], 'widthCu_err': perr[6],
        'data': data1d,
        'fit': profile(data1d['wavelength'], *popt),
    }, coords={k: i for k, i in data.coords.items() if i.ndim == 0})
    result['roi'] = roi
    result['los'] = LOS[roi] 
    result['line'] = 'CIII'
    # temperature
    lam0 = 465.0
    result['Ti'] = (
        (), 
        joule_to_eV((result['width'].item() * cvel / lam0)**2 * mC),
        {'units': 'eV'}
    )
    result['Ti_err'] = 2.0 * np.abs(
        squaresum(result['width_err'], WL_W0STD) / result['width'] * result['Ti']
    )
    result['Vi'] = result['shift'] / lam0 * cvel
    result['Vi_err'] = (result['shift_err'] + WL_DX0STD) / lam0 * cvel
    return result


def fit_HeII(data, roi):
    r''' For HeII line 
    
    Parameters
    ----------
    data1d: 1d xr.DataArray, with wavelength

    Returns
    -------
    xr.Dataset
    '''
    data1d = data.sel(wavelength=slice(468.3, 469))
    # calibrations
    instprofile = lambda x, A, x0, w, y0: twogauss(
        x, A, x0, w, y0,
        alpha=WL_ALPHA[roi],
        dx0=WL_DX0[roi], dx1=WL_DX1[roi],
        w0=WL_W0[roi], w1=WL_W1[roi]
    )    
    def profile(x, bg, dxHe, nHe, wHe):
        y = bg
        for wl, gA in [
            (468.5376850, 3.7560e+08),
            (468.5407226, 1.9628e+08),
            (468.5524404, 1.9594e+07),
            (468.5568006, 9.8160e+07),
            (468.5703850, 1.2363e+09),
            (468.5704380, 6.7608e+08),
            (468.5757080, 2.2258e+06),
            (468.5757975, 7.5120e+07),
            (468.5804092, 1.7661e+09),
            (468.5830890, 8.8302e+07),
            (468.5884123, 2.0036e+07),
            (468.5905553, 3.9204e+07),
            (468.5917885, 1.1136e+07)
        ]:
            y = y + instprofile(x, np.abs(nHe) * gA, wl + dxHe, wHe, 0)
        return y

    p0 = 0.0, 0.06, 1e-6, 0.03
    try:
        if data1d.max() > 1000:
            popt, pcov = optimize.curve_fit(
                profile, data1d['wavelength'].values, data1d.values, p0=p0
            )
        else:
            popt = np.full(4, np.nan)
            pcov = np.full((4, 4), np.nan)
    except RuntimeError:
        popt = np.full(4, np.nan)
        pcov = np.full((4, 4), np.nan)
    perr = np.sqrt(np.diagonal(pcov))

    result = xr.Dataset({
        'shift': popt[1], 'shift_err': perr[1],
        'intensity': popt[2] * 10, 'intensity_err': perr[2] * 10,
        'width': popt[3], 'width_err': perr[3],
        'data': data1d,
        'fit': profile(data1d['wavelength'], *popt),
    }, coords={k: i for k, i in data.coords.items() if i.ndim == 0})
    result['roi'] = roi
    result['los'] = LOS[roi]
    result['line'] = 'HeII'
    lam0 = 468.0
    result['Ti'] = (
        (), 
        joule_to_eV((result['width'].item() * cvel / lam0)**2 * mHe), 
            {'units': 'eV'}
    )
    result['Ti_err'] = 2.0 * np.abs(
        squaresum(result['width_err'], WL_W0STD) / result['width'] * result['Ti']
    )
    result['Vi'] = result['shift'] / lam0 * cvel
    result['Vi_err'] = (result['shift_err'] + WL_DX0STD) / lam0 * cvel
    return result


def fit_curvature(roi, shift, shift_err, use_predetermined_curvature=False):
    roi, shift, shift_err = xr.broadcast(roi, shift, shift_err)

    def func(x, a, b, c=0):
        return c * x * x + a * x + b

    popt, _ = optimize.curve_fit(
        func,
        roi.values.ravel(), shift.values.ravel(), p0=(0, 0),
        sigma=(shift_err / WEIGHT).values.ravel()
    )
    return func(roi, *popt)


def process(filename,args,savefig=True):
    ''' evaluate the Doppler shift
    '''

    data = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = spe_parser.xr_open(filename).isel(time=0, y=0).swap_dims({'x': 'wavelength'}).isel(roi=ROI)
    data.coords['filename'] = filename
    data.coords['los'] = 'roi', [LOS[r.item()] for r in data['roi']]

    resultsC = []
    for roi in range(data.sizes['roi']):
        resultsC.append(fit_CIII(data.isel(roi=roi), data.isel(roi=roi)['roi'].item()))
    resultsC = xr.concat(resultsC, dim='roi')
    resultsC['line'] = resultsC['line'][0].item()

    resultsHe = []
    for roi in range(data.sizes['roi']):
        resultsHe.append(fit_HeII(data.isel(roi=roi), data.isel(roi=roi)['roi'].item()))
    resultsHe = xr.concat(resultsHe, dim='roi')
    resultsHe['line'] = resultsHe['line'][0].item()

    results = xr.concat([resultsC, resultsHe], dim='line').sortby('los')
    # correct the velocity
    center = xr.concat([
        (results['Vi'].sel(line='CIII') * WEIGHT).mean() / np.mean(WEIGHT),
        (results['Vi'].sel(line='HeII') * WEIGHT).mean() / np.mean(WEIGHT),
    ], dim='line')
    results['Vi'] -= center
    return results


class Spectrometer:

    def __init__(self,shot,
                      root="/mnt/n/whamdata/optical_spectroscopy/SPEs",
                 ):

        self.shot = shot
        year = shot[:2]
        month = shot[2:4]
        day = shot[4:6]
        path = f"{root}/{year}/{month}/{day}"

        f1 = f"{path}/WHAM1_{shot}.spe"
        f2 = f"{path}/WHAM2_{shot}.spe"

        self.results = process(f2,shot,savefig=False)

        self.C = self.results.sel(line='CIII')
        self.LOS = LOS
        self.ROI = ROI

        self.gate_delay = float(self.C['data'].gate_delay)/1e6 # ms
        self.gate_width = float(self.C['data'].gate_width)/1e6 # ms
        # is the ICCD intensifier saved?

    def plot_Vi_T_CIII(self):

        C = self.results.sel(line='CIII')

        fig,axs = plt.subplots(2,1,sharex=True)

        axs[0].errorbar(C.los, C.Vi/1e3, yerr=C.Vi_err/1e3, fmt='o-', label='CIII')
        axs[1].errorbar(C.los, C.Ti, yerr=C.Ti_err, fmt='o-', color='C1')

        for a in axs:
            a.grid()

        axs[1].set_xlabel("impact parameter [mm]")
        axs[1].set_ylabel("Ti [eV]")
        axs[1].set_ylim(0,None)
        axs[0].set_ylabel("Ion Velocity [km/s]")
        axs[0].legend()

        tag = f"t_gate = {self.gate_width} ms, t_delay = {self.gate_delay} ms"
        axs[0].set_title(tag)
        fig.suptitle(self.shot)

    def plot_spectra(self):

        fig,axs = plt.subplots(1,1,figsize=(12,6))
        C = self.results.sel(line='CIII')

        def untangle(data):
            '''
            The spec file has a (N_chord, N_wavelengths) data matrix
            and the chords are out of order.

            This function sorts the chords, according to doppler shift
            '''
            arr = []
            for j,line in enumerate(data):
                k = np.argmax( np.array(line[:50]))
                L = np.array(self.C.wavelength[k])
                arr.append(L)

            arg = np.argsort(arr)
            return np.array(data[arg])

        #for c in [464.7418,464.92708, 465.0246,465.08384,465.1473]:
        for c in [464.7418, 465.0246,465.1473]:
            axs.axvline(c, ls='--', color='k', label=f"{c} nm")

        data = untangle(self.C.data)
        #data = np.array(self.C.data)[9, 8, 7, 1, 2, 0, 3, 4, 6, 5]
        shift = 0.1
        for j,spectra in enumerate(data):

            R = np.array(C.los[j])
            c_idx = get_color_index(R)
            axs.plot(C.wavelength - shift, spectra, 'o-', color=cmap[c_idx], label= f"{C.los[j]:.1f}, (roi,los)={self.ROI[j]},{self.LOS[j]:.1f}")
            #axs.plot(C.wavelength - shift, C.data[j], 'o-', color=cmap[c_idx], label= f"{C.los[j]:.1f}")
            #axs.plot(C.wavelength - shift, C.data[arg[j]], 'o-', color=cmap[c_idx], label= f"{C.los[j]:.1f}")


        axs.set_xlabel("wavelength [nm]")
        axs.set_ylabel("intensity [arb]")
        tag = f"t_gate = {self.gate_width} ms, t_delay = {self.gate_delay} ms"
        axs.set_title(tag)
        axs.legend(fontsize=10)
        axs.grid()

        fig.suptitle(self.shot)

    def plot_Vi(self,ax):

        C = self.C
        ax.errorbar(C.los, C.Vi/1e3, yerr=C.Vi_err/1e3, fmt='o-', label='self.shot')


