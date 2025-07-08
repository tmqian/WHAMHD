# This code was written by Keisuke Fujii


import sys
import os
import warnings
import numpy as np
from scipy import optimize
import xarray as xr
import matplotlib.pyplot as plt
import spe_parser

# Used ROI (ROI8 is blank)
ROI = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10])
WEIGHT = np.array([1, 1, 1, 1, 1, 1, 1, 1/2, 1/2, 1])
# LOS for the ROI
LOS = (np.array([
        6, 5, 8, 3, 10, 
        1,
        11, 2, #9,
        4, 7
]) - 5.8) * 15

CURVATURE = 0.00142597  # Currently assuming the 

def normal(x):
    return 1 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x ** 2)


def Gauss(x, A, x0, sigma, offset):
    """
    Standard Gaussian function with area A, centroid x0, 
    standard deviation sigma, and offset.
    """
    return normal((x - x0) / sigma) / sigma * A + offset


def get_centroid_CIII(data):
    r''' For CIII line 
    
    Parameters
    ----------
    data1d: 1d xr.DataArray, with wavelength

    Returns
    -------
    xr.Dataset
    '''
    data1d = data.sel(wavelength=slice(464.6, 465.4))
    
    def profile(x, bg, dxC, nC, wC, dxCu=0, nCu=0, wCu=1, dxO=0, nO=0, wO=1):
        y = bg
        for wl, gA in [
            (464.7418, 3.63e+08),
            (465.0246, 2.18e+08),
            (465.1473, 7.24e+07)
        ]:
            y = y + Gauss(x, np.abs(nC) * gA, wl + dxC, wC, 0)
        for wl, gA in [
            (464.927084, 1e+08),
        ]:
            y = y + Gauss(x, np.abs(nCu) * gA, wl + dxCu, wCu, 0)
        for wl, gA in [
            (465.08384, 1e8),
        ]:
            y = y + Gauss(x, nO * gA, wl + dxO, wO, 0)
        return y

    p0 = 0.0, 0.06, 1e-6, 0.03, 0.05, 1e-6, 0.02
    try:
        popt, pcov = optimize.curve_fit(
            profile, data1d['wavelength'].values, data1d.values, 
            p0=p0[:4]
        )
        popt, pcov = optimize.curve_fit(
            profile, data1d['wavelength'].values, data1d.values, 
            p0=tuple(popt) + (popt[1] - 0.01, popt[2] / 2) + tuple(popt[3:])
        )
    except RuntimeError:
        popt = np.full(8, np.nan)
        pcov = np.full((8, 8), np.nan)
    perr = np.sqrt(np.diagonal(pcov))

    result = xr.Dataset({
        'shiftC': popt[1], 'shiftC_err': perr[1],
        'intensityC': popt[2], 'intensityC_err': perr[2],
        'widthC': popt[3], 'widthC_err': perr[3],
        'shiftCu': popt[4], 'shiftCu_err': perr[4],
        'intensityCu': popt[5], 'intensityCu_err': perr[5],
        'widthCu': popt[6], 'widthCu_err': perr[6],
    }, coords={k: i for k, i in data.coords.items() if i.ndim == 0})
    fit = xr.Dataset({'data': data1d, 'fit': profile(data1d['wavelength'], *popt)})
    return result, fit


def get_centroid_HeII(data):
    r''' For HeII line 
    
    Parameters
    ----------
    data1d: 1d xr.DataArray, with wavelength

    Returns
    -------
    xr.Dataset
    '''
    data1d = data.sel(wavelength=slice(468.3, 469))
    
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
            y = y + Gauss(x, np.abs(nHe) * gA, wl + dxHe, wHe, 0)
        return y

    p0 = 0.0, 0.06, 1e-6, 0.03
    try:
        popt, pcov = optimize.curve_fit(
            profile, data1d['wavelength'].values, data1d.values, p0=p0
        )
    except RuntimeError:
        popt = np.full(4, np.nan)
        pcov = np.full((4, 4), np.nan)
    perr = np.sqrt(np.diagonal(pcov))

    result = xr.Dataset({
        'shiftHe': popt[1], 'shiftHe_err': perr[1],
        'intensityHe': popt[2] * 10, 'intensityHe_err': perr[2] * 10,
        'widthHe': popt[3], 'widthHe_err': perr[3]
    }, coords={k: i for k, i in data.coords.items() if i.ndim == 0})
    
    fit = xr.Dataset({'data': data1d, 'fit': profile(data1d['wavelength'], *popt)})
    return result, fit


def fit_curvature(roi, shift, shift_err, use_predetermined_curvature=False):
    roi, shift, shift_err = xr.broadcast(roi, shift, shift_err)

    def func(x, a, b, c=CURVATURE):
        return c * x * x + a * x + b

    popt, _ = optimize.curve_fit(
        func,
        roi.values.ravel(), shift.values.ravel(), p0=(CURVATURE, 0, 0),
        sigma=(shift_err / WEIGHT).values.ravel()
    )
    return func(roi, *popt)


def process(filenames):
    data = []
    for filename in filenames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da = spe_parser.xr_open(filename).isel(time=0, y=0).swap_dims({'x': 'wavelength'})
        da.coords['filename'] = filename
        data.append(da)
    data = xr.concat(data, dim='filename').isel(roi=ROI)#
    data.coords['los'] = 'roi', LOS
    data = data.stack(index=['filename', 'roi']).reset_index('index')

    results = []
    plt.figure(figsize=(15, 10))
    figname = filenames[0][:-4]
    if len(filenames) > 1:
        figname = figname + '-' + filename[-1][-7:-4]

    ax = plt.subplot(3, 2, 2)
    axint = plt.subplot(3, 2, 4)
    axwidth = plt.subplot(3, 2, 6)
    if (
        data['wavelength'].min() < 464.6 and 465.6 < data['wavelength'].max() and
        data.mean('index').max() > 1000  # hardcode for "intense enough" signal
    ):
        ax2 = plt.subplot(3, 2, 1)
        fits, results_C = [], []
        for ind in range(data.sizes['index']):
            result_C, fit = get_centroid_CIII(data.isel(index=ind))
            results_C.append(result_C)
            fits.append(fit)
            if ind in (0, 5):
                fit['data'].plot(color='C{}'.format(ind), marker='.', ls='')
                fit['fit'].plot(color='C{}'.format(ind), ls='-')
            ax2.set_xlabel('wavelength (nm)')
            ax2.set_ylabel('intensity (arb. units)')
            ax2.text(0.05, 0.95, 'CIII', va='top', transform=ax2.transAxes)

        fits = xr.concat(fits, dim=data['index'])
        results_C = xr.concat(results_C, dim=data['index'])
        # fit the curvature
        curv = fit_curvature(results_C['roi'], results_C['shiftC'], results_C['shiftC_err'])
        results_C['vC'] = (results_C['shiftC'] - curv) / 464 * 3e8
        results_C['vC_err'] = (results_C['shiftC_err']) / 464 * 3e8
        curv = fit_curvature(results_C['roi'], results_C['shiftCu'], results_C['shiftCu_err'])
        results_C['vCu'] = (results_C['shiftCu'] - curv) / 464 * 3e8
        results_C['vCu_err'] = (results_C['shiftCu_err']) / 464 * 3e8
        results.append(results_C.sortby('los'))

        ax.errorbar(LOS, results_C['vC'], yerr=results_C['vC_err'], fmt='o', label='CIII')
        axint.errorbar(
            LOS, results_C['intensityC'] / results_C['intensityC'].max(), 
            yerr=results_C['intensityC_err'] / results_C['intensityC'].max(), fmt='o', label='CIII')
        axwidth.errorbar(
            LOS, results_C['widthC'], yerr=results_C['widthC_err'], fmt='o', label='CIII')
        #ax.errorbar(LOS, results_C['vCu'], yerr=results_C['vCu_err'], fmt='o', label='Cu')

    if (
        (data['wavelength'].min() < 468.2 and 468.8 < data['wavelength'].max()) and
        data.sel(wavelength=slice(468.2, 468.8)).median('index').max() > 1000  # hardcode for "intense enough" signal
    ):
        print('analyzing HeII')
        ax2 = plt.subplot(3, 2, 3)
        fits, results_He = [], []
        for ind in range(data.sizes['index']):
            result_He, fit = get_centroid_HeII(data.isel(index=ind))
            results_He.append(result_He)
            fits.append(fit)
            if ind in (0, 5):
                fit['data'].plot(color='C{}'.format(ind), marker='.', ls='', ax=ax2)
                fit['fit'].plot(color='C{}'.format(ind), ls='-', ax=ax2)
            ax2.set_xlabel('wavelength (nm)')
            ax2.set_ylabel('intensity (arb. units)')
            ax2.text(0.05, 0.95, 'HeII', va='top', transform=ax2.transAxes)
            
        fits = xr.concat(fits, dim=data['index'])
        results_He = xr.concat(results_He, dim=data['index'])
        # fit the curvature
        curv = fit_curvature(results_He['roi'], results_He['shiftHe'], results_He['shiftHe_err'])
        results_He['vHe'] = (results_He['shiftHe'] - curv) / 464 * 3e8
        results_He['vHe_err'] = (results_He['shiftHe_err']) / 464 * 3e8
        results.append(results_He.sortby('los'))

        ax.errorbar(LOS, results_He['vHe'], yerr=results_He['vHe_err'], fmt='o', label='HeII')
        axint.errorbar(
      	    LOS, results_He['intensityHe'] / results_He['intensityHe'].max(), 
	    yerr=results_He['intensityHe_err'] / results_He['intensityHe'].max(), fmt='o', label='HeII')
        axwidth.errorbar(
            LOS, results_He['widthHe'], yerr=result_He['widthHe_err'], fmt='o', label='HeII')
    for a in [ax, axint, axwidth]:
        a.set_xlim(-90, 90)
        a.grid()
        a.legend()
        a.set_xlabel('approximated impact parameter (mm)')
    ax.set_ylim(-15000, 15000)
    ax.axhline(0, ls='--', color='k')
    ax.set_ylabel('v (m/s)')
    axint.set_ylim(0, 1.1)
    axint.set_ylabel('intensity (arb. units)')
    axwidth.set_ylim(0.022, 0.04)
    axwidth.set_ylabel('width (nm)')
    ax.set_title(figname)
    plt.savefig(figname + '.png', bbox_inches='tight')

if __name__ == '__main__':
    # if without the arguments, consider the latest file
    if len(sys.argv) == 1:
        filenames = sorted(
            [f for f in os.listdir() if f[-4:] == '.spe' and f[:5] == 'WHAM2']
        )[-1:]
        print(filenames)
    elif len(sys.argv) == 2:
        filenames = sys.argv[1:]
    else:
        raise NotImplementedError
        filenames = sys.argv[1:]
    
    process(filenames)
