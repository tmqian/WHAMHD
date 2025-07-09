import sys
import configparser
import os
import warnings

import numpy as np
from scipy import optimize
import xarray as xr

import spe_parser


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(THIS_DIR, 'config.ini')

CVEL = 2.99792458e8  # speed of light in m/s
J_TO_EV = 6.24150962915265e+18  # conversion

LINES = {
    # Line database. 
    'CIII_465': [
        # wavelength, gA
        (464.7418, 3.63e+08),
        (465.0246, 2.18e+08),
        (465.1473, 7.24e+07),
        ],
    'OII_460': [ # 2s22p2(1D)3p 	 2F°
        (459.0972, 7.08e+08),
        (459.5960, 2.92e+07),
        (459.6175, 5.00e+08),
    ],
    'OII_465': [  # 2s22p2(3P)3p 4D
        # wavelength, gA
        (463.88550, 1.44e+08),
        (464.18104, 3.51e+08),
        (464.91348, 6.27e+08),
        (465.08394, 1.34e+08),
        (466.16332, 1.62e+08),
        (467.37322, 2.48e+07), 
        (467.62350, 1.23e+08),
    ],
    'HeII_468': [
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
    ],
}

MASS = {
    'C': 1.9944236560726842e-26,  # mass of C atom in kg
    'O': 2.6567628733520366e-26,  # mass of O atom in kg
    'He': 6.646473665811757e-27  # mass of He atom in kg
}

def get_mass(line):
    '''return the mass of the line in kg, for given line
    
    CIII or HeII
    '''
    return MASS.get(line[:2], MASS.get(line[0]))


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


def two_gauss(x, A, x0, w, y0, alpha, dx0, dx1, w0, w1):
    return A * (
        Gauss(x, 1 - alpha, dx0 + x0, squaresum(w, w0), 0) + 
        Gauss(x, alpha, dx1 + x0, squaresum(w, w1), 0)
    ) + y0


def get_instrument_function(config, roi):
    if config['instrument_function_type'] == 'two_gauss':
        return (
            lambda x, A, x0, w, y0: getattr(sys.modules[__name__], 'two_gauss')(
                x, A, x0, w, y0, 
                alpha=config.get('inst_wl_alpha')[roi],
                dx0=config.get('inst_wl_dx0')[roi],
                dx1=config.get('inst_wl_dx1')[roi],
                w0=config.get('inst_wl_w0')[roi],
                w1=config.get('inst_wl_w1')[roi],
            )
        )
    raise NotImplementedError(
        f"Instrument function type '{config['instrument_function_type']}' is not implemented."
    )


def robust_curve_fit(func, xdata, ydata, p0=None, alpha=0, **kwargs):
    '''Regularized curve fitting with a penalty term for the parameters.
    '''
    ydata = np.concatenate([ydata, np.zeros(len(p0) - 1)])
    return optimize.curve_fit(
        lambda x, *p: np.concatenate(
            # penalize parameter to add robustness, except for last parameter (offset)
            [func(x, *p)] + [[alpha * p1] for p1 in p[:-1]]  
        ),
        xdata, ydata, p0=p0, **kwargs
    )


def fit_spectrum(spe_data, config):
    # choose lines to fit
    wlmin, wlmax = spe_data['wavelength'].min(), spe_data['wavelength'].max()
    
    lines = {}
    line_centers = []
    mass = []
    for name, wl_gA in LINES.items():
        wls = [wl for wl, gA in wl_gA]
        gAs = [gA for wl, gA in wl_gA]
        if (max(wls) > wlmin) or (min(wls) < wlmax):
            lines[name] = wl_gA
        line_centers.append(np.sum(np.array(wls) * np.array(gAs)) / np.sum(gAs))
        mass.append(get_mass(name))

    # function to be used to fit the spectrum
    def func(x, *params):
        y = params[-1] * 1e4
        for i, wl_gA in enumerate(lines.values()):
            A = params[i * 3]
            x0 = params[i * 3 + 1]
            w = params[i * 3 + 2]
            for wl, gA in wl_gA:
                y += inst_func(x, A * gA, wl + x0, w, 0)
        return y
    
    def func_intensity(x, *As):
        y = As[-1] * 1e4
        for i, wl_gA in enumerate(lines.values()):
            A = As[i]
            for wl, gA in wl_gA:
                y += inst_func(x, A * gA, wl, 0, 0)
        return y

    fit_result = []
    # loop over rois
    for roi in config['roi']:
        spec1 = spe_data.sel(roi=roi)
        idx = spec1 < 2**16 - 10  # filter out saturated pixels
        y = spec1.isel(wavelength=idx).values
        x = spec1['wavelength'].isel(wavelength=idx).values
        
        inst_func = get_instrument_function(config, roi)
        
        # first fit the intensity only to make the fit robust
        try:
            p0 = np.zeros(len(lines) + 1)
            popt, _ = optimize.curve_fit(func_intensity, x, y, p0=p0)

            p0 = []
            for i in range(len(lines)):
                p0.append(popt[i])
                p0.append(0.0)  # dx0
                p0.append(0.01)  # w
            p0.append(popt[-1])  # y0

            # fit with the full parameters
            popt, pcov = robust_curve_fit(
                func, x, y, p0=p0, maxfev=10000, alpha=config['fit_penalty']
            )
            perr = np.sqrt(np.diag(pcov))

        except (RuntimeError, ValueError) as e:
            popt = [np.nan] * len(p0)
            perr = [np.nan] * len(p0)

        # spec1.plot()
        # plt.plot(spec1['wavelength'], func(spec1['wavelength'].values, *popt))
        # plt.show()

        # fit by each line
        components = []
        for i in range(len(lines)):
            p = np.copy(popt)
            p[:-1:3] = 0
            p[i * 3] = popt[i * 3]
            components.append(func(spec1['wavelength'].values, *list(p)))

        fit_result.append(xr.Dataset({
            'data': ('wavelength', spec1.values),
            'fit': ('wavelength', func(spec1['wavelength'].values, *popt)),
            'fit_component': (('line', 'wavelength'), components),
            'intensity': ('line', popt[0:-1:3]),
            'shift': ('line', popt[1:-1:3]),
            'width': ('line', popt[2:-1:3]),
            'intensity_err': ('line', perr[0:-1:3]),
            'shift_err': ('line', perr[1:-1:3]),
            'width_err': ('line', perr[2:-1:3]),
            'offset': popt[-1], 'offset_err': perr[-1]
        }, coords={
            'wavelength': spec1['wavelength'], 'line': list(lines.keys()), 'roi': roi,
            'line_center': ('line', line_centers),
            'mass': ('line', mass)
        }))

    return xr.concat(fit_result, dim='roi')


def compensate_drift(fit, config):
    r''' Compensate the spectrometer drift by assuming the axisymmetry of the plasma

    fit: xr.Dataset
        should have
            - `shift` and `shift_err`, with the dimensions of ('time', 'roi', 'line').
            - `impact_parameter`, with the dimensions of `roi`.

    We would like to have a single `drift` component for each `line` and `roi`, shared by `time`.
    '''

    def func(x_impact_parameter, *p):
        ''' fitting function.
        
        The spectrometer drift is represented by
        a * x + b, 
        while the doppler shift is represented by 
        (
            d0 * impact_abs**6 + c0 * impact_abs**4 + b0 * impact_abs**2 + a0
        ) * impact_parameter,
        which is only-odd polynomial.
        '''
        x, impact_parameter = x_impact_parameter
        pol = p[:-2]
        a, b = p[-2:]
        return (
            a * x + b + np.poly1d(pol)(impact_parameter**2) * impact_parameter
        )

    drift = []
    for i in range(fit.sizes['line']):
        fit1 = fit.isel(line=i)
        roi, shift, shift_err, impact_parameter = xr.broadcast(
            fit1['roi'], fit1['shift'], fit1['shift_err'], fit1['impact_parameter']
        )
        #     a0, b0, c0, d0, a, b
        p0 = np.zeros(config['fit_velocity_order'] + 2)
        popt, pcov = optimize.curve_fit(
            func,
            (roi.values.ravel(), impact_parameter.values.ravel()), shift.values.ravel(), p0=p0,
            sigma=shift_err.values.ravel()
        )
        drift.append(fit1['roi'] * popt[-2] + popt[-1])

    fit['drift'] = xr.concat(drift, dim='line')
    return fit


def fit_spectra(spe_data, config):
    fits = []
    for t in range(spe_data.sizes['time']):
        fits.append(fit_spectrum(spe_data.isel(time=t, y=0), config))

    fits = xr.concat(fits, dim='time')
    fits.coords['date'] = spe_data['date']
    fits.coords['filename'] = spe_data['filename']
    fits.coords['impact_parameter'] = 'roi', config['impact_parameter']

    return fits


def cast_string(s):
    '''Cast a string to an appropriate type.'''
    if ',' in s:
        try:
            return np.array([int(v) for v in s.split(',')])
        except ValueError:
            return np.array([float(v) for v in s.split(',')])
    else:
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s


def load_config(spe_data):
    '''Load configuration parameters from the config file.
    
    Always store the latest parameters, but skip those with a date
    later than the date in the SPE file.
    '''
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    params = {}
    for key in config.keys():
        if key != 'DEFAULT':
            try:
                date = np.datetime64(key)
                if date > spe_data['date']:
                    continue
            except ValueError:
                pass
        
        for k, v in config[key].items():
            params[k] = cast_string(v)

    return params


def plot_fit(fit, config):
    nline = fit.sizes['line']
    fit = fit.sortby('impact_parameter')
    filename = fit['filename'].item()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    
    # spectrum
    for i in range(nline):
        ax = plt.subplot(nline, 2, i * 2 + 1)
        # wavelength width
        line = fit['line'].values[i].item()
        width = np.mean(config['inst_wl_w0']) + np.std(config['inst_wl_dx0'])
        wl_min = np.min([wl for wl, _ in LINES[line]]) - 5 * width
        wl_max = np.max([wl for wl, _ in LINES[line]]) + 5 * width

        for j, roi in enumerate(config['plot_roi']):
            color = 'C{}'.format(j)
            fit1 = fit.isel(time=config['plot_frame'], roi=roi).sel(
                wavelength=slice(wl_min, wl_max))
            fit1['fit'].plot(ax=ax, marker='', ls='-', color=color, lw=3, alpha=0.5)
            fit1['fit_component'].isel(line=i).plot(ax=ax, marker='', ls='-', color='0.5')
            fit1['data'].plot(ax=ax, marker='.', ls='', color=color)
        ax.set_xlim(wl_min, wl_max)
        ax.set_ylabel('intensity')
        ax.set_title(line)
    
    # intensity
    xmax = np.abs(fit['impact_parameter']).max().item() * 1.1
    ax = plt.subplot(nline, 2, 2)
    for i, line in enumerate(fit['line'].values):
        fit1 = fit.isel(time=config['plot_frame'], line=i)
        ax.errorbar(
            fit1['impact_parameter'], 
            fit1['intensity'] / np.abs(fit1['intensity'].max()),
            yerr=fit1['intensity_err'] / np.abs(fit1['intensity'].max()),
            color='C{}'.format(i), label=line, fmt='o-'
        )
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('impact_parameter')
    ax.set_ylabel('intensity (normalized)')
    ax.axvline(0, ls='--', color="0.5")        
    ax.legend()
    ax.set_title(filename)

    # velocity
    ax = plt.subplot(nline, 2, 4)
    for i, line in enumerate(fit['line'].values):
        fit1 = fit.isel(time=config['plot_frame'], line=i)
        if (
            ((fit1['Vi_err'].median() / np.abs(fit1['Vi']).median()) > 0.5)
        ):
            continue
        ax.errorbar(
            fit1['impact_parameter'], fit1['Vi'], yerr=np.abs(fit1['Vi_err']),
            color='C{}'.format(i), label=line,
            fmt='o-'
        )
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-config['plot_vmax'], config['plot_vmax'])
    ax.axhline(0, ls='--', color="0.5")        
    ax.axvline(0, ls='--', color="0.5")        
    ax.set_xlabel('impact_parameter')
    ax.set_ylabel('velocity (m/s)')

    # temperature
    ax = plt.subplot(nline, 2, 6)
    for i, line in enumerate(fit['line'].values):
        fit1 = fit.isel(time=config['plot_frame'], line=i)
        if (
            ((fit1['Vi_err'].median() / np.abs(fit1['Vi']).median()) > 0.5)
        ):
            continue
        ax.errorbar(
            fit1['impact_parameter'], fit1['Ti'], 
            yerr=np.abs(fit1['Ti_err']),
            color='C{}'.format(i), label=line,
            fmt='o-'
        )

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, config['plot_tmax'])
    ax.axvline(0, ls='--', color="0.5")        
    ax.set_xlabel('impact_parameter')
    ax.set_ylabel('temperature (eV)')

    plt.tight_layout()
    plt.savefig(os.path.join(
        config['save_dir'], 
        filename.replace('.spe', '.png').split(os.sep)[-1]
    ), bbox_inches='tight')


def run(filename):
    # open spe file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spe_data = spe_parser.xr_open(filename).swap_dims({'x': 'wavelength'})
    spe_data.coords['filename'] = filename
    
    # reading config file
    config = load_config(spe_data)

    # fit the spectrum
    fit = fit_spectra(spe_data, config)

    # compensate the instrumental drift in the Doppler shift, by assuming axisymmetric rotation
    fit = compensate_drift(fit, config)
    
    # calculate the physics quantity
    fit['Vi'] = (fit['shift'] - fit['drift']) / fit['line_center'] * CVEL
    fit['Vi_err'] = squaresum(
        fit['shift_err'], config['inst_wl_dx0std']
    ) / fit['line_center'] * CVEL
    fit['Ti'] = (
        fit['width'] / fit['line_center'] * CVEL
    )**2 * fit['mass'] * J_TO_EV
    fit['Ti_err'] = 2 * np.abs(
        fit['Ti'] * squaresum(fit['width_err'], config['inst_wl_w0std']) / fit['width']
    )

    # plot the result
    plot_fit(fit, config)

    # save the fit
    fit.to_netcdf(os.path.join(
        config['save_dir'], 
        filename.replace('.spe', '_fit.nc').split(os.sep)[-1]
    ))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python flometry.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    run(filename)
