from WhamData import WhamDiagnostic
import xarray as xr
import spe_parser

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
Refactoring of Keisuke's OES analysis class

TQ 8 July 2025
'''


# Used ROI (ROI8 is blank)
ROI = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10])

# How were these determined?
WEIGHT = np.array([1, 1, 1, 1/2, 1, 1, 1, 1/2, 1, 1])  # in TQ 9/24 APS static copy
#WEIGHT = np.array([1, 1, 1, 1, 1, 1, 1, 1/2, 1/2, 1]) # in Keisuke 9/18 static copy

# LOS for the ROI -- these are OLD LOS positions from circa 2024 September to 2025 March
LOS_2409 = (np.array([
        6, 5, 8, 3, 10, 1, 11, 2, 9, 4, 7 
        ]) - 5.8) * 15
# SECOND ATTEMPT, using photograph of Keisuke's lab notebook from 2025 March
# 07, mapping SPEC2 1 --> 1-6, 2 --> 1-5, 3 --> 3-8, et cetera
# posted in thread with Solomon Murdock on Slack #ornl_pdp channel.
# This one agrees better with Keisuke's plot WHAM_250306045.png on Slack.
LOS_2503 = np.array([
     14.3, # 1 <--> 1-6 (B6-6)
    -14.3, # 2 <--> 1-5 (B6-5)
     71.1, # 3 <--> 3-8 (B4-8)
    -71.1, # 4 <--> 3-3 (B4-3)
     126.7, # 5 <--> 1-10 (B6-10)
    -126.7, # 6 <--> 1-1 (...)
    153.7, # 7 <--> 1-11   # only fiber without a "paired" negative position
    -99.1, # 8 <--> 1-2
     99.1, # 9 <--> 3-9
    -42.7, # 10 <--> 1-4
     42.7, # 11 <--> 1-7
])

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

### for fits
def gaussian(x, area, center, sigma, offset):
    """
    Standard normalized Gaussian function:
        f(x) = (area / (sigma * sqrt(2π))) * exp(-0.5 * ((x - center)/sigma)^2) + offset

    Parameters
    ----------
    x : ndarray
        Input wavelength or coordinate axis
    area : float
        Integrated area under the Gaussian curve
    center : float
        Center (mean) of the Gaussian
    sigma : float
        Standard deviation (width parameter)
    offset : float
        Vertical offset (baseline)

    Returns
    -------
    ndarray : Gaussian evaluated at each x
    """
    xn = (x - center)/sigma
    norm = np.exp( -0.5 * xn**2 ) / np.sqrt(2.0 * np.pi)
    return norm * (area/sigma) + offset

def twogauss(
    x,              # input wavelength axis
    total_area,     # total area under both Gaussians
    center,         # central peak location
    width, # physical broadening (e.g. Doppler)
    baseline,       # additive offset
    alpha,          # fraction of area in second component
    delta0, delta1, # calibration-based centroid shifts
    sigma0, sigma1  # instrumental widths
):
    """
    Models an asymmetric spectral line as the sum of two Gaussians
    with separate centroid shifts and widths.

    Parameters
    ----------
    x : ndarray
        Wavelength or coordinate axis
    total_area : float
        Total integrated area (sum of both components)
    center : float
        Physical line center (unshifted)
    width : float
        Broadening due to plasma physics (Doppler, etc.)
    baseline : float
        Constant background offset
    alpha : float
        Fraction of area in the second Gaussian (0 < alpha < 1)
    delta0, delta1 : float
        Calibration offsets for component 1 and 2 (nm)
    sigma0, sigma1 : float
        Instrumental broadening widths for component 1 and 2 (nm)

    Returns
    -------
    ndarray : Model spectrum evaluated at x
    """
    # First Gaussian: (1 - alpha) fraction, shifted by delta0
    G1 = gaussian(
        x,
        area=(1 - alpha),
        center=center + delta0,
        sigma=np.sqrt(width**2 + sigma0**2),
        offset=0
    )

    # Second Gaussian: alpha fraction, shifted by delta1
    G2 = gaussian(
        x,
        area=alpha,
        center=center + delta1,
        sigma=np.sqrt(width**2 + sigma1**2),
        offset=0
    )
    return total_area * (G1 + G2) + baseline


### Physics constants
mC = 1.9944236560726842e-26  # mass of carbon in kg
mHe = 6.646473665811757e-27  # mass of helium in kg
cvel = 2.99792458e8  # light speed in m/s

def joule_to_eV(joule):
    '''return eV value for joule'''
    return joule * 6.24150962915265e+18

def squaresum(x, y):
    return np.sqrt(x * x + y * y)

### for colors
import matplotlib.pylab as pylab
N = 30
cmap = pylab.cm.coolwarm(np.linspace(0,1,N))
def get_color_index(V, M=8):
    input_ax = np.linspace(-M,M,N)
    idx = np.argmin(np.abs(V - input_ax))
    return idx

class Spectrometer(WhamDiagnostic):

    def __init__(self,shot,):
        super().__init__(shot)

        if not self.is_loaded:
            self.load()

    def load(self):
        self.openSPE()
        self.loadSpectra()

    def openSPE(self,
                root = "/mnt/n/whamdata/optical_spectroscopy/SPEs",
                ):

        shot_str = str(self.shot)
        yr = shot_str[:2]
        mn = shot_str[2:4]
        dy = shot_str[4:6]
        
        path = f"{root}/{yr}/{mn}/{dy}"
        f2 = f"{path}/WHAM2_{shot_str}.spe"

        datain = spe_parser.xr_open(f2)
        self.spe = datain
        self.fname2 = f2

    def loadSpectra(self):
        data = self.spe.isel(time=0, y=0).swap_dims({'x': 'wavelength'}).isel(roi=ROI)
        
        shot_str = str(self.shot)
        if int(shot_str[:6]) >= 250306:
            print('using new impact parameters from 2025 March (LOS_2503)')
            LOS = LOS_2503
        else:
            print('using old impact parameters from 2024 September (LOS_2409)')
            LOS = LOS_2409
        
#        data.coords['filename'] = filename
        data.coords['los'] = 'roi', [LOS[r.item()] for r in data['roi']]
        
        self.wavelength = data.wavelength.data
        self.spectra = data.data
        self.radius = LOS/10 # cm
        self.roi = data['roi'].data
        self.data = data


    def fit_CIII(self,
                 l_range=slice(464.6, 465.4),
                 lam0=465.0,
                 ):

        # Wavelength axis
        wl = self.data.wavelength.sel(wavelength=l_range).data

        # Prepare result storage
        results = {
            "radius": [],
            'roi': [], 
            'shift': [], 'shift_err': [],
            'intensity': [], 'intensity_err': [],
            'width': [], 'width_err': [],
            'Ti': [], 'Ti_err': [],
            'Vi': [], 'Vi_err': [],
            "success": [],
        }

        fit_outputs = {
            'wavelength': wl,
            'fits': [],  # shape: (n_roi, len(wl))
            'popt': [],    # shape: (n_roi, n_params)
            'pcov': [],    # optional
            'spectra': [],
        }

        for i,roi in enumerate(self.roi):
            radius = self.radius[i]
            spec = self.data.isel(roi=i).sel(wavelength=l_range).data

            ### construct fit target

            # instrument broadened gaussian (roi specific)
            instprofile = lambda x, A, x0, w, y0: twogauss(
                x, A, x0, w, y0,
                alpha=WL_ALPHA[roi],
                delta0=WL_DX0[roi], delta1=WL_DX1[roi],
                sigma0=WL_W0[roi], sigma1=WL_W1[roi]
            )
    
            def profile(x, bg, dxC, nC, wC, dxCu=0, nCu=0, wCu=1, dxO=0, nO=0, wO=1):
                '''
                bg: background
                dx: shift
                n: amplitude
                w: width

                for Carbon, Copper, Oxygen

                wl: wavelength
                gA: relative srength
                '''

                # background
                y = bg

                # add 3 C-III lines
                for wl, gA in [
                    (464.7418, 3.63e+08),
                    (465.0246, 2.18e+08),
                    (465.1473, 7.24e+07)
                ]:
                    y = y + instprofile(x, np.abs(nC) * gA, wl + dxC, wC, 0)

                # add Cu line
                for wl, gA in [(464.927084, 1e+08)]:
                    y = y + instprofile(x, np.abs(nCu) * gA, wl + dxCu, wCu, 0)

                # add O line
                for wl, gA in [(465.08384, 1e8)]:
                    y = y + instprofile(x, nO * gA, wl + dxO, wO, 0)

                return y
   

            # Initial parameter guesses
            # amplitude, cneter, width, offset, alpha,
            p0 = 0.0, 0.06, 1e-6, 0.03, 0.05, 1e-6, 0.02

            #amp_guess = spec.max() - np.median(spec)
            #offset_guess = np.median(spec)
            #center_guess = wl[np.argmax(spec)]
            #width_guess = 0.1  # nm
            #alpha_guess = 0.5
            #dx0_guess = dx1_guess = 0.0
            #w0_guess = w1_guess = 0.02
            #p0 = [amp_guess, center_guess, width_guess, offset_guess, alpha_guess, dx0_guess, dx1_guess, w0_guess, w1_guess]

            # Fit model
            try:
                if spec.max() > 1000:

                    # first fit Carbon only
                    popt, pcov = curve_fit(
                        profile, wl, spec,
                        p0=p0[:4]
                    )

                    # then perturbatively add impurity lines
                    popt, pcov = curve_fit(
                        profile, wl, spec,
                        p0=tuple(popt) + (popt[1] - 0.01, popt[2] / 2) + tuple(popt[3:])
                    )
                    results["success"].append(True)
                else:
                    popt = np.full(8, np.nan)
                    results["success"].append(False)

            except Exception as e:
                print(f"Fit failed for ROI {roi}: {e}")
                popt = np.full(8, np.nan)
                pcov = np.full((8, 8), np.nan)
                results[key].append(False)

            fit = profile(wl, *popt)
            fit_outputs['fits'].append(fit)
            fit_outputs['popt'].append(popt)
            fit_outputs['pcov'].append(pcov)
            fit_outputs['spectra'].append(spec)

            perr = np.sqrt(np.diagonal(pcov))
            shift = popt[1]; shift_err = perr[1]
            intensity = popt[2]; intensity_err = perr[2]
            width = popt[3]; width_err = perr[3]
        
            Vi = (shift/lam0) * cvel
            Vi_err = (shift_err + WL_DX0STD)/lam0 * cvel
        
            Ti = joule_to_eV((width/lam0 * cvel)**2 * mC) # mV^2
            Ti_err = 2.0 * np.abs(squaresum(width_err, WL_W0STD) / width * Ti)
        
            results['roi'].append(roi)
            results['shift'].append(shift)
            results['shift_err'].append(shift_err)
            results['intensity'].append(intensity)
            results['intensity_err'].append(intensity_err)
            results['width'].append(width)
            results['width_err'].append(width_err)
            results['Ti'].append(Ti)
            results['Ti_err'].append(Ti_err)
            results['Vi'].append(Vi)
            results['Vi_err'].append(Vi_err)
            results["radius"].append(radius)

        ### save
        coords = {
            "roi": results["roi"],
            "radius": ("roi", results["radius"]),
        }

        data_vars = {
            key: ("roi", val)
            for key, val in results.items()
            if key not in coords
        }
        data = xr.Dataset(data_vars, coords=coords).sortby('radius')

        # correct velocity offset
        center = (data['Vi'] * WEIGHT).mean() / np.mean(WEIGHT)
        data['Vi'] -= center
        
        self.CIII_fit_data = data
        self.fit_outputs = fit_outputs

    def plotSpectra(self):

        shift = 0.1
        for j,spectra in enumerate(self.spectra):
            r = self.radius[j]
            c_idx = get_color_index(r)

            tag = f"r = {r:.2f}"
            plt.plot(self.wavelength - shift, spectra, '.-', color=cmap[c_idx], label=tag, lw=0.5)

        plt.legend()
        plt.title(self.shot)

    def plotProfile(self):
        fig,axs = plt.subplots(3,1,figsize=(9,7))

        results = self.CIII_fit_data
        r = results['radius'].data

        idx = np.argsort(r)
        Vi = results['Vi'].data[idx]
        Ti = results['Ti'].data[idx]
        I = results['intensity'].data[idx]

        dVi = results['Vi_err'].data[idx]
        dTi = results['Ti_err'].data[idx]
        dI = results['intensity_err'].data[idx]

        rax = r[idx]
        axs[0].errorbar(rax, Vi, yerr=dVi)
        axs[1].errorbar(rax, Ti, yerr=dTi)
        axs[2].errorbar(rax, I , yerr=dI)

        axs[0].plot(rax, Vi, 'o')
        axs[1].plot(rax, Ti, 'o')
        axs[2].plot(rax, I, 'o')

        axs[0].set_title("Vi")
        axs[1].set_title("Ti")
        axs[2].set_title("I")

        for a in axs:
            a.grid()

        axs[-1].set_xlabel("radius (cm)")

        fig.suptitle(self.shot)

    def plotFits(self):
        fig,axs = plt.subplots(5,2,figsize=(9,7))
        results = self.CIII_fit_data
        radius = results['radius'].data
        fits = self.fit_outputs['fits']
        spectra = self.fit_outputs['spectra']
        wavelength = self.fit_outputs['wavelength']

        ax = axs.flatten()
        for j,r in enumerate(radius):
            ax[j].plot(wavelength,spectra[j], label=f"{r:.1f} cm")
            ax[j].plot(wavelength,fits[j], '--')
            ax[j].legend()
            ax[j].grid()

        fig.suptitle(self.shot)

    def stackProfiles(self, fig, axs, **kwargs):

        results = self.CIII_fit_data
        r = results['radius'].data

        idx = np.argsort(r)
        Vi = results['Vi'].data[idx]
        Ti = results['Ti'].data[idx]
        I = results['intensity'].data[idx]

        dVi = results['Vi_err'].data[idx]
        dTi = results['Ti_err'].data[idx]
        dI = results['intensity_err'].data[idx]

        rax = r[idx]
        axs[0].errorbar(rax, Vi, yerr=dVi, fmt='o-', label=self.shot,  **kwargs)
        axs[1].errorbar(rax, Ti, yerr=dTi, fmt='o-', **kwargs)
        axs[2].errorbar(rax, I , yerr=dI , fmt='o-', **kwargs)

        axs[0].set_title("Vi")
        axs[1].set_title("Ti")
        axs[2].set_title("I")

        axs[-1].set_xlabel("radius (cm)")



