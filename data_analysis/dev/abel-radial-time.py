import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from WhamData import ECH, BiasPPS, Interferometer, FluxLoop, EdgeProbes, NBI, AXUV, EndRing, Gas, adhocGas, IonProbe, ShineThrough, Dalpha
from DataSpec import Spectrometer
from multi import get_time_index 

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy import integrate
import abel 

'''
Updated 4/20/2025
'''

shot = 250220102
shot_ref = 250220099

shot = 250320104
shot_ref = 250320089

shot = 250324094
shot_ref = 250324080

class ChordData:
    """Store and plot chord data with consistent coloring"""
    
    def __init__(self, chord_RT, radius, time,
                  PLASMA_EDGE = 24,   # Edge of plasma in cm
                  R_VESSEL = 36,      # Vessel radius
                  XMESH = 101,        # Number of mesh points
                  RECON_CENTER = 0,   # Center reconstruction mode (0: fixed, 1: peak, 2: center of mass)
                 ):
        self.chord_RT = chord_RT  # RT: Radius-Time data
        self.time = time
        self.radius = radius
        
        # Generate radial coordinates
        self.chords_highres = np.linspace(-R_VESSEL, R_VESSEL, XMESH)
        self.n_highres = XMESH
        self.plasma_edge = PLASMA_EDGE

        # Create color maps for plots
        M = len(chord_RT)
        self.c_turbo = plt.cm.turbo(np.linspace(0, 0.8, M))
        self.c_plasma = plt.cm.plasma(np.linspace(0, 0.8, 7))  # For different time windows
    
    def plot_time_evolution(self, ax, t=3):
        """Plot time evolution of all chords"""
        for j, data in enumerate(self.chord_RT):
            ax.plot(self.time, data, color=self.c_turbo[j], lw=0.5)
        ax.axvline(t, color='k', ls='--', label=f"t = {t:.2f} ms")
    
    def plot_radial_profile(self, ax, t=3):
        """Plot radial profile at different time windows around time t"""
        tax = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5]  # Windows for averaging (seconds)
        
        for k, dt in enumerate(tax):
            # Find time indices for the window
            t1,t2 = get_time_index(self.time, t-dt/2, t+dt/2)
            
            # Average over the time window
            f = np.mean(self.chord_RT[:, t1:t2], axis=1)
            ax.plot(self.radius, f, 'o-', label=f"dt = {1000*dt} us", 
                   color=self.c_plasma[k], lw=0.7)

    def plot_abel_profile(self, ax, t=3, cm_to_m=False):
        """Plot radial profile at different time windows around time t"""
        tax = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5]  # Windows for averaging (seconds)
        
        r = self.chords_highres[self.n_highres//2:]
        for k, dt in enumerate(tax):
            # Find time indices for the window
            t1,t2 = get_time_index(self.time, t-dt/2, t+dt/2)
            
            # Average over the time window
            f = np.mean(self.radial_profile[t1:t2], axis=0)
            if cm_to_m:
                f = f*100
            ax.plot(r, f, 'o-', color=self.c_plasma[k], lw=0.7)

            if k == 5:
                # plot peaking factor
                n = f/np.max(f) # normalized profile
                a = np.argwhere(n<0.05)[0,0] # argument at edge
                #m = np.sum((n*r)[:a]) / np.sum(r[:a]) # area integrated mean
                m = np.mean(n[:a]) # line integrated mean
                edge = r[a]
                ax.axhline(np.max(f)*m, color='k', ls='--', lw=1, label=f"Peaking Factor {1/m:.2f}")
                ax.axvline(edge, color='k', ls='--', lw=1, label=f"edge {edge:.2f} cm")

    def plot_forward_model(self, ax,t=3):

        chords = self.chords_highres

        r = chords[self.n_highres//2:]
        dt = 0.2
        t1,t2 = get_time_index(self.time, t-dt/2, t+dt/2)
        fr = np.mean(self.radial_profile[t1:t2], axis=0)
        out = forward_model(r, fr, chords)
        ax.plot(chords, out, ls='--', color='k', label='fit')

    def extend_edge_data(self, plasma_edge, drop=[]):
        """
        Extend the data beyond plasma edge to improve boundary conditions
        Simplified version that ensures strictly increasing x coordinates
        
        Parameters:
        -----------
        y_data : 2D array of shape (n_radial, n_time)
            Signal intensity data
        x_coords : 1D array of shape (n_radial,)
            Radial coordinates
        plasma_edge : float
            Location of plasma edge
            
        Returns:
        --------
        x_padded, y_padded : Arrays with extended boundaries
        """
        # Make a copy to avoid modifying the original data
        x = self.radius.copy()
        y = self.chord_RT.copy()

        if drop != []:
            x = np.delete(x,drop)
            y = np.delete(y,drop,axis=0)
        
        # Sort coordinates if needed to ensure increasing order
        arg = np.argsort(x)
        x = x[arg]
        y = y[arg]
        
        # Number of time points
        n_time = y.shape[1]
        
        # Define edge padding - simpler approach
        # Add 5 points: 2 at each edge, plus zero padding
        n_padding = 5
        n_total = len(x) + n_padding
        
        # Create padded arrays
        x_padded = np.zeros((n_total, n_time))
        y_padded = np.zeros((n_total, n_time))
        
        # Add original data in the middle
        x_padded[2:-3] = np.tile(x[:, np.newaxis], (1, n_time))
        y_padded[2:-3] = y
        
        # Define edge positions with safety buffer
        x_min = x[0]
        x_max = x[-1]
        
        # Create edge padding x values - ensure strictly increasing
        # Left edge (smaller values)
        x_left_1 = x_min - 0.02 * (x_max - x_min)
        x_left_2 = x_min - 0.01 * (x_max - x_min)
        
        # Right edge (larger values)
        x_right_1 = x_max + 0.01 * (x_max - x_min)
        x_right_2 = x_max + 0.02 * (x_max - x_min)
        
        # Set edge x values
        for t in range(n_time):
            x_padded[0, t] = x_left_1 - 0.01  # Zero padding with slightly smaller x
            x_padded[1, t] = x_left_1
            x_padded[2, t] = x_left_2
            x_padded[-3, t] = x_right_1
            x_padded[-2, t] = x_right_2
            x_padded[-1, t] = x_right_2 + 0.01  # Zero padding with slightly larger x
        
        # Simple linear extrapolation for edge y values
        for t in range(n_time):
            # Calculate slopes at edges using first/last two points
            left_slope = (y[1, t] - y[0, t]) / (x[1] - x[0])
            right_slope = (y[-1, t] - y[-2, t]) / (x[-1] - x[-2])
            
            # Extrapolate to edge points
            y_padded[1, t] = max(0, y[0, t] - left_slope * (x[0] - x_left_1))
            y_padded[2, t] = max(0, y[0, t] - left_slope * (x[0] - x_left_2))
            
            y_padded[-3, t] = max(0, y[-1, t] + right_slope * (x_right_1 - x[-1]))
            y_padded[-2, t] = max(0, y[-1, t] + right_slope * (x_right_2 - x[-1]))
            
            # Zero padding at the very edges
            y_padded[0, t] = 0
            y_padded[-1, t] = 0
        
        # Verify that x coordinates are strictly increasing for each time point
        for t in range(n_time):
            if not np.all(np.diff(x_padded[:, t]) > 0):
                # Fix any issues with non-increasing values
                eps = 1e-6
                for i in range(1, n_total):
                    if x_padded[i, t] <= x_padded[i-1, t]:
                        x_padded[i, t] = x_padded[i-1, t] + eps

        self.radius_padded = x_padded
        self.chord_padded = y_padded
        return x_padded, y_padded

    def analyze_plasma(self, radial_coords):
        """
        Analyze plasma data to extract key parameters
        
        Parameters:
        -----------
        x_padded : 2D array of shape (n_radial_extended, n_time)
            Extended radial coordinates
        y_padded : 2D array of shape (n_radial_extended, n_time)
            Extended signal data
        radial_coords : 1D array
            High resolution radial coordinates
        time_coords : 1D array
            Time coordinates
            
        Returns:
        --------
        dict with plasma parameters
        """
        x_padded = self.radius_padded 
        y_padded = self.chord_padded 
        time_coords = self.time
        # Clean up signal and calculate total
        signal = np.maximum(y_padded, 0)
        total_signal = np.sum(signal, axis=0) + 1e-20
        
        # Use the first time point for x coordinates
        x_for_interp = x_padded[:, 0]
        
        # Check if x coordinates are already sorted
        if np.any(np.diff(x_for_interp) <= 0):
            # Sort the coordinates to ensure they're strictly increasing
            sort_idx = np.argsort(x_for_interp)
            x_for_interp = x_for_interp[sort_idx]
            
            # Apply the same sorting to the signal data (at all time points)
            sorted_signal = np.zeros_like(signal)
            for t in range(signal.shape[1]):
                sorted_signal[:, t] = signal[sort_idx, t]
            signal = sorted_signal
        
        # Compute center of mass and RMS radius
        mean_position = np.sum(signal * x_padded, axis=0) / total_signal
        rms_radius = 2 * np.sqrt(np.sum(signal * (x_padded - mean_position)**2, axis=0) / total_signal)
        
        # Create a direct mapping for high resolution without RectBivariateSpline
        # Instead, use linear interpolation which is simpler and more robust
        signal_hr = np.zeros((len(radial_coords), signal.shape[1]))
        
        for t in range(signal.shape[1]):
            signal_hr[:, t] = np.interp(radial_coords, x_for_interp, signal[:, t], left=0, right=0)
        
        # Compute high-resolution center of mass and RMS radius
        signal_hr_sum = np.sum(signal_hr, axis=0) + 1e-20
        mean_position_hr = np.sum(signal_hr * radial_coords.reshape(-1, 1), axis=0) / signal_hr_sum
        rms_radius_hr = 2 * np.sqrt(np.sum(signal_hr * (radial_coords.reshape(-1, 1) - mean_position_hr)**2, axis=0) / signal_hr_sum)
        
        # Decompose signal into symmetric and asymmetric parts
        mid_point = len(radial_coords) // 2
        center_shift = np.zeros(signal_hr.shape[1]) + mid_point
        
        # Extract symmetric and asymmetric components
        symmetric_component = np.zeros_like(signal_hr)
        asymmetric_component = np.zeros_like(signal_hr)
        dr = (radial_coords[-1] - radial_coords[0])/(len(radial_coords)-1)
       
        for t in range(signal_hr.shape[1]):
            # Center the signal
            indices = np.mod(np.arange(len(radial_coords)) - mid_point + int(center_shift[t]), len(radial_coords))
            centered_signal = signal_hr[indices, t]
            
            # Decompose into symmetric and asymmetric parts
            # Create reversed signal for symmetry check
            reversed_signal = centered_signal[::-1]
            
            # Calculate asymmetric component
            asymmetric = np.maximum(-reversed_signal + centered_signal, 0)
            
            # Calculate symmetric component
            symmetric = centered_signal - asymmetric
            
            # Smooth components
            symmetric_component[:, t] = gaussian_filter(symmetric, sigma=2)
            asymmetric_component[:, t] = gaussian_filter(asymmetric, sigma=2)
        
        # Perform Abel inversion for radial profile
        # Make sure we have valid data for Abel inversion
        try:
            projection = symmetric_component[mid_point:, :].T
            radial_profile = abel.basex.basex_transform(projection, direction="inverse", verbose=False, dr=dr)
        except Exception as e:
            print(f"Abel inversion failed: {e}")
            radial_profile = np.zeros((signal.shape[1], mid_point))
        self.radial_profile = radial_profile 
        return {
            'centroid': mean_position,
            'radius': rms_radius,
            'centroid_hr': mean_position_hr,
            'radius_hr': rms_radius_hr,
            'signal_hr': signal_hr,
            'symmetric': symmetric_component,
            'asymmetric': asymmetric_component,
            'radial_profile': radial_profile
        }

    def extend_edge_data_2(self, plasma_edge, mode="l", drop=[]):
        '''
        Extends plasma profile data beyond measurement range
        
        Parameters:
        plasma_edge: scalar - edge of plasma
        mode: string - mode of filling
        
        Returns:
        x_padded, y_padded: extended coordinate and signal arrays
        '''

        x = self.radius.copy()
        y = self.chord_RT.copy()

        # Ensure x is in descending order for consistent processing
        flip = 0
        if x[1] > x[0]:
            x = x[::-1]
            y = y[::-1]
            flip = 1
    
        if drop != []:
            x = np.delete(x,drop)
            y = np.delete(y,drop,axis=0)

        # Define plasma edge boundaries
        plasma_edge_p = plasma_edge
        plasma_edge_n = -plasma_edge
        
        # Adjust edges if needed
        if x[0] >= plasma_edge:
            plasma_edge_p = x[0] + 0.01  # 1cm
        if x[-1] <= -plasma_edge:
            plasma_edge_n = x[-1] - 0.01  # 1cm
        
        # Create intermediate points for padding
        left_x_mid = np.mean([plasma_edge_p, x[0]])
        right_x_mid = np.mean([plasma_edge_n, x[-1]])
        left_x_mid12 = np.linspace(plasma_edge_p, x[0], 4)[1:3]
        right_x_mid12 = np.linspace(plasma_edge_n, x[-1], 4)[1:3]
        left_xinter = plasma_edge_p
        right_xinter = plasma_edge_n
    
        # Calculate slopes using mean of adjacent edge elements
        leftslope = ((y[0] + y[1]) / 2 - (y[1] + y[2]) / 2) / \
                    ((x[0] + x[1]) / 2 - (x[1] + x[2]) / 2)
        
        rightslope = ((y[-1] + y[-2]) / 2 - (y[-2] + y[-3]) / 2) / \
                     ((x[-1] + x[-2]) / 2 - (x[-2] + x[-3]) / 2)
    
        # Gaussian falloff function for edge treatment
        def gaussian_falloff(x0, y0, m, plasma_edge, sigma_nm):
            sigma = np.abs(x0 - plasma_edge) / sigma_nm
            G_mu = x0 + (m / (y0 + 1e-20)) * sigma**2
            G_A = y0 * np.exp(m**2 * sigma**2 / (2 * y0**2 + 1e-20))
            return G_mu, G_A, sigma
        
        # Determine which points use Gaussian falloff
        total_signal = np.sum(y, 0)
        max_signal = np.max(y, 0)
        signal_threshold = 0.050
        min_total_signal = 0.02
        
        use_gaussian_right = (y[-1] / (max_signal + 1e-20) >= signal_threshold) & (total_signal > min_total_signal)
        use_gaussian_left = (y[0] / (max_signal + 1e-20) >= signal_threshold) & (total_signal > min_total_signal)
    
        # Calculate Gaussian parameters
        left_mu, left_A, left_sigma = gaussian_falloff(
            x[0], y[0, use_gaussian_left],
            leftslope[use_gaussian_left], plasma_edge=plasma_edge_p, sigma_nm=2
        )
        
        right_mu, right_A, right_sigma = gaussian_falloff(
            x[-1], y[-1, use_gaussian_right],
            rightslope[use_gaussian_right], plasma_edge=plasma_edge_n, sigma_nm=2
        )
    
        # Enforce slope direction
        leftslope[leftslope >= 0] = -leftslope[leftslope >= 0]
        rightslope[rightslope <= 0] = -rightslope[rightslope <= 0]
        
        # Calculate y-intercepts
        left_c = y[0] - leftslope * x[0]
        right_c = y[-1] - rightslope * x[-1]
        
        # Calculate y values for padding points using linear extrapolation
        left_y_mid = left_x_mid * leftslope + left_c
        right_y_mid = right_x_mid * rightslope + right_c
        left_y_mid12 = left_x_mid12.reshape(left_x_mid12.shape[0], 1) * leftslope + left_c
        right_y_mid12 = right_x_mid12.reshape(right_x_mid12.shape[0], 1) * rightslope + right_c
        
        # Ensure non-negative values
        left_y_mid12[left_y_mid12 < 0] = 0
        right_y_mid12[right_y_mid12 < 0] = 0
        left_y_mid[left_y_mid < 0] = 0
        right_y_mid[right_y_mid < 0] = 0
        
        # Apply Gaussian falloff where appropriate
        right_y_mid[use_gaussian_right] = right_A * np.exp(-((right_x_mid - right_mu)**2) / (2 * right_sigma**2))
        left_y_mid[use_gaussian_left] = left_A * np.exp(-((left_x_mid - left_mu)**2) / (2 * left_sigma**2))
        
        left_y_mid12[0, use_gaussian_left] = left_A * np.exp(-((left_x_mid12[0] - left_mu)**2) / (2 * left_sigma**2))
        right_y_mid12[0, use_gaussian_right] = right_A * np.exp(-((right_x_mid12[0] - right_mu)**2) / (2 * right_sigma**2))
        left_y_mid12[1, use_gaussian_left] = left_A * np.exp(-((left_x_mid12[1] - left_mu)**2) / (2 * left_sigma**2))
        right_y_mid12[1, use_gaussian_right] = right_A * np.exp(-((right_x_mid12[1] - right_mu)**2) / (2 * right_sigma**2))
        
        # Construct padded y array
        y_padded = np.concatenate([
            np.zeros((1, y.shape[1])),  # Top padding
            left_y_mid12,                         # Left padding
            y,                           # Original data
            right_y_mid12[::-1],                  # Right padding
            np.zeros((1, y.shape[1]))    # Bottom padding
        ], axis=0)
        
        # Adjust for minimum value
        y_padded = y_padded - np.max((y_padded[0, :], y_padded[-1, :]), 0)
        
        # Repeat x coordinates across all time points
        x_repeated = np.tile(x, (y.shape[1], 1)).T
        
        # Construct padded x array
        x_padded = np.concatenate([
            left_xinter + np.zeros((1, y.shape[1])),  # Top padding
            np.tile(left_x_mid12.reshape(2, 1), (1, y.shape[1])),  # Left padding
            x_repeated,  # Original data repeated
            np.tile(right_x_mid12[::-1].reshape(2, 1), (1, y.shape[1])),  # Right padding
            right_xinter + np.zeros((1, y.shape[1]))  # Bottom padding
        ], axis=0)
        
        # Restore original order if necessary
        if flip == 1:
            x_padded = x_padded[::-1]
            y_padded = y_padded[::-1]
            
        self.radius_padded = x_padded
        self.chord_padded = y_padded
        return x_padded, y_padded


# Function to perform forward modeling (inverse of Abel inversion)
def forward_model(r, emission_profile, chords):
    """
    Calculate chord-integrated measurements from radial emission profile
    
    Parameters:
    - r: array of radius values (cm)
    - emission_profile: radial emission profile (from Abel inversion)
    - chords: array of chord positions (impact parameters) (cm)
    
    Returns:
    - chord_integrated: array of line-integrated measurements
    """
    # Create interpolation function for emission profile
    # Use zero for extrapolation (outside the plasma)
    f_emission = interp1d(r, emission_profile, bounds_error=False, fill_value=0)
    
    chord_integrated = np.zeros_like(chords)
    
    for i, chord in enumerate(chords):
        # For each chord (impact parameter p), integrate along the line of sight
        # Use the relation r² = z² + p² where z is distance along the line of sight
        
        # Define integrand: 2 * emission(r) where r = sqrt(z² + p²)
        def integrand(z):
            radius = np.sqrt(z**2 + chord**2)
            return 2.0 * f_emission(radius)  # Factor 2 for symmetry
        
        # Integration limits - use a reasonable cutoff based on the max radius
        z_max = np.sqrt(max(r)**2 - chord**2) if chord < max(r) else 0
        
        if z_max > 0:
            # Integrate from 0 to z_max (take advantage of symmetry)
            result, _ = integrate.quad(integrand, 0, z_max)
            chord_integrated[i] = result
        else:
            chord_integrated[i] = 0.0
            
    return chord_integrated


tax = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5] # window for averaging fluctuations

da = Dalpha(shot)
chord_da = ChordData(da.data, da.radius, da.time)

axuv = AXUV(shot)
chord_axuv= ChordData(axuv.data, axuv.b*100, axuv.time, PLASMA_EDGE=15)

nbi = NBI(shot) # it seems for some NBI-only shots shineThrough is not populated
ref = NBI(shot_ref)


# calculate line density 
t0,t1 = get_time_index(nbi.time, nbi.t_start, nbi.t_stop)
# mask time when there is no beam
nbi.sigma_cx[:t0] = 1
nbi.sigma_cx[t1:] = 1
chord_nbi = np.log(ref.d_arr/nbi.d_arr)/nbi.sigma_cx
chord_nbi[:,:t0] = 0
chord_nbi[:,t1:] = 0

chord_shine = ChordData(chord_nbi, nbi.chord_radius, nbi.time, PLASMA_EDGE=15)

fig,axs = plt.subplots(3,3,figsize=(14, 8), sharex='col')

save = True
save = False
#time_axis = np.linspace(0,20,201)
time_axis = np.linspace(0,12,121)
time_axis = np.linspace(0,8,81)
time_axis = [4.25]
for t in time_axis:

    chords = [chord_axuv, chord_shine]
    chords = [chord_axuv, chord_shine, chord_da]
    for i, chord in enumerate(chords):
        chord.plot_time_evolution(axs[i, 0], t)
        chord.plot_radial_profile(axs[i, 1], t)
       
        # extend data at edge
        edge = chord.plasma_edge
        if i==0:
            # AXUV
            chord.extend_edge_data_2(edge)
        elif i==1:
            # NBI
            #chord.extend_edge_data(edge)
            chord.extend_edge_data_2(edge)
            chord.extend_edge_data_2(edge, drop=[11,12,13]) # for 0324094
        elif i==2:
            # H-alpha
            chord.extend_edge_data(edge,drop=[0])
        else:
            chord.extend_edge_data(edge)
        results = chord.analyze_plasma(chord.chords_highres)

        try:
            if i==1:
                chord.plot_abel_profile(axs[i, 2], t, cm_to_m=True)
            else:
                chord.plot_abel_profile(axs[i, 2], t)
            chord.plot_forward_model(axs[i, 1], t)
        except:
            print("abel failed")


    axs[-1,0].set_xlim(-2,22)
    axs[-1,1].set_xlim(-20,20)
    axs[-1,0].set_xlabel("time (ms)")
    axs[-1,1].set_xlabel("chord (cm)")
    axs[-1,2].set_xlabel("radius (cm)")
    axs[0,0].set_ylabel("axuv")
    axs[1,0].set_ylabel("nbi shinethrough")
    axs[2,0].set_ylabel("H-alpha")
#    axs[0,0].set_ylim(-0.5e-7,1.5e-7) # axuv, low impurity shot 
    axs[1,0].set_ylim(-0.5e17, 4e18) # nbi, low density shot, 0324094
    
    axs[0,0].legend(loc=2,fontsize=9)
    axs[0,1].legend(loc=1, fontsize=8)

    axs[0,2].legend(loc=1, fontsize=9)
    axs[1,2].legend(loc=1, fontsize=9)
    axs[2,2].legend(loc=1, fontsize=9)
   
    axs[0,1].set_ylim(bottom=0)
    axs[1,1].set_ylim(bottom=0)
    for a in axs.flat:
        a.grid()
    fig.suptitle(shot)

    if save:
        plt.savefig(f"out/{shot}_radial_t{t:.2f}.png")
        for a in axs.flat:
            a.clear()
        print(t)

plt.show()
