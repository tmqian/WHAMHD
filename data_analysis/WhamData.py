import MDSplus as mds

import numpy as np
from scipy.signal import savgol_filter as savgol
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

'''
This colletion of classes loads data from WHAM MDS+ tree

classes:
    BiasPPS
    Interferometer
    FluxLoop
    AXUV
    ECH
    EdgeProbes
    EndRings

Updated 15 January 2025
'''

class WhamDiagnostic:
    """Base class for all WHAM diagnostics"""
    def __init__(self, shot):
        self.shot = shot
        self.is_loaded = False
        self.load_status_message = ""

        try:
            self.load()
            self.is_loaded = True
        except Exception as e:
            self.load_status_message = str(e)
            print(f"Error loading {self.__class__.__name__} for shot {shot}: {str(e)}")

    def load(self):
        """Each subclass must implement this method to load its data"""
        raise NotImplementedError("Subclasses must implement load()")

    def to_dict(self, detail_level='summary'):
        """
        Convert diagnostic data to a dictionary
        
        Parameters:
        -----------
        detail_level : str
            Level of detail to include:
            - 'status': Only loading status
            - 'summary': Key metrics (default)
            - 'full': Complete dataset
            
        Returns:
        --------
        dict
            Dictionary representation of the diagnostic data
        """
        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }
        
        if not self.is_loaded:
            result["error"] = self.load_status_message
            return result
            
        # For higher detail levels, subclasses must implement
        if detail_level in ['summary', 'full']:
            raise NotImplementedError(f"Subclasses must implement to_dict() with {detail_level} detail level")
            
        return result

# Helper Functions

def zero_offset(f,idx=2000):
    f -= np.mean(f[-idx:])

def get_time_array(time, t_array):
    '''
    time is a t-axis
    t_array is a list or times in ms

    returns a list of indexes for t-axis
    '''
    return [np.argmin(np.abs(time-t)) for t in t_array]


def STFT(signal, time,
         W = 250, # number of windows
        ):

    '''
    Compute "short time fourier transform" over a signal f(t)

    Partition f into W windows of equal size S.
    The original time axis with N points (N = W*S)
    is sub-sampled into a new time axis T with S points.
    It is assumed that W evenly divdies N.

    Computes M fouier modes up to nyquist frequency (M = S/2)
    F is the corresponding frequency space axis.
    If time is in ms, then F is in kHz.
    '''

    # Compute integer array sizes
    N = len(time)
    S = N // W # samples per window
    M = S//2 # fourier modes

    # Set up time and spectral axes
    T = time[::S]
    dt = time[1] - time[0] # ms
    F = fftfreq(S,dt)[:M] # kHz

    # Batch Fourier Transform
    block = signal.reshape((W, S))
    G = fft(block,axis=1)[:,:M] # complex valued (W x M)
    return G, T, F

def STFT2(signal, time,
         W = 500, # number of windows
         L = 1 # Length of window in ms
        ):


    N = len(time)
    dT = time[-1] - time[0] # range, ms
    dt = time[1] - time[0] # step, ms
    f = round(N/dT) # kHz
    S = int(L * f) # Samples per window
    M = S//2 # fourier modes

    # Set up time and spectral axes
    tax = np.arange(0,N,N//W) # t start indices for W windows
    cut = round(S/(N//W))+1 
    tax = tax[:-cut] # remove the last windows that over flow the time axis
    T = time[tax]
    F = fftfreq(S,dt)[:M] # kHz

    # Batch Fourier Transform
    block = np.array([signal[t:t+S] for t in tax])
    G = fft(block,axis=1)[:,:M] # complex valued (W-cut x M)
    return G, T, F

### End Helper

class BiasPPS(WhamDiagnostic):

    def __init__(self, shot,
                       ILEM_gain=-500,
                       VLEM_gain=800,
                       VFBK_gain=-1,
                       no_dtacq = False
                 ):

        if shot < 240728011:
            # this is when I fixed the double (-) sign
            VLEM_gain *= -1

        self.ILEM_gain = ILEM_gain
        self.VLEM_gain = VLEM_gain
        self.VFBK_gain = VFBK_gain
        self.no_dtacq = no_dtacq

        super().__init__(shot)


    def load(self):

        if self.no_dtacq:
            self.load_labview_demand()
            return

        # set up tree
        tree = mds.Tree("wham",self.shot)

        # load data from nodes
        data = "bias"
        L_Dem = tree.getNode(f"{data}.PPS_L.demand.filtered").getData().data()
        L_ILem = tree.getNode(f"{data}.PPS_L.current.filtered").getData().data()
        L_VLem = tree.getNode(f"{data}.PPS_L.voltage.filtered").getData().data()
        L_Vpps = tree.getNode(f"{data}.PPS_L.voltage_PWM.signal").getData().data()
        L_VFB = tree.getNode(f"{data}.PPS_L.feedback_dmd.signal").getData().data()

        R_Dem = tree.getNode(f"{data}.PPS_R.demand.filtered").getData().data()
        R_ILem = tree.getNode(f"{data}.PPS_R.current.filtered").getData().data()
        R_VLem = tree.getNode(f"{data}.PPS_R.voltage.filtered").getData().data()
        R_Vpps = tree.getNode(f"{data}.PPS_R.voltage_PWM.signal").getData().data()
        R_VFB = tree.getNode(f"{data}.PPS_R.feedback_dmd.signal").getData().data()

        time = tree.getNode(f"{data}.PPS_L.demand.filtered").dim_of().data() * 1e3 # ms


        # save
        self.time = time
        self.L_Dem  = L_Dem
        self.L_ILem = L_ILem
        self.L_VLem = L_VLem
        self.L_Vpps = L_Vpps * self.VFBK_gain
        self.L_VFB  = L_VFB * self.VFBK_gain
        self.R_Dem  = R_Dem
        self.R_ILem = R_ILem
        self.R_VLem = R_VLem
        self.R_Vpps = R_Vpps * self.VFBK_gain
        self.R_VFB  = R_VFB * self.VFBK_gain

        self.load_labview_demand()

        # load raw data
        raw = "raw.acq196_370"
        self.raw_L_Dem = tree.getNode (f"{raw}.ch_01").getData().data()
        self.raw_L_ILem = tree.getNode(f"{raw}.ch_02").getData().data()
        self.raw_L_VLem = tree.getNode(f"{raw}.ch_03").getData().data()
        self.raw_L_Vpps = tree.getNode(f"{raw}.ch_04").getData().data()
        self.raw_L_VFB = tree.getNode (f"{raw}.ch_05").getData().data()
        self.raw_R_Dem = tree.getNode (f"{raw}.ch_06").getData().data()
        self.raw_R_ILem = tree.getNode(f"{raw}.ch_07").getData().data()
        self.raw_R_VLem = tree.getNode(f"{raw}.ch_08").getData().data()
        self.raw_R_Vpps = tree.getNode(f"{raw}.ch_09").getData().data()
        self.raw_R_VFB = tree.getNode (f"{raw}.ch_10").getData().data()
        self.trigger = tree.getNode (f"{raw}.ch_11").getData().data()

    def load_labview_demand(self):

        # load demand
        if self.shot > 250301000:
            tree = mds.Tree("wham",self.shot)
            tbias = tree.getNode("bias.bias_params.trig_time").getData().data() # t0 for bias demand waveform
            self.Ldem_T = tree.getNode("bias.bias_params.dmd_waveform.pps_L_T").getData().data() + tbias*1e3
            self.Ldem_V = tree.getNode("bias.bias_params.dmd_waveform.pps_L_V").getData().data()
            self.Rdem_T = tree.getNode("bias.bias_params.dmd_waveform.pps_R_T").getData().data() + tbias*1e3
            self.Rdem_V = tree.getNode("bias.bias_params.dmd_waveform.pps_R_V").getData().data()

    def plot_raw(self):
        fig, axs = plt.subplots(2,1,figsize=(8,5), sharex=True)

        axs[0].plot(self.time, self.raw_L_Dem, lw=0.5, label="Limiter Demand")
        axs[0].plot(self.time, self.raw_L_Vpps, lw=0.5, label="Voltage PPS")
        axs[0].plot(self.time, self.raw_L_ILem, lw=0.5, label="Current LEM")
        axs[0].plot(self.time, self.raw_L_VLem, lw=0.5, label="Voltage LEM")
        axs[0].plot(self.time, self.raw_L_VFB , lw=0.5, label="Feedback Output")

        axs[1].plot(self.time, self.raw_R_Dem, lw=0.5, label="Ring Demand")
        axs[1].plot(self.time, self.raw_R_Vpps, lw=0.5, label="Voltage PPS")
        axs[1].plot(self.time, self.raw_R_ILem, lw=0.5, label="Current LEM")
        axs[1].plot(self.time, self.raw_R_VLem, lw=0.5, label="Voltage LEM")
        axs[1].plot(self.time, self.raw_R_VFB , lw=0.5, label="Feedback Output")

        axs[0].set_title(f"Raw Data: {self.shot}")
        axs[-1].set_xlabel('ms')
        axs[0].legend()
        axs[1].legend()

        for a in axs:
            a.grid()


    def plot_PPS(self):

        fig, axs = plt.subplots(3,2, figsize=(10,8), sharex=True)
        axs[0,0].plot(self.time, self.L_Dem , 'C0', lw=0.5, label="L Demand [V]")
        axs[0,1].plot(self.time, self.R_Dem , 'C0', lw=0.5, label="R Demand [V]")
        axs[0,0].plot(self.time, self.L_VFB , 'C4', lw=0.5, label="Feedback Output [-V]")
        axs[0,1].plot(self.time, self.R_VFB , 'C4', lw=0.5, label="Feedback Output [-V]")

        axs[1,0].plot(self.time, self.L_ILem, 'C1', lw=0.5, label="L Current [A]")
        axs[1,1].plot(self.time, self.R_ILem, 'C1', lw=0.5, label="R Current [A]")

        axs[2,0].plot(self.time, self.L_VLem, 'C2', lw=0.5, label="L Voltage [V]")
        axs[2,1].plot(self.time, self.R_VLem, 'C2', lw=0.5, label="R Voltage [V]")

        axs[2,0].plot(self.time, self.L_Vpps, 'C3', lw=0.5, label="L Voltage PPS [-V]")
        axs[2,1].plot(self.time, self.R_Vpps, 'C3', lw=0.5, label="R Voltage PPS [-V]")

        fig.suptitle(self.shot)
        axs[0,0].set_title("Limiter")
        axs[0,1].set_title("Ring")
        axs[-1,0].set_xlabel('ms')
        axs[-1,1].set_xlabel('ms')

        for a in np.ndarray.flatten(axs):
            a.legend(loc=1)
            a.grid()

        fig.tight_layout()

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':
            summary = {}

            # this will cause a bug later
            Vlim = np.max(self.Ldem_V)
            Vring = np.min(self.Rdem_V)
            summary['limiter_bias'] = Vlim
            summary['ring_bias'] = Vring

        result['summary'] = summary
        return result

class Interferometer(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        linedens = tree.getNode("diag.interferomtr.linedens").getData().data() 
        offset = np.mean(linedens[-2000:])

        time = tree.getNode("diag.interferomtr.time").getData().data() * 1e3

        self.time = time
        self.linedens = linedens - offset

    def plot(self):
        fig,axs = plt.subplots(1,1,figsize=(10,5))
        axs.plot(self.time,self.linedens,'C1', label=r"Line Density [m$^{-2}$]")

        axs.set_xlabel("time [ms]")
        axs.legend()
        axs.grid()
        axs.set_title(self.shot)
        plt.show()

    def fix_fringe_skip(self, t0=10, # ms
                              N=1, # signed of number of fringes
                              dn=2.3e18, # hard coded
                              back = False, # propagate change back from t = t0
                              ):

        t = np.argmin(np.abs(self.time - t0))

        if back:
            self.linedens[:t] += N*dn
        else:
            # propagate change forward from t=t0
            self.linedens[t:] += N*dn

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            # smooth
            T = self.time
            nint = self.linedens
            navg = nint/(2*0.136)
            N = savgol(navg, 501,3)
            jm = np.argmax(N)
                           
            t1,t2,t3 = get_time_array(T, [4.8, 9.8, 14.8])
            summary['max line avg dens (m^-3)'] = N[jm]
            summary['max time (ms)'] = T[jm]
            summary['dens 4.8ms (m^-3)'] = N[t1]
            summary['dens 9.8ms (m^-3)'] = N[t2]
            summary['dens 14.8ms (m^-3)'] = N[t3]

            result['summary'] = summary
        return result

class FluxLoop(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)

        # convert from Weber to Maxwell
        FL1 = tree.getNode("diag.fluxloops.fl1").getData().data() * 1e8
        FL2 = tree.getNode("diag.fluxloops.fl2").getData().data() * 1e8
        FL3 = tree.getNode("diag.fluxloops.fl3").getData().data() * 1e8

        # the sign on flux loop 3 is questionable, so we integrate
        #sign = np.sign( np.sum(FL3) )
        #FL3 *= sign

        # shot 0801 on
        time = tree.getNode("diag.fluxloops.fl3").dim_of().data() * 1e3

        # zero offset
        def offset(f,idx=1000):
            f -= np.mean(f[:idx])
            return f

        self.FL1 = offset(FL1)
        self.FL2 = offset(FL2)
        self.FL3 = offset(FL3)
        self.time = time

    def calcPressure(self, psi=2e6,
                           B1=0.275,
                           B2=0.470,
                           B3=1.122,
                           ):
        mu0 = 4*np.pi/1e7
        U1 = B1**2 / (2*mu0)
        U2 = B2**2 / (2*mu0)
        U3 = B3**2 / (2*mu0)

        beta1 = 1 - (1 - self.FL1/psi)**2
        beta2 = 1 - (1 - self.FL2/psi)**2
        beta3 = 1 - (1 - self.FL3/psi)**2

        P1 = beta1*U1
        P2 = beta2*U2
        P3 = beta3*U3

        # save
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

        self.P1 = P1
        self.P2 = P2
        self.P3 = P3



    def plot(self):

        fig,axs = plt.subplots(1,1,figsize=(10,5))
        axs.plot(self.time,self.FL1,'C3', label="Flux Loop 1 [Mx]")
        axs.plot(self.time,self.FL2,'C2', label="Flux Loop 2 [Mx]")
        axs.plot(self.time,self.FL3,'C0', label="Flux Loop 3 [Mx]")

        axs.set_xlabel("time [ms]")
        axs.legend()
        axs.grid()
        axs.set_title(self.shot)

    def plotExtra(self):

        fig,axs = plt.subplots(1,3,figsize=(12,5), sharex=True)
        axs[0].plot(self.time,self.FL1,'C3', label="Flux Loop 1 [Mx]")
        axs[0].plot(self.time,self.FL2,'C2', label="Flux Loop 2 [Mx]")
        axs[0].plot(self.time,self.FL3,'C0', label="Flux Loop 3 [Mx]")

        axs[1].plot(self.time,self.beta1,'C3', label="beta 1")
        axs[1].plot(self.time,self.beta2,'C2', label="beta 2")
        axs[1].plot(self.time,self.beta3,'C0', label="beta 3")

        axs[2].plot(self.time,self.P1,'C3', label="Pressure 1 [Pa]")
        axs[2].plot(self.time,self.P2,'C2', label="Pressure 2 [Pa]")
        axs[2].plot(self.time,self.P3,'C0', label="Pressure 3 [Pa]")
        for a in axs:
            a.set_xlabel("time [ms]")
            a.legend()
            a.grid()
            #a.set_xlim(-2,16)

        fig.suptitle(self.shot)
        fig.tight_layout()
        return fig,axs

    def load_raw(self, axs=None, N=100, alpha=0.7):

        tree = mds.Tree("wham",self.shot)

        # convert from Weber to Maxwell
        raw = "raw.acq1001_631"
        sig1a = tree.getNode(f"{raw}.ch_01").getData().data()
        sig1b = tree.getNode(f"{raw}.ch_02").getData().data()
        sig2a = tree.getNode(f"{raw}.ch_03").getData().data()
        sig2b = tree.getNode(f"{raw}.ch_04").getData().data()
        sig3a = tree.getNode(f"{raw}.ch_05").getData().data()
        sig3b = tree.getNode(f"{raw}.ch_06").getData().data()

        axs[0,0].plot(sig1a[::N], label=f"{self.shot}", alpha=alpha)
        axs[1,0].plot(sig1b[::N], alpha=alpha)
        axs[0,1].plot(sig2a[::N], alpha=alpha)
        axs[1,1].plot(sig2b[::N], alpha=alpha)
        axs[0,2].plot(sig3a[::N], alpha=alpha)
        axs[1,2].plot(sig3b[::N], alpha=alpha)

        axs[0,0].set_title("Flux Loop 1")
        axs[0,1].set_title("Flux Loop 2")
        axs[0,2].set_title("Flux Loop 3")
        axs[0,0].set_ylabel("loop a")
        axs[1,0].set_ylabel("loop b")

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            # smooth
            T = self.time
            F1 = savgol(self.FL1/1e3, 501,3)
            F2 = savgol(self.FL2/1e3, 501,3)
            jm = np.argmax(F1)
                           
            t1,t2,t3 = get_time_array(T, [4.8, 9.8, 14.8])
            summary['max flux (kMx)'] = F1[jm]
            summary['max time (ms)'] = T[jm]

            summary['flux 4.8ms (kMx)'] = F1[t1]
            summary['flux 9.8ms (kMx)'] = F1[t2]
            summary['flux 14.8ms (kMx)'] = F1[t3]

            summary['flux2 4.8ms (kMx)'] = F2[t1]
            summary['flux2 9.8ms (kMx)'] = F2[t2]
            summary['flux2 14.8ms (kMx)'] = F2[t3]
            result['summary'] = summary

        return result

class AXUV(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):
        tree = mds.Tree("wham",self.shot)

        data = []
        R = []
        Phi = []
        Z = []
        b = []
        Ohm = []
        label = []

        # the mds node changed names on March 24 2025
        I_tag = "current"
        if self.shot < 250324000:
            I_tag = "photocurrent"

        for j in range(20):
            root = f"diag.axuv.DIODEARRAY1.CH_{j+1:02d}"
            data.append(tree.getNode(f'{root}.{I_tag}').getData().data())
            #data.append(tree.getNode(f'{root}.PHOTOCURRENT').getData().data())
            label.append(tree.getNode(f"{root}.DIODE_NUM").getData().data())

            R.append(tree.getNode(f'{root}.R').getData().data())
            Phi.append( tree.getNode(f'{root}.PHI').getData().data() )
            Z.append( tree.getNode(f'{root}.Z').getData().data() )
            b.append( tree.getNode(f'{root}.B_IMPACT').getData().data())
            Ohm.append( tree.getNode(f'{root}.RESISTOR').getData().data())

        time = tree.getNode(f'{root}.{I_tag}').dim_of().data() * 1e3
        #time = tree.getNode(f'{root}.PHOTOCURRENT').dim_of().data() * 1e3

        self.data = np.array(data)
        self.label = np.array(label)
        self.R = np.array(R)
        self.Phi = np.array(Phi)
        self.Z = np.array(Z)
        self.b = np.array(b)
        self.Ohm = np.array(Ohm)
        self.time = time


    def plot(self):
        cmap = plt.cm.turbo(np.linspace(0,1,20))

        fig,axs = plt.subplots(1,1,figsize=(12,8))
        for j in range(20):
            axs.plot(self.time,self.data[j], label=f"Ch {j+1}", color=cmap[j])

        axs.grid()
        axs.legend(loc=3, fontsize=6)
        axs.set_xlabel("time [ms]")
        axs.set_ylabel("axuv diode [arb]")

        fig.suptitle(self.shot)


class ECH(WhamDiagnostic):

    def __init__(self, shot,
                       downsample_rate = 40,
                       fft = True,
                       median = True,
                       mean = True,
            ):

        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)

        source = "ech.ech_proc"
        self.Fwg_filt = tree.getNode(f"{source}.wg_monitor_f.filtered").getData().data()
        self.Rwg_filt = tree.getNode(f"{source}.wg_monitor_r.filtered").getData().data()
        self.Vs_filt =  tree.getNode(f"{source}.ves_monitor.filtered").getData().data()
        self.time = tree.getNode(f"{source}.wg_monitor_f.filtered").dim_of().data() * 1e3 # ms

        # Read gyrotron parameters from MDSplus
        params = "ech.ech_params"
        self.cryo_I = tree.getNode(f"{params}.cryomag_I").getData().data()
        self.fil_I = tree.getNode(f"{params}.filament_I").getData().data()
        self.gun_I = tree.getNode(f"{params}.gun_coil_I").getData().data()
        self.HVPS_V = tree.getNode(f"{params}.HVPS_V").getData().data()
        self.ech_target = tree.getNode(f"{params}.ech_target").getData().data()
        self.pol_ang_1 = tree.getNode(f"{params}.pol_ang_1").getData().data()
        self.pol_ang_2 = tree.getNode(f"{params}.pol_ang_2").getData().data()

    def plot(self):
        fig,axs = plt.subplots(3,1, sharex=True, figsize=(8,6))
        axs[0].plot(self.time, self.Fwg_filt, label="Waveguide Forward Power [kW]")
        axs[1].plot(self.time, self.Rwg_filt, label="Waveguide Reflected Power [arb]")
        axs[2].plot(self.time, self.Vs_filt, label="Vessel Stray Power [arb]")

        for a in axs:
            a.grid()
            a.legend(loc=1)
        axs[0].set_title(f"HVPS {self.HVPS_V} V")
        axs[-1].set_xlabel("time [ms]")
        fig.suptitle(self.shot)

        plt.show()

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            T = self.time
            P = self.Fwg_filt
            t1,t2 = get_time_array(T, [2.5,4.5])
            Pech = np.mean(P[t1:t2])
                           
            summary['P ECH (kW)'] = Pech
            # todo: get pulse duration

            result['summary'] = summary

            if Pech < 10:
                # correct for blank data
                retult['is_loaded'] = False

        return result


class EdgeProbes(WhamDiagnostic):

    def __init__(self, shot, R1=270e3,
                             R2=2.7e3,
                             ):

        self.V_factor = (R1+R2)/R2
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        source = "diag.probe_ring"

        self.ProbeArr = [ tree.getNode(f"{source}.P{j*30:03d}.voltage.signal").getData().data() for j in range(12) ]
        self.time = tree.getNode(f"{source}.P000.voltage.signal").dim_of().data() * 1e3 # ms

    def plot(self):
        fig,axs = plt.subplots(1,1,figsize=(13,10),sharex=True)
        for j in range(12):
            axs.plot(self.time, self.ProbeArr[j], label=f"Probe {j}")

        axs.grid()
        axs.legend()
        axs.set_xlabel("time [ms]")
        axs.set_xlim(-5,35)
        fig.suptitle(self.shot)

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            T = self.time
            V = self.ProbeArr[6]

            t1,t2 = get_time_array(T, [2,9])
            Vf = np.mean(V[t1:t2])
                           
            summary['V float (V)'] = Vf

            if np.abs(Vf) < 1:
                # correct for blank data
                retult['is_loaded'] = False
            result['summary'] = summary

        return result

class NBI(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        self.d_arr = np.array([tree.getNode(f"diag.shinethru.detector_{j+1:02d}").getData().data() for j in range(15)]) * 1e3
        d2 = tree.getNode("diag.shinethru.detector_02").getData().data() 
        d5 = tree.getNode("diag.shinethru.detector_05").getData().data() 
        d10 = tree.getNode("diag.shinethru.detector_10").getData().data() 
        time = tree.getNode("diag.shinethru.time").getData().data() * 1e3
        
        I_Beam = tree.getNode("nbi.i_beam").getData().data() 
        V_Beam = tree.getNode("nbi.v_beam").getData().data() / 1e3
        
        t_nbi = tree.getNode("nbi.time_slow").getData().data() * 1e3

        self.d2 = d2
        self.d5 = d5
        self.d10 = d10
        self.I_Beam = I_Beam
        self.V_Beam = V_Beam
        self.time = time

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            T = self.time
            I = self.I_Beam
            V = self.V_Beam
            P = I*V

            t1,t2 = get_time_array(T, [4,12])
            Pnbi = np.mean(P[t1:t2])
            Vnbi = np.mean(V[t1:t2])
            Inbi = np.mean(I[t1:t2])
                           
            summary['P NBI (kW)'] = Pnbi
            summary['V NBI (kW)'] = Vnbi
            summary['I NBI (kW)'] = Inbi
            # todo: get pulse duration

            if Pnbi < 50:
                # correct for blank data
                retult['is_loaded'] = False
            result['summary'] = summary

        return result

class Radiation(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)

        try:
            soft_xray = tree.getNode("diag.soft_xray.spectrum").getData().data()
            E_soft = tree.getNode("diag.soft_xray.spectrum").dim_of().data() # 8192 points, approximately 0 to 80 keV
        except:
            print("issue loading soft xray")
    
        try:
            # Kunal's plastic scintillator in lead house
            neutron = tree.getNode("diag.neutrondet.calib_signal").getData().data()
            t_neutron = tree.getNode("diag.neutrondet.calib_signal").dim_of().data()
        except:
            print("issue loading neutrons")
    
    
        try:
            # sometime before 0924 hard xray switched from ch 03 to ch 02
            hard_xray = tree.getNode("raw.mason_ds1000.ch_02.signal").getData().data()
            t_hard = tree.getNode("raw.mason_ds1000.ch_02.signal").dim_of().data() 
            h_freq = tree.getNode("raw.mason_ds1000.ch_02.freq").getData().data() # NOT included in dim_of (!)
            h_offset = tree.getNode("raw.mason_ds1000.ch_02.offset").getData().data() # already included in signal
            h_delay = tree.getNode("raw.mason_ds1000.ch_02.delay").getData().data()
            h_trig = tree.getNode("raw.mason_ds1000.trig_time").getData().data()

            time = (t_hard / h_freq) + h_trig + h_delay
            self.t_hard = time * 1e3

            offset = np.mean(hard_xray[:10000])
            self.hard_xray = hard_xray  - offset

        except:
            print("issue loading hard xray")

    def plot(self, fig=None, axs=None):

        s = self

        # the data is in pulse height volts.
        # approximately 100 keV / 100 mV.
        axs.plot(s.t_hard, s.hard_xray * 1e3, label=f"shot {s.shot}")

        axs.legend()
        axs.set_title("hard xray")

        axs.set_ylabel("Pulse Height [keV]")
        axs.set_xlabel("time [ms]")

        #fig.suptitle(s.shot)
        fig.tight_layout()


class Bolometer(WhamDiagnostic):
    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        node = "diag.cu_bolom"
        self.data = np.array([tree.getNode(f"{node}.ch_{j+1:02d}").getData().data()  for j in range(7)])
        self.time = tree.getNode(f"{node}.time").getData().data()

    def plot(self):

        fig,axs = plt.subplots(7,1,figsize=(8,9), sharex=True)
        for j in range(7):
            data = self.data[j]
            dT = np.max(data) - np.min(data)
            axs[j].plot(self.time, data, label=rf"$\Delta T$ = {dT:.2f}")
            axs[j].axhline(np.min(data), ls='--', color='C1', lw=0.5)
            axs[j].axhline(np.max(data), ls='--', color='C1', lw=0.5)
            axs[j].set_ylabel(f"Ch {j+1}")

        for a in axs:
            a.grid()
            a.legend()

        axs[-1].set_xlabel("time [ms]")

        fig.suptitle(self.shot)

    def plotQ(self, axs, N=6, # N channels
              vessel_D = 60, # cm, diameter
              vessel_L = 150, # cm, length
              foil_area = 3.9, # cm2
              foil_mass = 0.36, # g
              specific_heat = 0.385, # J/g/C
            ):

        arr = []
        for j in np.arange(N):
            data = self.data[j+1] # skip ch 0
            T = np.max(data) - np.min(data)
            arr.append(T)

        dT = np.array(arr)

        dE = foil_mass*specific_heat*dT
        Q = dE / foil_area *1e3 #mJ/cm2

        surface = np.pi*vessel_D*vessel_L #cm2
        E = Q*surface / 1e6 # kJ

        rax = [-50, -30, -10, 10, 30, 50]
        #rax = np.arange(N)+1
        #axs.plot(rax,Q,'o-',label=self.shot)
        axs.plot(rax,Q,'o-',label=f"{np.sum(E):.1f} kJ")
        axs.set_xlabel("Z [cm]")
        axs.set_ylabel("Fast Neutral Flux [mJ/cm2]")

        axs.grid()
        axs.set_title(self.shot)
        axs.legend()

    def plotCombo(self, N=6, # N channels
              vessel_D = 60, # cm, diameter
              vessel_L = 150, # cm, length
              foil_area = 3.9, # cm2
              foil_mass = 0.36, # g
              specific_heat = 0.385, # J/g/C
            ):

        fig = plt.figure(figsize=(12,7))
        gs = GridSpec(N, 2, figure=fig)

        ax0 = fig.add_subplot(gs[:,1])
        axs = [ fig.add_subplot(gs[k,0]) for k in range(N) ]

        arr = []
        for j in range(N):
            data = self.data[j+1] # skip ch 0
            dT = np.max(data) - np.min(data)
            axs[j].plot(self.time, data, label=rf"$\Delta T$ = {dT:.2f}")
            axs[j].axhline(np.min(data), ls='--', color='C1', lw=0.5)
            axs[j].axhline(np.max(data), ls='--', color='C1', lw=0.5)
            axs[j].set_ylabel(f"Ch {j+1}")
            arr.append(dT)

        dT = np.array(arr)

        dE = foil_mass*specific_heat*dT
        Q = dE / foil_area *1e3 #mJ/cm2

        surface = np.pi*vessel_D*vessel_L #cm2
        E = Q*surface / 1e6 # kJ

        rax = np.arange(N)+1
        ax0.plot(rax,Q,'C1o-', label=f"Average CX Loss: {np.mean(E):.2f} kJ")
        ax0.set_title(r"Neutral Heat Flux [mJ / cm$^2$]")
        ax0.set_xlabel("Ch #")
        ax0.grid()
        ax0.legend()

        for a in axs:
            a.legend()
            #a.sharex(axs[-1])
            a.minorticks_on()
            a.grid(which='both')
            a.grid(which='minor', linestyle=":", linewidth=0.5)

        for a in axs[:-1]:
            a.xaxis.set_ticklabels([])

        axs[-1].set_xlabel("time [ms]")

        fig.suptitle(self.shot)
        plt.show()

class adhocGas(WhamDiagnostic):
    # temp for week of 25-0220
    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        raw = "raw.gas_scope"
        
        data = [tree.getNode(f"{raw}.ch_{j+1:02d}.signal").getData().data() for j in range(4)]
        freq = [tree.getNode(f"{raw}.ch_{j+1:02d}.freq").getData().data() for j in range(4)]
        delay = [tree.getNode(f"{raw}.ch_{j+1:02d}.delay").getData().data() for j in range(4)]
        offset = [tree.getNode(f"{raw}.ch_{j+1:02d}.offset").getData().data() for j in range(4)]
        scale = [tree.getNode(f"{raw}.ch_{j+1:02d}.scale").getData().data() for j in range(4)]
        
        N = len(data[0])
        trig_time = tree.getNode(f"fueling.trig_times.main").getData().data() * 1e3 # ms
        dt = 1/freq[0] *1e3 # ms
        scope_delay = delay[0] * 1e3 # ms
        time = np.arange(N)*dt + trig_time + scope_delay
        
        ### need to process the data
        def adjust(data):
            offset = np.mean(data[-1000:])
            data -= offset
        
            scale = np.max(data)
            data /= scale
            return data

        self.time = time
        self.trig = adjust(data[0])
        self.ring = adjust(data[1])
        self.nec = adjust(data[2])
        self.sec = adjust(data[3])

    def plot(self):
        plt.figure()
        s = self
        plt.plot(s.time, s.trig, label="Ch 1 Trig")
        plt.plot(s.time, s.ring, label="Ch 2 Ring")
        plt.plot(s.time, s.nec, label="Ch 3 North CC")
        plt.plot(s.time, s.sec, label="Ch 4 South CC")
        plt.grid()
        plt.legend()
        plt.title(self.shot)

class Gas(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)

        gasDmd = tree.getNode("fueling.cmd_wvfrms.main").getData().data() 
        mainDmd = tree.getNode("fueling.cmd_wvfrms.main").getData().data() 
        baffleDmd = tree.getNode("fueling.cmd_wvfrms.baffle").getData().data() 
        necDmd = tree.getNode("fueling.cmd_wvfrms.nec").getData().data() 
        secDmd = tree.getNode("fueling.cmd_wvfrms.sec").getData().data() 
        secondaryDmd = tree.getNode("fueling.cmd_wvfrms.secondary").getData().data() 

        gasDmd = mainDmd
        t_puff = tree.getNode("fueling.cmd_wvfrms.main").dim_of().data()
        t_nec = tree.getNode("fueling.cmd_wvfrms.nec").dim_of().data() 

        try:
            asdex1 = tree.getNode("fueling.oscar_gauge.x10.signal").getData().data()
            asdex2 = tree.getNode("fueling.oscar_gauge.x1.signal").getData().data()
            t_asdex = tree.getNode("fueling.oscar_gauge.x1.signal").dim_of().data()
        except:
            print("oscar gauge not found, trying asdex gauge")
            asdex1 = tree.getNode("fueling.oscar_gauge.10x.signal").getData().data()
            asdex2 = tree.getNode("fueling.oscar_gauge.1x.signal").getData().data()
            t_asdex = tree.getNode("fueling.asdex_gauge.1x.signal").dim_of().data()

        trig = tree.getNode("raw.diag_rp_01.trig_time").getData().data()

        self.t_puff = t_puff * 1e3 # ms
        self.t_asdex = t_asdex * 1e3 # ms
        self.t_nec = t_nec * 1e3 # ms

        self.asdex_lo = asdex1 # Torr
        self.asdex_hi = asdex2
        self.gasDmd = gasDmd
        self.mainDmd = mainDmd
        self.baffleDmd = baffleDmd
        self.necDmd = necDmd
        self.secDmd = secDmd
        self.secondaryDmd = secondaryDmd


    def plot(self, fig=None, axs=None):

        s = self

        if fig == None:
            fig,axs = plt.subplots(3,1)
            for a in axs:
                a.grid()

        axs[0].plot(s.t_puff, s.gasDmd)
        #axs[1].plot(s.t_asdex, s.asdex_hi)
        axs[1].plot(s.t_asdex, s.asdex_lo *1e3, label=f"shot {s.shot}")
        axs[1].plot(s.t_asdex, s.asdex_hi *1e3)
        axs[2].plot(s.t_redIon, s.redIon)

        axs[1].legend()
        axs[0].set_title("gas puff demand")
        axs[1].set_title("asdex gauge")
        axs[2].set_title("shielded 'red' ion gauge")

        axs[0].set_ylabel("arb")
        axs[1].set_ylabel("mTorr")
        axs[2].set_ylabel("Torr")

        #fig.suptitle(s.shot)
        fig.tight_layout()

class ShineThrough(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        path = "diag.shinethru.linedens"

        self.nt = tree.getNode(f"{path}.central_dens").getData().data()
        self.nr = tree.getNode(f"{path}.density_prof").getData().data().T # [time, radius]
        self.time = tree.getNode(f"{path}.central_dens").dim_of().data() * 1e3 # ms
        self.radius = tree.getNode(f"{path}.density_prof").dim_of(0).data() * 100 # cm
        
        self.chord_radius = tree.getNode(f"{path}.detector_pos").getData().data() * 100 # cm

    def plot(self, t_array=[6]):

        fig, axs = plt.subplots(1,1)
        arr = get_time_array(self.time, t_array)

        for k,t in enumerate(t_array):
            axs.plot(self.radius, self.nr[arr[k]], 'o-', label=f"{t} ms")

        axs.grid()
        axs.legend()
        axs.set_title(self.shot)
        axs.set_xlabel("radius [cm]")
        axs.set_ylabel("density [m^-3]")

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            # smooth
            T = self.time
            N = self.nt
            jm = np.argmax(N)
                           
            t1,t2,t3 = get_time_array(T, [4.8, 9.8, 14.8])
            summary['max shinethrough dens (m^-3)'] = N[jm]
            summary['max time (ms)'] = T[jm]
            summary['dens 4.8ms (m^-3)'] = N[t1]
            summary['dens 9.8ms (m^-3)'] = N[t2]
            summary['dens 14.8ms (m^-3)'] = N[t3]

            result['summary'] = summary
        return result

class IonProbe(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        tree = mds.Tree("wham",self.shot)
        path = "diag.ion_probe"
        
        self.Icol = tree.getNode(f"{path}.I_col").getData().data() 
        self.time = tree.getNode(f"{path}.I_col").dim_of().data() * 1e3 # ms

class EndRing(WhamDiagnostic):

    def __init__(self, shot):
        super().__init__(shot)

    def load(self):

        # set up tree
        tree = mds.Tree("wham",self.shot)
        source = "bias.end_rings"

        ProbeArr = np.array([tree.getNode(f"{source}.N{j}.voltage.signal").getData().data() for j in range(10)])
        SmoothArr = np.array([tree.getNode(f"{source}.N{j}.voltage.filtered").getData().data() for j in range(10)])
        time = tree.getNode(f"{source}.N0.voltage.signal").dim_of().data() * 1e3 # ms

        # temp
        #SmoothArr[0] = bias.RVs
        #ProbeArr[0] = bias.R_VLem

        # use bottom radii for rings 1-9, and middle for disk 0
        rax = np.array([4.0,8.0,11.1,13.4,15.3,16.9,18.3,19.5,20.6,21.6]) # end cell plane

        self.radii = rax
        self.mid = (rax[1:] + rax[:-1])/2
        self.ProbeArr = ProbeArr
        self.SmoothArr = SmoothArr
        self.time = time # ms

    def calc_rotation(self,t, B0=0.08):
        '''
        calculate the rotation profile 
        - for given time t [ms]
        - assuming B [T]
        '''

        def get_time_index(t):
            # get the index at time t, ms
            j = np.argmin(np.abs(self.time - t))
            return j

        j = get_time_index(t)
        V = self.SmoothArr[:,j]
        rax = self.radii

        Er = -np.diff(V)/np.diff(rax)  # V / cm
        vphi = Er*100 / B0 / 1e3   # km/s
        omega = vphi / rax[:-1] * 100  # rad / ms

        self.V_ring = V
        self.Vphi = vphi
        self.Er = Er
        self.Omega = omega

    def plot_rotation(self,
                      time_slices = [5,8,15], # array, ms
            ):

        # make plots
        fig = plt.figure(figsize=(15,8))
        gs = GridSpec(5,3, figure=fig)

        ax0 = fig.add_subplot(gs[:2,:])
        ax1 = fig.add_subplot(gs[2:,0])
        ax2 = fig.add_subplot(gs[2:,1])
        ax3 = fig.add_subplot(gs[2:,2])
        axs = np.array([ax1,ax2,ax3])

        def get_time_index(t):
            # get the index at time t, ms
            j = np.argmin(np.abs(self.time - t))
            return j

        # plot time trace
        N_ring = len(self.ProbeArr)
        cjet = plt.cm.jet(np.linspace(0,1,10))
        for j in range(N_ring):
            ax0.plot(self.time, self.ProbeArr[j], color=cjet[j], lw=0.5)
            if j>0:
                ax0.plot(self.time, self.SmoothArr[j], color=cjet[j], label=f"Ring {j}")
            else:
                ax0.plot(self.time, self.SmoothArr[j], color=cjet[j], label=f"Disk 0")

        ax0.set_xlabel("time [ms]")
        ax0.set_title("floating potential [V]", fontsize=12)
        ax0.set_xlim(-1,32)
        ax0.set_ylim(1.05 * np.min(self.ProbeArr), np.max(self.ProbeArr)*1.5)
        ax0.grid()
        ax0.legend(loc=1, fontsize=7)

        # plot time slices
        rax = self.radii
        for i,t in enumerate(time_slices):

            self.calc_rotation(t)
            ax0.axvline(t,color=f'C{i}', ls='--')

            ax1.plot(rax,self.V_ring,'o-',label=f"t = {t} ms")
            ax2.plot(rax[:-1],self.Er,'o-')
            ax3.plot(rax[:-1],self.Omega,'o-')

        ax1.set_title("Floating Potential [V]", fontsize=12)
        ax2.set_title("Radial Electric Field [V/cm]", fontsize=12)
        ax3.set_title("ExB Rotation [rad/ms]", fontsize=12)
        ax1.legend()

        for a in axs:
            a.grid()
            a.set_xlabel("Ring Middle Radius [cm]")

        fig.suptitle(self.shot)
        fig.tight_layout()

        return fig,axs, ax0


class Dalpha(WhamDiagnostic):

    def __init__(self,shot):
        super().__init__(shot)

    def load(self):
        tree = mds.Tree("wham",self.shot)

        # load data from nodes
        root = "diag.midplane_da" 
        data = np.array([tree.getNode(f"{root}.los_{j+1:02d}").getData().data() for j in range(8)])
        time = tree.getNode(f"{root}.los_01").dim_of().data() * 1e3 #ms
        radius = tree.getNode(f"{root}.impact_param").getData().data() / 10 # cm

        # filter non-physical peaks from D-alpha
        arg = np.argwhere( np.abs(data) > 1e20)
        data.T[arg] = 0

        # save
        self.data = data[:-2] # ignore last two channels for now
        self.radius = radius[:-2]
        self.time = time

    def plot(self):

        fig,axs = plt.subplots(1,1)
        axs.contourf(self.time, self.radius, self.data)
        axs.set_title(self.shot)
        axs.set_ylabel("radius [cm]")
        axs.set_xlabel("time [ms]")

class Bdot(WhamDiagnostic):

    def __init__(self,shot):
        super().__init__(shot)

    def load(self,
        source = "raw.acq2206_043",
           ):

        tree = mds.Tree("wham",self.shot)
        ch_arr = [1,2,3,4]
        node = [tree.getNode(f"{source}.ch_{j:02d}") for j in ch_arr]
        data = [n.getData().data() for n in node]
        
        time = node[0].dim_of().data() * 1e3
        diff = -(data[0] + data[3])/2 * 1e3 # mV

        self.data = data
        self.Br = diff
        self.time = time

    def to_dict(self, detail_level='summary'):

        # Always include status information
        result = {
            "is_loaded": self.is_loaded,
        }

        if detail_level == 'summary':

            summary = {}
            
            T = self.time
            t1,t2 = get_time_array(T, [2,9])
            dB = np.mean(self.Br[t1:t2])
                           
            if np.abs(dB) < 1:
                # correct for blank data
                retult['is_loaded'] = False
            result['summary'] = summary

        return result
