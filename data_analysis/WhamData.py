import MDSplus as mds

import numpy as np
from scipy.signal import savgol_filter as savgol
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as pl

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

# Helper Functions

def zero_offset(f,idx=2000):
    f -= np.mean(f[-idx:])

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

### End Helper

class BiasPPS:

    def __init__(self, shot,
                       ILEM_gain=-500,
                       VLEM_gain=800,
                       VFBK_gain=-1):

        self.shot = shot
        if shot < 240728011:
            # this is when I fixed the double (-) sign
            VLEM_gain *= -1

        self.ILEM_gain = ILEM_gain
        self.VLEM_gain = VLEM_gain
        self.VFBK_gain = VFBK_gain

        self.load()

    def load(self, t_max=100 #ms
            ):

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

class Interferometer:

    def __init__(self, shot):

        self.shot = shot
        self.load()

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


class FluxLoop:

    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self):

        tree = mds.Tree("wham",self.shot)

        # convert from Weber to Maxwell
        FL1 = tree.getNode("diag.fluxloops.fl1").getData().data() * 1e8
        FL2 = tree.getNode("diag.fluxloops.fl2").getData().data() * 1e8
        FL3 = tree.getNode("diag.fluxloops.fl3").getData().data() * 1e8

        # the sign on flux loop 3 is questionable, so we integrate
        sign = np.sign( np.sum(FL3) )
        FL3 *= sign

        #FL3 *= -1 # used for 0721, but not 0729
        #if self.shot > 240809000 and self.shot < 240830000:
        #    FL3 *= -1

        # convert to ms
        #time = tree.getNode("diag.fluxloops.time").getData().data() * 1e3
        #time = (time+5) * (2./3) - 8 # ad hoc, for shots 0728, not perfect
        #time = (time-6)/2 # ad hoc for shots 0721
        #time = (time+2)*2 # ad hoc for shots 0729045+

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

class AXUV:

    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self):
        tree = mds.Tree("wham",self.shot)

        data = []
        R = []
        Phi = []
        Z = []
        b = []
        Ohm = []
        label = []
        for j in range(20):
            root = f"diag.axuv.DIODEARRAY1.CH_{j+1:02d}"
            data.append(tree.getNode(f'{root}.PHOTOCURRENT').getData().data())
            label.append(tree.getNode(f"{root}.DIODE_NUM").getData().data())

            R.append(tree.getNode(f'{root}.R').getData().data())
            Phi.append( tree.getNode(f'{root}.PHI').getData().data() )
            Z.append( tree.getNode(f'{root}.Z').getData().data() )
            b.append( tree.getNode(f'{root}.B_IMPACT').getData().data())
            Ohm.append( tree.getNode(f'{root}.RESISTOR').getData().data())

        time = tree.getNode(f'{root}.PHOTOCURRENT').dim_of().data() * 1e3

        self.data = np.array(data)
        self.label = np.array(label)
        self.R = np.array(R)
        self.Phi = np.array(Phi)
        self.Z = np.array(Z)
        self.b = np.array(b)
        self.Ohm = np.array(Ohm)
        self.time = time


    def plot(self):
        cmap = pl.cm.turbo(np.linspace(0,1,20))

        fig,axs = plt.subplots(1,1,figsize=(12,8))
        for j in range(20):
            axs.plot(self.time,self.data[j], label=f"Ch {j+1}", color=cmap[j])

        axs.grid()
        axs.legend(loc=3, fontsize=6)
        axs.set_xlabel("time [ms]")
        axs.set_ylabel("axuv diode [arb]")

        fig.suptitle(self.shot)


class ECH:

    def __init__(self, shot,
                       downsample_rate = 40,
                       fft = True,
                       median = True,
                       mean = True,
            ):

        self.shot = shot
        self.load()

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


class EdgeProbes:

    def __init__(self, shot, R1=270e3,
                             R2=2.7e3,
                             ):

        self.shot = shot
        self.V_factor = (R1+R2)/R2
        self.load()

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

class NBI:

    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self):

        tree = mds.Tree("wham",self.shot)
        self.d_arr = np.array([tree.getNode(f"diag.shinethru.detector_{j+1:02d}").getData().data() for j in range(14)]) * 1e3
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

class Radiation:

    def __init__(self, shot):

        self.shot = shot
        self.load()

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

class Bolometer:
    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self):

        tree = mds.Tree("wham",self.shot)
        node = "diag.cu_bolom"
        self.data = np.array([tree.getNode(f"{node}.ch_{j+1:02d}").getData().data()  for j in range(7)])
        self.time = tree.getNode(f"{node}.time").getData().data()

    def plot(self):

        fig,axs = plt.subplots(7,1,figsize=(8,9), sharex=True)
        for j in range(7):
            data = bolo.data[j]
            dT = np.max(data) - np.min(data)
            axs[j].plot(bolo.time, data, label=rf"$\Delta T$ = {dT:.2f}")
            axs[j].axhline(np.min(data), ls='--', color='C1', lw=0.5)
            axs[j].axhline(np.max(data), ls='--', color='C1', lw=0.5)
            axs[j].set_ylabel(f"Ch {j+1}")

        for a in axs:
            a.grid()
            a.legend()

        axs[-1].set_xlabel("time [ms]")

        fig.suptitle(self.shot)
        plt.show()

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

        rax = np.arange(N)+1
        axs.plot(rax,Q,'o-',label=self.shot)

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

class Gas:
    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self):

        gasDmd = tree.getNode("fueling.cmd_wvfrms.main").getData().data() 
        mainDmd = tree.getNode("fueling.cmd_wvfrms.main").getData().data() 
        baffleDmd = tree.getNode("fueling.cmd_wvfrms.baffle").getData().data() 
        necDmd = tree.getNode("fueling.cmd_wvfrms.nec").getData().data() 
        secDmd = tree.getNode("fueling.cmd_wvfrms.sec").getData().data() 
        secondaryDmd = tree.getNode("fueling.cmd_wvfrms.secondary").getData().data() 

        gasDmd = mainDmd
        t_puff = tree.getNode("fueling.cmd_wvfrms.main").dim_of().data()
        t_nec = tree.getNode("fueling.cmd_wvfrms.nec").dim_of().data() 
        t_asdex = tree.getNode("fueling.asdex_gauge.1x.signal").dim_of().data()
        t_redIon = tree.getNode("raw.diag_rp_01.rp_04.ch_01").dim_of().data()

        asdex1 = tree.getNode("fueling.asdex_gauge.10x.signal").getData().data()
        asdex2 = tree.getNode("fueling.asdex_gauge.1x.signal").getData().data()
        redIon = tree.getNode("raw.diag_rp_01.rp_04.ch_01").getData().data()

        freq2 = tree.getNode("raw.diag_rp_01.rp_02.freq").getData().data()
        freq4 = tree.getNode("raw.diag_rp_01.rp_04.freq").getData().data()
        trig = tree.getNode("raw.diag_rp_01.trig_time").getData().data()

        self.t_redIon = (t_redIon / freq4 + trig) * 1e3
        self.t_puff = t_puff * 1e3 # ms
        self.t_asdex = t_asdex * 1e3 # ms
        self.t_nec = t_nec * 1e3 # ms

        self.asdex_lo = asdex1 # Torr
        self.asdex_hi = asdex2
        self.redIon = redIon * -2.5e-4 # Torr
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


class EndRing:

    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self, R1=270e3, # upper voltage divider resistor, Ohm
                   R2=2.7e3, # lower voltage divider resistor, Ohm
                   win = 101, # savgol window
                   pol = 3,   # savgol polynomial
                   ):

        # set up tree
        tree = mds.Tree("wham",self.shot)
        source = "raw.acq196_370"

        try:
            # used for voltage
            bias = BiasPPS(self.shot)
        except:
            print(f"Issue with Bias {self.shot}")
        # get source nodes, these are uploaded by the DTACQ
        if self.shot < 241218000:
            # this is the setting used for APS Sept 2024
            idx = np.arange(20,30)[::-1] + 1 # channels 21-30
            dtacq_arr = [ tree.getNode(f"{source}.ch_{j+1:02d}") for j in idx ]
        elif self.shot >= 241218000:
            idx = np.arange(21,31)
            dtacq_arr = [ tree.getNode(f"{source}.ch_{j:02d}") for j in idx ]

        t_delay = tree.getNode(f"{source}.trig_time").getData().data()
        dtacq_time = dtacq_arr[0].getData().dim_of().data()
        time = (dtacq_time + t_delay) * 1e3

        # Get data
        V_factor = (R1+R2)/R2

        # zero offset
        def zero_offset(f,idx=1000):
            f -= np.mean(f[:idx])

        # smooth signals
        ProbeArr = []
        SmoothArr = []
        N_ring = 10
        for j in range(N_ring):
            Vf = dtacq_arr[j].getData().data() * V_factor
            zero_offset(Vf)
            Vs = savgol(Vf,win,pol)

            ProbeArr.append(Vf)
            SmoothArr.append(Vs)
        ProbeArr = np.array(ProbeArr)
        SmoothArr = np.array(SmoothArr)

        # temp
        SmoothArr[0] = bias.RVs
        ProbeArr[0] = bias.R_VLem

        rax = np.array([4.0,8.0,11.1,13.4,15.3,16.9,18.3,19.5,20.6,21.6]) # use bottom radii for rings 2-10, and middle for disk 1

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
        for j in range(N_ring):
            ax0.plot(self.time, self.ProbeArr[j], f"C{j}", lw=0.5)
            ax0.plot(self.time, self.SmoothArr[j], f"C{j}", label=f"Ring {j+1}")

        ax0.set_xlabel("time [ms]")
        ax0.set_title("floating potential [V]", fontsize=12)
        ax0.set_xlim(-1,20)
        ax0.set_ylim(1.05 * np.min(self.ProbeArr), np.max(self.ProbeArr)*1.2)
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
