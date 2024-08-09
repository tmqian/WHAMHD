import MDSplus as mds
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol

'''
This colletion of classes loads data from WHAM MDS+ tree

classes:
    BiasPPS
    Interferometer
    FluxLoop
    AXUV
    ECH
    Edge Probes

8 August 2024
'''

class BiasPPS:

    def __init__(self, shot,
                       ILEM_gain=-500,
                       VLEM_gain=800,
                       VFBK_gain=-1):

        self.shot = shot
        if shot < 240728011:
            # this is when I fixed the double (-) sign
            VLEM_gain *= -1
            #ILEM_gain *= -1
        #else:
        #    ILEM_gain *= -1

        self.ILEM_gain = ILEM_gain
        self.VLEM_gain = VLEM_gain
        self.VFBK_gain = VFBK_gain

        self.load()
        self.smooth()

    def load(self, t_max=100 #ms
            ):

        # set up tree
        tree = mds.Tree("wham",self.shot)

        # load data from nodes
        data = "bias.bias_raw"
        raw = "raw.acq196_370"
        raw_L_Dem = tree.getNode(f"{data}.ch01_l_dem").getData().data()
        raw_L_ILem = tree.getNode(f"{data}.ch02_l_ilem").getData().data()
        raw_L_VLem = tree.getNode(f"{data}.ch03_l_vlem").getData().data()
        raw_L_Vpps = tree.getNode(f"{data}.ch04_l_fdbk").getData().data()
        raw_R_Dem = tree.getNode(f"{data}.ch05_r_dem").getData().data()
        raw_R_ILem = tree.getNode(f"{data}.ch06_r_ilem").getData().data()
        raw_R_VLem = tree.getNode(f"{data}.ch07_r_vlem").getData().data()
        raw_R_Vpps = tree.getNode(f"{data}.ch08_r_fdbk").getData().data()
        raw_L_VFB = tree.getNode(f"{raw}.ch_09").getData().data()
        raw_R_VFB = tree.getNode(f"{raw}.ch_10").getData().data()

        #t_delay = tree.getNode(f"{raw}.delay").getData().data() # ms, positive
        t_delay = tree.getNode(f"bias.bias_params.trig_time").getData().data() *-1e3 # ms, positive

        # calibrate data (this should be added to MDS+)

        L_Dem = raw_L_Dem
        R_Dem = raw_R_Dem
        
        L_ILem = raw_L_ILem * self.ILEM_gain
        R_ILem = raw_R_ILem * self.ILEM_gain
        
        L_VLem = raw_L_VLem * self.VLEM_gain
        R_VLem = raw_R_VLem * self.VLEM_gain
        L_Vpps = raw_L_Vpps * self.VLEM_gain
        R_Vpps = raw_R_Vpps * self.VLEM_gain
        
        L_VFB = raw_L_VFB * self.VFBK_gain 
        R_VFB = raw_R_VFB * self.VFBK_gain 

        # zero offset
        def zero_offset(f,idx=2000):
            f -= np.mean(f[-idx:])
        
        zero_offset(L_Dem)
        zero_offset(R_Dem)
        zero_offset(L_ILem)
        zero_offset(R_ILem)
        zero_offset(L_VLem)
        zero_offset(R_VLem)
        zero_offset(L_Vpps)
        zero_offset(R_Vpps)
        zero_offset(L_VFB)
        zero_offset(R_VFB)

        # time
        N_points = len(L_Dem)
        time = np.linspace(0,t_max, N_points) - t_delay

        # save
        self.time = time

        self.raw_L_Dem  = raw_L_Dem
        self.raw_L_ILem = raw_L_ILem
        self.raw_L_VLem = raw_L_VLem
        self.raw_L_Vpps = raw_L_Vpps
        self.raw_R_Dem  = raw_R_Dem
        self.raw_R_ILem = raw_R_ILem
        self.raw_R_VLem = raw_R_VLem
        self.raw_R_Vpps = raw_R_Vpps
        self.raw_L_VFB  = raw_L_VFB 
        self.raw_R_VFB  = raw_R_VFB

        self.L_Dem  = L_Dem
        self.L_ILem = L_ILem
        self.L_VLem = L_VLem
        self.L_Vpps = L_Vpps
        self.R_Dem  = R_Dem
        self.R_ILem = R_ILem
        self.R_VLem = R_VLem
        self.R_Vpps = R_Vpps
        self.L_VFB  = L_VFB 
        self.R_VFB  = R_VFB

    def smooth(self, win=51, poly=3):

        self.LVs = savgol(self.L_VLem,win,poly)
        self.LIs = savgol(self.L_ILem,win,poly)
        self.RVs = savgol(self.R_VLem,win,poly)
        self.RIs = savgol(self.R_ILem,win,poly)

    def plot_raw(self):
        fig, axs = plt.subplots(2,1,figsize=(8,5))

        axs[0].plot(self.time, self.raw_L_Dem, lw=0.3, label="Limiter Demand")
        axs[0].plot(self.time, self.raw_L_Vpps, lw=0.3, label="Voltage PPS")
        axs[0].plot(self.time, self.raw_L_ILem, lw=0.3, label="Current LEM")
        axs[0].plot(self.time, self.raw_L_VLem, lw=0.3, label="Voltage LEM")
        axs[0].plot(self.time, self.raw_L_VFB , lw=0.3, label="Feedback Output")
        
        axs[1].plot(self.time, self.raw_R_Dem, lw=0.3, label="Ring Demand")
        axs[1].plot(self.time, self.raw_R_Vpps, lw=0.3, label="Voltage PPS")
        axs[1].plot(self.time, self.raw_R_ILem, lw=0.3, label="Current LEM")
        axs[1].plot(self.time, self.raw_R_VLem, lw=0.3, label="Voltage LEM")
        axs[1].plot(self.time, self.raw_R_VFB , lw=0.3, label="Feedback Output")
        
        axs[0].set_title(f"Raw Data: {self.shot}")
        axs[-1].set_xlabel('ms')
        axs[0].legend()
        axs[1].legend()
        
        for a in axs:
            a.grid()


    def plot_PPS(self):

        fig, axs = plt.subplots(3,2, figsize=(10,8), sharex=True)
        axs[0,0].plot(self.time, self.L_Dem , 'C0', lw=0.3, label="L Demand [V]")
        axs[0,0].plot(self.time, self.L_VFB , 'C4', lw=0.3, label="Feedback Output [-V]")
        axs[0,1].plot(self.time, self.R_VFB , 'C4', lw=0.3, label="Feedback Output [-V]")
        axs[0,1].plot(self.time, self.R_Dem , 'C0', lw=0.3, label="R Demand [V]")
        
        axs[1,0].plot(self.time, self.L_ILem, 'C1', lw=0.3, label="L Current [A]")
        axs[1,1].plot(self.time, self.R_ILem, 'C1', lw=0.3, label="R Current [A]")
        
        axs[2,0].plot(self.time, self.L_Vpps, 'C3', lw=0.3, label="L Voltage PPS [V]")
        axs[2,1].plot(self.time, self.R_Vpps, 'C3', lw=0.3, label="R Voltage PPS [V]")
        
        axs[2,0].plot(self.time, self.L_VLem, 'C2', lw=0.3, label="L Voltage [V]")
        axs[2,1].plot(self.time, self.R_VLem, 'C2', lw=0.3, label="R Voltage [V]")
        
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
        time = tree.getNode("diag.interferomtr.time").getData().data() * 1e3

        #time = time - 7.5 # ad hoc 0728
        #time = 1.1*time + 1.2 # ad hoc 0721
        #time = time - 6.5 # ad hoc 0730
        #linedens *= -1

        self.time = time
        self.linedens = linedens

    def plot(self):
        fig,axs = plt.subplots(1,1,figsize=(10,5))
        axs.plot(self.time,self.linedens,'C1', label=r"Line Density [m$^{-2}$]")

        axs.set_xlabel("time [ms]")
        axs.legend()
        axs.grid()
        axs.set_title(self.shot)


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

        #FL3 *= -1 # used for 0721, but not 0729

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

        fig,axs = plt.subplots(1,3,figsize=(10,5))
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
            a.set_xlim(-2,16)

        fig.suptitle(self.shot)



class AXUV:

    def __init__(self, shot):

        self.shot = shot
        self.load()

    def load(self):

        tree = mds.Tree("wham",self.shot)

        raw = "raw.acq1001_639"
        '''
        64 channels
        60 populated, 20 per diode

        53-62 ordered center to edge
        62 is the BOTTOM edge (z0_p10)
        '''

        self.data = [-tree.getNode(f"{raw}.ch_{j+1:02d}").getData().data() for j in range(64)]
        self.freq = tree.getNode(f"{raw}.frequency").getData().data() 
        self.trig = tree.getNode(f"{raw}.trig_time").getData().data() 
        self.decimation = tree.getNode(f"{raw}.decimation").getData().data() 
        self.nsamp = tree.getNode(f"{raw}.numsamples").getData().data() 
        time = tree.getNode(f"{raw}.ch_01").dim_of().data() * 1e3 # ms

        self.time = time

    def plot(self):
        fig,axs = plt.subplots(1,1,figsize=(12,8))
        for j in range(52,62):
            axs.plot(self.time,self.data[j], label=f"Ch {j+1}")

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

        # filter options
        self.fft = fft
        self.median = median
        self.mean = mean
        self.downsample_rate = downsample_rate
  
        import ech_plotting_helpers as f_ech
        self.func = f_ech

        self.old = False # legacy
        self.load()

    def load(self):

        tree = mds.Tree("wham",self.shot)

        kwargs = { "do_fft_filt" : self.fft,
                   "do_med_filt" : self.median,
                   "do_mean_filt" : self.mean,
                   "downsample_rate" : self.downsample_rate,
                   "old" : self.old }
        shot = self.shot

        # Gyrotron Voltage
        V_t,V_filt,V = self.func.get_ech_signal("gyrotron_v",shot, **kwargs)
        self.V_filt = V_filt / 1e3 # convert to kV
        self.V = V / 1e3 # convert to kV

        # Gyrotron Current
        It,I_filt,I = self.func.get_ech_signal("gyrotron_i",shot, **kwargs)
        # wave guide forward power
        Fwg_t,Fwg_filt,Fwg = self.func.get_ech_signal("WG_monitor_F",shot, **kwargs)
        # wave guide reflected power
        Rwg_t,Rwg_filt,Rwg = self.func.get_ech_signal("WG_monitor_R",shot, **kwargs)
        # wave guide bias voltage
        Vwg_t,Vwg_filt,Vwg = self.func.get_ech_signal("wgd_bias_v",shot, **kwargs)
        # dummy load forward power
        Fdl_t,Fdl_filt,Fdl = self.func.get_ech_signal("DL_monitor_F",shot, **kwargs)
        # vessel stray power
        Vs_t,Vs_filt,Vs = self.func.get_ech_signal("ves_monitor",shot, **kwargs)

        # save
        self.time = V_t
        self.t_bias = Vwg_t

        self.I_filt = I_filt
        self.Fwg_filt = Fwg_filt
        self.Rwg_filt = Rwg_filt
        self.Fdl_filt = Fdl_filt
        self.Vwg_filt = Vwg_filt
        self.Vs_filt = Vs_filt

        self.I = I
        self.Fwg = Fwg
        self.Rwg = Rwg
        self.Fdl = Fdl
        self.Vwg = Vwg
        self.Vs = Vs


        #Read gyrotron parameters from MDSplus
        self.cryo_I = tree.getNode("ech.ech_params.cryomag_I").getData().data()
        self.fil_I = tree.getNode("ech.ech_params.filament_I").getData().data()
        self.gun_I = tree.getNode("ech.ech_params.gun_coil_I").getData().data()
        self.HVPS_V = tree.getNode("ech.ech_params.HVPS_V").getData().data()
        self.ech_target = tree.getNode("ech.ech_params.ech_target").getData().data()
        self.pol_ang_1 = tree.getNode("ech.ech_params.pol_ang_1").getData().data()
        self.pol_ang_2 = tree.getNode("ech.ech_params.pol_ang_2").getData().data()
        self.mir_ang = tree.getNode("ech.ech_params.mirror_ang").getData().data()


    def plot(self):
        fig,axs = plt.subplots(3,1, sharex=True)
        axs[0].plot(self.time, self.Fwg_filt, label="Waveguide Forward Power [kW]")
        axs[1].plot(self.time, self.Rwg_filt, label="Waveguide Reflected Power [kW]")
        axs[2].plot(self.time, self.Vs_filt, label="Vessel Stray Power [kW]")

        
        for a in axs:
            a.grid()
            a.legend()
        axs[0].set_title(f"HVPS {self.HVPS_V} V")
        axs[-1].set_ylabel("time [ms]")
        fig.suptitle(self.shot)


class EdgeProbes:

    def __init__(self, shot, R1=108e3,
                             R2=2.7e3,
                             RP = False,
                             ):

        self.shot = shot
        self.V_factor = (R1+R2)/R2

        if RP:
            self.load_RP()
        else:
            self.load()

    def load(self):
        '''
        loads from dtacq
        should add alternative option for RedPitay and/or Rigol scope
        '''

        tree = mds.Tree("wham",self.shot)
        raw = "raw.acq196_370"

        ProbeArr = []
        for j in range(20,32):
            data = tree.getNode(f"{raw}.ch_{j+1:02d}").getData().data()
            Vf = data * self.V_factor
            #Vs = savgol(Vf,win,pol)
        
            ProbeArr.append(Vf)
            #SmoothArr.append(Vs)
        self.ProbeArr = np.array(ProbeArr)
        #SmoothArr = np.array(SmoothArr)

    def load_RP(self):
        '''
        loads from RedPitaya
        should add alternative option for RedPitay and/or Rigol scope
        '''

        tree = mds.Tree("wham",self.shot)

        def getScope(scope="TQ_SCOPE",ch=1):
            root = f"raw.{scope}"
            node = f"{root}.ch_{ch:02d}"
            trig = tree.getNode(f"{root}.trig_time").getData().data() # scalar offset [s]
            freq = tree.getNode(f"{node}.freq").getData().data()
            delay = tree.getNode(f"{node}.delay").getData().data()
            arr = tree.getNode(f"{node}.signal").dim_of().data()
            data = tree.getNode(f"{node}.signal").getData().data()

            time = (arr / freq + trig + delay) * 1e3 # convert to ms
            signal = data * self.V_factor
            return signal, time

        def getRP(rack=1,RP=1,ch=1):
            root = f"raw.diag_rp_{rack:02d}"
            node = f"{root}.rp_{RP:02d}"
            trig = tree.getNode(f"{root}.trig_time").getData().data() # scalar offset [s]
            freq = tree.getNode(f"{node}.freq").getData().data()
            arr = tree.getNode(f"{node}.ch_{ch:02d}").dim_of().data()
            data = tree.getNode(f"{node}.ch_{ch:02d}").getData().data()

            time = (arr / freq + trig) * 1e3 # convert to ms
            signal = data * self.V_factor
            return signal, time

        # for 0803
        P01, T01 = getRP(RP=7,ch=1)
        P02, T02 = getRP(RP=7,ch=2)
        P03, T03 = getRP(RP=5,ch=1)
        P04, T04 = getRP(RP=5,ch=2)
        P05, T05 = getRP(RP=3,ch=1)
        P06, T06 = getRP(RP=3,ch=2)
        P07, T07 = getRP(RP=8,ch=1)
        P08, T08 = getRP(RP=8,ch=2)
        P09, T09 = getScope(scope="TQ_SCOPE",ch=1)
        P10, T10 = getScope(scope="TQ_SCOPE",ch=2)
        P11, T11 = getScope(scope="TQ_SCOPE",ch=3)
        P12, T12 = getScope(scope="mason_scope",ch=4)
        # for 0730
        #P01, T01 = getRPdata(RP=7,ch=1)
        #P02, T02 = getRPdata(RP=7,ch=2)
        #P03, T03 = getRPdata(RP=5,ch=1)
        #P04, T04 = getRPdata(RP=5,ch=2)
        #P05, T05 = getRPdata(RP=8,ch=1)
        #P06, T06 = getRPdata(RP=8,ch=2)
        #P07, T07 = getRPdata(RP=3,ch=2)
        #P08, T08 = getRPdata(RP=1,ch=1)
        #P09, T09 = getRPdata(RP=2,ch=2)
        #P10, T10 = getRPdata(RP=3,ch=1)
        #P11 = np.zeros_like(P01)
        #P12 = np.zeros_like(P01)
        self.ProbeArr = [P01, P02, P03, P04, 
                                  P05, P06, P07, P08, 
                                  P09, P10, P11, P12]
        self.TimeArr = [T01, T02, T03, T04, 
                                 T05, T06, T07, T08, 
                                 T09, P10, T11, T12]

    def plot(self):
        fig,axs = plt.subplots(1,1,figsize=(13,10),sharex=True)
        for j in range(12):
            axs.plot(self.TimeArr[j], self.ProbeArr[j], label=f"Probe {j}")

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

