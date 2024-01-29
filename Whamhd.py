import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pylab import cm as colormap

from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, ifft, fftfreq, fftshift

# user def
path = "/Users/tqian/Documents/WHAM/Wham-Theory/simulation/gkeyll/2023-1005-gkyl-data"

# constants
mass_ratio = 1840
e = 1.6e-19
Mi = 1.67e-27


class SyntheticDiagnostic:

    def __init__(self):

        print("Hello World")

    def loadGkylPack(self,fin):
        '''
        This is a 'pack' because it loads packaged post-processed as opposed to raw Gkyl data.
        '''

        data = np.load(fin, allow_pickle=True)

        # these are 1D arrays
        self.t = data['t']
        self.x = data['x']
        self.y = data['y']
        self.z = data['z']

        # these are 4D arrays (t,z,y,x)
        self.Ne = data['ne']
        self.Te = data['Te']
        self.Ti = data['Ti']
        self.Phi = data['phi']

    def interpolate(self):
        '''
        The function interperloates data on a recular grid (T,Z,Y,X)
        '''
        
        grid = (self.t, self.z, self.y, self.x)
        f_Ne = RegularGridInterpolator(grid, self.Ne)
        f_Te = RegularGridInterpolator(grid, self.Te)
        f_Ti = RegularGridInterpolator(grid, self.Ti)
        f_Phi = RegularGridInterpolator(grid, self.Phi)
        
        probes = self.probes
        self.sample_Ne = np.array( [[ f_Ne( (t,*p) ) for t in self.t] for p in probes] )
        self.sample_Te = np.array( [[ f_Te( (t,*p) ) for t in self.t] for p in probes] )
        self.sample_Ti = np.array( [[ f_Ti( (t,*p) ) for t in self.t] for p in probes] )
        self.sample_Phi = np.array( [[ f_Phi( (t,*p) ) for t in self.t] for p in probes] )

        self.f_Ne  = f_Ne 
        self.f_Te  = f_Te
        self.f_Ti  = f_Ti
        self.f_Phi = f_Phi

    def initProbeCircle(self, z_idx, R=0.20,M=4):
        '''
        init a circle of M probes

        at Z with radius R
        '''

        Z = self.z[z_idx]
    
        tax = np.linspace(0,np.pi*2,M,endpoint=False)
        x = R*np.cos(tax)
        y = R*np.sin(tax)
        z = Z*np.ones(M)

        self.probes = np.transpose([z,y,x])
        self.M = M
        self.z_idx = z_idx
        self.R_probes = R

    def sampleProbes(self,area=1e-6):
        '''
        probe area in m^2
        '''
        ne = self.sample_Ne
        Te = self.sample_Te
        phi = self.sample_Phi

        self.Isat = 0.6*e*ne*np.sqrt(Te/Mi) * area
        self.Vf = phi + (Te/e) * np.log(0.6*np.sqrt(2*np.pi/mass_ratio))

    def GridPlot(self, t, fig=None):
        '''
        Make plot of data, sampling from probes, without inferring Isat or Vfloat
        '''
        if fig==None:
            fig = plt.figure(figsize=(10,5))
        M = self.M
        z = self.z_idx
        probes = self.probes

        T = self.t[t]
        Z = self.z[z]
        gs = gridspec.GridSpec(M, 2, figure=fig)
        ax0 = fig.add_subplot(gs[:,0])
        ax0.contourf(self.x, self.y, self.Ne[t,z],20,cmap='inferno')
        ax0.set_title(f"Ne: t = {T:.3e},  Z = {Z:.2f}")
        for j in np.arange(M):
            ax = fig.add_subplot(gs[j,1])
            ax.plot(self.t, self.sample_Ne[j])
            ax.axvline(T, ls='--', color='C2')
            ax.set_title("X = {:.2f}, Y = {:.2f}, Z = {:.2f}".format(*probes[j][::-1]))
        
            _,py,px = probes[j]
            ax0.plot(px,py,'C2o')
        
        fig.suptitle(f"frame {t} : t = {T:.3e}")
        fig.tight_layout()

    def ProbePlot(self,t,cmap,fig=None):
        '''
        Make plot of data, sampling from probes, with Isat and Vfloat
        '''
        if fig==None:
            fig = plt.figure(figsize=(10,5))
        M = self.M
        z = self.z_idx
        probes = self.probes

        T = self.t[t]
        Z = self.z[z]
        t_axis = self.t * 1e3
        gs = gridspec.GridSpec(M, 3, figure=fig)
    
        ax0 = fig.add_subplot(gs[:,0])
        ax0.contourf(self.x*100, self.y*100, self.Ne[t,z],20,cmap='inferno')
        ax0.set_title(f"Ne: t = {T:.3e},  Z = {Z:.2f}")
        ax0.set_xlabel("X [cm]")
        ax0.set_ylabel("Y [cm]")
    
        k_grid = M // 2
        ax_d1 = fig.add_subplot(gs[:k_grid,2])
        ax_d2 = fig.add_subplot(gs[k_grid:,2])
    
        for j in np.arange(M):
            ax = fig.add_subplot(gs[j,1])
            ax.plot(t_axis, self.sample_Ne[j], 'C1')
            ax.axvline(T*1e3, ls='--', color=cmap[j])
            ax.set_title("X = {:.2f}, Y = {:.2f}, Z = {:.2f}".format(*probes[j][::-1]))
        
            _,py,px = probes[j] * 100
            ax0.plot(px,py,'o', color=cmap[j])
    
            ax_d1.plot(t_axis, self.Vf[j], color=cmap[j])
            ax_d2.plot(t_axis, self.Isat[j]*1e3, color=cmap[j])
    
        ax_d1.set_title("Floating Potential [V]")
        ax_d2.set_title("Ion Saturation [mA]")
    
        ax.set_xlabel("time [ms]")
        ax_d2.set_xlabel("time [ms]")
        ax_d1.axvline(T*1e3, ls='--', color='C1')
        ax_d2.axvline(T*1e3, ls='--', color='C1')
    
        ax_d1.grid()
        ax_d2.grid()
        ax0.set_aspect('equal')
    
        fig.suptitle(f"frame {t} : t = {T:.3e}")
        fig.tight_layout()


    def plot_kTime(self,signals):
        '''
        plot the magnitude and phase components of spatial FT as function of time
        '''
        M = len(signals)
        m_wave = fftfreq(M, 1/M) 
        kspec = np.transpose( [fft(s) for s in signals.T] )
        time = self.t * 1e3

        # plot
        warm = colormap.autumn(np.linspace(0, 1, M))
        fig,axs = plt.subplots(3,1, figsize=(14,6))
        for j in np.arange(M):
            if j==0:
                axs[0].plot(time, np.abs(kspec[j]), label=f"m = {m_wave[j]}", color=warm[j])
            else:
                axs[1].plot(time, np.abs(kspec[j]), label=f"m = {m_wave[j]:.1f}", color=warm[j])
            axs[2].plot(time, np.angle(kspec[j]), label=f"m = {m_wave[j]:.1f}", color=warm[j])
        
        axs[0].set_ylabel('magnitude')
        axs[1].set_ylabel('magnitude')
        axs[2].set_ylabel('phase')
        axs[-1].set_xlabel('time [ms]')
        [a.legend(fontsize=8) for a in axs[:-1]]
        fig.tight_layout()
        
    def contour_mt(self,signals, ax=None):
        '''
        contour plot of wavenumber evolving in time
        removing the (m=0) DC amplitude
        '''
        if ax == None:
            fig,ax = plt.subplots(1,1)
        M = len(signals)
        time = self.t * 1e3
        m_wave = fftfreq(M, 1/M) 
        warm = colormap.autumn(np.linspace(0, 1, M))

        dc = np.mean(signals,axis=0)
        kspec = np.transpose( [fft(s) for s in (signals-dc).T] )
        ax.contourf(time,fftshift(m_wave),fftshift( np.abs(kspec),axes=0),25, cmap='hot')
        ax.set_ylabel("wave number")
        ax.set_xlabel('time [ms]')

        
    def probeSpectralAnalysis(self,t,fig=None):
        '''
        Make plot of data, sampling from probes, without inferring Isat or Vfloat
        '''
        if fig==None:
            fig = plt.figure(figsize=(16,7))

        M = self.M
        z = self.z_idx
        probes = self.probes

        T = self.t[t]
        Z = self.z[z]
        R = self.R_probes 
        time = self.t * 1e3
        signals = self.sample_Ne
        raw = self.Ne

        gs = gridspec.GridSpec(4, 3, figure=fig)
        ax_signal = fig.add_subplot(gs[0,:])
        ax_plasma = fig.add_subplot(gs[1:,0])
        ax_spectra = fig.add_subplot(gs[1:,-1])
        ax_rad = fig.add_subplot(gs[1,1])
        ax_amp = fig.add_subplot(gs[2,1])
        ax_ang = fig.add_subplot(gs[3,1])
        cool = colormap.winter(np.linspace(0, 1, M))

        # space
        fig.suptitle(f"Ne frame {t}: t = {T:.3e},  Z = {Z:.2f}, R = {R:.2f}")
        ax_plasma.contourf(self.x*100, self.y*100, raw[t,z],20,cmap='inferno')
        ax_plasma.set_xlabel("x [cm]")
        ax_plasma.set_ylabel("y [cm]")
        
        for j in np.arange(M):
            _,py,px = probes[j] * 100
            ax_plasma.plot(px,py,'o', color=cool[j])

            if (j % 2):
                continue
            ax_signal.plot(time, signals[j], color=cool[j], label=f"probe {j+1}")
        ax_signal.axvline(time[t], ls='--', color='C2')
        ax_signal.set_xlabel("time [ms]")
        ax_signal.legend(loc=1, fontsize=8)

        # spectral
        m_wave = fftshift( fftfreq(M, 1/M) )
        warm = colormap.autumn(np.linspace(0, 1, M))
        r_axis = np.arange(M)
        ax_rad.plot( r_axis+1, signals[:,t], 'C2o-' )
        for j in r_axis:
            ax_rad.plot(j+1, signals[j,t], 'o', color=cool[j])
        ax_rad.set_xlabel("probe number")

        dc = np.mean(signals,axis=0)
        kspec = fftshift( np.transpose( [fft(s) for s in (signals-dc).T] ), axes=0 ) 
        ax_spectra.contourf(time, m_wave, np.abs(kspec),25, cmap='hot')
        ax_spectra.set_ylabel("wave number")
        ax_spectra.set_xlabel('time [ms]')
        ax_spectra.axvline(time[t], color='C2', ls='--')

        ax_amp.plot(m_wave, np.abs(kspec[:,t]), 'C1o--', label='amplitude')
        ax_ang.plot(m_wave, np.angle(kspec[:,t]), 'C3o--', label='phase')
        ax_ang.set_xlabel("wave number")
        ax_amp.legend(loc=1,fontsize=10)
        ax_ang.legend(loc=1,fontsize=10)

        ax_plasma.set_aspect('equal')
        ax_spectra.set_aspect('equal')
        fig.tight_layout()
