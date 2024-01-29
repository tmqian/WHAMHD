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
        
        gkyl = self
        grid = (gkyl.t, gkyl.z, gkyl.y, gkyl.x)
        f_Ne = RegularGridInterpolator(grid, gkyl.Ne)
        f_Te = RegularGridInterpolator(grid, gkyl.Te)
        f_Ti = RegularGridInterpolator(grid, gkyl.Ti)
        f_Phi = RegularGridInterpolator(grid, gkyl.Phi)
        
        probes = self.probes
        self.sample_Ne = np.array( [[ f_Ne( (t,*p) ) for t in gkyl.t] for p in probes] )
        self.sample_Te = np.array( [[ f_Te( (t,*p) ) for t in gkyl.t] for p in probes] )
        self.sample_Ti = np.array( [[ f_Ti( (t,*p) ) for t in gkyl.t] for p in probes] )
        self.sample_Phi = np.array( [[ f_Phi( (t,*p) ) for t in gkyl.t] for p in probes] )

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
        if fig==None:
            fig = plt.figure(figsize=(10,5))
        M = self.M
        gkyl = self
        z = self.z_idx
        probes = self.probes

        T = gkyl.t[t]
        Z = gkyl.z[z]
        gs = gridspec.GridSpec(M, 2, figure=fig)
        ax0 = fig.add_subplot(gs[:,0])
        ax0.contourf(gkyl.x, gkyl.y, gkyl.Ne[t,z],20,cmap='inferno')
        ax0.set_title(f"Ne: t = {T:.3e},  Z = {Z:.2f}")
        for j in np.arange(M):
            ax = fig.add_subplot(gs[j,1])
            ax.plot(gkyl.t, self.sample_Ne[j])
            ax.axvline(T, ls='--', color='C2')
            ax.set_title("X = {:.2f}, Y = {:.2f}, Z = {:.2f}".format(*probes[j][::-1]))
        
            _,py,px = probes[j]
            ax0.plot(px,py,'C2o')
        
        fig.suptitle(f"frame {t} : t = {T:.3e}")
        fig.tight_layout()

    def ProbePlot(self,t,cmap,fig=None):
        if fig==None:
            fig = plt.figure(figsize=(10,5))
        M = self.M
        gkyl = self
        z = self.z_idx
        probes = self.probes

        T = gkyl.t[t]
        Z = gkyl.z[z]
        t_axis = gkyl.t * 1e3
        gs = gridspec.GridSpec(M, 3, figure=fig)
    
        ax0 = fig.add_subplot(gs[:,0])
        ax0.contourf(gkyl.x*100, gkyl.y*100, gkyl.Ne[t,z],20,cmap='inferno')
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

    def spectralAnalysis(self,signals):

        M = len(signals)
        R = self.R_probes
        ds = np.pi*2*R/M
        self.m_wave = fftfreq(M,ds) * (2*np.pi*R)
        self.kspec = np.transpose( [fft(s) for s in signals.T] )


    def plot_kTime(self,signals):
        '''
        plot the magnitude and phase components of spatial FT as function of time
        '''
        time = self.t * 1e3
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
        '''
        if ax == None:
            fig,ax = plt.subplots(1,1)
        M = len(signals)
        R = self.R_probes
        ds = np.pi*2*R/M
        time = self.t * 1e3
        m_wave = fftfreq(M,ds) * (2*np.pi*R)
        warm = colormap.autumn(np.linspace(0, 1, M))

        dc = np.mean(signals,axis=0)
        kspec = np.transpose( [fft(s) for s in (signals-dc).T] )
        ax.contourf(time,fftshift(m_wave),fftshift( np.abs(kspec),axes=0),25, cmap='hot')
        ax.set_ylabel("wave number")
        ax.set_xlabel('time [ms]')

        
