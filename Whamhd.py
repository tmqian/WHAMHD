import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pylab import cm as cmap

# user def
path = "/Users/tqian/Documents/WHAM/Wham-Theory/simulation/gkeyll/2023-1005-gkyl-data"

# constants
A_probe = 9.5e-6 # m2
mass_ratio = 1840
pi = np.pi
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
        
        from scipy.interpolate import RegularGridInterpolator
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

    def GridPlot(self, t, fig=None):
        if fig==None:
            fig = plt.figure(figsize=(10,5))
        M = self.M
        gkyl = self
        z = self.z_idx
        probes = self.probes

        gs = gridspec.GridSpec(M, 2, figure=fig)
        ax0 = fig.add_subplot(gs[:,0])
        T = gkyl.t[t]
        Z = gkyl.z[z]
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
