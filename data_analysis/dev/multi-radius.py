import numpy as np
import matplotlib.pyplot as plt
import sys

from WhamData import ECH, BiasPPS, Interferometer, FluxLoop, EdgeProbes, NBI, AXUV, EndRing, Gas, adhocGas, IonProbe, ShineThrough, Dalpha
from DataSpec import Spectrometer
from multi import get_time_index 
from FitData import *


shot = int(sys.argv[1])

tax = [3,4.8,7]
cmap = plt.cm.plasma(np.linspace(0,0.8,len(tax)))

line = np.linspace(-30, 30, 100)
radius = np.linspace(-20, 20, 50)

# 3-parameter Gaussian
def m1(r,r0=2,w=10,A=1):
    return A*np.exp(-((r-r0)/w)**2)

try:
    axuv = AXUV(shot)
    axuv_data = True
except:
    print('no axuv')
    axuv_data = False

try:
    shine = ShineThrough(shot)
    nbi = NBI(shot)
    shine_data = True
except:
    shine_data = False

try:
    bias = EndRing(shot)
    bias_data = True
except:
    bias_data = False

try:
    da = Dalpha(shot)
    dalpha_data = True
except:
    dalpha_data = False


fig,axs = plt.subplots(4,3,figsize=(12,8), sharex=True)
dt = 0.1 # window for averaging fluctuations
for k,t in enumerate(tax):

    if axuv_data:
        t1,t2 = get_time_index(axuv.time, t-dt/2,t+dt/2)

        r_axuv = axuv.b*100
        f_axuv = np.mean(axuv.data[:,t1:t2],axis=1)
        axs[0,0].plot(r_axuv, f_axuv, 'o', label=f"t = {t:.1f} ms", color=cmap[k])
        # axuv.data [20,25000] R,T
        opt = scipy_min(r_axuv, f_axuv, m1, 
                  param_names = ["A", "w", "r0"],
                  initial_guess = [1e-7, 5.1, 0],
                  method="Nelder-Mead",
        ) 
        predicted_profile = m1(radius, **opt)
        predicted_integrals = forward_integrate(m1, radius, line, **opt)

        axs[0,0].plot(radius, predicted_integrals, '--', color=cmap[k]) 
        axs[0,1].plot(radius, predicted_profile, color=cmap[k]) 

    if shine_data:
        t1,t2 = get_time_index(shine.time, t,t)
        axs[1,2].plot(shine.radius, shine.nr[t1,:], color=cmap[k])  
        # shine.nr [400,100] T,R
    
        # nbi [10,40000] R,T
        t1,t2 = get_time_index(nbi.time, t,t)

        f_shine = nbi.d_arr[:,t1]
        r_shine = shine.chord_radius
        axs[1,0].plot(r_shine, f_shine, 'o', color=cmap[k])  

        opt = scipy_min(r_shine, f_shine, m1, 
                  param_names = ["A", "w", "r0"],
                  initial_guess = [1.0, 5, 0],
                  method="Nelder-Mead",
        ) 
        predicted_profile = m1(radius, **opt)
        predicted_integrals = forward_integrate(m1, radius, line, **opt)

        axs[1,0].plot(radius, predicted_integrals, '--', color=cmap[k]) 
        axs[1,1].plot(radius, predicted_profile, color=cmap[k]) 

    if bias_data:
        try:
            rax = bias.radii * (13./22) # approximate map
            t1,t2 = get_time_index(bias.time, t,t)
            v1 = bias.ProbeArr[:,t1]
            cat = np.concatenate
            axs[3,1].plot(cat([-rax[::-1],rax]), cat([v1[::-1],v1]), 'o-', color=cmap[k])
            # endring [10,27000] R,T
        except:
            pass

    if dalpha_data:
        t1,t2 = get_time_index(da.time, t,t)

        r_dalpha = da.radius
        f_dalpha = da.data[:,t1]
        axs[2,0].plot(r_dalpha, f_dalpha, 'o', color=cmap[k])  
        # da.data [6,350000] R,T

        opt = scipy_min(r_dalpha, f_dalpha, m1, 
                  param_names = ["A", "w", "r0"],
                  initial_guess = [1e9, 5, 2],
                  method="Nelder-Mead",
        ) 
        predicted_profile = m1(radius, **opt)
        predicted_integrals = forward_integrate(m1, radius, line, **opt)

        axs[2,0].plot(radius, predicted_integrals, '--', color=cmap[k]) 
        axs[2,1].plot(radius, predicted_profile, color=cmap[k]) 

axs[0,0].set_title("measured chords")
axs[0,1].set_title("fitted profiles")
axs[0,2].set_title("physics")

axs[0,0].set_ylabel("AXUV")
axs[1,0].set_ylabel("NBI Shinethrough")
axs[2,0].set_ylabel("D-alpha")
axs[3,0].set_ylabel("Endring Voltage")

axs[0,0].legend(fontsize=8)

axs[1,0].set_ylim(bottom=0)
axs[1,1].set_ylim(bottom=0)
axs[1,2].set_ylim(bottom=0)

axs[-1,0].set_xlabel("midplane radius [cm]")
axs[-1,1].set_xlabel("midplane radius [cm]")
axs[-1,2].set_xlabel("midplane radius [cm]")

for a in np.ndarray.flatten(axs):
    a.minorticks_on()
    a.grid(which='both')
    a.grid(which='minor', linestyle=":", linewidth=0.5)

fig.suptitle(shot)
fig.tight_layout()

plt.show()
