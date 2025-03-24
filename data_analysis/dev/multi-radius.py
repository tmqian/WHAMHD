import numpy as np
import matplotlib.pyplot as plt
import sys

from WhamData import ECH, BiasPPS, Interferometer, FluxLoop, EdgeProbes, NBI, AXUV, EndRing, Gas, adhocGas, IonProbe, ShineThrough, Dalpha
from DataSpec import Spectrometer

from multi import get_time_index 
shot = int(sys.argv[1])

axuv = AXUV(shot)
shine = ShineThrough(shot)
nbi = NBI(shot)
bias = EndRing(shot)

if shot > 250306000:
    da = Dalpha(shot)
    d_alpha = True
else:
    d_alpha = False

tax = [6,9,11]
tax = [4,5,6]
tax = [3,4.8,7]

fig,axs = plt.subplots(4,2,figsize=(10,8), sharex=True)
for t in tax:

    t1,t2 = get_time_index(axuv.time, t,t)
    axs[0,0].plot(axuv.b*100, axuv.data[:,t1], 'o-', label=f"t = {t:.1f} ms")
    # axuv.data [20,25000] R,T

    t1,t2 = get_time_index(shine.time, t,t)
    axs[1,1].plot(shine.radius, shine.nr[t1,:])  
    # shine.nr [400,100] T,R

    ch = np.array([i for i in range(11) if i != 5])
    dShine = nbi.d_arr[ch]
    # nbi [10,40000] R,T
    t1,t2 = get_time_index(nbi.time, t,t)
    axs[1,0].plot(nbi.radius, dShine[:,t1], 'o-')  

    rax = bias.radii * (13./22) # approximate map
    t1,t2 = get_time_index(bias.time, t,t)
    v1 = bias.ProbeArr[:,t1]
    cat = np.concatenate
    axs[3,1].plot(cat([-rax[::-1],rax]), cat([v1[::-1],v1]), 'o-')
    # endring [10,27000] R,T

    if d_alpha:
       t1,t2 = get_time_index(da.time, t,t)
       axs[2,0].plot(da.radius, da.data[:,t1], 'o-')  
       # shine.nr [6,350000] R,T

axs[0,0].set_title("chord, un-inverted")
axs[0,1].set_title("radius, inverted")
axs[0,0].set_ylabel("AXUV")
axs[1,0].set_ylabel("NBI Shinethrough")
axs[2,0].set_ylabel("D-alpha")
axs[3,0].set_ylabel("Endring Voltage")

axs[0,0].legend(fontsize=8)

axs[1,0].set_ylim(bottom=0)
axs[1,1].set_ylim(bottom=0)

axs[-1,0].set_xlabel("radius [cm]")
axs[-1,1].set_xlabel("radius [cm]")
for a in np.ndarray.flatten(axs):
    a.grid()
fig.suptitle(shot)
plt.show()
import pdb
pdb.set_trace()
