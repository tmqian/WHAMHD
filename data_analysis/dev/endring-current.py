import MDSplus as mds
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.signal import savgol_filter as savgol
import sys

'''
TQ 18 Feb 2025
'''

shot = int(sys.argv[1])

tree = mds.Tree("wham",shot)

# get raw data
raw = "raw.acq196_370"
node = [tree.getNode(f"{raw}.ch_{j+1:02d}") for j in range(64,84)]

t0 = tree.getNode(f"{raw}.trig_time").getData().data() # this is the dtacq trig time, NOT the bias trig time.
time = 1e3 * (t0 + node[0].dim_of().data()) #ms

I_factor = -1530/50
Iring = np.array([I_factor * n.getData().data() for n in node])
# first 10 are S, second 10 are N

## Current Rings
def getLabel(j):
    if j < 10:
        return f"S{j}"
    else:
        return f"N{j-10}"


fig,axs = plt.subplots(5,2,figsize=(10,8),sharey=True, sharex=True)
ax = np.ndarray.flatten(axs)
for j,I in enumerate(Iring):
    k = j%10
    ax[k].plot(time,I, label=getLabel(j))
for a in ax:
    a.minorticks_on()
    a.grid(which='both')
    a.grid(which='minor', linestyle=":", linewidth=0.5)
    a.legend(fontsize=8)
ax[0].set_ylabel("Ring Current [A]")
ax[-1].set_xlabel("Time [ms]")

ax[1].set_xlim(-3,33)
fig.suptitle(f"{shot}")


## plot 2
S,N = np.reshape(Iring,(2,10,27000))
I_circ = (N-S)/2
I_tot = N+S

fig,axs = plt.subplots(5,2,figsize=(10,8),sharey=True, sharex=True)
ax = np.ndarray.flatten(axs)
for j in range(10):

    if j==0:
        ax[j].plot(time,I_circ[j], label="I_circ = (N-S)/2")
        ax[j].plot(time,I_tot[j], label="I_tot = N+S")
        ax[j].set_title(f"Disk {j}")
    else:
        ax[j].plot(time,I_circ[j])
        ax[j].plot(time,I_tot[j])
        ax[j].set_title(f"Ring {j}")

for a in ax:
    a.minorticks_on()
    a.grid(which='both')
    a.grid(which='minor', linestyle=":", linewidth=0.5)

ax[0].legend(fontsize=8)
axs[0,0].set_ylabel("Ring Current [A]")
axs[1,0].set_ylabel("Ring Current [A]")
axs[2,0].set_ylabel("Ring Current [A]")
axs[3,0].set_ylabel("Ring Current [A]")
axs[4,0].set_ylabel("Ring Current [A]")
axs[-1,0].set_xlabel("Time [ms]")
axs[-1,1].set_xlabel("Time [ms]")

ax[1].set_xlim(-3,33)
fig.suptitle(f"{shot}")
fig.tight_layout()


# plot 3
N_ring = 10
cvir = plt.cm.viridis(np.linspace(0,1,N_ring))
cjet = plt.cm.jet(np.linspace(0,1,N_ring))
#fig,axs = plt.subplots(2,1,figsize=(11,8))
fig = plt.figure(figsize=(15,8))

gs = GridSpec(2,4, figure=fig)
ax0 = fig.add_subplot(gs[0,:-1])
ax1 = fig.add_subplot(gs[1,:-1])
ax2 = fig.add_subplot(gs[0,-1])
ax3 = fig.add_subplot(gs[1,-1])
axs = np.array([ax0,ax1,ax2,ax3])

for j in range(N_ring):
    axs[0].plot(time, I_tot[j], color=cjet[j], lw=0.5)
    if j>0:
        axs[1].plot(time, I_circ[j], color=cjet[j], label=f"Ring {j}", lw=0.5)
    else:
        axs[1].plot(time, I_circ[j], color=cjet[j], label=f"Disk 0", lw=0.5)

def get_time_index(t):
    # get the index at time t, ms
    j = np.argmin(np.abs(time - t))
    return j

radii = np.array([4.0,8.0,11.1,13.4,15.3,16.9,18.3,19.5,20.6,21.6]) 

time_slices = [3,4,8,11,15]
Nt = len(time_slices)
cpla = plt.cm.plasma(np.linspace(0,1,Nt))
cpla = plt.cm.autumn(np.linspace(0,1,Nt))
cpla = plt.cm.cividis(np.linspace(0,1,Nt))
for i,t in enumerate(time_slices):

    ts = get_time_index(t)
    axs[0].axvline(t,color=cpla[i], ls='--')
    axs[1].axvline(t,color=cpla[i], ls='--')
    axs[2].plot(radii,I_tot[:,ts],'o-', color=cpla[i], label=f"t = {t} ms")
    axs[3].plot(radii,I_circ[:,ts],'o-', color=cpla[i])

for a in axs:
    a.minorticks_on()
    a.grid(which='both')
    a.grid(which='minor', linestyle=":", linewidth=0.5)
axs[1].legend(fontsize=8)
axs[2].legend(fontsize=8)
axs[1].set_title("Circulating Current (N-S)/2")
axs[0].set_title("Total Current (N+S)")

axs[0].set_xlim(-3,17)
axs[1].set_xlim(-3,17)

axs[0].set_ylabel("Current [A]")
axs[1].set_ylabel("Current [A]")
axs[1].set_xlabel("Time [ms]")
axs[3].set_xlabel("Radius [cm]")

fig.suptitle(f"{shot}")

fig,axs = plt.subplots(2,2, figsize=(10,10))

ax = axs[0,0]
C = ax.contourf(time,radii, S,20,cmap='inferno')
fig.colorbar(C,ax=ax)
ax.set_xlim(-3,17)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Radius [cm]")
ax.set_title("South End Cell [A]")
ax.axhline(radii[4], color='w', ls='--', lw=2)

ax = axs[0,1]
C = ax.contourf(time,radii, N,20,cmap='inferno')
fig.colorbar(C,ax=ax)
ax.set_xlim(-3,17)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Radius [cm]")
ax.set_title("North End Cell [A]")
ax.axhline(radii[4], color='w', ls='--', lw=2)

ax = axs[1,0]
C = ax.contourf(time,radii[1:], I_circ[1:],20,cmap='turbo')
fig.colorbar(C,ax=ax)
ax.set_xlim(-3,17)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Radius [cm]")
ax.set_title("Circulating Current [A]")
ax.axhline(radii[4], color='w', ls='--', lw=2)

ax = axs[1,1]
C = plt.contourf(time,radii, I_tot,20,cmap='turbo')
fig.colorbar(C,ax=ax)
ax.set_xlim(-3,17)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Radius [cm]")
ax.set_title("Total Current [A]")
ax.axhline(radii[4], color='w', ls='--', lw=2)

fig.suptitle(f"{shot}")

plt.show()
