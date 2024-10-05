import MDSplus as mds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from WhamData import EdgeProbes

import sys
'''
Plot Floating Probe (edge fluctuation array) data
usage: python script.py 240803094

T Qian - 9/30/2024
'''

# Kunal Sanwalka- Added function to save edge fluctuation data as an npz file for analysis with other scripts
def save_npz(shot_num):

    # Edge probe data object
    edge = EdgeProbes(shot_num, RP=True)

    # Directory in which to save the .npz file
    dataDest = '/home/WHAMdata/python/oneOffScripts/data/'

    # Number of edge fluctuation probes
    N = 12

    # Array to store edge probe data
    probe0Data = edge.ProbeArr[0]
    # Associated time array
    probe0Time = edge.TimeArr[0]
    edgeData = np.zeros(shape=(N, len(probe0Data)))
    for i in range(N):
        probeData = edge.ProbeArr[i]
        timeArr = edge.TimeArr[i]

        # Interpolate so they all have the same time basis
        try:
            probeData = np.interp(probe0Time, timeArr, probeData)
        except:
            probeData = np.zeros(len(probe0Time))
        edgeData[i] = probeData

    np.savez(dataDest+'{}_edge_probe_data.npz'.format(shot_num), edgeData=edgeData, timeArr=probe0Time)

    return

# use parser to get shot number
shot_num = int(sys.argv[1])
edge = EdgeProbes(shot_num, RP=True)

# Kunal Sanwalka- Call function to save the data as an .npz file
save_npz(shot_num)

fig = plt.figure(figsize=(15,8))
gs = GridSpec(3,3, figure=fig)

ax0 = fig.add_subplot(gs[0,:])

ax21 = fig.add_subplot(gs[1,0])
ax22 = fig.add_subplot(gs[1,1])
ax23 = fig.add_subplot(gs[1,2])
ax31 = fig.add_subplot(gs[2,0])
ax32 = fig.add_subplot(gs[2,1])
ax33 = fig.add_subplot(gs[2,2])

axs = [ax21, ax22, ax23, ax31, ax32, ax33]
N = 12
for j in range(N):
    ax0.plot(edge.TimeArr[j], edge.ProbeArr[j], lw=0.5, label=f"Edge Probe {j}")

for j in range(N-2):
    axs[j//2].plot(edge.TimeArr[j], edge.ProbeArr[j], f"C{j}", alpha=0.3 )
    axs[j//2].plot(edge.TimeArr[j], edge.ProbeArr[j], f"C{j}", label=f"Edge Probe {j}")
axs[4].plot(edge.TimeArr[10], edge.ProbeArr[10], f"C10", label=f"Edge Probe 10")
axs[5].plot(edge.TimeArr[11], edge.ProbeArr[11], f"C11", label=f"Edge Probe 11")


ax32.set_xlabel("time [ms]")
ax0.set_ylabel("floating potential [V]")
ax0.set_xlim(-7,100)
ax0.set_xlim(-1,15)
ax0.grid()
ax0.legend(loc=1, fontsize=7)

for a in axs:
    a.grid()
    a.legend(loc=1)
    a.sharex(ax0)
    a.sharey(ax0)

fig.suptitle(shot_num)
plt.show()
