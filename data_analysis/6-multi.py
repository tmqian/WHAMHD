import numpy as np
import matplotlib.pyplot as plt
import sys

from multi import plot6
 

plotLimiter = True
print("shot number, Ring [V], Ring [A], ECH [kW], Flux [kMx], Nint [m^-2]")


# plot
fig, axs = plt.subplots(6,1,sharex=True, figsize=(11,10))

for shot in sys.argv[1:]:
    plot6(shot, axs=axs, plotLimiter=plotLimiter)

axs[0].legend(loc=4, fontsize=7)
for a in axs:
    a.minorticks_on()
    a.grid(which='both')
    a.grid(which='minor', linestyle=":", linewidth=0.5)

axs[-1].set_xlabel("time [ms]")
axs[-1].set_xlim(-1,20)

fig.tight_layout()
#plt.savefig(f"out/shot-{shot}.png")
plt.show()


