import matplotlib.pyplot as plt

from WhamData import FluxLoop
import sys

shot_num = int(sys.argv[1])

plotPressure = False

flux = FluxLoop(shot_num)
flux.plot()
plt.xlim(-3,30)

if plotPressure:
    flux.calcPressure()
    fig,axs = flux.plotExtra()
    for a in axs:
        a.set_xlim(-2,16)

plt.show()


