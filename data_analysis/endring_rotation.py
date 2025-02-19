from WhamData import EndRing
import sys

import matplotlib.pyplot as plt

shot = int(sys.argv[1])
save = False

ring = EndRing(shot)

tax = [2.5,5,8,15] # time index, ms
ring.plot_rotation(time_slices = tax)

if save:
    plt.savefig(f"out/rot-{shot}.png")
else:
    plt.show()
