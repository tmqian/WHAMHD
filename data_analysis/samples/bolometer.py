from WhamData import Bolometer

import matplotlib.pyplot as plt
import sys

shot = int(sys.argv[1])
bolo = Bolometer(shot)
#bolo.plot()

fig,axs = plt.subplots(1,1)
bolo.plotQ(axs)
plt.show()
