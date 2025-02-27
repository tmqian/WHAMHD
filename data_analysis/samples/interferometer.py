import numpy as np
import matplotlib.pyplot as plt

import sys
from WhamData import Interferometer

#shot = int(sys.argv[1])
#intf= Interferometer(shot)
#intf.plot()

fig,axs = plt.subplots(1,1,figsize=(10,5))
for shot in sys.argv[1:]:

    self = Interferometer(int(shot))
    axs.plot(self.time,self.linedens, label=f"{self.shot}")

axs.set_title(r"Line Integrated Density [m$^{-2}$]")
axs.set_xlabel("time [ms]")
axs.legend()
axs.grid()
plt.show()
