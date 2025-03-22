from WhamData import Dalpha
import matplotlib.pyplot as plt
import sys

shot = int(sys.argv[1])
da = Dalpha(shot)
da.plot()
plt.show()
