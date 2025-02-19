from DataSpec import Spectrometer
import sys
import matplotlib.pyplot as plt

'''
Works on Jack, reading data from NAS
2/18/2025 TQ
'''

shot = sys.argv[1]
spec = Spectrometer(shot)

spec.plot_Vi_T_CIII()
spec.plot_spectra()

plt.show()
