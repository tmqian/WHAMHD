from DataSpec import Spectrometer
import sys
import matplotlib.pyplot as plt

'''
Copied from 0924 analysis
'''

shot = int(sys.argv[1])
spec = Spectrometer(shot)

spec.plot_Vi_T_CIII()
spec.plot_spectra()

plt.show()
