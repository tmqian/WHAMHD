from DataSpec import Spectrometer
import sys

import numpy as np
import matplotlib.pyplot as plt

shot = sys.argv[1]

spec = Spectrometer(shot)
spec.load_Carbon()

arr = []
for j,line in enumerate(spec.C.data):
    k = np.argmax( np.array(line[:50]))
    L = np.array(spec.C.wavelength[k])
    print(j,L)
    arr.append(L)

print(np.argsort(arr))


plt.show()
