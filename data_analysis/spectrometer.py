from DataSpec import Spectrometer
import sys
import matplotlib.pyplot as plt
from pathlib import Path


nasPath = '/mnt/n/whamdata/shot_plots/' 

'''
Works on Jack, reading data from NAS
2/18/2025 TQ
'''

shot = int(sys.argv[1])
spec = Spectrometer(shot)

spec.plot_Vi_T_CIII()
# DE added 25/04/03
shotstr = str(shot)
directory = nasPath + shotstr[:2]+'/'+shotstr[2:4]+'/'+shotstr[4:6]+'/OES/'
Path(directory).mkdir(parents=True,exist_ok=True)

plt.tight_layout()
plt.savefig(directory+'oes_'+shotstr+'_ViTiCIII.png',dpi=300,bbox_inches='tight')

spec.plot_spectra()

#plt.show()
plt.close()

# DE added 25/04/03 to make single plot on request for Craig Jacobson, for documentation of plasma performance to DOE
spec.plot_Vi_single()
plt.savefig(directory+'oes_'+shotstr+'_ViCIII.png',dpi=300,bbox_inches='tight'); print('Figure saved as ',shotstr,'_ViCIII.png')
#plt.show()
plt.close()

