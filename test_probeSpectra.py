from Whamhd import SyntheticDiagnostic
import numpy as np
import matplotlib.pyplot as plt

# output single frame or 1000
video = False

# user defined path
#    download data here: https://uwmadison.box.com/s/uuzej9m1uvh7h3vhehz9aip4hmh4ikq6
path = "/Users/tqian/Documents/WHAM/Wham-Theory/simulation/gkeyll/2023-1005-gkyl-data"
fin = f"{path}/32.npz"

# load data from file
data = SyntheticDiagnostic()
data.loadGkylPack(fin)

# set up probe locations
M = 12
z_idx = 10
data.initProbeCircle(z_idx,M=M)

# sample data at probe locations
data.interpolate()
A_probe = 9.5e-6 # m2
data.sampleProbes(area=A_probe)


# output
if video:
    # plot many frames to make a video, make sure outdir exists
    outdir = "spec_01" 
    N = len(data.t)
    fig = plt.figure(figsize=(16,7))
    for t in np.arange(N)[::2]:
        data.probeSpectralAnalysis(t,fig=fig)
        plt.savefig(f"{outdir}/{t:04d}.png")
        fig.clear()
    # use this line from terminal to turn set of PNG into GIF
    ### convert -delay 5 -loop 1 *.png gk32_ne-fast.gif

else:
    # plot single frame at t=500
    data.probeSpectralAnalysis(500)
    plt.show()
