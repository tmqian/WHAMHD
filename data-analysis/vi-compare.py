from DataSpec import Spectrometer
from WhamData import EndRing
import sys

import matplotlib.pyplot as plt
import numpy as np

fig,axs = plt.subplots(2,2, figsize=(10,8))

for shot in sys.argv[1:]:
    shot = int(shot)

    try:
        spec = Spectrometer(shot)
        C = spec.C
        axs[1,1].errorbar(C.los/10, C.Vi/1e3, yerr=C.Vi_err/1e3, fmt='o-')
        axs[0,1].errorbar(C.los/10, C.Ti, yerr=C.Ti_err, fmt='o-')

        Vi1 = C.Vi[-1]/1e3
        Ti1 = C.Ti[-1]
        Vi0 = C.Vi[5]/1e3
        Ti0 = C.Ti[5]

        axs[0,1].axhline(Ti1, ls='--', color='C1', label=rf"$T_i$ = ({Ti0:.1f}, {Ti1:.1f}) eV")
        axs[0,1].axhline(Ti0, ls='--', color='C1')
        axs[1,1].axhline(Vi1, ls='--', color='C1', label=r"$V_{ti}$" + f" = {Vi1:.2f} km/s")

        print(f"{shot}, {Vi0:.2f}, {Vi1:.2f}, {Ti0:.1f}, {Ti1:.1f}")

    except:
        print(f"issue with spec {shot}")

    ring = EndRing(shot)
    axs[0,0].plot(ring.radius, ring.V_ring, 'o-')
    #axs[0,0].plot(ring.radius, ring.V_ring, 'o-', label=shot)
    axs[1,0].plot(ring.mid[:-1], ring.Vphi[:-1], 'x-', label=shot)

    V = ring.V_ring[0]
    axs[0,0].axhline(V, ls='--', color='C1', label=r"$V_{ring}$" + f" = {V:.1f} V")

axs[0,0].set_title("Floating Potential [V]")
axs[0,1].set_title("Ti C-iii [eV]")
axs[1,0].set_title("ExB rotation [km/s]")
axs[1,1].set_title("Ion C-III rotation [km/s]")

axs[1,0].set_xlabel("End Ring radius [cm]")
axs[1,1].set_xlabel("Mid Plane radius [cm]")

axs[0,0].sharex(axs[1,0])

axs[0,0].legend()
axs[1,0].legend()
axs[0,1].legend()
axs[1,1].legend()

for a in np.ndarray.flatten(axs):
    a.grid()


#plt.savefig(f"out/combo-rot-{shot}.png")
plt.show()

