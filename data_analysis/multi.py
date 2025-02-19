import numpy as np
import matplotlib.pyplot as plt
import sys

from WhamData import ECH, BiasPPS, Interferometer, FluxLoop, EdgeProbes, NBI, AXUV, EndRing, Gas
from DataSpec import Spectrometer

from scipy.signal import savgol_filter as savgol
axuv_map = plt.cm.turbo(np.linspace(0,1,20))
probe_map = plt.cm.hsv(np.linspace(0,1,12))

'''
This package uses the classes in WhamData
to plot multiple diagnostics.

plot9 - (3x3) figure, many traces per plot, single shot
plot6 - (6x1) figure, single trace per plot, multi or single shot
'''

def toBool(data):
    # convert MDS database string to T/F
    if str(data) == 'On':
        return True
    else:
        return False

def get_time_index(time_axis, t1, t2):

    j1 = np.argmin(np.abs(time_axis - t1))
    j2 = np.argmin(np.abs(time_axis - t2))
    return j1,j2

def plot9(shot, fout="", plot_limiter_bias=False, tag=""):

    fig, axs = plt.subplots(3,3,sharex=True, figsize=(13,9))

    # ECH plots
    try:
        ech = ECH(shot)
        ax = axs[0,0]
        ax.plot(ech.time,ech.Fwg_filt,color='blue', label="ECH Forward Power [kW]")
        ax.plot(ech.time,ech.Rwg_filt,color='red', label="ECH Reverse Power [kW]")
    except:
        print(f"Issue with ECH {shot}")
    
    # bias plots
    try:
        bias = BiasPPS(shot)
        ax = axs[2,1]
        ax.plot(bias.time, bias.L_Dem*1e3, 'C0', label="Limiter Demand [mV]")
        ax.plot(bias.time, bias.L_VLem, 'C1', label="Limiter Potential [V]")
        ax.plot(bias.time, bias.L_ILem, 'C5', label="Limiter Current [A]")
        ax = axs[2,2]
        ax.plot(bias.time, bias.R_Dem*1e3, 'C0', label="Ring Demand [mV]")
        ax.plot(bias.time, bias.R_VLem, 'C1', label="Ring Potential [V]")
        ax.plot(bias.time, bias.R_ILem, 'C5', label="Ring Current [A]")
    except:
        print(f"Issue with Bias {shot}")
    
    # flux plot
    try:
        flux = FluxLoop(shot)
        ax = axs[0,1]
        ax.plot(flux.time,flux.FL1/1e3,'C3', label="Flux Loop 1 [kMx]")
        ax.plot(flux.time,flux.FL2/1e3,'C2', label="Flux Loop 2 [kMx]")
        ax.plot(flux.time,flux.FL3/1e3,'C0', label="Flux Loop 3 [kMx]")
    except:
        print(f"Issue with Flux {shot}")
    
    # density
    try:
        intf = Interferometer(shot)
        axs[1,1].plot(intf.time, intf.linedens,'C2', label=r"Line Integrated Density [m$^{-2}$]")
    except:
        print(f"Issue with Interferometer {shot}")
    
    ## end ring
    #try:
    #    ring = EndRing(shot)
    #    axs[1,2].plot(ring.time, ring.SmoothArr[1],'C1', label=r"End Ring 2 [V]")
    #except:
    #    print(f"Issue with End Ring {shot}")

    # edge probe
#    try:
#        edge = EdgeProbes(shot)
#        for j in [0,3,11]:
#            axs[1,2].plot(edge.time, edge.ProbeArr[j], color=probe_map[j], label=f"Edge Probe {j} [V]")
#    except:
#        print(f"Issue with Edge probe {shot}")

    # gas
    try:
        gas = Gas(shot)
        ax = axs[2,0]
        ax.plot(gas.t_asdex, gas.asdex_hi*1e3, 'C2', label=r"asdex [mTorr]")
        ax.plot(gas.t_asdex, gas.asdex_lo*1e3, 'C1')
    except:
        print(f"Issue with gas {shot}")
    
    # axuv
    try:
        axuv = AXUV(shot)
        ax = axs[0,2]
        for j in range(20):
            if j in [2,10,17]:
                ax.plot(axuv.time, axuv.data[j], color=axuv_map[j], label=f"AXUV {j+1} [arb]")
            else:
                ax.plot(axuv.time, axuv.data[j], color=axuv_map[j])
    except:
        print(f"Issue with axuv {shot}")

    # NBI
    try:
        nbi = NBI(shot)
        ax = axs[1,0]
        ax.plot(nbi.time, nbi.V_Beam, label="NBI Voltage [kV]")
        ax.plot(nbi.time, nbi.I_Beam, label="NBI Current [A]")

        ax = axs[1,2]
        ax.plot(nbi.time, nbi.d2 *1e3, 'C0', label="Shine Through 2 [mA]")
        ax.plot(nbi.time, nbi.d5 *1e3, 'C4', label="Shine Through 5 [mA]")
        ax.plot(nbi.time, nbi.d10 *1e3, 'C2', label="Shine Through 10 [mA]")
    except:
        print(f"Issue with NBI {shot}")


    for a in np.ndarray.flatten(axs):
        a.legend(loc=1, fontsize=8)
        a.minorticks_on()
        a.grid(which='both')
        a.grid(which='minor', linestyle=":", linewidth=0.5)

    axs[-1,0].set_xlabel("time [ms]")
    axs[-1,1].set_xlabel("time [ms]")
    axs[-1,2].set_xlabel("time [ms]")
    axs[-1,0].set_xlim(-2,24)

    if tag == "":
        fig.suptitle(shot)
    else:
        fig.suptitle(tag)

    fig.tight_layout()

    if fout != "":
        plt.savefig(fout)
        plt.close()
        print(f"Saved {fout}")
    else:
        plt.show()


def vi_compare(shot, fout="", noPlot=False):
    fig,axs = plt.subplots(2,2, figsize=(10,8))
    
    
    try:
        spec = Spectrometer(shot)
        has_rot = True
        if noPlot:
            return has_rot
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
        has_rot = False
        if noPlot:
            return has_rot
    
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

    fig.suptitle(shot)
    
    if fout != "":
        plt.savefig(fout)
        plt.close()
        print(f"Saved {fout}")
    else:
        plt.show()

    return has_rot


# plot
def plot6(shot, axs=None, plotLimiter=True):
    shot = int(shot)

    if axs is None:
        fig, axs = plt.subplots(6,1,sharex=True, figsize=(11,10))

    # Bias plots
    try:
        bias = BiasPPS(shot)


        if plotLimiter:
            j1,j2 = get_time_index(bias.time, 7,9)
            V_ring = np.mean(bias.LVs[j1:j2])
            I_ring = np.mean(bias.LIs[j1:j2])
            axs[3].plot(bias.time, bias.LVs)
            axs[3].set_title(r"Limiter Potential [V]")
            axs[4].plot(bias.time, bias.LIs)
            axs[4].set_title(r"Limiter Current [A]")

        else:

            j1,j2 = get_time_index(bias.time, 7,9)
            V_ring = np.mean(bias.RVs[j1:j2])
            I_ring = np.mean(bias.RIs[j1:j2])
            axs[3].plot(bias.time, bias.RVs)
            axs[3].set_title(r"Ring Potential [V]")
            axs[4].plot(bias.time, bias.RIs)
            axs[4].set_title(r"Ring Current [A]")

    except:
        print(f"Issue with Bias {shot}")

    # ECH plots
    try:
        ech = ECH(shot)
        axs[0].plot(ech.time,ech.Fwg_filt, label=f"{shot}")
        axs[0].set_title("ECH Forward Power [kW]")

        j1,j2 = get_time_index(ech.time, 3,9)
        P_ech = np.mean(ech.Fwg_filt[j1:j2])
    except:
        print(f"Issue with ECH {shot}")

    try:
        flux = FluxLoop(shot)
        axs[1].plot(flux.time,flux.FL1/1000)
        axs[1].set_title("Flux Loop 1 [kMx]")

        win = 201; poly = 3
        F = savgol(flux.FL1/1e3, win,poly)
        j1,j2 = get_time_index(flux.time, 8,11)
        F1 = np.max(F[j1:j2])
    except:
        print(f"Issue with Flux Loop {shot}")

    try:
        intf = Interferometer(shot)
        axs[2].plot(intf.time, intf.linedens)
        axs[2].set_title(r"Line integrated density [m$^{-2}$]")

        j1,j2 = get_time_index(intf.time, 5,9)
        N_int = np.mean(intf.linedens[j1:j2])
    except:
        print(f"Issue with Interferometer {shot}")

    try:
        gas = Gas(shot)
        axs[5].plot(gas.t_asdex, gas.asdex_hi*1e3)
        axs[5].set_title(r"asdex [mTorr]")
    except:
        print(f"Issue with gas {shot}")

    print(f"{shot}, {V_ring:.1f}, {I_ring:.2f}, {P_ech:.1f}, {F1:.2f}, {N_int:.1e}")

    if axs is None:
        axs[0].legend(loc=4, fontsize=7)
        for a in axs:
            a.minorticks_on()
            a.grid(which='both')
            a.grid(which='minor', linestyle=":", linewidth=0.5)

        axs[-1].set_xlabel("time [ms]")
        axs[-1].set_xlim(-1,20)

        fig.tight_layout()
        #plt.savefig(f"out/shot-{shot}.png")
        plt.show()


