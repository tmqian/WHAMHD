import numpy as np
import matplotlib.pyplot as plt
import sys

from WhamData import ECH, BiasPPS, Interferometer, FluxLoop, EdgeProbes, NBI, AXUV, EndRing, ShineThrough, Gas, adhocGas, Dalpha
from DataSpec import Spectrometer

from scipy.signal import savgol_filter as savgol
axuv_map = plt.cm.turbo(np.linspace(0,1,20))
probe_map = plt.cm.hsv(np.linspace(0,1,12))
alpha_map = plt.cm.plasma(np.linspace(0,1,6))

'''
This package uses the classes in WhamData
to plot multiple diagnostics.

plot9 - (3x3) figure, many traces per plot, single shot
plot6 - (6x1) figure, single trace per plot, multi or single shot
plot8 - (4x2) figure, contains NBI
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

def readLog(csv):
    '''
    Loads a csv shot log
    Expects YYMMDD-shotlog.csv

    First column of csv is 3-digit shot number
    All other columns are concatenated as the log entry
    '''

    with open(csv) as f:
        data = f.readlines()
    day = int(csv[:6])*1000
    
    log = {}
    for line in data:
        k = line.find(',')
        key = day + int(line[:k])
        val = line[k+1:].strip()
        log[key] = val

    return log

def plot9(shot, save="", plot_limiter_bias=False, tag=""):

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

        LV = bias.L_VLem
        LI = bias.L_ILem - LV/1.4
        RV = bias.R_VLem
        RI = bias.R_ILem 
        time = bias.time

        ax = axs[2,1]
        ax.plot(time, bias.L_Dem*1e3, 'C0', label="Limiter Demand [mV]")
        ax.plot(time, LV, 'C1', label="Limiter Potential [V]")
        ax.plot(time, LI, 'C5', label="Limiter Current [A]")

        ax = axs[2,2]
        ax.plot(time, bias.R_Dem*1e2, 'C0', label="Ring Demand [cV]")
        ax.plot(time, RV, 'C1', label="Ring Potential [V]")
        ax.plot(time, RI*10, 'C5', label="Ring Current [10x A]")
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
        r0 = 0.136 # hard coded
        ne = intf.linedens / (2*r0)
        axs[1,1].plot(intf.time, ne,'C2', label=r"Line Averaged Density [m$^{-3}$]")
        #axs[1,1].plot(intf.time, intf.linedens,'C2', label=r"Line Integrated Density [m$^{-2}$]")
    except:
        print(f"Issue with Interferometer {shot}")
    
    # Shine Through
    try:
        shine = ShineThrough(shot)
        j1,j2 = get_time_index(shine.time,1,14.5)
        axs[1,1].plot(shine.time[j1:j2], shine.nt[j1:j2], 'C1', label=r"Shine Through Density [m$^{-3}$]")
    except:
        print(f"Issue with ShineThrough {shot}")

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
    '''
    replacing with demand, because gauge signal is now low (25/02/20)
    '''
#    try:
#        gas = Gas(shot)
#        ax = axs[2,0]
#        ax.plot(gas.t_asdex, gas.asdex_hi*1e3, 'C2', label=r"asdex [mTorr]")
#        ax.plot(gas.t_asdex, gas.asdex_lo*1e3, 'C1')
#    except:
#        print(f"Issue with gas {shot}")
    try:
        gas = adhocGas(shot)
        ax = axs[2,0]
        ax.plot(gas.time, gas.ring, 'C1', label='ring')
        ax.plot(gas.time, gas.sec, 'C2', label='CC-S')
        ax.plot(gas.time, gas.nec, 'C3', label='CC-N')
    except:
        print(f"Issue with gas demand {shot}")
    
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
    axs[-1,0].set_xlim(-6,27)

    if tag == "":
        fig.suptitle(shot)
    else:
        fig.suptitle(tag)

    fig.tight_layout()

    if save != "":
        plt.savefig(save)
        plt.close()
        print(f"Saved {save}")
    else:
        plt.show()

def plot12(shot, save="", plot_limiter_bias=False, tag=""):

    fig, axs = plt.subplots(3,4,sharex=True, figsize=(13,9))

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

        LV = bias.L_VLem
        LI = bias.L_ILem - LV/1.4
        RV = bias.R_VLem
        if shot > 250319000:
            RI = bias.R_ILem  - RV/1.4
        else:
            RI = bias.R_ILem 
        time = bias.time

        ax = axs[2,1]
        ax.plot(bias.Ldem_T, bias.Ldem_V, 'C0', label="Limiter Demand [V]")
        ax.plot(time, LV, 'C1', label="Limiter Potential [V]")
        ax.plot(time, LI, 'C5', label="Limiter Current [A]")

        ax = axs[2,2]
        ax.plot(bias.Ldem_T, bias.Rdem_V, 'C0', label="Ring Demand [V]")
        ax.plot(time, RV, 'C1', label="Ring Potential [V]")
        #ax.plot(time, RI, 'C5', label="Ring Current [A]")
        ax.plot(time, RI*10, 'C5', label="Ring Current [10x A]")
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
        #axs[1,1].plot(intf.time, intf.linedens,'C2', label=r"Line Integrated Density [m$^{-2}$]")

        r0 = 0.136 # hard coded
        ne = intf.linedens / (2*r0)
        axs[1,1].plot(intf.time, ne,'C2', label=r"Line Averaged Density [m$^{-3}$]")
        eV = 1.609e-19
        # need to interpolate on to flux time
        flux.calcPressure()
        t = intf.time

        p1 = np.interp(t, flux.time, flux.P1)
        p2 = np.interp(t, flux.time, flux.P2)
        p3 = np.interp(t, flux.time, flux.P3)
        ax = axs[0,3]
        ax.plot(flux.time,flux.P1,'C3', label="Flux Loop 1 [Pa]")
        ax.plot(flux.time,flux.P2,'C2', label="Flux Loop 2 [Pa]")
        ax.plot(flux.time,flux.P3,'C0', label="Flux Loop 3 [Pa]")

        ax = axs[1,3]
        ax.plot(t,p1/ne/eV,'C3', label="Flux Loop 1 [eV]")
        ax.plot(t,p2/ne/eV,'C2', label="Flux Loop 2 [eV]")
        ax.plot(t,p3/ne/eV,'C0', label="Flux Loop 3 [eV]")
        ax.set_ylim(0,6000)
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
    '''
    replacing with demand, because gauge signal is now low (25/02/20)
    '''
#    try:
#        gas = Gas(shot)
#        ax = axs[2,0]
#        ax.plot(gas.t_asdex, gas.asdex_hi*1e3, 'C2', label=r"asdex [mTorr]")
#        ax.plot(gas.t_asdex, gas.asdex_lo*1e3, 'C1')
#    except:
#        print(f"Issue with gas {shot}")
    try:
        gas = adhocGas(shot)
        ax = axs[2,0]
        ax.plot(gas.time, gas.ring, 'C1', label='ring')
        ax.plot(gas.time, gas.sec, 'C2', label='CC-S')
        ax.plot(gas.time, gas.nec, 'C3', label='CC-N')
    except:
        print(f"Issue with gas demand {shot}")
    
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

        #shine = ShineThrough(shot)
        #ax = axs[1,1]
        #j1,j2 = get_time_index(shine.time,1,14.5)
        #ax.plot(shine.time[j1:j2], shine.nt[j1:j2], 'C1', label=r"Shine Through Density [m$^{-3}$]")

        ax = axs[1,2]
        ax.plot(nbi.time, nbi.d2 *1e3, 'C0', label="Shine Through 2 [mA]")
        ax.plot(nbi.time, nbi.d5 *1e3, 'C4', label="Shine Through 5 [mA]")
        ax.plot(nbi.time, nbi.d10 *1e3, 'C2', label="Shine Through 10 [mA]")
    except:
        print(f"Issue with NBI {shot}")

#    # Ion Probe
#    try:
#        ip = IonProbe(shot)
#        ax = axs[2,3]
#        ax.plot(ip.time, ip.Icol *1e6, label="Collector Current [uA]")
#    except:
#        print(f"Issue with ion probe {shot}")

    # D-alpha
    try: 
        da = Dalpha(shot)
        ax = axs[2,3]
        for k in [0,1,2]:
            ax.plot(da.time, da.data[k], color=alpha_map[k], label=f"H-alpha {k}")
    except:
        print(f"Issue with D-alpha {shot}")

    # gas
    try:
        ax = axs[2,3]
        ax.plot(gas.t_asdex, gas.asdex_hi*1e3)
        ax.set_title(r"asdex [mTorr]")
        ax.set_ylim(-0.1,1.5)
    except:
        print(f"Issue with gas {shot}")

    for a in np.ndarray.flatten(axs):
        a.legend(loc=1, fontsize=8)
        a.minorticks_on()
        a.grid(which='both')
        a.grid(which='minor', linestyle=":", linewidth=0.5)

    axs[-1,0].set_xlabel("time [ms]")
    axs[-1,1].set_xlabel("time [ms]")
    axs[-1,2].set_xlabel("time [ms]")
    axs[-1,3].set_xlabel("time [ms]")
    axs[-1,0].set_xlim(-6,27)

    if tag == "":
        fig.suptitle(shot)
    else:
        fig.suptitle(tag)

    fig.tight_layout()

    if save != "":
        plt.savefig(save)
        plt.close()
        print(f"Saved {save}")
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

    tag = f"{shot}"
    # Bias plots
    try:
        bias = BiasPPS(shot)

        if plotLimiter:
            j1,j2 = get_time_index(bias.time, 7,9)
            V_ring = np.mean(bias.L_VLem[j1:j2])
            I_ring = np.mean(bias.L_ILem[j1:j2])
            axs[3].plot(bias.time, bias.L_VLem)
            axs[3].set_title(r"Limiter Potential [V]")
            axs[4].plot(bias.time, bias.L_ILem)
            axs[4].set_title(r"Limiter Current [A]")

        else:

            j1,j2 = get_time_index(bias.time, 7,9)
            V_ring = np.mean(bias.R_VLem[j1:j2])
            I_ring = np.mean(bias.R_ILem[j1:j2])
            axs[3].plot(bias.time, bias.R_VLem)
            axs[3].set_title(r"Ring Potential [V]")
            axs[4].plot(bias.time, bias.R_ILem)
            axs[4].set_title(r"Ring Current [A]")

        tag += f", {V_ring:.1f}, {I_ring:.2f}"
    except:
        print(f"Issue with Bias {shot}")
        tag += f", -, -"

    # ECH plots
    try:
        ech = ECH(shot)
        axs[0].plot(ech.time,ech.Fwg_filt, label=f"{shot}")
        axs[0].set_title("ECH Forward Power [kW]")

        j1,j2 = get_time_index(ech.time, 3,9)
        P_ech = np.mean(ech.Fwg_filt[j1:j2])
        tag += f", {P_ech:.1f}"
    except:
        print(f"Issue with ECH {shot}")
        tag += f", -"

    try:
        flux = FluxLoop(shot)
        axs[1].plot(flux.time,flux.FL1/1000)
        axs[1].set_title("Flux Loop 1 [kMx]")

        win = 201; poly = 3
        F = savgol(flux.FL1/1e3, win,poly)
        j1,j2 = get_time_index(flux.time, 8,11)
        F1 = np.max(F[j1:j2])
        tag += f", {F1:.2f}"
    except:
        print(f"Issue with Flux Loop {shot}")
        tag += f", -"

    try:
        intf = Interferometer(shot)
        axs[2].plot(intf.time, intf.linedens)
        axs[2].set_title(r"Line integrated density [m$^{-2}$]")

        j1,j2 = get_time_index(intf.time, 5,9)
        N_int = np.mean(intf.linedens[j1:j2])
        tag += f", {N_int:.1e}"
    except:
        print(f"Issue with Interferometer {shot}")
        tag += f", -"

    try:
        gas = Gas(shot)
        axs[5].plot(gas.t_asdex, gas.asdex_hi*1e3)
        axs[5].set_title(r"asdex [mTorr]")
    except:
        print(f"Issue with gas {shot}")

    print(tag)

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


# plot
def plot8(shot, axs=None, plotLimiter=True):
    shot = int(shot)

    tag = f"{shot}"
    # Bias plots
    try:
        bias = BiasPPS(shot)

        if plotLimiter:

            V = bias.L_VLem
            I = bias.L_ILem - V/1.4

            axs[2,0].set_title(r"Limiter Potential [V]")
            axs[3,0].set_title(r"Limiter Current [A]")

        else:
            V = bias.R_VLem
            I = bias.R_ILem 

            axs[2,0].set_title(r"Ring Potential [V]")
            axs[3,0].set_title(r"Ring Current [A]")

        time = bias.time
        axs[2,0].plot(time, V)
        axs[3,0].plot(time, I)

        j1,j2 = get_time_index(time, 7,9)
        V_ref = np.mean(V[j1:j2])
        I_ref = np.mean(I[j1:j2])
        tag += f", {V_ref:.1f}, {I_ref:.2f}"
    except:
        print(f"Issue with Bias {shot}")
        tag += f", -, -"

    # ECH plots
    try:
        ech = ECH(shot)
        ax = axs[0,0]
        ax.plot(ech.time,ech.Fwg_filt, label=f"{shot}")
        ax.set_title("ECH Power [kW]")

        j1,j2 = get_time_index(ech.time, 3,9)
        P_ech = np.mean(ech.Fwg_filt[j1:j2])
        tag += f", {P_ech:.1f}"
    except:
        print(f"Issue with ECH {shot}")
        tag += f", -"


    # NBI
    try:
        nbi = NBI(shot)
        ax = axs[1,0]

        P = nbi.V_Beam * nbi.I_Beam
        ax.plot(nbi.time, P)
        ax.set_title("NBI Power [kW]")
    
        j1,j2 = get_time_index(nbi.time, 6,12)
        P_nbi = np.mean(P[j1:j2])
        tag += f", {P_nbi:.1f}"
    except:
        print(f"Issue with NBI {shot}")
        tag += f", -"

    # Flux
    try:
        flux = FluxLoop(shot)
        axs[0,1].plot(flux.time,flux.FL1/1000)
        axs[0,1].set_title("Flux Loop 1 [kMx]")
        axs[1,1].plot(flux.time,flux.FL2/1000)
        axs[1,1].set_title("Flux Loop 2 [kMx]")

        win = 201; poly = 3
        j1,j2 = get_time_index(flux.time, 8,11)

        F = savgol(flux.FL1/1e3, win,poly)
        F1 = np.max(F[j1:j2])
        tag += f", {F1:.2f}"

        F = savgol(flux.FL2/1e3, win,poly)
        F2 = np.max(F[j1:j2])
        tag += f", {F2:.2f}"

    except:
        print(f"Issue with Flux Loop {shot}")
        tag += f", -"
        tag += f", -"

    # intf
    try:
        intf = Interferometer(shot)

        if shot == 250214066:
            intf.fix_fringe_skip(t0=6, N=-1, back=True)
        if shot == 250214041:
            intf.fix_fringe_skip(t0=7.7, N=1, back=True)

        axs[2,1].plot(intf.time, intf.linedens)
        axs[2,1].set_title(r"Line integrated density [m$^{-2}$]")

        j1,j2 = get_time_index(intf.time, 5,9)
        N_int = np.mean(intf.linedens[j1:j2])
        tag += f", {N_int:.1e}"
    except:
        print(f"Issue with Interferometer {shot}")
        tag += f", -"

    try:
        gas = Gas(shot)
        ax = axs[3,1]
        ax.plot(gas.t_asdex, gas.asdex_hi*1e3)
        ax.set_title(r"asdex [mTorr]")
        ax.set_ylim(-0.1,1.5)
    except:
        print(f"Issue with gas {shot}")

    print(tag)

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


