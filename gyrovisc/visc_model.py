import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

'''
Updated 11 August 2025
works for gyrovisc model A
'''

_save = False
plot_IV = True
plot_rotation = True
plot_model=True

def save_plot(fig, name):
    fig.savefig(name)
    print(f"saved {name}")

from scipy.integrate import cumulative_trapezoid as trap
# import mac color scheme
mac = { 
        "red-orange" :(246/255, 75/255, 47/255),    
        "blue" :(0/255, 120/255, 210/255),    
        "green" :(106/255, 194/255, 72/255), 
}

e = 1.609e-19 # C
me = 9.11e-31 # kg
mi = me * 1840
eps0 = 8.85e-12 # SI

def profile(r, A=1, B=0.1,k=2):
    '''
    assumes r is (0,1)
    A = r(0), B = r(1)
    k is peaking factor
    '''

    return (A-B)*(1 - r**k) + B


class ViscModel:

    def __init__(self, n0=1e19,
                       n1=2e18,
                       Te0=50,
                       Te1=5,
                       Ti0=50,
                       Ti1=5,
#                 a = 0.025, # inner boundary, original
#                 a = 0.0425, # map 7.5 cm disk outer radius by sqrt (2500/800 G)
                 a = 0.03, 
                 B=0.25,
                 L=0.5,
                 I=10,
                 ):

        self.n0 = n0
        self.Ti0 = Ti0
        self.Te0 = Te0

        ### init profiles
        rax = np.linspace(0,1,100)
        r = rax * 0.13 + 1e-15 # m
        
        ne = profile(rax, A=n0, B=n1, k=10)
        Te = profile(rax, A=Te0, B=Te1, k=2)
        Ti = profile(rax, A=Ti0, B=Ti1, k=2)

        print(f"I (A): {I}")
        print(f"L (m): {L}")
        print(f"mi/me: {mi/me:.0f}")

        self.Ti = Ti
        self.Te = Te
        self.ne = ne
        self.L = L
        self.B = B
        self.a0 = a
        self.I = I


        ### level 2
        pi = ne*Ti * e # Pa
        pe = ne*Te * e
        
        # need wpe/2pi < f_ECH
        wpe = np.sqrt( e**2 * ne / me /eps0)
        
        # tau ei, eq 3.183 https://farside.ph.utexas.edu/teaching/plasma/Plasma/node41.html
        lamb = 17
        tau_ei = (6*np.sqrt(2*np.pi**3*me*(Te*e)**3)*eps0**2) / (lamb*e**4*ne)
        nu_ei = 1/tau_ei/2
        
        Omega_ci = e*B/mi
        
        k = I/2/np.pi/L # for Jr

        self.nu_ei = nu_ei
        self.nu_ii = nu_ei * np.sqrt(me/mi)
        self.pi = pi
        self.pe = pe
        
        ###
        print(f"ion cyc (Hz): {Omega_ci/2/3.1415:e}")
        print(f"k = rJ_r (A/m): {k}")

        ### Level 3
        sigma = e**2 * ne / me / nu_ei
        #sigma = eps0 * wpe**2 / nu_ei
        eta = pi/Omega_ci
        
        Jr = np.where(r < a, 0, k/r)
        
        gradPi = np.gradient(pi, r)
        gradPe = np.gradient(pe, r)
        Jt = (gradPi + gradPe)/B

        self.gradPi = gradPi
        self.gradPe = gradPe
        self.Jt = Jt
        self.Jr = Jr
        self.eta = eta
        self.sigma = sigma
        self.nu_ei = nu_ei
        

        ### Level 4
        Ver = - Jt/sigma/B
        
        f = np.where(r<a,0, (1-a/r)/eta)
        I_visc = trap(f, r, initial=0) # viscosity integral
        Vit = -k*B * I_visc
        
        Vir = Jr/e/ne + Ver
        Vet = Vit - Jt/e/ne

        self.Vit = Vit
        self.Vir = Vir
        self.Vet = Vet
        self.Ver = Ver
        
        
        ### Level 5
        Er = - Vit*B + Jr/sigma + gradPi/e/ne
        Er2 = - Vet*B + Jr/sigma - gradPe/e/ne
        phi = -trap(Er,r, initial=0) 

        self.Er = Er
        self.phi = phi

        ### Level 6
        
        # interesting to note phi(0) < phi(a), there may be a local max
        V_bias = phi[0] - phi[-1]
        
        Sigma = 1/trap( -Vit*B + Jr/sigma,r, initial=0)[-1]
        Vp = -trap(gradPi/e/ne, r, initial=0)[-1]
        
        print("Sigma:", Sigma)
        print("Vp:", Vp)
        
        
        vax = np.linspace(-500,700,50)
        Iax = (2*np.pi*L*Sigma) * (vax + Vp)


        ### Level 7
        C_rot = -Sigma * I_visc # radial function : maps volts to v_itheta

        self.vax = vax # voltage axis
        self.Iax = Iax
        self.rax = rax
        self.radius = r
        self.Sigma = Sigma
        self.Vp = Vp
        self.k = 2*np.pi*L
        # I = k Sigma ( V + Vp)

        #self.pe = pe
        self.I_visc = I_visc
        self.C_rot = C_rot


        # compute div Pi theta, after the fact
        divPi = 1/r * np.gradient(r*eta*np.gradient(Vit, r),r)
        self.divPi = divPi


    def plotTiVi(self, axs):
        '''
        Plots Ti, eta, Vi_theta, Jr
        to explore relation between FLR and viscosity
        '''

        axs[0].plot(self.rax, self.Ti, label=f"Ti = {self.Ti0}")
        axs[1].plot(self.rax, self.eta)
        axs[2].plot(self.rax, self.Vit)
        axs[3].plot(self.rax, self.Jr)

    def plotProfile(self):

        fig,axs = plt.subplots(1,1)
        axs.plot(self.rax, self.Ti/self.Ti0, label=r'$T_{i0}$ = ' + f'{self.Ti0} eV')
        axs.plot(self.rax, self.Te/self.Te0, ls='--', label=r'$T_{e0}$ = ' + f'{self.Te0} eV')
        axs.plot(self.rax, self.ne/self.n0, label=r'$n_{e0}$ = ' + f'{self.n0} ' + r'm$^{-3}$')
        axs.grid()
        axs.legend()
        axs.set_ylim(bottom=0)

        axs.set_title("Model (n,T) Profiles", fontsize=16)
        axs.set_ylabel("normalized profile", fontsize=14)
        axs.set_xlabel("normalized radius", fontsize=14)

        if _save:
            save_plot(fig, "gyrovisc-modelA-1-profile.pdf")


    def plotEr(self):

        fig,axs = plt.subplots(2,1, figsize=(5,7))
        axs[0].plot(self.rax, self.Er)

        offset = np.min(self.phi)
        axs[1].plot(self.rax, self.phi-offset, label=f"I = {10} A")
        axs[1].legend()

        axs[0].set_title(r"Radial Electric Field [V/m]", fontsize=14)
        axs[1].set_title(r"Potential [V]", fontsize=14)
        axs[1].set_xlabel(r"normalized radius", fontsize=14)

        for a in axs:
            a.grid()

        if _save:
            save_plot(fig, "gyrovisc-modelA-6-Er.pdf")

    def plotPlasmaParams(self):

        fig,axs = plt.subplots(3,1, figsize=(5,7))
        axs[0].plot(self.rax, self.nu_ii)
        axs[1].plot(self.rax, self.sigma)
        axs[2].plot(self.rax, self.eta)

        axs[0].set_title(r"$\nu_{ii}$ Collisionality [Hz]", fontsize=14)
        axs[1].set_title(r"$\sigma$ Conductivity [($\Omega \cdot$m)$^{-1}$]", fontsize=14)
        axs[2].set_title(r"$\eta_\wedge^i$ Viscosity [Pa$\cdot$s]", fontsize=14)

        for a in axs:
            a.grid()
            a.set_ylim(bottom=0)


        fig.suptitle("Plasma Parameters", fontsize=12)
        #axs.set_ylabel("normalized profile", fontsize=14)
        axs[-1].set_xlabel("normalized radius", fontsize=14)

        fig.tight_layout()

        if _save:
            save_plot(fig, "gyrovisc-modelA-2-plasma.pdf")

    def plotForce(self):
        '''
        a pair of 2x2 plots
        in momentum units
        '''

        fig,axs = plt.subplots(2,2, figsize=(8,8))
        axs[0,0].set_title(r"ion radial force [N/m$^{3}$]", fontsize=14)
        axs[0,1].set_title(r"ion azimuthal force [N/m$^{3}$]", fontsize=14)
        axs[1,0].set_title(r"electron radial force [N/m$^{3}$]", fontsize=14)
        axs[1,1].set_title(r"electron azimuthal force [N/m$^{3}$]", fontsize=14)


        ne = self.ne
        B = self.B

        axs[0,0].plot(self.rax, -self.gradPi, label=r"$-\nabla p_i$")
        axs[0,0].plot(self.rax, e*ne*self.Er, label=r"$e n_e E_r$")
        axs[0,0].plot(self.rax, e*ne*self.Vit*B, label=r"$e n_e V_{i\theta} B_z$")
        axs[0,0].plot(self.rax, -self.nu_ei*me*self.Jr/e,label=r"$ - \nu_{ei} m_e J_r / e$")

        axs[1,0].plot(self.rax, -self.gradPe, label=r"$-\nabla p_e$")
        axs[1,0].plot(self.rax, -e*ne*self.Er, label=r"$-e n_e E_r$")
        axs[1,0].plot(self.rax, -e*ne*self.Vet*B, label=r"$-e n_e V_{e\theta} B_z$")
        axs[1,0].plot(self.rax, self.nu_ei*me*self.Jr/e,label=r"$\nu_{ei} m_e J_r / e$")

        axs[0,1].plot(self.rax[:-2], -self.divPi[:-2], label=r"$-(\nabla \cdot \Pi)_\theta$")
        axs[0,1].plot(self.rax, -e*ne*self.Vir*B, label=r"$-e n_e V_{ir} B_z$")
        axs[0,1].plot(self.rax, -self.nu_ei*me*self.Jt/e,label=r"$-\nu_{ei} m_e J_\theta / e$")

        axs[1,1].plot(self.rax, e*ne*self.Ver*B, label=r"$e n_e V_{er} B_z$")
        axs[1,1].plot(self.rax, self.nu_ei*me*self.Jt/e,label=r"$\nu_{ei} m_e J_\theta / e$")

        for a in axs.flatten():
            a.legend()
            a.grid()

        fig.suptitle("force balance", fontsize=12)
        axs[-1,0].set_xlabel("normalized radius", fontsize=14)
        axs[-1,1].set_xlabel("normalized radius", fontsize=14)

        fig.tight_layout()

        if _save:
            save_plot(fig, "gyrovisc-modelA-4-force.pdf")

    def plotVelocity(self):
        '''
        a pair of 2x2 plots
        in velocity units
        '''

        fig,axs = plt.subplots(2,2, figsize=(8,8))
        axs[0,0].set_title("ion azimuthal velocity [m/s]", fontsize=14)
        axs[1,0].set_title("electron azimuthal velocity [m/s]", fontsize=14)
        axs[0,1].set_title("ion radial velocity [m/s]", fontsize=14)
        axs[1,1].set_title("electron radial velocity [m/s]", fontsize=14)

        ne = self.ne
        B = self.B
        sigma = self.sigma

        axs[0,0].plot(self.rax, self.Vit, 'k', label=r"$V_{i\theta}$")
        axs[0,0].plot(self.rax, -self.Er/B, '--', label=r"$-E_r/B_z$")
        axs[0,0].plot(self.rax, self.Jr/sigma/B, '--', label=r"$J_r/\sigma B_z$")
        axs[0,0].plot(self.rax, self.gradPi/(e*ne*B), '--', label=r"$\nabla p_i/en_e B_z$")

        axs[1,0].plot(self.rax, self.Vet, 'k', label=r"$V_{e\theta}$")
        axs[1,0].plot(self.rax, -self.Er/B, '--', label=r"$-E_r/B_z$")
        axs[1,0].plot(self.rax, self.Jr/sigma/B, '--', label=r"$J_r/\sigma B_z$")
        axs[1,0].plot(self.rax, -self.gradPe/(e*ne*B), '--', label=r"$-\nabla p_e/en_e B_z$")

        axs[0,1].plot(self.rax, self.Vir, 'k', label=r"$V_{ir}$")
        axs[0,1].plot(self.rax, -self.Jt/sigma/B, 'C1--', label=r"$-J_\theta / \sigma B_z$")
        axs[0,1].plot(self.rax[:-2], -(self.divPi/e/ne/B)[:-2], 'C4--', label=r"$-(\nabla \cdot \Pi)_\theta / e n_e B_z$")

        axs[1,1].plot(self.rax, self.Ver, 'k', label=r"$V_{er}$")
        axs[1,1].plot(self.rax, -self.Jt/sigma/B, 'C1--', label=r"$-J_\theta / \sigma B_z$")

        for a in axs.flatten():
            a.legend()
            a.grid()

        fig.suptitle("transverse flow", fontsize=12)
        axs[-1,0].set_xlabel("normalized radius", fontsize=14)
        axs[-1,1].set_xlabel("normalized radius", fontsize=14)

        fig.tight_layout()

        if _save:
            save_plot(fig, "gyrovisc-modelA-3-velocity.pdf")

    def plotCurrent(self):
        fig,axs = plt.subplots(2,2, figsize=(8,8))

        axs[0,0].set_title("radial", fontsize=14)
        axs[0,1].set_title("azimuthal",fontsize=14)

        ne = self.ne

        axs[0,0].plot(self.rax, self.Jr, 'k', label=r"$J_{r}$")
        axs[0,0].plot(self.rax, self.Vir*e*ne, 'C1--', label=r"$e n_eV_{ir} $")
        axs[0,0].plot(self.rax, -self.Ver*e*ne, 'C0--', label=r"$-e n_eV_{er}$")

        axs[0,1].plot(self.rax, self.Jt, 'k', label=r"$J_{\theta}$")
        axs[0,1].plot(self.rax, self.Vit*e*ne, 'C1--', label=r"$e n_e V_{i\theta}$")
        axs[0,1].plot(self.rax, -self.Vet*e*ne, 'C0--', label=r"$-e n_eV_{e\theta}$")

        axs[1,0].plot(self.rax, self.Jr/e/ne, 'k', label=r"$J_{r}/e n_e$")
        axs[1,0].plot(self.rax, self.Vir, 'C1--', label=r"$V_{ir}$")
        axs[1,0].plot(self.rax, -self.Ver, 'C0--', label=r"$-V_{er}$")

        axs[1,1].plot(self.rax, self.Jt/e/ne, 'k', label=r"$J_{\theta}/en_e$")
        axs[1,1].plot(self.rax, self.Vit, 'C1--', label=r"$V_{i\theta}$")
        axs[1,1].plot(self.rax, -self.Vet, 'C0--', label=r"$-V_{e\theta}$")

        for a in axs.flatten():
            a.legend()
            a.grid()

        fig.suptitle("plasma current", fontsize=12)
        axs[-1,0].set_xlabel("normalized radius", fontsize=14)
        axs[-1,1].set_xlabel("normalized radius", fontsize=14)
        axs[0,0].set_ylabel(r"current [A/m$^2$]", fontsize=14)
        axs[1,0].set_ylabel(r"flow [m/s]", fontsize=14)

        fig.tight_layout()

        if _save:
            save_plot(fig, "gyrovisc-modelA-5-current.pdf")

v_ring, I_ring = np.loadtxt("IV.csv", delimiter=",", skiprows=1).T
v2 = ViscModel(n0=4e18, Ti0=25, Te0=25)
v3 = ViscModel(n0=9e18, Ti0=40, Te0=40, n1=3e18, Ti1=10, Te1=10)
#v3 = ViscModel(n0=9e18, Ti0=40, Te0=40) # old, "works"


if plot_model:
    v3.plotProfile()
    v3.plotPlasmaParams()
    v3.plotVelocity()
    v3.plotForce()
    v3.plotCurrent()
    v3.plotEr()

shift_A = 3 # factor for phi = AT

if plot_IV:
    fig,axs = plt.subplots(1,1, figsize=(6,6))
    
    
    # I(V) plot
    axs.plot(v_ring, I_ring, 'o', color=mac['red-orange'], label="experiment")
    for v,c in zip([v3,v2], ['C1', 'C0']):
        shift = shift_A * v.Te0 
        # I(V) plot
        tag = f"model: ne = {v.n0}, Ti = {v.Ti0}, Te = {v.Te0}"
        #tag = f"(n, Ti, Te) = {v.n0}, {v.Ti0}, {v.Te0}"
        axs.plot(v.vax + shift, v.Iax, color=c, label=tag)
        axs.plot(v.vax , v.Iax, ':', lw=0.7, color=c)
    axs.set_title(r"I(V$_{bias}$) :" + f" L = {v.L}")
    axs.grid()
    axs.legend()
    axs.set_xlabel("Bias Voltage", fontsize=12)
    axs.set_ylabel("Bias Current", fontsize=12)
    axs.set_xlim(-120,620)
    axs.set_ylim(-1,6)


    if _save:
        save_plot(fig, "gyrovisc-fit.pdf")


# rotation plot
if plot_rotation:
    fig2,ax2 = plt.subplots(1,2, figsize=(12,6))

    # (8/8) this experimental data load is pretty confusing
    # ah I see. This has all shots. All biases and one radial location.
    v_disc, Vi_m78 = np.loadtxt("bias-rot-m78.csv", delimiter=",", skiprows=1).T
    # index near 7.2 cm
    ax2[0].scatter(v_disc, -Vi_m78, color='C4', label="experimental data")
    #ax2[0].scatter(v_disc, -Vi_m78, color='C4', label="experimental data (r = 7.2 cm)")
    
    # (8/8) this json dict is much more carefully structured
    # this has 5 biases, but all radial locations.
    # first I need to match them
    with open("velocity_profile.json", "r") as f:
        raw = json.load(f)
    vdata_in = {k: np.array(v) for k, v in raw.items()}
    bias = np.array(list(vdata_in.keys())[:-1], float)
    bias_str = np.array(list(vdata_in.keys())[:-1])
    chord_radius = vdata_in['radial_chords']
    velocity_arr = np.array([vdata_in[s] for s in bias_str])

    # compare json with csv data
    #ax2[0].plot(bias, -velocity_arr[:,0], 'C0*', ms=10, label=f"r = {chord_radius[0]}")
    #ax2[0].plot(bias, velocity_arr[:,-1], 'C1^', ms=10, label=f"r = {chord_radius[-1]}")

    _shift = True
    v = v3
    

    i = np.argmin(np.abs(v.radius - 0.072)) 
    for v in [v2,v3]:
        if _shift:
            shift = shift_A * v.Te0
        else:
            shift = 0
        tag = f"(n, Ti, Te) = {v.n0}, {v.Ti0}, {v.Te0}"
        ax2[0].plot(v.vax, -v.C_rot[i] * (v.Vp + v.vax + shift), '--', label=tag)
    
    ax2[0].legend()
    ax2[0].set_xlim(-240,640)
    ax2[0].set_ylim(-8000,24000)
    
    ax2[0].set_xlabel("Bias Voltage [V]", fontsize=14)
    ax2[0].set_ylabel("Ion Rotation Velocity [m/s]", fontsize=14)

    # plot raw data
    r_chords = vdata_in.pop('radial_chords')/10 # cm
    for j,varr in enumerate(vdata_in.values()):
        ax2[1].plot(r_chords, varr, color=f"C{j}")
        ax2[1].plot(-r_chords, -varr,'o-', color=f"C{j}")
        if j==4:
            ax2[1].plot(r_chords, varr,'o', color=f"C{j}", label="experimental data")
        else:
            ax2[1].plot(r_chords, varr,'o', color=f"C{j}")

    # plot model
    idx = [np.argmin(np.abs(v.radius - r)) for r in v.radius]
    for j,V_str in enumerate(vdata_in.keys()):
    #for j,V in enumerate([-200, 0, 200, 400, 600]):
        V = int(V_str)

        if _shift:
            shift = shift_A * v.Te[idx]  # profile dependent shift
        else:
            shift = 0
        #V = 0

        model_vel = -v.C_rot * (v.Vp + (V+shift))
        ax2[1].plot(v.radius*100, model_vel, '--', label=f"{V} V", color=f"C{j}")

        # highlight data
        #ax2[0].plot(V, -vdata_in[V_str][0], f"C{j}x", ms=10)
   
    ax2[1].set_xlabel("Chord Radius [cm]", fontsize=14)
    ax2[1].set_ylabel("Ion Rotation Velocity [m/s]", fontsize=14)

    ax2[1].axvline(v3.a0*100, color='k', ls='--', lw=0.7, label=f"a = {v3.a0}")

    ax2[1].set_ylim(-1e4,3e4)
    ax2[1].set_xlim(-1, 10)
    #ax2[1].set_xlim(-0.01, 0.1)
    for a in ax2:
        a.grid()

    ax2[1].legend()
    #if _shift:
    #    fig2.suptitle("shift")
    #else:
    #    fig2.suptitle("shift disabled")

    fig2.tight_layout()

    if _save:
        save_plot(fig2, "gyrovisc-rotation.pdf")


plt.show()
