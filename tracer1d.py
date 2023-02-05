import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Define upwind flux operation given fields at - (uM) and + (uP)
# cell edges and a constant 'wind' speed, c and direction sign(c).
def upwind_flux_1d(uM, uP, normal, c, Nx, abs_tol=1e-15):
    flux = np.zeros((Nx,))
    normal_velocity = normal * c

    msk = normal_velocity >= abs_tol
    flux[msk] = uM[msk]

    msk = normal_velocity < -abs_tol
    flux[msk] = uP[msk]

    return flux


# Define central flux operation given fields at - (uM) and + (uP)
# cell edges.
def central_flux_1d(uM, uP, normal, c, Nx):
    flux = np.zeros((Nx,))
    flux = 0.5*(uM + uP)
    return flux


Nx = 128
Lx = 3000

dx = Lx / Nx
x = np.linspace(0.5*dx, Lx-0.5*dx, Nx)

# Set flow velocity, where positive is left-to-right flow.
c = 0.2

# Example 1:
T = np.cos(2*np.pi*x/Lx)

# Example 2:
# tracer_scale = Lx / 10
# T = 1.0*np.exp(-((x-0.6*Lx)/tracer_scale)**2.0)

# Should scale the grid-level time-scale by some C (e.g., 0.45) 
# between 0 and 1 (non-inclusive).
dt = 0.45*dx/np.abs(c)

NUMSTEPS = 2000
NUMOUTS = 50
OUT_INTERVAL = int(NUMSTEPS/NUMOUTS)

plt.figure()
t = 0.0
output_times = []
for n in range(0, NUMSTEPS):
    cT = c*T
    Tin = np.cos(2*np.pi*(-0.5*dx-c*t)/Lx)
    cTL = np.concatenate([[c*Tin], cT[:-1]])
    cTR = np.concatenate([cT[1:], [c*T[-1]]])

    Fjph = upwind_flux_1d(cT, cTR, c,  1.0, Nx)
    Fjmh = upwind_flux_1d(cT, cTL, c, -1.0, Nx)

    # Always use upwinding at the cell that lies against i
    # the right-hand boundary.
    if c >= 0:
        Fjph[-1] = cT[-1]
        Fjmh[-1] = cTL[-1]
    else:
        Fjph[-1] = cTR[-1]
        Fjmh[-1] = cT[-1]

    T -= (dt/dx)*(Fjph - Fjmh)
    t += dt

    if n % OUT_INTERVAL == 0:
        plt.plot(x, T)
        plt.ylim([-1.005, 1.005])
        plt.ylabel('$\overline{T}$')
        plt.xlabel('x (m)')
        plt.grid('on')
        plt.ion()
        plt.draw()
        plt.show()
        plt.ioff()
        output_times.append(int(t))
        if len(output_times) == 6:
            legend_entries = (f"t={output_times[0]}", f"t={output_times[1]}",
                              f"t={output_times[2]}", f"t={output_times[3]}",
                              f"t={output_times[4]}", f"t={output_times[5]}")

            plt.legend(legend_entries)
            with PdfPages("1d_upwinding_cosine.pdf") as export_pdf:
                export_pdf.savefig()
            exit()

        plt.pause(0.5)
