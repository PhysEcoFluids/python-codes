import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


def fourier_diff_x(u, ks):
    return np.real(ifft(1.j*ks*fft(u)))


def fourier_op1d(u, op):
    return np.real(ifft(op*fft(u)))


# Set number of grid points and length of domain (Lx).
Nx = 512
Lx = 4800
pi = np.pi

dx = Lx/Nx
x = np.arange(0, Nx)*dx

dk = 2*pi/Lx  # Nyquist wavenumber

# Define wavenumbers
ks = np.hstack((np.arange(0, Nx/2+1), np.arange(-Nx/2+1, 0)))*dk

# Set up filter.
kmax = np.max(ks)
cutoff = 0.65
kcrit = kmax*cutoff
f_order = 4 
epsf = 1e-16

myfilt = np.ones((Nx, ))
mymask = (np.abs(ks) < kcrit)
myfilt = myfilt*(mymask +
                 (1 - mymask) *
                 np.exp(np.log(epsf)*(1.*(np.abs(ks)-kcrit)/(np.max(ks)-kcrit)
                                      )**f_order))
g = 9.81
H = 12.5
gamma = (H*H)/6

eta = 3.5*np.exp(-((x-0.75*Lx)/(0.075*Lx))**2.0)
h = H + eta
c = np.sqrt(g*h)
u = (-c/h)*eta
hu = h*u

NHOP = np.ones((Nx,)) + gamma*ks*ks
NHOP = 1/NHOP

CFL = 0.25
dt = CFL*np.min(dx/c)
t = 0

j = 0
plt.figure()
etam1 = eta.copy()
etap1 = eta.copy()
hm1 = etam1 + H
hp1 = etap1 + H
um1 = u.copy()
up1 = u.copy()

while t < 2500:
    if j % 25 == 0:
        xFront = 0.75*Lx - np.sqrt(g*H)*t
        plt.clf()
        plt.plot(x, eta, [xFront, xFront], [np.min(eta), np.max(eta)], '--')
        plt.ion()
        plt.title(f"t={int(t)}")
        plt.draw()
        plt.show()
        plt.ioff()
        plt.savefig(f"frame{str(j).zfill(7)}.png", format="png")
        plt.pause(1e-2)

    if j == 0:
        hp1 = h - dt*fourier_diff_x(hu, ks)
        u_rhs = -dt*u*fourier_diff_x(u, ks) - dt*g*fourier_diff_x(eta, ks)
        # up1 = u + u_rhs # -- hydrostatic
        up1 = u + fourier_op1d(up1, NHOP)
    else:
        hp1 = hm1 - 2*dt*fourier_diff_x(hu, ks)
        u_rhs = -2*dt*u*fourier_diff_x(u, ks) - 2*dt*g*fourier_diff_x(eta, ks)
        # up1 = um1 + u_rhs # -- hydrostatic
        up1 = um1 + fourier_op1d(u_rhs, NHOP)

    etap1 = hp1 - H

    if t > 160:
        np.save(f"fourier_N{Nx}_eta_t={int(t)}_nonlinear.npy", eta)
        exit()

    j += 1
    t += dt

    # filter Fourier fields
    etap1 = fourier_op1d(etap1, myfilt)
    up1 = fourier_op1d(up1, myfilt)

    hp1 = H + etap1
    hup1 = hp1*up1

    if np.any(h < 0):
        print("negative h")
        exit(-2)

    if np.isnan(dt):
        print("nan time-step")
        exit(-1)

    hm1 = h
    h = hp1
    etam1 = eta
    eta = etap1
    um1 = u
    u = up1
    hum1 = hu
    hu = hup1
