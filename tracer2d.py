'''
   tracer2d.py
   Copyright 2023 Derek Steinmoeller and Marek Stastna
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
import time


Nx = 256
Ny = 256

Lx = 3e3
Ly = 3e3

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

dx = Lx / Nx
dy = Ly / Ny

xx, yy = np.meshgrid(x, y)

vel = 0.20

# incompressible field.

# constant 'wind'
# u = np.zeros((Ny, Nx)) + vel
# v = np.zeros((Ny, Nx))

# shear flow
u = vel*(np.tanh((yy-0.5*Lx)/(0.1*Ly)))
v = np.zeros((Ny, Nx))

fig, axs = plt.subplots(2, 2, figsize=(7, 6))

plt.set_cmap('jet')

subplot = axs[0][0].pcolor(xx/Lx, yy/Ly, u)
axs[0][0].set_ylabel('y')
axs[0][0].set_title('u')
divider = make_axes_locatable(axs[0][0])

cax = divider.append_axes('bottom', size='15%', pad=0.45)
fig.colorbar(subplot, cax=cax, orientation='horizontal', cmap='jet')


# solid body vortex
# u = -vel*(yy - 0.5*Ly)/Ly
# v = vel*(xx - 0.5*Lx)/Lx

Tlen = Lx / 20
T = 1.0*np.exp(-(((xx-0.5*Lx)/Tlen)**2.0 + ((yy-0.5*Ly)/Tlen)**2.0)) + \
    1.0*np.exp(-(((xx-0.5*Lx)/Tlen)**2.0 + ((yy-0.75*Ly)/Tlen)**2.0)) + \
    1.0*np.exp(-(((xx-0.5*Lx)/Tlen)**2.0 + ((yy-0.25*Ly)/Tlen)**2.0))

subplot = axs[0][1].pcolor(xx/Lx, yy/Ly, T)
axs[0][1].set_title('T @ t=0')
divider = make_axes_locatable(axs[0][1])

cax = divider.append_axes('bottom', size='15%', pad=0.45)
fig.colorbar(subplot, cax=cax, orientation='horizontal', cmap='jet')


spd_max = np.max(np.sqrt(u**2.0 + v**2.0).flatten())
dt = 0.45*np.min([dx, dy]) / spd_max

NUMSTEPS = 2000

uR = np.column_stack([u[:, 1:], u[:, -1]])
uL = np.column_stack([u[:, 0], u[:, :-1]])

vS = np.row_stack([v[1:, :], v[-1, :]])
vN = np.row_stack([v[0, :], v[:-1, :]])


@njit()
def upwind_flux_1d_numba(uM, uP, normal, Nx, Ny, tol=0):
    flux = np.zeros((Ny, Nx))
    neg_tol = -tol
    for i in range(Ny):
        for j in range(Nx):
            val = uM[i, j]
            scaled_val = val*normal

            if scaled_val > tol:
                flux[i, j] = val
            elif scaled_val < neg_tol:
                flux[i, j] = uP[i, j]
    return flux


def upwind_flux_1d_numpy(uM, uP, normal, Nx, Ny, tol=0):
    flux = np.zeros((Ny, Nx))
    scaled_uM = normal * uM

    msk = scaled_uM > tol
    flux[msk] = uM[msk]

    msk = scaled_uM < -tol
    flux[msk] = uP[msk]

    return flux


wall_t1 = time.perf_counter()
for j in range(0, NUMSTEPS):

    Tu = T*u
    Tv = T*v

    TuR = np.column_stack([Tu[:, 1:], Tu[:, -1]])
    TuL = np.column_stack([Tu[:, 0], Tu[:, :-1]])

    TvN = np.row_stack([Tv[1:, :], Tv[-1, :]])
    TvS = np.row_stack([Tv[0, :], Tv[:-1, :]])
    tol = 1.e-15

    Fjph = upwind_flux_1d_numba(Tu, TuR, 1, Nx, Ny)
    Fjmh = upwind_flux_1d_numba(Tu, TuL, -1, Nx, Ny)

    Giph = upwind_flux_1d_numba(Tv, TvS, -1, Nx, Ny)
    Gimh = upwind_flux_1d_numba(Tv, TvN, 1, Nx, Ny)

    T -= (dt/dx)*(Fjph - Fjmh) + (dt/dy)*(Gimh - Giph)

    if j % 100 == 0:
        plotnum = int(j/100) - 1
        subplot = axs[1][plotnum].pcolor(xx/Lx, yy/Ly, T)
        axs[1][plotnum].set_xlabel('x')
        if plotnum == 0:
            axs[1][plotnum].set_ylabel('y')
        axs[1][plotnum].set_title(f"T @ t={int(dt*j)}")
        divider = make_axes_locatable(axs[1][plotnum])

        if plotnum == 1:
            plt.draw()
            plt.savefig("2d_tracer_shear.png", dpi=300)
            print('done export')

wall_t2 = time.perf_counter()

print(wall_t2 - wall_t1)
