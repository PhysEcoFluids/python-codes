#!/usr/bin/env python3

'''
   sw_1d_nonhydro_fv.py
   Copyright 2023 Derek Steinmoeller and Marek Stastna

solve:
eta_t + (hu)_x = 0
u_t + u_u_x = - g eta_x + H^2/6 u_xxt

or equivalently,

h_t + (hu)_x = 0
(hu)_t + (hu^2)_x = -(0.5*g*h^2)_x + g*h*H_x + H^2/6 (hu)_xxt

Soln:
----    #hu = hu -  dt*hu_rhs


Use splitting, so that we first solve
(1) Hyperbolic step:

h^{n+1} = h^n - dt*F1(h, hu)_x
(hu)^{n+1} = (hu)^n - dt*F2(h, hu)_x + S(h),

where F is the flux vector, and S(h) is the source term.
Problem: Since S(h) might vary wildly between grid cells, so we need
to ensure no spurious flow is generated while applying the terms. It may be
included in a flux-differencing  formulation via
(F2_R* - F2_L*) = (F2_R + g*h*HR) - (F2_L + g*h*HL) and projected onto the
eigenstructure as-per the f-wave formulation (LeVeque).

Alternatively, we can use the approximate HLLC flux which can take into
account the contact discontinuity at bathymetry jumps at F2R* and F2L*.

(2) Elliptic step (Non-hydrostatic pressure terms):
(I - gamma*Dxx)*hu^{n+1} = hu^{*} - gamma*Dxx*hu^n, where hu^{*} is the
solution of the hyperbolic system at t = t_{n+1}

==
DS.
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def build_poisson1d(N, dx, K, left_bc_type="neumann", right_bc_type="neumann",
                    helmholtz=False):
    # Set up sparse Poisson operator, use triplets.
    rows = []
    cols = []
    data = []
    dx2 = dx*dx
    helm = 0.0
    if helmholtz:
        helm = 1.0
    for i in range(0, N):
        if i == 0:
            if left_bc_type == "neumann":
                rows.append(i)
                cols.append(i)
                data.append(K[i]*1.0/dx)

                rows.append(i)
                cols.append(i+1)
                data.append(-K[i]*1.0/dx)
            else:
                rows.append(i)
                cols.append(i)
                data.append(1.0)
        elif i == N-1:
            if right_bc_type == "neumann":
                rows.append(i)
                cols.append(i)
                data.append(-K[i]*1.0/dx)

                rows.append(i)
                cols.append(i-1)
                data.append(K[i]*1.0/dx)
            else:
                rows.append(i)
                cols.append(i)
                data.append(1.0)
        else:
            rows.append(i)
            cols.append(i)
            data.append(helm - K[i]*-2.0/dx2)

            rows.append(i)
            cols.append(i-1)
            data.append(-K[i]*1.0/dx2)

            rows.append(i)
            cols.append(i+1)
            data.append(-K[i]*1.0/dx2)

    return csc_matrix((data, (rows, cols)), shape=(N, N))


L = 4800
Nx = 4096

dx = L / Nx
x = np.linspace(0.5*dx, L - 0.5*dx, Nx)


g = 9.81
H = 12.5*np.ones((Nx,))
# H[int(Nx/2):] = 12.5

gamma = (H*H)/6

helmholtz_OP = build_poisson1d(Nx, dx, gamma, helmholtz=True)
linsolver = splu(helmholtz_OP)

eta = 3.5*np.exp(-((x-0.75*L)/(0.075*L))**2.0)

# eta = 5.5*(x/L)
# eta = 0.0
h = H + eta
c = np.sqrt(g*h)
u = (-c/h)*eta
hu = h*u
hu_hyd = hu + 0.0

CFL = 0.25
dt = CFL*np.min(dx/c)

NUMSTEPS = 100000

H_ghost = np.hstack((H[0], H, H[-1]))
nh_press = 0*eta
t = 0.0
for j in range(0, NUMSTEPS):

    eta = h - H
    u = hu / h

    if j % 200 == 0:
        plt.figure(1)
        plt.clf()
        plt.ion()
        # plt.plot(x[1:-1], hu[1:-1], '.-', x[1:-1], hu_hyd[1:-1], '.-')
        xFront = 0.75*L - np.sqrt(g*H)*t 
        plt.plot(x, eta, '.-', [xFront, xFront], [np.min(eta), np.max(eta)], 
                 '--')
        plt.title(f"t={int(t)}")
        # plt.ylim((-12.5, 3.5))
        plt.draw()
        plt.show()
        plt.pause(1e-3)
        plt.ioff()

    if t > 160:
        np.save(f"FV_N{Nx}_eta_t={int(t)}_nonlinear.npy", eta)
        exit()

    # Apply reflective BC's to solution in the ghost cells.
    eta_ghost = np.hstack((eta[0], eta, eta[-1]))
    h_ghost = H_ghost + eta_ghost
    hu_ghost = np.hstack((-hu[0], hu, -hu[-1]))

    # compute fluxes
    F1 = hu_ghost
    F2 = (hu_ghost*hu_ghost)/h_ghost + 0.5*g*h_ghost*h_ghost

    # Grab Right and Left states
    F1R = F1[1:]
    F1L = F1[:-1]
    F2R = F2[1:]
    F2L = F2[:-1]

    # Get state jumps
    dF1 = F1R - F1L
    # dF2 = F2R - F2L

    # above not general; deal with bathymetry in momentum equation:
    h_unique = 0.5*(h_ghost[1:] + h_ghost[:-1])
    dF2 = (F2R - g*h_unique*np.hstack((H, H[-1]))) \
        - (F2L - g*h_unique*np.hstack((H[0], H)))

    # Forme the eigenvectors
    u = hu_ghost / h_ghost
    c = np.sqrt(g*h_ghost)

    alphas = np.zeros((2, Nx+2))
    for j in range(0, Nx+1):
        # set up 2x2 matrix for each interface, Solve R*alpha_vec = dF_vec
        # to project onto eigenvecs.
        mat = np.ndarray((2, 2))
        mat[0, 0] = 1
        mat[0, 1] = 1
        mat[1, 0] = u[j] - c[j]
        mat[1, 1] = c[j] + u[j]

        RHS = np.row_stack((
            dF1[j],
            dF2[j],
        ))

        alphas[:, j] = np.squeeze(np.linalg.solve(mat, RHS))

    # Form the f-waves.
    W1L = alphas[0, :]*1
    W2L = alphas[0, :]*(-c + u)  # leftward wave.

    W1R = alphas[1, :]*1
    W2R = alphas[1, :]*(c + u)  # rightward wave.

    # Evolve waves.
    h -= (dt/dx)*(W1R[:-2] + W1L[1:-1])

    hu_rhs = (W2R[:-2] + W2L[1:-1])/dx

    nh_rhs = -dt*hu_rhs
    nh_rhs[0] = 0
    nh_rhs[-1] = 0

    OPtimesRHS = linsolver.solve(nh_rhs)
    # up1 = um1 + u_rhs # -- hydrostatic
    # hu = hu -  dt*hu_rhs
    # hu_hyd = hu - dt*hu_rhs
    hu = hu + OPtimesRHS

    t += dt
