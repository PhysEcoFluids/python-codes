#!/usr/bin/python3
'''
Copyright (C) 2017-2019  Waterloo Quantitative Consulting Group, Inc.
See COPYING and LICENSE files at project root for more details.
'''

import numpy as np
import pyblitzdg as dg
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.interpolate import splev, splrep, interp1d, griddata

def positivityPreservingLimiter2D(h, hu, hv):
    Np, K = h.shape
    hmin = np.tile(np.min(h, axis=0), (Np, 1))
    hmin[hmin < 1e-3] = 1e-3

    hmean = np.tile(np.mean(h, axis=0), (Np, 1))

    theta = np.ones((Np, K))
    theta = hmean / (hmean - hmin + 1e-4)
    
    theta[theta > 1] = 1.0
    humean = np.tile(np.mean(hu, axis=0), (Np, 1))
    hvmean = np.tile(np.mean(hv, axis=0), (Np, 1))

    h  = hmean  + theta*(h  - hmean)
    hu = humean + theta*(hu - humean)
    hv = hvmean + theta*(hv - hvmean)

    return h, hu, hv


def minmod(a, b):
    soln = np.zeros(a.shape)
    for i, _ in enumerate(a):
        if a[i] < b[i] and a[i]*b[i] > 0:
            soln[i] = a[i]
        elif b[i] < a[i] and a[i]*b[i] > 0:
            soln[i] = b[i]
        else:
            soln[i] = 0.0
    
    return soln



def surfaceReconstruction(etaM, hM, etaP, hP):
    # get bed elevations
    zM = etaM - hM
    zP = etaP - hP

    dz = (zP - 0.5*minmod(zP - zM, 1e-3*np.ones(zM.shape))) - (zM + 0.5*minmod(zM-zP, -1e-3*np.ones(zM.shape)))


    # flux limit
    #etaCorrM = zP - zM - dz
    #for i,_ in enumerate(etaCorrM):
    #    if etaCorrM[i] > (etaP[i] - etaM[i]):
    #        etaCorrM[i] = etaP[i] - etaM[i]

    #    if etaCorrM[i] < 0:
    #        etaCorrM[0] = 0.0    
    
    #etaM += etaCorrM

    etaCorrP = zM - zP - dz
    for i, _ in enumerate(etaCorrP):
        if etaCorrP[i] > (etaM[i] - etaP[i]):
            etaCorrP[i] = etaM[i] - etaP[i]
            
        if etaCorrP[i] <= 0:
            etaCorrP[i] = 0.0
        else:
            etaP[i] += etaCorrP[i]


    # Get corrected bed elevation
    #zM = etaM - hM
    zP = etaP - hP

    # enforce non-negativity
    maxz = zM
    for i, _ in enumerate(zM):
        if zP[i] > zM[i]:
            maxz[i] = zP[i]

    hM = etaM - maxz
    hM[hM <= 1e-3] = 1e-3
    hP = etaP - maxz
    hP[hP <= 1e-3] = 1e-3

    return hM, hP


def sw2dComputeFluxes(h, hu, hv, hN, g, H):
    #h equation
    F1 = hu
    G1 = hv

    # Get velocity fields
    u = hu / h
    v = hv / h

    # hu equation

    F2 = hu*u + 0.5*g*h*h
    G2 = hu*v

    # hv equation
    F3 = G2
    G3 = hv*v + 0.5*g*h*h

    # N (tracer) equation
    F4 = hN*u
    G4 = hN*v

    return ((F1,F2,F3,F4),(G1,G2,G3,G4))

def sw2dComputeRHS(h, hu, hv, hN, zx, zy, g, H, f, ctx, vmapM, vmapP):
    #vmapM = ctx.vmapM
    #vmapP = ctx.vmapP
    BCmap = ctx.BCmap
    nx = ctx.nx
    ny = ctx.ny
    rx = ctx.rx
    sx = ctx.sx
    ry = ctx.ry
    sy = ctx.sy
    Dr = ctx.Dr
    Ds = ctx.Ds
    Nfp = ctx.numFacePoints

    Lift = ctx.Lift
    Fscale = ctx.Fscale

    hC = h.flatten('F')
    huC = hu.flatten('F')
    hvC = hv.flatten('F')
    hNC = hN.flatten('F')
    nxC = nx.flatten('F')
    nyC = ny.flatten('F')
    
    mapW = BCmap[3]

    # get field values along elemental faces.
    hM = hC[vmapM]
    hP = hC[vmapP]

    eta = h - H
    etaC = eta.flatten('F')
    #etaM = etaC[vmapM]
    #etaP = etaC[vmapP]

    uM = huC[vmapM] / hC[vmapM]
    uP = huC[vmapP] / hC[vmapP]

    vM = hvC[vmapM] / hC[vmapM]
    vP = hvC[vmapP] / hC[vmapP]

    hNM = hNC[vmapM]
    hNP = hNC[vmapP]

    nxW = nxC[mapW]
    nyW = nyC[mapW]

    # hM, hP = surfaceReconstruction(etaM, hM, etaP, hP)
    # h = np.reshape(hC, (Np, K), order='F')

    # re-form conserved transport from corrected 
    # water column heights.
    huM = hM*uM
    hvM = hM*vM

    huP = hP*uP
    hvP = hP*vP

    # set bc's (no normal flow thru the walls).
    huP[mapW] = huM[mapW] - 2*nxW*(huM[mapW]*nxW + hvM[mapW]*nyW)
    hvP[mapW] = hvM[mapW] - 2*nyW*(huM[mapW]*nxW + hvM[mapW]*nyW)

    # compute jump in states
    dh = hM - hP
    dhu = huM - huP
    dhv = hvM - hvP
    dhN = hNM - hNP

    ((F1M,F2M,F3M,F4M),(G1M,G2M,G3M,G4M)) = sw2dComputeFluxes(hM, huM, hvM, hNM, g, H)
    ((F1P,F2P,F3P,F4P),(G1P,G2P,G3P,G4P)) = sw2dComputeFluxes(hP, huP, hvP, hNP, g, H)
    ((F1,F2,F3,F4),(G1,G2,G3,G4)) = sw2dComputeFluxes(h, hu, hv, hN, g, H)

    uM = huM/hM 
    vM = hvM/hM

    uP = huP/hP
    vP = hvP/hP

    spdM = np.sqrt(uM*uM + vM*vM) + np.sqrt(g*hM)
    spdP = np.sqrt(uP*uP + vP*vP) + np.sqrt(g*hP)

    spdMax = np.max(np.array([spdM, spdP]), axis=0)

    # spdMax = np.max(spdMax)
    lam = np.reshape(spdMax, (ctx.numFacePoints, ctx.numFaces*ctx.numElements), order='F')
    lamMaxMat = np.outer(np.ones((Nfp, 1), dtype=np.float), np.max(lam, axis=0))
    spdMax = lamMaxMat.flatten('F')

    # strong form: Compute flux jump vector. (fluxM - numericalFlux ) dot nW
    dFlux1 = 0.5*((F1M - F1P)*nxC + (G1M-G1P)*nyC - spdMax*dh)
    dFlux2 = 0.5*((F2M - F2P)*nxC + (G2M-G2P)*nyC - spdMax*dhu)
    dFlux3 = 0.5*((F3M - F3P)*nxC + (G3M-G3P)*nyC - spdMax*dhv)
    dFlux4 = 0.5*((F4M - F4P)*nxC + (G4M-G4P)*nyC - spdMax*dhN)

    dFlux1Mat = np.reshape(dFlux1, (Nfp*ctx.numFaces, K), order='F')
    dFlux2Mat = np.reshape(dFlux2, (Nfp*ctx.numFaces, K), order='F')
    dFlux3Mat = np.reshape(dFlux3, (Nfp*ctx.numFaces, K), order='F')
    dFlux4Mat = np.reshape(dFlux4, (Nfp*ctx.numFaces, K), order='F')

    # Flux divergence:
    RHS1 = -(rx*np.dot(Dr, F1) + sx*np.dot(Ds, F1))
    RHS1+= -(ry*np.dot(Dr, G1) + sy*np.dot(Ds, G1))

    RHS2 = -(rx*np.dot(Dr, F2) + sx*np.dot(Ds, F2))
    RHS2+= -(ry*np.dot(Dr, G2) + sy*np.dot(Ds, G2))

    RHS3 = -(rx*np.dot(Dr, F3) + sx*np.dot(Ds, F3))
    RHS3+= -(ry*np.dot(Dr, G3) + sy*np.dot(Ds, G3))

    RHS4 = -(rx*np.dot(Dr, F4) + sx*np.dot(Ds, F4))
    RHS4+= -(ry*np.dot(Dr, G4) + sy*np.dot(Ds, G4))

    surfaceRHS1 = Fscale*dFlux1Mat
    surfaceRHS2 = Fscale*dFlux2Mat
    surfaceRHS3 = Fscale*dFlux3Mat
    surfaceRHS4 = Fscale*dFlux4Mat

    RHS1 += np.dot(Lift, surfaceRHS1)
    RHS2 += np.dot(Lift, surfaceRHS2)
    RHS3 += np.dot(Lift, surfaceRHS3)
    RHS4 += np.dot(Lift, surfaceRHS4)

    # Add source terms
    RHS2 += f*hv
    RHS3 -= f*hu
    RHS2 -= g*h*zx
    RHS3 -= g*h*zy

    return (RHS1, RHS2, RHS3, RHS4)

# Main solver:
# set scaled density jump.
drho = 1.00100 - 1.000

# compute reduced gravity
g = drho*9.81

H0 = 7.5
c0 = np.sqrt(g*H0)

finalTime = 24*3600
numOuts = 200
t = 0.0

meshManager = dg.MeshManager()
meshManager.readMesh('input/headlands3.msh')

Verts = meshManager.vertices
EToV = meshManager.elements
bcType = meshManager.bcType

# Correct boundary condition table for the different types
# TODO, store magic integers like 6 - Dirichlet and 7 - Neuman in an enum prop.
faces = np.array([[0, 1], [1, 2], [2, 0]])
for e in range(0, len(bcType)):
    for (faceInd, f) in enumerate(faces):
        v1 = EToV[e, f[0]]
        v2 = EToV[e, f[1]]
        v1x = Verts[v1, 0]
        v1y = Verts[v1, 1]
        v2x = Verts[v2, 0]
        v2y = Verts[v2, 1]

        midx = 0.5*(v1x + v2x)
        midy = 0.5*(v1y + v2y)

        if  np.abs(midx - 0.0) < 1e-6:
            bcType[e, faceInd] = 2   # outflow
        elif np.abs(midx - 8000.0) < 1e-6:
            bcType[e, faceInd] = 2   # outflow

meshManager.setBCType(bcType)

# Numerical parameters:
NOrder = 8

filtOrder = 4
filtCutoff = 0.6*NOrder

nodes = dg.TriangleNodesProvisioner(NOrder, meshManager)
nodes.buildFilter(filtCutoff, filtOrder)

outputter = dg.VtkOutputter(nodes)

ctx = nodes.dgContext()

x = ctx.x
y = ctx.y

BCmap = ctx.BCmap
mapW = ctx.BCmap[3]
mapO = ctx.BCmap[2]

vmapO = ctx.vmapM[mapO]

vmapW = ctx.vmapM[mapW]

xFlat = x.flatten('F')
yFlat = y.flatten('F')


xO = xFlat[vmapO]
yO = yFlat[vmapO]


vmapOM = []
vmapOP = []
for i in vmapO:
    xi = xFlat[i]
    yi = yFlat[i]

    ips = np.where( np.logical_and(np.abs(yO - yi) < 1e-7, np.abs(xO - xi) > 1000))[0]

    vmapOM.append(i)
    vmapOP.append(vmapO[ips])

lookup = dict.fromkeys(vmapOM)

# Figure out look-up for periodic BCs
vmapOPflat = []
for i, l in enumerate(vmapOP):
    if len(l) == 1:
        vmapOPflat.append(l[0])
    else:
        if i==0 and (abs(vmapOP[i+1] -l[0]) <= 1):
            vmapOPflat.append(l[0])
        elif (i > 0 and i < len(vmapOP)-1) and (abs(vmapOP[i-1][0] -l[0]) <= 1 or abs(vmapOP[i+1][0] -l[0]) <= 1):
            vmapOPflat.append(l[0])
        elif (i==len(vmapOP)-1) and (abs(vmapOP[i-1] -l[0]) <= 1):
            vmapOPflat.append(l[0])
        else:
            vmapOPflat.append(l[1])

vmapOP = vmapOPflat


for i, key in enumerate(lookup.keys()):
    lookup[key] = vmapOP[i]

vmapM = ctx.vmapM
vmapP = ctx.vmapP
# Need to mutate vmapP
for i  in range(0, len(vmapM)):
    if vmapM[i] in lookup.keys():
        vmapP[i] = lookup[vmapM[i]]

xF = ctx.x.flatten('F')
yF = ctx.y.flatten('F')


indN, indK = np.where(np.hypot(x, y) < 1.9e1)
centreIndN = indN[0]
centreIndK = indK[0]

xW = xFlat[vmapW]
yW = yFlat[vmapW]

topInds = np.logical_and(yW > 200, np.logical_and(xW > 3250, xW < 4750))

xtop = xW[topInds]
ytop = yW[topInds]

isort = np.argsort(xtop)
xtop2 = xtop[isort]
ytop2 = ytop[isort]

# build arc-length parametrization
s = [0.0]
for i in range(1, len(xtop2)):
    d = np.hypot(xtop2[i]-xtop2[i-1], ytop2[i]-ytop2[i-1])
    s.append(s[i-1] + d)

s = np.array(s)

ss = np.linspace(s[0], s[-1], 15)

ssfine = np.linspace(s[0], s[-1], 256)
xx = griddata(s, xtop2, ss)
yy = griddata(s, ytop2, ss)
# build parametric curve
splx = splrep(ss, xx)
sply = splrep(ss, yy)


xTopSmooth = splev(ssfine, splx)
yTopSmooth = splev(ssfine, sply)
plt.figure(num=1)
plt.plot(xTopSmooth, yTopSmooth, 'og')
#plt.show()


bcInds = np.where(bcType.flatten('F') > 0)
bcFaces = np.transpose(np.unravel_index(bcInds, (ctx.numElements, ctx.numFaces), order='F'))
#print(np.transpose(bcFaces))

modifiedVerts = np.zeros(Verts.shape[0])

curvedFaces = []
for bcFace in bcFaces:

    el = bcFace[0][0]
    face = bcFace[0][1]
    # Get vertices along face
    v1ind = EToV[el, face]
    v2ind = EToV[el, (face+1) % ctx.numFaces]

    v1 = Verts[v1ind, :]
    v2 = Verts[v2ind, :]

    hyps1 = np.hypot(xTopSmooth - v1[0], yTopSmooth-v1[1])
    hyps2 = np.hypot(xTopSmooth - v2[0], yTopSmooth-v2[1])

    minInd1 = np.argmin(hyps1)
    minInd2 = np.argmin(hyps2)


    mytol = 200


    if hyps1[minInd1] > mytol or hyps2[minInd2] > mytol:
        # nothing to do here
        continue

    curvedFaces.append([el, face])

    plt.figure(num=1)
    plt.plot([v1[0], v2[0]], [v1[1], v2[1]], '.b')

    # set new vertex coordinates to closest spline point coordinates
    newx1 = xTopSmooth[minInd1] 
    newy1 = yTopSmooth[minInd1]

    newx2 = xTopSmooth[minInd2]
    newy2 = yTopSmooth[minInd2]

    plt.plot([newx1, newx2], [newy1, newy2], '-r')

    # update mesh vertex locations
    Verts[v1ind, 0] = newx1
    Verts[v1ind, 1] = newy1

    Verts[v2ind, 0] = newx2
    Verts[v2ind, 1] = newy2

    modifiedVerts[v1ind] = 1  
    modifiedVerts[v2ind] = 1


for face in curvedFaces:
    k = face[0]
    f = face[1]

    if f==0:
       v1 = EToV[k, 0]
       v2 = EToV[k, 1]
       vr = ctx.r
    elif f==1:
        v1 = EToV[k, 1]
        v2 = EToV[k, 2]
        vr = ctx.s
    elif f==2:
        v1 = EToV[k, 0]
        v2 = EToV[k, 2]
        vr = ctx.s

    fr = vr[ctx.Fmask[:, f]]
    x1 = Verts[v1, 0]
    y1 = Verts[v1, 1]
    x2 = Verts[v2, 0]
    y2 = Verts[v2, 1]

    v1_dists2 = (x1-xTopSmooth)**2 + (y1-yTopSmooth)**2
    v2_dists2 = (x2-xTopSmooth)**2 + (y2-yTopSmooth)**2

    v1s_inds = np.where(np.sqrt(v1_dists2) < 1.0e-8)
    v2s_inds = np.where(np.sqrt(v2_dists2) < 1.0e-8)

    if len(v1s_inds) == 0 or len(v2s_inds) == 0:
        continue

    if len(v1s_inds) > 1 and len(v2s_inds) > 1:
        raise Exception('bad parameterization.')
    
    if len(v1s_inds) == 1 and len(v2s_inds) == 1:
        t1 = ssfine[v1s_inds[0]] # set end-points of parameter-space interval.
        t2 = ssfine[v2s_inds[0]]
    else:
        raise Exception("whoops!")

    tLGL = 0.5*t1*(1-fr) + 0.5*t2*(1+fr)

    # Basically xnew - xold
    # where xnew is evaluated using the parameterization
    fdx = splev(tLGL, splx) - x[ctx.Fmask[:,f], k]
    fdy = splev(tLGL, sply) - y[ctx.Fmask[:,f], k]

    # build 1D Vandermonde matrix for face nodes and volume nodes
    vand = dg.VandermondeBuilder()
    Vface, Vfinv = vand.buildVandermondeMatrix(fr, True, NOrder)
    Vvol,  = vand.buildVandermondeMatrix(vr, False, NOrder)

    # compute unblended volume deformations
    vdx = np.dot(Vvol, np.dot(Vfinv, fdx))
    vdy = np.dot(Vvol, np.dot(Vfinv, fdy))

    # compute blending functions
    r = ctx.r
    s = ctx.s
    ids = np.where(np.abs(1-vr) > 1.e-7)[0]
    blend = np.zeros(ids.shape)
    if f==0: blend = -(r[ids]+s[ids])/(1-vr[ids])
    if f==1: blend = (r[ids]+1)/(1-vr[ids])
    if f==2: blend = -(r[ids]+s[ids])/(1-vr[ids])
    
    # blend deformation to volume interior
    x[ids, k] += blend*vdx[ids]
    y[ids, k] += blend*vdy[ids]


nodes.setCoordinates(x, y)
# cub_ctx   = nodes.buildCubatureVolumeMesh(3*(NOrder)+1)
gauss_ctx = nodes.buildGaussFaceNodes(2*(NOrder+1))


# plt.figure(1)
# plt.plot(x.flatten('F'), y.flatten('F'), '.')
# plt.draw()
# plt.show()




Np = ctx.numLocalPoints
K = ctx.numElements

Filt = ctx.filter
#Filt = np.eye(Np)

#eta = -0.1*(x/8000.0)
f=7.8825e-5
amp = .065*H0
L = 500
W = 200
eta = amp*np.exp(-((y-L)/W)**2)
etay = -2*amp*(y-L)*np.exp(-((y-L)/W)**2) / W**2
u = (-g/f)*etay

umax =np.max(u.flatten('F')) 
print("umax: ", umax)
print("rad:" , c0/f)
print("froude: ", umax/c0)
v = 0*eta
# distTransect, etaTransect = getMaxEtaTransect(eta, x, y, vmapW)
#write1dField("distTransect0000000.asc", distTransect)
#write1dField("etaTransect0000000.asc", etaTransect)

r = np.sqrt(x*x + y*y)

# u   = np.zeros([Np, K], dtype=np.float, order='C')
# v   = np.zeros([Np, K], dtype=np.float, order='C')

#H = 9.5*(1-(r/8000)*(r/8000)) + .5
H = H0 + 0*x
Dr = ctx.Dr 
Ds = ctx.Ds
rx = ctx.rx
ry = ctx.ry
sx = ctx.sx 
sy = ctx.sy

z = -H
zx = (rx*np.dot(Dr, z) + sx*np.dot(Ds, z))
zy = (ry*np.dot(Dr, z) + sy*np.dot(Ds, z))

Nrad = 2e3
Nx = 2000.0
Ny = 2500.0
# N   = np.exp(-(((x-Nx)/Nrad)**2 + ((y-Ny)/Nrad)**2))
# N   = np.exp(-((y-Ny)/Nrad)**2)
N = 0*eta

h = H + eta
#h = 5*(0.5*(1 - np.tanh(x/Nrad)))
h[h < 1e-3] = 1e-3
#H = h - eta
eta = h - H
hu = h*u
hv = h*v
hN = h*N

# setup fields dictionary for outputting.
fields = dict()
fields["eta"] = eta
fields["u"] = u
fields["v"] = v
# fields["N"] = N
fields["h"] = h
outputter.writeFieldsToFiles(fields, 0)

Hbar = np.mean(H)
c = np.sqrt(g*Hbar)*np.ones((Np, K))
CFL = 0.35
# dt = CFL / np.max( ((NOrder+1)**2)*0.5*np.abs(ctx.Fscale.flatten('F'))*(c.flatten('F')[vmapM]  + np.sqrt(((u.flatten('F'))[vmapM])**2 + ((v.flatten('F'))[vmapM])**2)))
dt = 1.1

numSteps = int(np.ceil(finalTime/dt))
#outputInterval = int(numSteps / numOuts)
outputInterval = 10

step = 0
print("Entering main time-loop")
while t < finalTime:

    (RHS1,RHS2,RHS3,RHS4) = sw2dComputeRHS(h, hu, hv, hN, zx, zy, g, H, f, ctx, vmapM, vmapP)

    RHS1 = np.dot(Filt, RHS1)
    RHS2 = np.dot(Filt, RHS2)
    RHS3 = np.dot(Filt, RHS3)
    RHS4 = np.dot(Filt, RHS4)
    
    # predictor
    h1  = h + 0.5*dt*RHS1
    hu1 = hu + 0.5*dt*RHS2
    hv1 = hv + 0.5*dt*RHS3
    hN1 = hN + 0.5*dt*RHS4

    h1, hu1, hv1 = positivityPreservingLimiter2D(h1, hu1, hv1)
    h1[h1 < 1e-3] = 1e-3

    (RHS1,RHS2,RHS3,RHS4) = sw2dComputeRHS(h1, hu1, hv1, hN1, zx, zy, g, H, f, ctx, vmapM, vmapP)

    RHS1 = np.dot(Filt, RHS1)
    RHS2 = np.dot(Filt, RHS2)
    RHS3 = np.dot(Filt, RHS3)
    RHS4 = np.dot(Filt, RHS4)

    # corrector - Update solution
    h += dt*RHS1
    hu += dt*RHS2
    hv += dt*RHS3
    hN += dt*RHS4


    h, hu, hv = positivityPreservingLimiter2D(h, hu, hv)


    drycells = h <= 1e-3
    h[drycells] = 1e-3

    hu[drycells] = 0.0
    hv[drycells] = 0.0

    u = hu / h
    v = hv / h
    dt = CFL / np.max( ((NOrder+1)**2)*0.5*np.abs(ctx.Fscale.flatten('F'))*(c.flatten('F')[vmapM]  + np.sqrt(((u.flatten('F'))[vmapM])**2 + ((v.flatten('F'))[vmapM])**2)))


    h_max = np.max(np.abs(h))
    if h_max > 1e8  or np.isnan(h_max):
        raise Exception("A numerical instability has occurred.")

    t += dt
    step += 1

    if step % outputInterval == 0 or step == numSteps:
        print('Outputting at t=' + str(t))
        eta = h-H
        fields["eta"] = eta
        fields["u"] = hu/h
        fields["v"] = hv/h
        # fields["N"] = hN/h
        fields["h"] = h
        outputter.writeFieldsToFiles(fields, step)
    
    #if step == 10080 or step == 15120 or step == 5145:
        # distTransect, etaTransect = getMaxEtaTransect(eta, x, y, vmapW)
        #write1dField(f"distTransect{step:07d}.asc", distTransect)
        #write1dField(f"etaTransect{step:07d}.asc", etaTransect)
