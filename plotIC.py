import matplotlib.pyplot as plt
from jaxpm.painting import cic_paint
from jax.numpy import array as jarray

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#%pylab inline
from tqdm import tqdm
import cmasher as cmr
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import readgadget
import jax_cosmo as jc
import numpy as np
import glob
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint
import cmasher as cmr
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import functools
from pm import make_neural_ode_fn
import itertools
from base import PRNGSequence, PGDCorrection
from jaxpm.pm import linear_field, lpt, make_ode_fn, pm_forces
from jaxpm.painting import cic_paint, cic_read, compensate_cic,cic_paint_2d
from nn import NeuralSplineFourierFilter
from kernels import fftk, gradient_kernel, invlaplace_kernel, longrange_kernel
from jaxpm.utils import power_spectrum, cross_correlation_coefficients
from jax.numpy import stack


mesh_shape= [64, 64, 64]
box_size  = [25., 25., 25.]
cosmo = jc.Planck15(Omega_c= 0.3 - 0.049, Omega_b=0.049, n_s=0.9624, h=0.6711, sigma8=0.8)

print("Cosmology:", cosmo)

init_cond = '/gpfs02/work/diffusion/ics_stellar/ic/ics'

header   = readgadget.header(init_cond)
BoxSize  = header.boxsize/1e3  #Mpc/h
Nall     = header.nall         #Total number of particles
Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
Omega_m  = header.omega_m      #value of Omega_m
Omega_l  = header.omega_l      #value of Omega_l
h        = header.hubble       #value of h
redshift = header.redshift     #redshift of the snapshot
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

ptype = [1] #dark matter is particle type 1
ids_i = np.argsort(readgadget.read_block(init_cond, "ID  ", ptype)-1)  #IDs starting from 0
pos_i = readgadget.read_block(init_cond, "POS ", ptype)[ids_i]/1e3     #positions in Mpc/h
vel_i = readgadget.read_block(init_cond, "VEL ", ptype)[ids_i]         #peculiar velocities in km/s

pos_i = pos_i.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
vel_i = vel_i.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
pos_i = (pos_i/BoxSize*64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
vel_i = (vel_i / 100 * (1./(1+redshift)) / BoxSize*64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
a_i   = 1./(1+redshift)

scales = []
#scales.append(a_i)
poss = []
vels = []

snap_dir= '/gpfs02/work/diffusion/neural_ode/CV0'
snap_files = glob.glob(os.path.join(snap_dir, 'snapshot_*.hdf5'))
# dropping .hdf5.1 etc
snap_files = [f for f in snap_files if not f.endswith('.hdf5.1')]
snap_files.sort()

for snapfile in tqdm(snap_files):
    
    snapshot = snapfile[:-len('.hdf5')]
    header   = readgadget.header(snapshot)
    
    redshift = header.redshift     #redshift of the snapshot
    h        = header.hubble       #value of h
    
    ptype = [1] #dark matter is particle type 1
    ids = np.argsort(readgadget.read_block(snapshot, "ID  ", ptype)-1)     #IDs starting from 0
    pos = readgadget.read_block(snapshot, "POS ", ptype)[ids] / 1e3        #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)[ids]              #peculiar velocities in km/s

    # Reordering data for simple reshaping
    pos = pos.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    vel = vel.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    
    pos = (pos / BoxSize * 64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
    vel = (vel / 100 * (1./(1+redshift)) / BoxSize*64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
    
    scales.append((1./(1+redshift)))
    poss.append(pos)
    vels.append(vel)


rng_seq = PRNGSequence(1) # endless stream of random keys

print(pos_i)
print(pos_i.shape)
print(vel_i)
print(vel_i.shape)
print(poss[0])
print(poss[0].shape)
print(vels[0])  
print(vels[0].shape)

cmap = cmr.eclipse  
col = cmr.eclipse([0.,0,0.55,0.85]) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0., vmax=1))

# Target redshifts and scale factors
target_redshifts = [6, 3, 0]
target_scales = [1 / (1 + z) for z in target_redshifts]

# Convert scales to JAX array for easier comparison
scales_np = np.array(scales)

# Get indices of snapshots closest to the target redshifts
#target_indices = [np.argmin(np.abs(scales_np - a)) - 1 for a in target_scales]
#snapshot_scales_np = np.array(scales[1:])
snapshot_scales_np = np.array(scales)
target_indices = [np.argmin(np.abs(snapshot_scales_np - a)) - 1 for a in target_scales]
matched_redshifts = [1. / snapshot_scales_np[i] - 1 for i in target_indices]

'''
print("Available snapshot redshifts:")
for i, a in enumerate(scales[1:]):
    print(f"Snapshot {i}: z = {1/a - 1:.2f}")

print("Matched redshifts and indices:")
for i, z in zip(target_indices, matched_redshifts):
    print(f"Index in poss: {i}, z ≈ {z:.2f}")
'''
tolerance = 0.02

for target_z, target_a in zip(target_redshifts, target_scales):
    diffs = np.abs(scales_np - target_a)
    best_idx = np.argmin(diffs)
    if diffs[best_idx] > tolerance:
        print(f"⚠️  No match found for z = {target_z} (a = {target_a:.5f}) within tolerance.")
        continue
    target_indices.append(best_idx)
    matched_redshifts.append(1. / scales_np[best_idx] - 1)

print("Matched redshifts and indices:")
for i, z in zip(target_indices, matched_redshifts):
    print(f"Index in resi: {i}, z ≈ {z:.2f}")

# Print matched redshifts
#print("Matched redshifts:")
#for idx in target_indices:
    #matched_z = 1. / scales_np[idx] - 1.
   # print(f"Index: {idx}, z ≈ {matched_z:.2f}")

# Set up color normalization
resi = odeint(make_ode_fn(mesh_shape), [pos_i, vel_i], jnp.array(scales), cosmo, rtol=1e-5, atol=1e-5)

'''
norm = colors.LogNorm(
    vmin=min(cic_paint(jnp.zeros(mesh_shape), poss[idx]).sum(axis=0).min() for idx in target_indices),
    vmax=max(cic_paint(jnp.zeros(mesh_shape), poss[idx]).sum(axis=0).max() for idx in target_indices)
)

# Plot the selected redshifts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, idx, z in zip(axes, target_indices, target_redshifts):
    grid = cic_paint(jnp.zeros(mesh_shape), poss[idx]).sum(axis=0)
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title(f"$z \\approx {z}$")
    ax.set_xticks([])
    ax.set_yticks([])
'''


norm = colors.LogNorm(
    vmin=min(cic_paint(jnp.zeros(mesh_shape), resi[i][0]).sum(axis=0).min() for i in target_indices),
    vmax=max(cic_paint(jnp.zeros(mesh_shape), resi[i][0]).sum(axis=0).max() for i in target_indices)
)

fig, axes = plt.subplots(1, len(target_indices), figsize=(5 * len(target_indices), 5))
if len(target_indices) == 1:
    axes = [axes]

for ax, i, z in zip(axes, target_indices, matched_redshifts):
    density = cic_paint(jnp.zeros(mesh_shape), resi[i][0]).sum(axis=0)
    ax.imshow(density, cmap=cmap, norm=norm)
    ax.set_title(f"$z \\approx {z:.2f}$")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
plt.suptitle("Original")
plt.savefig("images/grids_at_z_27_6_0_IC.png")
