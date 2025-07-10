import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

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
#from jaxpm.utils import power_spectrum, cross_correlation_coefficients
from jaxpm.utils import power_spectrum
from jax.numpy import stack
'''
def power_spectrum_1d(field, boxsize, kmin=0.0, dk=0.03):
    shape = field.shape
    boxsize = np.asarray(boxsize)
    N = np.prod(shape)

    # FFT and normalize
    delta_k = jnp.fft.rfftn(field) / N
    power = (delta_k * delta_k.conj()).real

    # Create k-grid
    kx = np.fft.fftfreq(shape[0], d=boxsize[0] / shape[0])
    ky = np.fft.fftfreq(shape[1], d=boxsize[1] / shape[1])
    kz = np.fft.rfftfreq(shape[2], d=boxsize[2] / shape[2])  # full for binning
    KX, KY, KZ = np.meshgrid(kx, ky, kz[:shape[2]//2+1], indexing='ij')
    k_mag = 2 * np.pi * np.sqrt(KX**2 + KY**2 + KZ**2)

    # Flatten arrays
    k_flat = k_mag.ravel()
    P_flat = power.ravel()

    # Bin power into 1D spectrum
    kmax = k_flat.max()
    kbins = np.arange(kmin, kmax + dk, dk)
    inds = np.digitize(k_flat, kbins) - 1

    Psum = np.bincount(inds, weights=P_flat, minlength=len(kbins))
    Nsum = np.bincount(inds, minlength=len(kbins))

    # Avoid div-by-zero
    valid = (Nsum > 0) & (np.arange(len(Nsum)) < len(kbins) - 1)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])[valid[:-1]]

    return k_centers, (Psum[valid] / Nsum[valid]) * np.prod(boxsize)
'''
def power_spectrum_1d(field, boxsize, kmin=0.0, dk=0.03, kbins=None):
    shape = field.shape
    boxsize = np.asarray(boxsize)
    N = np.prod(shape)

    # FFT and normalize
    delta_k = jnp.fft.rfftn(field) / N
    power = (delta_k * delta_k.conj()).real

    # Create k-grid
    kx = np.fft.fftfreq(shape[0], d=boxsize[0] / shape[0])
    ky = np.fft.fftfreq(shape[1], d=boxsize[1] / shape[1])
    kz = np.fft.rfftfreq(shape[2], d=boxsize[2] / shape[2])  # full for binning
    KX, KY, KZ = np.meshgrid(kx, ky, kz[:shape[2]//2+1], indexing='ij')
    k_mag = 2 * np.pi * np.sqrt(KX**2 + KY**2 + KZ**2)

    # Flatten arrays
    k_flat = k_mag.ravel()
    P_flat = power.ravel()

    # Generate or use provided kbins
    if kbins is None:
        kmax = k_flat.max()
        kbins = np.arange(kmin, kmax + dk, dk)

    # Bin power into 1D spectrum
    inds = np.digitize(k_flat, kbins) - 1
    Psum = np.bincount(inds, weights=P_flat, minlength=len(kbins))
    Nsum = np.bincount(inds, minlength=len(kbins))

    # Avoid div-by-zero
    valid = (Nsum > 0) & (np.arange(len(Nsum)) < len(kbins) - 1)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])[valid[:-1]]

    return k_centers, (Psum[valid] / Nsum[valid]) * np.prod(boxsize)

def cross_correlation_coefficients(field_a,
                                   field_b,
                                   boxsize,
                                   kmin=None,
                                   dk=None,
                                   kbins=None):
    """
    Calculate the cross-correlation coefficient between two real space fields.

    Args:
        field_a: real-valued field (e.g. PM density)
        field_b: real-valued field (e.g. stellar density)
        boxsize: physical box size in each dimension
        kmin: optional, minimum k value if kbins is not provided
        dk: optional, bin width in k if kbins is not provided
        kbins: optional, array of k bin edges

    Returns:
        k_centers: central k values for each bin
        cross_power: cross power spectrum normalized
    """
    shape = field_a.shape
    boxsize = np.asarray(boxsize)
    N = np.prod(shape)

    # Fourier transforms
    delta1 = jnp.fft.fftn(field_a) / N
    delta2 = jnp.fft.fftn(field_b) / N
    cross = (delta1 * delta2.conj()).real

    # Construct |k| array
    kx = np.fft.fftfreq(shape[0], d=boxsize[0] / shape[0])
    ky = np.fft.fftfreq(shape[1], d=boxsize[1] / shape[1])
    kz = np.fft.fftfreq(shape[2], d=boxsize[2] / shape[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = 2 * np.pi * np.sqrt(KX**2 + KY**2 + KZ**2)

    # Flatten for binning
    k_flat = k_mag.ravel()
    cross_flat = cross.ravel()

    # Determine bin edges
    if kbins is None:
        assert kmin is not None and dk is not None, "Must provide kmin and dk if kbins not given"
        kmax = k_flat.max()
        kbins = np.arange(kmin, kmax + dk, dk)

    inds = np.digitize(k_flat, kbins) - 1

    # Bin cross power
    Psum = np.bincount(inds, weights=cross_flat, minlength=len(kbins))
    Nsum = np.bincount(inds, minlength=len(kbins))

    valid = (Nsum > 0) & (np.arange(len(Nsum)) < len(kbins) - 1)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])[valid[:-1]]

    cross_power = (Psum[valid] / Nsum[valid]) * np.prod(boxsize)
    return k_centers, cross_power

############################################################# pm component 
mesh_shape= [128, 128, 128]
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

pos_i = (pos_i / BoxSize * 128).reshape([256,256,256,3])[::2,::2,::2,:].reshape([-1,3])
vel_i = (vel_i / 100 * (1./(1+redshift)) / BoxSize * 128).reshape([256,256,256,3])[::2,::2,::2,:].reshape([-1,3])
a_i   = 1./(1+redshift)

#target_redshifts = [2, 1.5, 1.0, 0.5, 0.0]
#scales = [1 / (1 + z) for z in target_redshifts]

scales = []
scales.append(a_i)
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
    
    redshift = header.redshift    
    h        = header.hubble    
    
    ptype = [1]
    #ids = np.argsort(readgadget.read_block(snapshot, "ID  ", ptype)-1)    
    #pos = readgadget.read_block(snapshot, "POS ", ptype)[ids] / 1e3      
    #vel = readgadget.read_block(snapshot, "VEL ", ptype)[ids]            

    #pos = pos.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    #vel = vel.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    
    #pos = (pos / BoxSize * 64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
    #vel = (vel / 100 * (1./(1+redshift)) / BoxSize*64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
    
    scales.append((1./(1+redshift)))
    #poss.append(pos)
    #vels.append(vel)

rng_seq = PRNGSequence(1) 

resi = odeint(make_ode_fn(mesh_shape), [pos_i, vel_i], jnp.array(scales), cosmo) # pm

target_z = 2.0
target_a = 1 / (1 + target_z)

# Only search in scales[1:], which correspond to resi[1:], resi[2:], ...
scales_np = np.array(scales)
idx = np.argmin(np.abs(scales_np - target_a))
#idx = idx_local + 1  # shift because we skipped the IC

matched_z = 1 / scales_np[idx] - 1
print(f"Matched redshift: z ≈ {matched_z:.2f}, index = {idx}")


# downsample
pos_target = resi[0][idx]

pm_scaled = pos_target / BoxSize * 128
pm_field = cic_paint(jnp.zeros([128, 128, 128]), pm_scaled)


############################################################## stellar component
# Grids_Mstar_IllustrisTNG_CV_128_z=0.0.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=0.5.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=1.0.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=1.5.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=2.0.npy

filename = 'Grids_Mstar_IllustrisTNG_CV_128_z=2.0.npy'
DIR = "/gpfs02/work/diffusion/gridsMstar"
OUT_DIR  = "images" # saving stuff here
os.makedirs(OUT_DIR, exist_ok=True)

suite   = "IllustrisTNG"   
subset  = "CV"            
grid    = 128               
z       = 1.5          
proj_ax = 2                 # axis projecting along
cmap    = "inferno"

fullpath = os.path.join(DIR, filename)
if not os.path.exists(fullpath):
    raise FileNotFoundError(f"Couldn’t find {fullpath}")


grid3dstellar = np.load(fullpath, mmap_mode="r")   

if grid3dstellar.ndim == 4:
    grid3dstellar = grid3dstellar[0] 

delta_nods = np.array(grid3dstellar)

k_nods, Pk_nods = power_spectrum(
    delta_nods,
    boxsize=np.array([25.0,25.0,25.0]),
    kmin=np.pi/25.0,
    dk=2*np.pi/25.0,
)

# surf = grid3dstellar.sum(axis=proj_ax)
# Now making 64^2 slice
stellar_lowres = grid3dstellar.reshape(64,2,64,2,64,2).mean(axis=(1,3,5))
pm_lowres = pm_field.reshape(64,2,64,2,64,2).mean(axis=(1,3,5))
surf_stellar = stellar_lowres.sum(axis=proj_ax)
surf_pm = pm_lowres.sum(axis=proj_ax)

# Plot 2 downsampled 
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(np.log10(surf_pm + 1e-5), cmap='inferno')
axs[0].set_title("PM Field (64² slice)")

im1 = axs[1].imshow(np.log10(surf_stellar + 1e-5), cmap='inferno')
axs[1].set_title("Stellar Field (64² slice)")

fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"images/pm_vs_stellar_slice_z{target_z}_downsampled.png")
plt.show()


################# cross corr
print("pm_field shape:", pm_field.shape)
print("stellar_field shape:", grid3dstellar.shape)
print("BoxSize:", box_size)

kmin = 0.0
dk = 0.10
kmax = np.pi * (64 / 25)
kbins = np.arange(kmin, kmax + dk, dk)

# 64^2 version
downsampled_box = np.array(box_size) 

k_pm, P_pm = power_spectrum_1d(pm_lowres, boxsize=downsampled_box, kbins = kbins) #kmin=kmin, dk=dk)
k_star, P_star = power_spectrum_1d(stellar_lowres, boxsize=downsampled_box, kbins=kbins) #kmin=kmin, dk=dk)

k_vals, corr_vals = cross_correlation_coefficients(pm_lowres, stellar_lowres, boxsize=downsampled_box, kbins=kbins) #kmin=kmin, dk=dk)


''' # 128
box_size = np.array(box_size)
k_pm, P_pm = power_spectrum_1d(pm_field, boxsize=box_size,  kmin=kmin,
    dk=dk)
k_star, P_star = power_spectrum_1d(grid3dstellar, boxsize=box_size,  kmin=kmin,
    dk=dk)

k_vals, corr_vals = cross_correlation_coefficients(pm_field, grid3dstellar, boxsize = box_size,  kmin=kmin,
    dk=dk) 
'''
print("cross corelation:", corr_vals)
###### plot
r_k = corr_vals / np.sqrt(P_pm * P_star)

fig, ax = plt.subplots(figsize = (16,8))
ax.plot(k_vals, r_k)
ax.set_xscale('log')
ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$', fontsize=16)
ax.axhline(1, linestyle='--', color='k') #, alpha=0.5)
ax.set_ylabel(r"$r(k)$", fontsize=16)
ax.set_title(rf"Cross-corr $r(k)$ PM vs Stellar Field at $z = {target_z}$", fontsize=18)
#ax.set_title(f"Cross-correlation for PM vs Stellar Field at $z = {target_z}$")
ax.grid(True, which='both', ls=':')
fig.tight_layout()
fig.savefig(f"images/cross_corr_pm_vs_stellar_z{target_z}.png")
fig.show()


