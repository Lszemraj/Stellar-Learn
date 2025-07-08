import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage


#PATTERN = os.path.join(DIR, "stars_z*.npy")

#data = np.load("/gpfs02/work/diffusion/gridsMstar/Grids_Mstar_IllustrisTNG_1P_128_z=0.0.npy", allow_pickle=True)

#print("shape", data.shape)
#print("data", data)
#A = np.load("/gpfs02/work/diffusion/gridsMstar/Grids_Mstar_IllustrisTNG_CV_512_z=0.5.npy", allow_pickle=False)

# Grids_Mstar_IllustrisTNG_CV_128_z=0.0.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=0.5.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=1.0.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=1.5.npy
# Grids_Mstar_IllustrisTNG_CV_128_z=2.0.npy

filename = 'Grids_Mstar_IllustrisTNG_CV_128_z=1.5.npy'
DIR = "/gpfs02/work/diffusion/gridsMstar"
OUT_DIR  = "images" # saving stuff here
os.makedirs(OUT_DIR, exist_ok=True)

suite   = "IllustrisTNG"   
subset  = "CV"            
grid    = 128               
z       = 1.5          
proj_ax = 2                 # axis projecting along
cmap    = "inferno"


#filename = f"Grids_Mstar_{suite}_{subset}_{grid}_z={z:.2f}.npy"
fullpath = os.path.join(DIR, filename)
if not os.path.exists(fullpath):
    raise FileNotFoundError(f"Couldn’t find {fullpath}")


grid3d = np.load(fullpath, mmap_mode="r")   

if grid3d.ndim == 4:
    grid3d = grid3d[0]    # pick the first volume


# Suppose your original is 512³ and you want 128³:
#factor = 1/4
#zoom_factors = (factor, factor, factor)

# 1) (Optional) Gaussian‐smooth to avoid aliasing.
#    A σ of ~ half the downsample factor helps.
#sigma = 0.5 * (1/factor - 1)  # e.g. 0.5*(4 - 1) = 1.5
#grid_smooth = ndimage.gaussian_filter(grid3d, sigma=sigma)

# 2) Zoom with your choice of interpolation order:
#    order=0 → nearest neighbor (blocky but preserves total sums per voxel)
#    order=1 → trilinear (more smooth, less blocky)
#    order=3 → cubic spline (the default, but can overshoot)
#grid_small = ndimage.zoom(grid_smooth, zoom_factors, order=1)

#print("zoomed shape:", grid_small.shape)

#grid3d = ndimage.zoom(grid3d,1/4)

#print("Loaded grid3d:", grid3d.shape, "dtype:", grid3d.dtype)

# powerspectrum 
#boxsize = np.array([25.0, 25.0, 25.0])
#kmin = np.pi / boxsize[0]
#dk   = 2*np.pi / boxsize[0]

# if grid3d is NumPy, convert to JAX:
delta_nods = np.array(grid3d)

k_nods, Pk_nods = power_spectrum(
    delta_nods,
    boxsize=np.array([25.0,25.0,25.0]),
    kmin=np.pi/25.0,
    dk=2*np.pi/25.0,
)


#delta_ds = np.array(grid_small)

#k_ds, Pk_ds = power_spectrum(
    #delta_ds,
    #boxsize=np.array([25.0,25.0,25.0]),
    #kmin=np.pi/25.0,
    #dk=2*np.pi/25.0,
#)

plt.figure(figsize=(6,4))
plt.loglog(k_nods, Pk_nods )#, label="Original")
#plt.loglog(k_ds,  Pk_ds,  label="Downsampled", linestyle='-')

plt.xlabel("k")
plt.ylabel("P(k)")
plt.title(f"Power spectrum z = {z}")
plt.grid(which="both", ls=":")
#plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(f"images/powerspectrum_{subset}_{grid}_{z}_grid.png", dpi=300)
plt.show()


'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)


axes[0].loglog(k_nods, Pk_nods)
axes[0].set_title("Original $P(k)$")
axes[0].set_xlabel(r"$k$")
axes[0].set_ylabel(r"$P(k)$")
axes[0].grid(which="both", ls=":")


axes[1].loglog(k_ds, Pk_ds)
axes[1].set_title("Downsampled $P(k)$")
axes[1].set_xlabel(r"$k$")
# no need to set ylabel again if sharey=True, but you can
# axes[1].set_ylabel(r"$P(k)$")
axes[1].grid(which="both", ls=":")

plt.tight_layout()
plt.savefig("images/powerspectrum_compare.png", dpi=300)
plt.show()
'''

# to 2D
#surf = grid3d.sum(axis=proj_ax)  # now shape is (128,128)
#surf_ds = grid_small.sum(axis=proj_ax)
surf = grid3d.sum(axis=proj_ax)
'''
mask = (surf == 0)
surf_masked = np.ma.array(surf, mask=mask)

norm = LogNorm(vmin=surf[surf>0].min(), vmax=surf.max(), clip=True)
cmap = plt.cm.inferno
cmap.set_bad("black")

im = plt.imshow(
    surf_masked.T,
    origin="lower",
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
)
'''
# plotting here

#positive_ds = surf_ds[surf_ds > 0]
positive = surf[surf > 0]
#vmin = positive.min()  
#print("vmin:", vmin)
#print("max:", surf.max())
#vmax = surf.max()
vmin = 233429710.0
vmax = 41136576000000.0
norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
fig, ax = plt.subplots(1, 1, figsize=(6,6), constrained_layout=True)


im = ax.imshow(
    surf.T,
    origin="lower",
    cmap="inferno",
    norm=norm,
    interpolation="nearest",
)
ax.set_title(f"$z={z:.2f}$")



cbar = fig.colorbar(
    im,
    ax=ax,
    location="right",
    fraction=0.046,
    pad=0.02,
)
cbar.set_label(r"$\Sigma_\star$ [mass per voxel]")
    

outname = f"Mstar_{suite}_{subset}_{grid}_z{z:.2f}_grid.png"
plt.savefig(os.path.join(OUT_DIR, outname), dpi=300, bbox_inches="tight")
plt.show()

'''
fig, axes = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True )

for ax, surf, title in zip(axes, [surf_nods, surf_ds], ["Original", "Downsampled"]):
    im = ax.imshow(
        surf.T,
        origin="lower",
        cmap="inferno",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(f"{title}\n$z={z:.2f}$")  

# single colorbar for both

cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.02)
cbar.set_label(r"$\Sigma_\star$ [mass per voxel]")


outname = f"Mstar_{suite}_{subset}_{grid}_z{z:.2f}_comparison.png"
plt.savefig(os.path.join(OUT_DIR, outname), dpi=300, bbox_inches="tight")
plt.show()

'''


'''
# making zeros black
cmap = plt.cm.inferno
cmap.set_bad(color="black")  


plt.figure(figsize=(6,6))


im = plt.imshow(
    np.ma.masked_less_equal(surf, vmin).T,
    origin="lower",
    cmap=cmap,
    norm=norm,
)


plt.title(f"z={z}")
#plt.axis("off")
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label(r"$\Sigma_\star$ [mass per voxel]")
outname = f"Mstar_{suite}_{subset}_{grid}_z{z:.2f}_downsample.png"
plt.savefig(os.path.join(OUT_DIR, outname), dpi=300, bbox_inches="tight")
plt.show()
'''


'''

fig, ax = plt.subplots(figsize=(6,6))
# mask zeros, set up norm, cmap etc.
surf_masked = np.ma.array(surf, mask=(surf==0))
vmin = surf[surf>0].min(); vmax = surf.max()
norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
cmap = plt.cm.inferno; cmap.set_bad("black")

im = ax.imshow(
    surf_masked.T,
    origin="lower",
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)
ax.axis("off")
fig.suptitle(f"z = {z}")
# Now add the colorbar on *this* figure, tied to that axis:
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\Sigma_\star$ [mass per voxel]")
outname = f"Mstar_{suite}_{subset}_{grid}_z{z:.2f}.png"
plt.savefig(os.path.join(OUT_DIR, outname), dpi=300, bbox_inches="tight")
plt.tight_layout(pad=0)
'''