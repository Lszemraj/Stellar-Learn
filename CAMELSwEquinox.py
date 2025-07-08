import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
#from pm import make_neural_ode_fn
import itertools
#from base import PRNGSequence, PGDCorrection
from jaxpm.pm import linear_field, lpt, make_ode_fn, pm_forces
from jaxpm.painting import cic_paint, cic_read, compensate_cic,cic_paint_2d
#from nn import NeuralSplineFourierFilter
from kernels import fftk, gradient_kernel, invlaplace_kernel, longrange_kernel
from jaxpm.utils import power_spectrum, cross_correlation_coefficients
from jax.numpy import stack
import equinox as eqx
#from equinox.experimental import filter_odeint
from equinox.experimental import filter_odeint
from equinox import tree_serialise_leaves, tree_deserialise_leaves

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
poss = []
vels = []

snap_dir = '/gpfs02/work/diffusion/neural_ode/CV0'
snap_files = sorted([f for f in glob.glob(os.path.join(snap_dir, 'snapshot_*.hdf5')) if not f.endswith('.hdf5.1')])



#for i in tqdm(range(34)):
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



class EquinoxSplineFourierFilter(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, n_knots=16, latent_size=32, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        mlp_key = key
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=latent_size,
            depth=3,
            activation=jax.nn.swish,
            key=mlp_key
        )

    def __call__(self, k, a):
        x = jnp.log1p(k)[:, None]
        return self.mlp(x).squeeze()
    
def neural_nbody_ode_eqx(state, a, cosmo, model):
    pos, vel = state
    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    delta_k = jnp.fft.rfftn(delta)
    kvec = fftk(delta.shape)

    pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
    kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
    filt = model(kk, jnp.atleast_1d(a))
    pot_k *= (1.0 + filt)

    forces = jnp.stack([
        cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos) for i in range(3)
    ], axis=-1)
    forces *= 1.5 * cosmo.Omega_m

    Esqr = jc.background.Esqr(cosmo, a)
    dpos = vel / (a**3 * jnp.sqrt(Esqr))
    dvel = forces / (a**2 * jnp.sqrt(Esqr))
    return dpos, dvel

key = jax.random.PRNGKey(0)
model = EquinoxSplineFourierFilter(key=key)
res = filter_odeint(
    lambda state, a, args: neural_nbody_ode_eqx(state, a, *args),
    (poss[0], vels[0]),
    jnp.array(scales),
    (cosmo, model),
    rtol=1e-5,
    atol=1e-5
)


model_path = "model.eqx"
tree_serialise_leaves(model_path, model)
model_loaded = tree_deserialise_leaves(model_path, model)

# Plotting results
cmap = cmr.eclipse
col = cmr.eclipse([0., 0.13, 0.55, 0.85])

im1 = cic_paint(jnp.zeros(mesh_shape), poss[-1]).sum(axis=0)
im2 = cic_paint(jnp.zeros(mesh_shape), res[0][-1]).sum(axis=0)
TI = ['CAMELS', 'PM+NN']
image_paths = [im1, im2]

norm = colors.LogNorm(vmax=im2.max(), vmin=im2.min())
f, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), dpi=90)

for imp, ax, ci in zip(image_paths, axes.ravel(), TI):
    ax.imshow(imp, cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_title(ci, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.)
    cb = f.colorbar(ax.imshow(imp, cmap=cmap, norm=norm), cax=cax)
    cb.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('equinox_output_2D.png')


