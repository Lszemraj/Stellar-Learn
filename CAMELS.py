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
from matplotlib import gridspec

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
scales.append(a_i)
poss = []
vels = []

snap_dir= '/gpfs02/work/diffusion/neural_ode/CV0'
snap_files = glob.glob(os.path.join(snap_dir, 'snapshot_*.hdf5'))
# dropping .hdf5.1 etc
snap_files = [f for f in snap_files if not f.endswith('.hdf5.1')]
snap_files.sort()


#for i in tqdm(range(34)):
for snapfile in tqdm(snap_files):
    
    snapshot = snapfile[:-len('.hdf5')]
    header   = readgadget.header(snapshot)
    
    redshift = header.redshift    
    h        = header.hubble    
    
    ptype = [1]
    ids = np.argsort(readgadget.read_block(snapshot, "ID  ", ptype)-1)    
    pos = readgadget.read_block(snapshot, "POS ", ptype)[ids] / 1e3      
    vel = readgadget.read_block(snapshot, "VEL ", ptype)[ids]            

    pos = pos.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    vel = vel.reshape(4,4,4,64,64,64,3).transpose(0,3,1,4,2,5,6).reshape(-1,3)
    
    pos = (pos / BoxSize * 64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
    vel = (vel / 100 * (1./(1+redshift)) / BoxSize*64).reshape([256,256,256,3])[::4,::4,::4,:].reshape([-1,3])
    
    scales.append((1./(1+redshift)))
    poss.append(pos)
    vels.append(vel)


rng_seq = PRNGSequence(1) 

print(pos_i)
print(pos_i.shape)
print(vel_i)
print(vel_i.shape)
print(poss[0])
print(poss[0].shape)
print(vels[0])  
print(vels[0].shape)
#resi = odeint(make_ode_fn(mesh_shape), [poss[0], vels[0]], jnp.array(scales), cosmo, rtol=1e-5, atol=1e-5)
resi = odeint(make_ode_fn(mesh_shape), [pos_i, vel_i], jnp.array(scales), cosmo, rtol=1e-5, atol=1e-5)

cmap = cmr.eclipse  
col = cmr.eclipse([0.,0,0.55,0.85]) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0., vmax=1))

norm=colors.LogNorm(vmax=cic_paint(jnp.zeros(mesh_shape), poss[::2][-1]).sum(axis=0).max(),
                          vmin=cic_paint(jnp.zeros(mesh_shape),poss[::2][-1]).sum(axis=0).min())
'''
figure(figsize=[10,10])
for i in range(16):
    subplot(4,4,i+1)
    imshow(cic_paint(jnp.zeros(mesh_shape), poss[::2][i]).sum(axis=0), cmap=cmap,norm=norm)
'''

fig = plt.figure(figsize=(10, 10))

# Loop over 16 subplots
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    img = cic_paint(jnp.zeros(mesh_shape), poss[::2][i]).sum(axis=0)
    im = ax.imshow(img, cmap=cmap, norm=norm)
    ax.set_xticks([])  
    ax.set_yticks([])

plt.tight_layout()
plt.show()
plt.savefig("images/high_res_ics_i_ls.png")
print("done with loop and first image")

# using Flax filter for NeuralSplineFourierFilter
model = NeuralSplineFourierFilter(n_knots=16, latent_size=32)

# init
x_dummy = jnp.ones((1,))      
a_dummy = jnp.array(1.0)
key = jax.random.PRNGKey(0)
variables = model.init(key, x_dummy, a_dummy)
params = variables['params']


def neural_nbody_ode(state, a, cosmo, params):
    pos, vel = state
    #grid = jnp.zeros(mesh_shape, dtype=jnp.complex64)
    #kvec = fftk(grid)
    #kvec    = fftk(mesh_shape)
    delta   = cic_paint(jnp.zeros(mesh_shape), pos)
    delta_k = jnp.fft.rfftn(delta)
    kvec   = fftk(delta.shape)
    # gravitational potential
    pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

    # learned spline filter
    kk     = jnp.sqrt(sum((ki/np.pi)**2 for ki in kvec))
    filt   = model.apply({'params': params}, kk, jnp.atleast_1d(a))
    pot_k  = pot_k * (1.0 + filt)

    # compute forces
    forces = jnp.stack([
        cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos)
        for i in range(3)
    ], axis=-1)
    forces = forces * 1.5 * cosmo.Omega_m

    # drift (dpos) and kick (dvel)
    Esqr = jc.background.Esqr(cosmo, a)
    dpos = vel / (a**3 * jnp.sqrt(Esqr))
    dvel = forces / (a**2 * jnp.sqrt(Esqr))
    return dpos, dvel


pk_c_arr    = []
cross_c_arr = []


################### NN LOOP
#pk_pgd_arr    = []
#cross_pgd_arr = []

# loop over your saved Flax params

for filename in os.listdir("correction_params_NoPowerSpec/"):
    print("filename = ", filename)
    params =pickle.load(open(os.path.join('correction_params_NoPowerSpec/', filename), 'rb'))
    # integrate the N-body ODE with these params
    #ode_rhs_fn = neural_nbody_ode(model, mesh_shape, cosmo, params)
    #ode_rhs_fn = make_neural_ode_fn(model, mesh_shape, cosmo, params)
    base_ode   = make_neural_ode_fn(model, mesh_shape)
    ode_rhs_fn = lambda state, a: base_ode(state, a, cosmo, params)
    # 3) pack (pos, vel, cosmo, params) into your initial "state"
    #init_state = (poss[0], vels[0])
    init_state = (pos_i, vel_i)
    '''
    res = odeint(
    ode_rhs,
    (poss[0], vels[0]),
    jnp.array(scales),
    rtol=1e-5,
    atol=1e-5
) '''
    '''
    res = odeint(
        neural_nbody_ode,
        (poss[0], vels[0]),           # initial (pos, vel)
        jnp.array(scales),             # scale factors array
        cosmo,                         # cosmology object
        params,                        # your Flax params dict
        rtol=1e-5,
        atol=1e-5,
    )'''
    res = odeint(
    ode_rhs_fn,
    init_state,
    jnp.array(scales),
    rtol=1e-5,
    atol=1e-5
    )

    # final density field
    final_delta_jax = cic_paint(jnp.zeros(mesh_shape), res[0][-1])
    final_delta     = np.array(final_delta_jax)
    # power spectrum 
    _, pk_c = power_spectrum(
        final_delta,
        boxsize = np.array([25.0, 25.0, 25.0]),
        kmin    = np.pi / 25.0,
        dk      = 2 * np.pi / 25.0,
    )
    pk_c_arr.append(pk_c)

    # cross-correlation
    init_delta_jax = cic_paint(jnp.zeros(mesh_shape), poss[-1])
    init_delta     = np.array(init_delta_jax)
    _, cross_c = cross_correlation_coefficients(
        init_delta,
        final_delta,
        boxsize = np.array([25.0, 25.0, 25.0]),
        kmin    = np.pi / 25.0,
        dk      = 2 * np.pi / 25.0,
    )
    pk_c_arr.append(pk_c)
    cross_c_arr.append(cross_c)

print("pk array", pk_c_arr)

# converted this section to use numpy bc it was not working
pk_c        = np.mean( jnp.stack(pk_c_arr),    axis=0 )
cross_c     = np.mean( jnp.stack(cross_c_arr), axis=0 )
pk_c_std    = np.std(  jnp.stack(pk_c_arr),    axis=0 )
cross_c_std = np.std(  jnp.stack(cross_c_arr), axis=0 )


k, pk_ref = power_spectrum(
    cic_paint(np.zeros(mesh_shape), poss[-1]),
    boxsize=np.array([25.0,25.0,25.0]),
    kmin=np.pi/25.0,
    dk=2*np.pi/25.0,
)
k, pk_i = power_spectrum(
    cic_paint(np.zeros(mesh_shape), resi[0][-1]),
    boxsize=np.array([25.0,25.0,25.0]),
    kmin=np.pi/25.0,
    dk=2*np.pi/25.0,
)
k, cross_i = cross_correlation_coefficients(
    cic_paint(np.zeros(mesh_shape), poss[-1]),
    cic_paint(np.zeros(mesh_shape), resi[0][-1]),
    boxsize=np.array([25.0,25.0,25.0]),
    kmin=np.pi/25.0,
    dk=2*np.pi/25.0,
)


# PLOTTTTT 1
col = cmr.eclipse([0.,0.13,0.55,0.85])  
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1],hspace=0) 
ax0 = plt.subplot(gs[0])
ax0.loglog(k, pk_ref,'--',  label='CAMELS',color=col[0])
ax0.loglog(k, pk_i,label='PM without correction',color=col[1])
ax0.loglog(k, pk_c,  label='PM with NN-correction',color=col[2])
ax0.label_outer()
plt.legend(fontsize='large')
ax0.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",fontsize=14)
ax0.set_ylabel(r"$P(k)$", fontsize=14)
ax1 = plt.subplot(gs[1])
ax1.semilogx(k, (pk_i/pk_ref)-1,label='PM without correction',color=col[1])
ax1.semilogx(k, (pk_c/pk_ref)-1,label='PM with NN-correction',color=col[2])   
ax1.set_ylabel(r"$ (P(k) \ / \ P^{Camels}(k))-1$",fontsize=14)
ax1.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",fontsize=14)
ax1.set_ylim(-1,1)
plt.tight_layout()
plt.grid(True)
plt.savefig("images/pm_comparison_i_NPS.png")



##### PLOTTTTT 2
plt.semilogx(k, (pk_i/pk_ref)-1,label='PM without correction',color=col[1])
plt.semilogx(k, (pk_c/pk_ref)-1,label='PM with NN-correction',color=col[2])   
plt.fill_between(k, ((pk_c/pk_ref)-1)-(pk_c_std/pk_c),   ((pk_c/pk_ref)-1)+(pk_c_std/pk_c), alpha=.1,color=col[1])

plt.ylabel(r"$ (P(k) \ / \ P^{Camels}(k))-1$",fontsize=14)
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",fontsize=14)
plt.ylim(-1.5,1.5)
plt.grid(True)
plt.legend(fontsize='large')
plt.tight_layout()
plt.savefig('images/camels_residual_err_CV_0_i_NPS.png')


# PLOTTTTTTTT 3 
plt.semilogx(k, cross_i/np.sqrt(pk_ref*pk_i),label='PM without correction',color=col[1])
plt.semilogx(k, cross_c/np.sqrt(pk_ref*pk_c),label='PM with NN correction',color=col[2])

plt.fill_between(k, (cross_c/np.sqrt(pk_ref*pk_c))-(cross_c_std/cross_c),  (cross_c/np.sqrt(pk_ref*pk_c))+(cross_c_std/cross_c), alpha=.1,color=col[1])

plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",fontsize=14)
plt.axhline(y=1,color=col[0],linestyle='dashed')
plt.ylabel(r"$ P^{cross}(k) \ / \ \sqrt{P^{Camels}(k))P^{PM}(k)}$",fontsize=14)
plt.grid(True)
plt.ylim(.4,1.2)
plt.legend(fontsize='large')
plt.savefig('images/cross_corr_err_CV_0_i_NPS.png')

###### PLOTTTTTT 4
im1=cic_paint(jnp.zeros(mesh_shape), poss[-1]).sum(axis=0)
im2=cic_paint(jnp.zeros(mesh_shape), resi[0][-1]).sum(axis=0)
im3=cic_paint(jnp.zeros(mesh_shape), res[0][-1]).sum(axis=0)
TI=['CAMELS','PM','PM+NN']
image_paths=[im1,im2,im3]

norm=colors.LogNorm(vmax=cic_paint(jnp.zeros(mesh_shape), res[0][-1]).sum(axis=0).max(),
                          vmin=cic_paint(jnp.zeros(mesh_shape), res[0][-1]).sum(axis=0).min())

cmap = cmr.eclipse
ticks_size = 18
f, axes = plt.subplots(1, 3, sharey=True, figsize=(16,5), dpi=90)

for imp, ax, ci in zip(image_paths, axes.ravel(),TI):
    ax.imshow(imp, cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_title(ci, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.)
    cb = f.colorbar(ax.imshow(imp, cmap=cmap, norm=norm), cax=cax)
    cb.ax.tick_params(labelsize=ticks_size)
fig.tight_layout()
plt.savefig('images/cluster_3D_CV_0_i_NPS.png')


################# PLOTTTTT 4
im1=cic_paint_2d(jnp.zeros([256,256]), poss[-1][...,:2]*4, weight=None)
im2=cic_paint_2d(jnp.zeros([256,256]), resi[0][-1][...,:2]*4, weight=None)
im3=cic_paint_2d(jnp.zeros([256,256]), res[0][-1][...,:2]*4, weight=None)
TI=['CAMELS','PM','PM+NN']
image_paths=[im1,im2,im3]


cmap = cmr.eclipse
norm=colors.LogNorm(vmax=cic_paint_2d(jnp.zeros([128,128]), res[0][-1][...,:2]*2, weight=None).max(),
                          vmin=cic_paint_2d(jnp.zeros([128,128]), res[0][-1][...,:2]*2, weight=None).min())
ticks_size = 18
f, axes = plt.subplots(1, 3, sharey=True, figsize=(16,5), dpi=90)


for imp, ax, ci in zip(image_paths, axes.ravel(),TI):
    ax.imshow(imp, cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_title(ci, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.)
    cb = f.colorbar(ax.imshow(imp, cmap=cmap, norm=norm), cax=cax)
    cb.ax.tick_params(labelsize=ticks_size)

plt.savefig('images/cluster_2D_CV_0_i_NPS.png') 
