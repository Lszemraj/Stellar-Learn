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

from stellar_nn import StarCNN
from stellar_utils import make_nn_stellar_ode_fn 


model = StarCNN(num_channels=1, num_layers=2, kernel_size=3)
dummy_input = jnp.ones((1, 64, 64, 64, 1))
rng = jax.random.PRNGKey(0)
params_dummy = model.init(rng, dummy_input)


pk_c_arr = []
cross_c_arr = []

mesh_shape = [64, 64, 64]
box_size = [64., 64., 64.]
snapshots = jnp.array([0.1, 0.5, 1.0])
cosmo = jc.Planck15(Omega_c=0.2, sigma8=0.8)

for filename in os.listdir("training/"):
    print("filename = ", filename)

    with open(os.path.join("training/", filename), "rb") as f:
        params = pickle.load(f)

    # Construct RHS function using CNN model and params
    ode_rhs_fn = make_nn_stellar_ode_fn(mesh_shape, model, params)

    init_state = (poss[0], vels[0], jnp.zeros(mesh_shape))

    # Evolve using odeint
    res = odeint(
        ode_rhs_fn,
        init_state,
        jnp.array(scales),
        cosmo,
        rtol=1e-5,
        atol=1e-5
    )

    # Final CNN-predicted stellar density field
    final_density = np.array(res[0][-1])  # sff at final step

    # Power spectrum
    _, pk_c = power_spectrum(
        final_density,
        boxsize = np.array([25.0, 25.0, 25.0]),
        kmin    = np.pi / 25.0,
        dk      = 2 * np.pi / 25.0,
    )
    pk_c_arr.append(pk_c)

    # Cross-correlation with initial density field
    init_delta = np.array(cic_paint(jnp.zeros(mesh_shape), poss[-1]))
    _, cross_c = cross_correlation_coefficients(
        init_delta,
        final_density,
        boxsize = np.array([25.0, 25.0, 25.0]),
        kmin    = np.pi / 25.0,
        dk      = 2 * np.pi / 25.0,
    )
    cross_c_arr.append(cross_c)


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


version = 1

####################################################################### PLOTTING 

#col = cmr.eclipse([0., 0.13, 0.55, 0.85])
version = version or "combined"


fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

# power spectra
ax = axes[0]
ax.loglog(k, pk_ref, "--", label="CAMELS", color = 'g' )#= col[0])
ax.loglog(k, pk_i,   "-",  label="PM",      color = 'r' )#=col[1])
ax.loglog(k, pk_c,   "-",  label="PM+NN",   color = 'b' )#=col[2])
ax.set_title(r"$P(k)$", fontsize=14)
ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
ax.grid(True, which="both", ls=":")
ax.legend(frameon=False)

# fractional matter power spectrum
ax = axes[1]
res_i = (pk_i/pk_ref) - 1
res_c = (pk_c/pk_ref) - 1
ax.semilogx(k, res_i, label="PM − 1",   color= 'r' ) #=col[1])
ax.semilogx(k, res_c, label="PM+NN − 1",color= 'b' ) #col[2])
# ±1σ band around PM+NN
ax.fill_between(k,
    res_c - (pk_c_std/pk_c),
    res_c + (pk_c_std/pk_c),
    color=col[2], alpha=0.2
)
ax.set_title("Fractional $P(k)/P_{\\rm CAMELS}-1$", fontsize=14)
ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$(P/P_{\rm ref}) - 1$")
ax.set_ylim(-1.5, 1.5)
ax.grid(True, which="both", ls=":")
ax.legend(frameon=False)

# Cross-correlation coefficient
ax = axes[2]
cc_i = cross_i/np.sqrt(pk_ref*pk_i)
cc_c = cross_c/np.sqrt(pk_ref*pk_c)
ax.semilogx(k, cc_i, label="PM",   color= 'r' )   #color=col[1])
ax.semilogx(k, cc_c, label="PM+NN",  color= 'b' ) # color=col[2])
# ±1σ band around PM+NN
ax.fill_between(k,
    res_c - (pk_c_std/ pk_ref),  # ← correct denominator
    res_c + (pk_c_std/ pk_ref),
    color=col[2], alpha=0.2
)
ax.axhline(1.0, color=col[0], ls="--")
ax.set_title(r"Cross-corr $r(k)$", fontsize=14)
ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$r(k)$")
ax.set_ylim(0.4, 1.2)
ax.grid(True, which="both", ls=":")
ax.legend(frameon=False)

plt.suptitle("Original")
plt.tight_layout()

outpath = f"images/summary_PS_xcorr_{version}.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
