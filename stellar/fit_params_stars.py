import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ[
    'XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
from jax.experimental.ode import odeint
from base import PRNGSequence
from jaxpm.painting import cic_paint, cic_read
from pm import make_neural_ode_fn
from kernels import fftk, gradient_kernel, invlaplace_kernel as laplace_kernel, longrange_kernel
from nn import NeuralSplineFourierFilter  # the Flax version
from jaxpm.utils import power_spectrum
import glob
from stellar_nn import StarCNN
import readgadget
import optax
from functools import partial
from flax import linen as nn
from flax.training import train_state
from jaxpm.pm import linear_field, lpt, make_ode_fn
from stellar_utils import make_nn_stellar_ode_fn 

############## TRAINING CODE
#@jax.jit(static_argnames=["model"])
def loss_fn(params, nn_model, initial_state, snapshots, target, cosmo):
    ode_fn = make_nn_stellar_ode_fn(mesh_shape, nn_model, params)
    result = odeint(ode_fn, initial_state, snapshots, cosmo, rtol=1e-5, atol=1e-5)
    pred_density = result[-1][-1]  # final stellar density
    return jnp.mean((pred_density - target)**2)  # MSE loss

#@jax.jit
#@jax.jit(static_argnames=["nn_model"])
def train_step(params, model, initial_state, snapshots, target, cosmo):
    grads = jax.grad(loss_fn)(params, model, initial_state, snapshots, target, cosmo)
    return jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, params, grads)  # manual SGD


loss_fn = jax.jit(loss_fn, static_argnames=["nn_model"])
train_step = jax.jit(train_step, static_argnames=["model"])


camels_baseline = np.load("/gpfs02/work/diffusion/gridsMstar/Grids_Mstar_IllustrisTNG_CV_128_z=0.0.npy")
camels_baseline_64 = camels_baseline.reshape(27, 64, 2, 64, 2, 64, 2).mean(axis=(2, 4, 6))


mesh_shape = [64, 64, 64]
box_size = [64., 64., 64.]
snapshots = jnp.array([0.1, 0.5, 1.0])

k = jnp.logspace(-4, 1, 128)
pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=0.2, sigma8=0.8), k)
pk_fn = lambda x: jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

initial_conditions = linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(0))
particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),axis=-1).reshape([-1,3])
cosmo = jc.Planck15(Omega_c=0.2, sigma8=0.8)
dx, p, f = lpt(cosmo, initial_conditions, particles, a=0.1)

rng = jax.random.PRNGKey(0)
mesh_shape = (64, 64, 64)
model = StarCNN(num_channels=1, num_layers=2, kernel_size=3)

dummy_input = jnp.ones((1, 64, 64, 64, 1))  # batch=1, channels=1
params = model.init(rng, dummy_input)
#params = model.init(rng, (particles + dx, p, jnp.zeros(mesh_shape)), 0.1, cosmo)

initial_state = (particles + dx, p, jnp.zeros(mesh_shape))
target = camels_baseline_64[0]
snapshots = jnp.array([0.1, 0.5, 1.0])

losses = []

for epoch in range(100):
    params = train_step(params, model, initial_state, snapshots, target, cosmo)
    if epoch % 10 == 0:
        loss_val = loss_fn(params, model, initial_state, snapshots, target, cosmo)
        print(f"Epoch {epoch}: Loss = {loss_val:.6f}")
        losses.append(float(loss_val))

out_path = "training/starcnn_params.pkl"
out_dir  = os.path.dirname(out_path)

if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump({"params": params}, f)

print(f"Saved trained params to {out_path}")
