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
from nn import SimpleCNN
#import partial

import readgadget
import optax
from functools import partial

from flax import linen as nn
from flax.training import train_state

FLAGS = flags.FLAGS

flags.DEFINE_string("filename", "correction_params_stellar/starnn_flax.params", "Output filename")
flags.DEFINE_string("training_sims",'/gpfs02/work/diffusion/neural_ode/CV0',"Simulations used")
flags.DEFINE_float("Omega_m", 0.3 - 0.049, "")
flags.DEFINE_float("Omega_b",0.049, "")
flags.DEFINE_float("sigma8", 0.8, "")
flags.DEFINE_float("n_s", 0.9624, "")
flags.DEFINE_float("h", 0.6711, "")
flags.DEFINE_integer("mesh_shape", 64, "")
flags.DEFINE_float("box_size", 25., "")
flags.DEFINE_integer("niter", 500, "")
flags.DEFINE_float("learning_rate", 0.01, "")
flags.DEFINE_boolean("custom_weight", True, "")
flags.DEFINE_float("lambda_2", 1., "")
flags.DEFINE_float("lambda_1", 0.1, "")


# adding in for initial conditions

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




def loss_fn(params, cosmo, target_pos, target_vel, target_pk, scales, model):
    # integrate the neural ODE
    print("loss function called")
    ode_fn = make_neural_ode_fn(model,
                                [FLAGS.mesh_shape]*3)
    res = odeint(ode_fn,
                 (target_pos[0], target_vel[0]),
                 jnp.array(scales),
                 cosmo, 
                 params,
                 rtol=1e-5, atol=1e-5)
    # weight by displacement
    dist2 = jnp.sum((res[0] - target_pos)**2, axis=-1)
    w     = jnp.where(dist2 < 100, dist2, 0.)
    # spectrum as in prev code
    k, pk = jax.vmap(lambda x: power_spectrum(
        cic_paint(jnp.zeros((FLAGS.mesh_shape,)*3), x),
        boxsize=np.array([FLAGS.box_size]*3),
        kmin=jnp.pi/FLAGS.box_size, dk=2*jnp.pi/FLAGS.box_size

    ))(res[0])
    #spec_loss = jnp.mean(jnp.sum((pk/target_pk - 1)**2, axis=-1)) # commented out to turn off powerspec compoenent
    if FLAGS.custom_weight:
        return FLAGS.lambda_2*jnp.mean(w) #+ FLAGS.lambda_1*spec_loss
    else:
        return #FLAGS.lambda_1*spec_loss

@jax.jit
def train_step(state, cosmo, target_pos, target_vel, target_pk, scales):
    loss, grads = jax.value_and_grad(loss_fn)(state.params,
                                               cosmo,
                                               target_pos,
                                               target_vel,
                                               target_pk,
                                               scales,
                                               state.apply_fn)
    state = state.apply_gradients(grads=grads)
    print(f"Loss: {loss} | ")
    return state, loss

class CNNTrainState(train_state.TrainState):
    pass


def create_train_state(rng, model_loaded):
    dummy_x = jnp.zeros((FLAGS.mesh_shape,)*3)
    dummy_a = jnp.ones((1,))
    params = model.init(rng, dummy_x, dummy_a)['params']
    tx = optax.adam(FLAGS.learning_rate)
    return CNNTrainState.create(apply_fn=model.apply, params=params, tx=tx)


def main(_):
    # cosmology
    cosmo = jc.Planck15(Omega_c=FLAGS.Omega_m-FLAGS.Omega_b,
                        Omega_b=FLAGS.Omega_b,
                        n_s=FLAGS.n_s, h=FLAGS.h, sigma8=FLAGS.sigma8)

    scales = []
    scales.append(a_i)
    poss   = []
    poss.append(pos_i)
    vels   = []
    vels.append(vel_i)
    re = 256 // FLAGS.mesh_shape

    snap_paths = sorted(glob.glob(
        os.path.join(FLAGS.training_sims, "snapshot_*.hdf5")
    ))#[:2]
'''
    for snapshot in tqdm(snap_paths, desc="Loading snapshots"):
        header   = readgadget.header(snapshot)
        redshift = header.redshift
        BoxSize  = header.boxsize / 1e3  # Mpc/h

        # sort by ID
        ids     = np.argsort(readgadget.read_block(snapshot, "ID  ", [1]) - 1)
        raw_pos = readgadget.read_block(snapshot, "POS ", [1])[ids] / 1e3
        raw_vel = readgadget.read_block(snapshot, "VEL ", [1])[ids]

        # reshape downsample
        pos = (
            raw_pos
            .reshape(re, re, re, FLAGS.mesh_shape, FLAGS.mesh_shape, FLAGS.mesh_shape, 3)
            .transpose(0,3,1,4,2,5,6)
            .reshape(-1,3)
        )
        vel = (
            raw_vel
            .reshape(re, re, re, FLAGS.mesh_shape, FLAGS.mesh_shape, FLAGS.mesh_shape, 3)
            .transpose(0,3,1,4,2,5,6)
            .reshape(-1,3)
        )

        # to mesh units
        pos = (
            pos / BoxSize * FLAGS.mesh_shape
        ).reshape(256,256,256,3)[::re,::re,::re].reshape(-1,3)
        vel = (
            vel / 100 * (1.0/(1+redshift)) / BoxSize * FLAGS.mesh_shape
        ).reshape(256,256,256,3)[::re,::re,::re].reshape(-1,3)

        scales.append(1.0 / (1.0 + redshift))
        poss.append(pos)
        vels.append(vel)
    '''
    #reference arrays
    ref_pos = jnp.stack(poss, axis=0)
    ref_vel = jnp.stack(vels, axis=0)
    ref_pk  = jax.vmap(lambda x: power_spectrum(
                  cic_paint(jnp.zeros((FLAGS.mesh_shape,)*3), x),
                  boxsize=np.array([FLAGS.box_size]*3),
                  kmin=jnp.pi/FLAGS.box_size,
                  dk=2*jnp.pi/FLAGS.box_size
               )[1])(ref_pos)
    
    rng_seq = PRNGSequence(0)
    init_key = next(rng_seq)
    #state    = create_train_state(init_key, model)

    model = SimpleCNN(num_channels=64, num_layers=3)
    optimizer = optax.adam(FLAGS.learning_rate) # define optimizer
    
    # try to load existing parameters
    try:
        with open(FLAGS.filename, "rb") as f:
            loaded_params = pickle.load(f)
        print(f"▶️ Loaded pretrained params from {FLAGS.filename}")
        # if found make train state
        state = NNTrainState.create(
            apply_fn=model,
            params=loaded_params,
            tx=optimizer
        )
    except FileNotFoundError:
        print("❗ No pretrained params found, initializing fresh.")
        state = create_train_state(init_key, model)  # model.init + optimizer.init

    
    losses = []
    for step in range(FLAGS.niter):
        print("step:", step)
        state, loss = train_step(
            state,
            cosmo,
            ref_pos,
            ref_vel,
            ref_pk,
            scales
        )
        print("train step done")
        losses.append(loss)

    out_path = FLAGS.filename
    out_dir  = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(FLAGS.filename, "wb") as f:
        pickle.dump(state.params, f)
    print(f"Saved params to {FLAGS.filename}")


if __name__ == "__main__":
    app.run(main)