import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
#import partial

import readgadget
import optax
from functools import partial

from flax import linen as nn
from flax.training import train_state

import jax.numpy as jnp
import jax_cosmo as jc

from distributed import fft3d, ifft3d, normal_field
from jaxpm.growth import (dGf2a, dGfa, growth_factor, growth_factor_second,
                          growth_rate, growth_rate_second)
from jaxpm.kernels import (PGD_kernel, fftk, gradient_kernel,
                           invlaplace_kernel, longrange_kernel)
from jaxpm.painting import cic_paint, cic_read
from stellar_nn import StarCNN

from flax.training import train_state

from jaxpm.pm import linear_field, lpt, make_ode_fn


def make_nn_stellar_ode_fn(mesh_shape, model, params):

    def neural_stellarbody_ode(state, a, cosmo):
        """
        state is (position, velocities, density)
        """

        pos, vel, den = state
        delta = cic_paint(jnp.zeros(mesh_shape), pos)
        print(delta.shape, pos.shape, vel.shape)
        delta_k = jnp.fft.rfftn(delta)
        #print(delta_k.shape)
        kvec = fftk(delta.shape)

        # gravitational potential calc
        pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec,
                                                                     r_split=0)

        #  correction filter
        kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))

        # gravitational force calc
        forces = jnp.stack([
            cic_read(jnp.fft.irfftn(-gradient_kernel(kvec, i) * pot_k), pos)
            for i in range(3)
        ],
                           axis=-1)

        forces = forces * 1.5 * cosmo.Omega_m
        
        delta_cnn_input = delta[None, ..., None]  # shape: (1, 64, 64, 64, 1) 
        sff = model.apply(params, delta_cnn_input)[0, ..., 0]

        #sff = 0.0001*delta**2.0 # star formation field? placeholder as I don't think this is right
        # position update
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # velovity update 
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel, sff
    
    return neural_stellarbody_ode