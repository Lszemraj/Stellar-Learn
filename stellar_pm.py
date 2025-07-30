import jax.numpy as jnp
import jax_cosmo as jc

from distributed import fft3d, ifft3d, normal_field
from jaxpm.growth import (dGf2a, dGfa, growth_factor, growth_factor_second,
                          growth_rate, growth_rate_second)
from jaxpm.kernels import (PGD_kernel, fftk, gradient_kernel,
                           invlaplace_kernel, longrange_kernel)
from jaxpm.painting import cic_paint, cic_read
from stellar_nn import StarCNN



def make_nn_stellar_ode_fn(model, mesh_shape):

    def neural_stellarbody_ode(state, a, cosmo, params):
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
        pot_k = pot_k * (1. + model.apply({'params': params}, kk, jnp.atleast_1d(a)))

        # gravitational force calc
        forces = jnp.stack([
            cic_read(jnp.fft.irfftn(-gradient_kernel(kvec, i) * pot_k), pos)
            for i in range(3)
        ],
                           axis=-1)

        forces = forces * 1.5 * cosmo.Omega_m

        delta_den = cic_paint(jnp.zeros(mesh_shape), den) # evolved stellar field

        sff = StarCNN(delta, delta_den) # star formation field? placeholder as I don't think this is right
        div = sff / delta 

        ddden = cic_read(div, delta) # delta density
        print("dden", dden)
        # position update
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # velovity update 
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel, ddden

    return neural_stellarbody_ode