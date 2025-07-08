import functools
import itertools
import jax
import jax.numpy as jnp
import jaxpm
from kernels import fftk, gradient_kernel, invlaplace_kernel, longrange_kernel
import numpy as np
from jaxpm.painting import cic_paint, cic_read, compensate_cic,cic_paint_2d
from flax import linen as nn

def PRNGSequence(seed):
    """Yields an endless sequence of PRNGKeys."""
    rng = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
    while True:
        rng, key = jax.random.split(rng)
        yield key

class PGDCorrection(nn.Module):
    mesh_shape: tuple  # e.g. (nx, ny, nz)

    @staticmethod
    def PGD_kernel(kvec, kl, ks):
        """Vectorized PGD filter in Fourier space."""
        kk = sum(ki**2 for ki in kvec)
        # avoid division by zero at k=0
        zero_mask = (kk == 0)
        kk = jnp.where(zero_mask, 1.0, kk)

        kl2 = kl**2
        ks4 = ks**4

        # two Gaussian falloffs
        v = jnp.exp(-kl2 / kk) * jnp.exp(-kk**2 / ks4)
        # zero out the k=0 mode
        v = jnp.where(zero_mask, 0.0, v)
        return v

    @nn.compact
    def __call__(self, pos, cosmo):
        """
        pos: [npart, 3] positions array
        cosmo: cosmology object
        """
        # 1) learnable parameters
        α  = self.param('alpha', nn.initializers.ones, ())  # scalar
        kl = self.param('kl',    nn.initializers.ones, ())  # scalar
        ks = self.param('ks',    nn.initializers.ones, ())  # scalar

        # 2) build k-vector grid
        kvec = fftk(self.mesh_shape)  # your existing FFT grid utility

        # 3) paint density and FFT → δ_k
        delta   = cic_paint(jnp.zeros(self.mesh_shape), pos)
        delta_k = jnp.fft.rfftn(delta)

        # 4) apply Laplace kernel + PGD filter
        pgd_range = PGDCorrection.PGD_kernel(kvec, kl, ks)
        pot_k_pgd = delta_k * invlaplace_kernel(kvec) * pgd_range

        # 5) back to real space forces
        forces_pgd = jnp.stack([
            cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k_pgd), pos)
            for i in range(3)
        ], axis=-1)

        # 6) scaled drift term
        dpos_pgd = α * forces_pgd
        return dpos_pgd


