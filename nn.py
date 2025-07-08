import jax
import jax.numpy as jnp
from flax import linen as nn


def _deBoorVectorized(x, t, c, p):
    """
    Evaluates S(x).

    Args
    ----
    x: position
    t: array of knot positions, needs to be padded as described above
    c: array of control points
    p: degree of B-spline
    """
    k = jnp.digitize(x, t) - 1

    d = [c[j + k - p] for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter(nn.Module):
    n_knots:    int = 8
    latent_size:int = 32 # decreased from 32 # increased from 16 to improve performance

    @nn.compact
    def __call__(self, x, a):
        """
        x: array of scales (normalized to fftfreq default)
        a: scalar scale factor
        """
        # pass 'a' through a tiny two-layer net with sin activations
        a_vec = jnp.atleast_1d(a)
        net   = jnp.sin( nn.Dense(self.latent_size)(a_vec) )
        net   = jnp.sin( nn.Dense(self.latent_size)(net) )

        # predict weights and knot-spacings
        w = nn.Dense(self.n_knots + 1)(net)   # control-point magnitudes
        k = nn.Dense(self.n_knots - 1)(net)   # interior knot deltas

        # build a valid knot vector in [0,1]:
        k = jnp.concatenate([ jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k)) ])

        # build control points (we prepend a zero so len(w)==n_knots+1)
        w = jnp.concatenate([ jnp.zeros((1,)), w ])

        # pad knots at both ends with multiplicity p=3
        ak = jnp.concatenate([
            jnp.zeros((3,)),   # three zeros
            k,                  # interior knots
            jnp.ones((3,))      # three ones
        ])

        # evaluate the cubic B-spline (p=3), clamping x into [0,1-ε]
        xp = jnp.clip(x / jnp.sqrt(3), 0.0, 1.0 - 1e-4)
        return _deBoorVectorized(xp, ak, w, 3)