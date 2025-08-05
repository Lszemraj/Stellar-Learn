import jax
import jax.numpy as jnp
from flax import linen as nn


#CNN for Star model
class StarCNN(nn.Module):
    num_channels: int = 1  #channels
    num_layers: int = 1     #conv layers
    kernel_size: int = 1

    def periodic_pad(self, x, pad):
        return jnp.pad(
            x,
            ((0, 0),      # batch dim
             (pad, pad),  # height
             (pad, pad),  # width
             (pad, pad),  # depth
             (0, 0)),     # channels
         mode='wrap'
     )

    @nn.compact
    def __call__(self, x):
        pad = self.kernel_size // 2  
        for _ in range(self.num_layers):
            x = self.periodic_pad(x, pad)
            x = nn.Conv(
                features=self.num_channels,
                kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                strides=(1, 1, 1),
                padding='VALID' 
            )(x)
            x = nn.selu(x)
        return x
         

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
    
        a_vec = jnp.atleast_1d(a)
        net   = jnp.sin( nn.Dense(self.latent_size)(a_vec) )
        net   = jnp.sin( nn.Dense(self.latent_size)(net) )

        w = nn.Dense(self.n_knots + 1)(net)  
        k = nn.Dense(self.n_knots - 1)(net)  

        k = jnp.concatenate([ jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k)) ])
        w = jnp.concatenate([ jnp.zeros((1,)), w ])

        ak = jnp.concatenate([
            jnp.zeros((3,)),  
            k,               
            jnp.ones((3,))    
        ])

        xp = jnp.clip(x / jnp.sqrt(3), 0.0, 1.0 - 1e-4)
        return _deBoorVectorized(xp, ak, w, 3)