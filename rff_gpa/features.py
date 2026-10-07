"""Random Fourier features and the RBF spectral density."""

import jax
import jax.numpy as jnp


def project_qkv(x, w_q, w_k, w_v):
    """Project tokens into queries, keys, and values, Eq. (1).

    The PDF writes Q = X W_Q, K = X W_K, V = X W_V with each weight in
    R^{D x D_*}. A later sentence writes V = X W_V^T. This follows Eq. (1).
    """
    return x @ w_q, x @ w_k, x @ w_v


def rff_features(x, omega):
    """Cosine/sine random Fourier feature map, Eq. (13).

    Parameters
    ----------
    x : array, shape (..., D)
        Inputs.
    omega : array, shape (n_freq, D)
        Frequencies. Eq. (13) uses M/2 frequencies to build M features.

    Returns
    -------
    phi : array, shape (..., M)
        M = 2 * n_freq. Column order is cos(w_1), sin(w_1), ..., cos(w_{M/2}),
        sin(w_{M/2}), each scaled by sqrt(2/M).
    """
    if omega.ndim != 2:
        raise ValueError("omega must have shape (n_freq, D)")
    if x.shape[-1] != omega.shape[-1]:
        raise ValueError("x and omega must share the last dimension")
    proj = x @ jnp.swapaxes(omega, -1, -2)
    paired = jnp.stack((jnp.cos(proj), jnp.sin(proj)), axis=-1)
    flat = paired.reshape(x.shape[:-1] + (paired.shape[-2] * paired.shape[-1],))
    m = flat.shape[-1]
    return jnp.sqrt(2.0 / m) * flat


def sample_rbf_frequencies(key, n_freq, input_dim, lengthscale):
    """Draw frequencies from N(0, ell^{-2} I).

    The PDF does not write this law. It is the spectral density of the
    unit-variance RBF in `rbf_kernel`, so Eq. (12) and Eq. (13) estimate that
    kernel. `lengthscale` is a caller argument, not a paper constant.
    """
    if n_freq < 1:
        raise ValueError("n_freq must be positive")
    ell = jnp.asarray(lengthscale)
    omega = jax.random.normal(key, (n_freq, input_dim), dtype=ell.dtype)
    return omega / ell


def rbf_kernel(x, y, lengthscale):
    """Unit-variance RBF Gram matrix, exp(-||x - y||^2 / (2 ell^2)).

    The PDF does not write this closed form. It is the Bochner transform
    (Eq. 11) of N(0, ell^{-2} I), which `rff_features` estimates.
    x has shape (N, D) and y has shape (M, D).
    """
    ell = jnp.asarray(lengthscale, dtype=x.dtype)
    diff = x[:, None, :] - y[None, :, :]
    sq = jnp.sum(diff * diff, axis=-1)
    return jnp.exp(-sq / (2.0 * ell * ell))
