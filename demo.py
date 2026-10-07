"""Print a tiny RFF-GPA mean, diagonal variance, and Woodbury relative error.

Lengthscale and sigma^2 below are not stated in the paper.
"""

import jax
import jax.numpy as jnp

from rff_gpa import (
    exact_gp_attention,
    gram_from_features,
    rff_features,
    rff_gp_attention,
    sample_rbf_frequencies,
)


def main():
    key = jax.random.PRNGKey(0)
    length = 6
    dim = 4
    value_dim = 3
    n_freq = 8
    lengthscale = 1.0
    sigma2 = 0.25
    k_q, k_k, k_v, k_w = jax.random.split(key, 4)
    q = jax.random.normal(k_q, (length, dim))
    k = jax.random.normal(k_k, (length, dim))
    v = jax.random.normal(k_v, (length, value_dim))
    omega = sample_rbf_frequencies(k_w, n_freq, dim, lengthscale)
    phi_q = rff_features(q, omega)
    phi_k = rff_features(k, omega)
    mean, diag_var = rff_gp_attention(phi_q, phi_k, v, sigma2)
    k_qq, k_qk, k_kk = gram_from_features(phi_q, phi_k)
    mean_exact, _, _ = exact_gp_attention(k_qq, k_qk, k_kk, v, sigma2)
    rel = jnp.linalg.norm(mean - mean_exact) / jnp.linalg.norm(mean_exact)
    print("mean shape", tuple(mean.shape))
    print("diag variance shape", tuple(diag_var.shape))
    print("relative error", float(rel))


if __name__ == "__main__":
    main()
