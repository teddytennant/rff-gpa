"""Exact GP attention and its Woodbury random-feature form."""

import jax.numpy as jnp


def gram_from_features(phi_q, phi_k):
    """RFF Gram matrices K_QQ, K_QK, K_KK from Eq. (16).

    Returns the three matrices in that order. K_KQ is K_QK transposed.
    """
    k_qq = phi_q @ jnp.swapaxes(phi_q, -1, -2)
    k_qk = phi_q @ jnp.swapaxes(phi_k, -1, -2)
    k_kk = phi_k @ jnp.swapaxes(phi_k, -1, -2)
    return k_qq, k_qk, k_kk


def exact_gp_attention(k_qq, k_qk, k_kk, v, sigma2):
    """GP attention mean and covariance, Eq. (14) and Eq. (15).

    These are Eq. (9) and Eq. (10) with queries as test inputs, keys as
    training inputs, and values as noisy observations. K_KQ is taken to be
    K_QK transposed, which holds for the symmetric kernel in the paper.

    Returns the predictive mean (L_q, D_v), the full covariance (L_q, L_q),
    and diag(Sigma) of shape (L_q,). The diagonal is not clamped.
    """
    _check_kernel_shapes(k_qq, k_qk, k_kk, v)
    sigma2 = jnp.asarray(sigma2, dtype=k_kk.dtype)
    eye = jnp.eye(k_kk.shape[0], dtype=k_kk.dtype)
    system = k_kk + sigma2 * eye
    mean = k_qk @ jnp.linalg.solve(system, v)
    cov = k_qq - k_qk @ jnp.linalg.solve(system, jnp.swapaxes(k_qk, -1, -2))
    return mean, cov, jnp.diag(cov)


def woodbury_inverse(phi_k, sigma2):
    """Eq. (17) and Eq. (18), the L by L inverse via an M by M factor.

    A = (1/sigma^2) I - (1/sigma^4) Phi_K (I + (1/sigma^2) Phi_K^T Phi_K)^{-1} Phi_K^T.

    This materializes the L by L matrix so tests can compare it with a dense
    inverse. Posterior inference should use `rff_gp_attention`, which does not.
    """
    if phi_k.ndim != 2:
        raise ValueError("phi_k must have shape (L, M)")
    sigma2 = jnp.asarray(sigma2, dtype=phi_k.dtype)
    length, width = phi_k.shape
    eye_l = jnp.eye(length, dtype=phi_k.dtype)
    eye_m = jnp.eye(width, dtype=phi_k.dtype)
    gram = jnp.swapaxes(phi_k, -1, -2) @ phi_k
    inner = eye_m + gram / sigma2
    inner_inv = jnp.linalg.inv(inner)
    correction = phi_k @ inner_inv @ jnp.swapaxes(phi_k, -1, -2)
    return eye_l / sigma2 - correction / (sigma2 * sigma2)


def rff_gp_attention(phi_q, phi_k, v, sigma2):
    """Linear-time posterior mean and diagonal variance, Eq. (19) and Eq. (20).

    Substituting Eq. (18) and cancelling terms yields the equivalent system
    that inverts only an M by M factor:

        mean = Phi_Q (Phi_K^T Phi_K + sigma^2 I)^{-1} Phi_K^T V
        diag(Sigma)_i = sigma^2 phi_Q,i^T (Phi_K^T Phi_K + sigma^2 I)^{-1} phi_Q,i

    The returned diagonal is not clamped. The paper does not clamp it.
    """
    if phi_q.ndim != 2 or phi_k.ndim != 2 or v.ndim != 2:
        raise ValueError("phi_q, phi_k, and v must be rank-2")
    if phi_q.shape[-1] != phi_k.shape[-1]:
        raise ValueError("phi_q and phi_k must share the feature dimension")
    if phi_k.shape[0] != v.shape[0]:
        raise ValueError("phi_k and v must share the key length")
    sigma2 = jnp.asarray(sigma2, dtype=phi_k.dtype)
    eye = jnp.eye(phi_k.shape[-1], dtype=phi_k.dtype)
    system = jnp.swapaxes(phi_k, -1, -2) @ phi_k + sigma2 * eye
    mean = phi_q @ jnp.linalg.solve(system, jnp.swapaxes(phi_k, -1, -2) @ v)
    solved_q = jnp.linalg.solve(system, jnp.swapaxes(phi_q, -1, -2))
    diag_var = sigma2 * jnp.sum(phi_q * jnp.swapaxes(solved_q, -1, -2), axis=-1)
    return mean, diag_var


def _check_kernel_shapes(k_qq, k_qk, k_kk, v):
    if k_kk.ndim != 2 or k_kk.shape[0] != k_kk.shape[1]:
        raise ValueError("k_kk must be square")
    if k_qq.ndim != 2 or k_qq.shape[0] != k_qq.shape[1]:
        raise ValueError("k_qq must be square")
    if k_qk.shape != (k_qq.shape[0], k_kk.shape[0]):
        raise ValueError("k_qk must have shape (L_q, L_k)")
    if v.shape[0] != k_kk.shape[0]:
        raise ValueError("v must have the same length as k_kk")
