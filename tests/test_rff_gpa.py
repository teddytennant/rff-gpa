"""Numerical checks of the RFF-GPA identities in arXiv:2610.08578."""

import jax
import jax.numpy as jnp
import pytest

from rff_gpa import (
    exact_gp_attention,
    gram_from_features,
    init_train_params,
    mc_elbo,
    negative_mc_elbo,
    project_qkv,
    rbf_kernel,
    rff_features,
    rff_gp_attention,
    sample_frequencies,
    sample_predictive,
    sample_rbf_frequencies,
    train_step,
    woodbury_inverse,
)

jax.config.update("jax_enable_x64", True)


def _features(key, length, dim, n_freq):
    k_q, k_k, k_v, k_w = jax.random.split(key, 4)
    q = jax.random.normal(k_q, (length, dim))
    k = jax.random.normal(k_k, (length, dim))
    v = jax.random.normal(k_v, (length, dim))
    omega = jax.random.normal(k_w, (n_freq, dim))
    return q, k, v, rff_features(q, omega), rff_features(k, omega)


def test_feature_map_matches_cosine_average():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 3))
    omega = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    phi = rff_features(x, omega)
    assert phi.shape == (5, 16)
    diff = x[:, None, :] - x[None, :, :]
    phase = jnp.einsum("fd,nmd->nmf", omega, diff)
    expected = jnp.mean(jnp.cos(phase), axis=-1)
    assert jnp.allclose(phi @ phi.T, expected, atol=1e-12, rtol=1e-12)


def test_rbf_approximation_error_decreases_with_m():
    # Fixed seed. Nested frequencies, so larger M reuses the smaller draw.
    # Feature counts 8, 16, 32 stay inside the requested range.
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
    lengthscale = 1.0
    omega = sample_rbf_frequencies(jax.random.PRNGKey(100), 16, 3, lengthscale)
    target = rbf_kernel(x, x, lengthscale)
    scale = jnp.linalg.norm(target)
    errors = []
    for n_freq in (4, 8, 16):
        phi = rff_features(x, omega[:n_freq])
        assert phi.shape[-1] == 2 * n_freq
        errors.append(float(jnp.linalg.norm(phi @ phi.T - target) / scale))
    assert errors[0] > errors[1] > errors[2]
    assert all(jnp.isfinite(err) for err in errors)


def test_woodbury_matches_dense_inverse():
    _, _, _, _, phi_k = _features(jax.random.PRNGKey(2), length=6, dim=3, n_freq=8)
    sigma2 = 0.3
    dense = jnp.linalg.inv(phi_k @ phi_k.T + sigma2 * jnp.eye(phi_k.shape[0]))
    low_rank = woodbury_inverse(phi_k, sigma2)
    assert low_rank.shape == dense.shape
    assert jnp.allclose(low_rank, dense, atol=1e-8, rtol=1e-8)


def test_predictive_mean_matches_exact_gp_attention():
    _, _, v, phi_q, phi_k = _features(jax.random.PRNGKey(3), length=5, dim=4, n_freq=6)
    sigma2 = 0.4
    k_qq, k_qk, k_kk = gram_from_features(phi_q, phi_k)
    mean_exact, _, _ = exact_gp_attention(k_qq, k_qk, k_kk, v, sigma2)
    mean_lr, _ = rff_gp_attention(phi_q, phi_k, v, sigma2)
    # Eq. (19) with the materialized Woodbury inverse, Eq. (18).
    mean_woodbury = phi_q @ phi_k.T @ woodbury_inverse(phi_k, sigma2) @ v
    assert mean_lr.shape == (5, 4)
    assert jnp.allclose(mean_lr, mean_exact, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(mean_woodbury, mean_exact, atol=1e-8, rtol=1e-8)


def test_diag_variance_matches_exact_and_is_nonnegative():
    _, _, v, phi_q, phi_k = _features(jax.random.PRNGKey(4), length=7, dim=3, n_freq=8)
    sigma2 = 0.2
    k_qq, k_qk, k_kk = gram_from_features(phi_q, phi_k)
    _, cov, diag_exact = exact_gp_attention(k_qq, k_qk, k_kk, v, sigma2)
    _, diag_lr = rff_gp_attention(phi_q, phi_k, v, sigma2)
    factor = woodbury_inverse(phi_k, sigma2)
    cov_woodbury = phi_q @ phi_q.T - phi_q @ phi_k.T @ factor @ phi_k @ phi_q.T
    assert diag_lr.shape == (7,)
    assert jnp.allclose(diag_lr, diag_exact, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(cov_woodbury, cov, atol=1e-8, rtol=1e-8)
    # The paper does not clamp. Require non-negativity within 1e-5.
    assert jnp.min(diag_lr) >= -1e-5
    assert jnp.min(diag_exact) >= -1e-5
    # Diagonal variance does not depend on V. Eq. (15) and Eq. (20).
    v_other = jax.random.normal(jax.random.PRNGKey(5), v.shape)
    _, diag_other = rff_gp_attention(phi_q, phi_k, v_other, sigma2)
    assert jnp.allclose(diag_other, diag_lr, atol=1e-12, rtol=1e-12)


def test_queries_and_keys_may_differ_in_length():
    key = jax.random.PRNGKey(6)
    q = jax.random.normal(jax.random.fold_in(key, 1), (4, 3))
    k = jax.random.normal(jax.random.fold_in(key, 2), (8, 3))
    v = jax.random.normal(jax.random.fold_in(key, 3), (8, 2))
    omega = jax.random.normal(jax.random.fold_in(key, 4), (8, 3))
    phi_q = rff_features(q, omega)
    phi_k = rff_features(k, omega)
    sigma2 = 0.5
    k_qq, k_qk, k_kk = gram_from_features(phi_q, phi_k)
    mean_exact, _, diag_exact = exact_gp_attention(k_qq, k_qk, k_kk, v, sigma2)
    mean_lr, diag_lr = rff_gp_attention(phi_q, phi_k, v, sigma2)
    assert mean_lr.shape == (4, 2)
    assert jnp.allclose(mean_lr, mean_exact, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(diag_lr, diag_exact, atol=1e-8, rtol=1e-8)
    assert jnp.min(diag_lr) >= -1e-5


def test_reparameterization_and_mc_sum():
    mu = jnp.array([0.2, -0.4, 0.1])
    s_diag = jnp.array([0.25, 1.0, 0.5])
    key = jax.random.PRNGKey(7)
    omega = sample_frequencies(mu, s_diag, key, n_freq=5)
    xi = jax.random.normal(key, (5, 3))
    assert jnp.allclose(omega, mu + jnp.sqrt(s_diag) * xi, atol=1e-12, rtol=1e-12)

    mean = jnp.ones((4, 2))
    diag_var = jnp.array([1.0, 4.0, 0.25, 0.0])
    y_hat = sample_predictive(mean, diag_var, key, n_samples=3)
    eps = jax.random.normal(key, (3, 4, 2))
    expected = mean + jnp.sqrt(diag_var)[:, None] * eps
    assert jnp.allclose(y_hat, expected, atol=1e-12, rtol=1e-12)

    log_liks = jnp.array([0.1, -0.2, 0.3])
    total = mc_elbo(log_liks)
    assert jnp.allclose(total, jnp.sum(log_liks))
    assert not jnp.allclose(total, jnp.mean(log_liks))


def test_project_qkv_matches_eq1():
    x = jnp.arange(12.0).reshape(4, 3)
    w_q = jnp.ones((3, 2))
    w_k = jnp.full((3, 2), 2.0)
    w_v = jnp.full((3, 1), 0.5)
    q, k, v = project_qkv(x, w_q, w_k, w_v)
    assert jnp.allclose(q, x @ w_q)
    assert jnp.allclose(k, x @ w_k)
    assert jnp.allclose(v, x @ w_v)


def test_train_step_decreases_mc_loss():
    key = jax.random.PRNGKey(0)
    length, dim, value_dim, n_classes, n_freq = 4, 3, 4, 3, 4
    k_q, k_k, k_v, k_p, k_step = jax.random.split(key, 5)
    q = jax.random.normal(k_q, (length, dim))
    k = jax.random.normal(k_k, (length, dim))
    v = jax.random.normal(k_v, (length, value_dim))
    label = jnp.array(1)
    params = init_train_params(k_p, dim, value_dim, n_classes, dtype=jnp.float64)
    loss0 = negative_mc_elbo(q, k, v, label, params, k_step, n_freq, n_samples=1)
    updated, step_loss = train_step(
        q, k, v, label, params, k_step, n_freq, learning_rate=0.5, n_samples=1
    )
    loss1 = negative_mc_elbo(q, k, v, label, updated, k_step, n_freq, n_samples=1)
    assert step_loss.shape == ()
    assert jnp.isfinite(step_loss)
    assert jnp.allclose(step_loss, loss0)
    assert float(loss1) < float(loss0)


def test_train_step_rejects_bad_sample_count():
    with pytest.raises(ValueError):
        sample_predictive(jnp.zeros((2, 2)), jnp.ones((2,)), jax.random.PRNGKey(0), 0)
