"""Monte Carlo ELBO draw and one training step."""

import jax
import jax.numpy as jnp

from rff_gpa.attention import rff_gp_attention
from rff_gpa.features import rff_features


def sample_frequencies(mu, s_diag, key, n_freq):
    """Reparameterized spectral draw, Eq. (24).

    Eq. (22) is a product of independent Gaussians, so each frequency gets its
    own noise vector. The PDF writes xi^{(s)} without a frequency index.
    mu and s_diag have shape (D,). s_diag holds the diagonal of S and must be
    positive. Returns omega of shape (n_freq, D).
    """
    if n_freq < 1:
        raise ValueError("n_freq must be positive")
    xi = jax.random.normal(key, (n_freq, mu.shape[0]), dtype=mu.dtype)
    return mu + jnp.sqrt(s_diag) * xi


def sample_predictive(mean, diag_var, key, n_samples):
    """Reparameterized predictive draw, Eq. (25).

    The paper writes the draw column by column and treats the D output
    coordinates as independent GPs, so the noise is independent across those
    columns. mean has shape (L, D_v) and diag_var has shape (L,).
    Returns samples of shape (n_samples, L, D_v).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    eps = jax.random.normal(key, (n_samples,) + mean.shape, dtype=mean.dtype)
    std = jnp.sqrt(diag_var)[:, None]
    return mean[None, :, :] + std[None, :, :] * eps


def linear_head_log_prob(y_hat, label, w_head):
    """Categorical log-likelihood of one sequence label.

    The PDF defines this term as the prediction-head likelihood p(labels | Y).
    This stand-in reads the first token of Y_hat and applies a linear map.
    It is not the paper's ViT [CLS] head and not W_o.
    """
    logits = y_hat[0] @ w_head
    return jax.nn.log_softmax(logits)[jnp.asarray(label)]


def mc_elbo(log_liks):
    """Eq. (26): sum of per-sample log-likelihoods, not the average."""
    return jnp.sum(log_liks)


def init_train_params(key, input_dim, value_dim, n_classes, dtype=None):
    """Unconstrained parameters for `train_step`.

    mu starts at 0. s_raw and sigma_raw start at 0, so softplus gives about
    0.693 for the spectral diagonal and for sigma^2. w_head is normal with
    scale 0.1. None of these initial values are stated in the paper.
    """
    if dtype is None:
        dtype = jnp.result_type(float)
    weight_key, _ = jax.random.split(key)
    return {
        "mu": jnp.zeros((input_dim,), dtype=dtype),
        "s_raw": jnp.zeros((input_dim,), dtype=dtype),
        "sigma_raw": jnp.zeros((), dtype=dtype),
        "w_head": 0.1 * jax.random.normal(weight_key, (value_dim, n_classes), dtype=dtype),
    }


def negative_mc_elbo(q, k, v, label, params, key, n_freq, n_samples=1):
    """Negation of Eq. (26), the scalar minimized by `train_step`.

    params uses unconstrained s_raw and sigma_raw. Positive S and sigma^2 are
    softplus(s_raw) and softplus(sigma_raw). The paper does not state that
    parameterization. S in the PDF is the diagonal covariance, not this raw value.
    """
    elbo = _paired_elbo(q, k, v, label, params, key, n_freq, n_samples)
    return -elbo


def train_step(q, k, v, label, params, key, n_freq, learning_rate, n_samples=1):
    """One SGD step on the negative MC ELBO.

    Algorithm 1 uses Adam. This step is plain SGD so the package depends only
    on jax and numpy. learning_rate is a caller argument, not a paper constant.
    Returns the updated params and the pre-step loss, which is a finite scalar
    when the draw is finite.
    """
    loss, grads = jax.value_and_grad(_loss_of_params)(
        params, q, k, v, label, key, n_freq, n_samples
    )
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return updated, loss


def _loss_of_params(params, q, k, v, label, key, n_freq, n_samples):
    return negative_mc_elbo(q, k, v, label, params, key, n_freq, n_samples)


def _paired_elbo(q, k, v, label, params, key, n_freq, n_samples):
    s_diag = jax.nn.softplus(params["s_raw"])
    sigma2 = jax.nn.softplus(params["sigma_raw"])
    key_w, key_e = jax.random.split(key)
    keys_w = jax.random.split(key_w, n_samples)
    keys_e = jax.random.split(key_e, n_samples)

    def one_sample(key_omega, key_eps):
        omega = sample_frequencies(params["mu"], s_diag, key_omega, n_freq)
        phi_q = rff_features(q, omega)
        phi_k = rff_features(k, omega)
        mean, diag_var = rff_gp_attention(phi_q, phi_k, v, sigma2)
        y_hat = sample_predictive(mean, diag_var, key_eps, 1)[0]
        return linear_head_log_prob(y_hat, label, params["w_head"])

    log_liks = jax.vmap(one_sample)(keys_w, keys_e)
    return mc_elbo(log_liks)
