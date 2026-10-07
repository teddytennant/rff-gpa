JAX implementation of random feature Gaussian process attention from https://arxiv.org/abs/2610.08578.

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

## What does not match the paper

The feature map is Eq. (13): M/2 frequencies, cosine and sine pairs, scale sqrt(2/M). The phase-shift map sqrt(2/M) cos(omega^T x + b) is not in the PDF, so it is not implemented.

The PDF does not write a closed-form RBF kernel or a lengthscale. `rbf_kernel` and `sample_rbf_frequencies` are the unit-variance squared exponential whose spectral density is N(0, ell^{-2} I). That is the Bochner specialization of Eq. (11) that makes Eq. (13) an unbiased estimator of that kernel. Callers pass `lengthscale`. It is not a paper hyperparameter. The demo uses lengthscale 1 and sigma^2 0.25. Neither value is stated in the paper.

Eq. (22) writes a product of M Gaussians. Eq. (13) draws M/2 frequencies, and Appendix D draws 32 frequencies to form 64 cosine and sine features. This code draws one frequency per factor and builds 2 * n_freq features, matching Eq. (13) and Appendix D. Each frequency gets its own noise vector. Eq. (24) omits the frequency index on xi.

Eq. (26) is a sum of log-likelihoods, not an average. The training loss is the negation of that sum. Algorithm 1 uses Adam. The train step is plain SGD, so the package depends only on jax and numpy. The learning rate is an argument. The number of Monte Carlo samples is an argument. Appendix D uses one sample at training time. That choice is not fixed here.

Positive S and sigma^2 are obtained with softplus on unconstrained raw parameters. The paper does not state that parameterization. Initial mu is 0, the raw spectral and noise parameters are 0, and the linear head is normal with scale 0.1. None of those initial values are in the paper.

The categorical term uses a linear map on the first token of the predictive sample. That is a stand-in for the prediction-head likelihood in the text around Eq. (21) and Eq. (26). It is not a ViT, not a [CLS] token, and not the Transformer output projection W_o. Multi-head attention is omitted. No image or text model is implemented, and no accuracy, ECE, or calibration number from the paper is reproduced.

RFF-CGP, Eq. (27) through Eq. (30), is omitted.

Appendix D adds a numerical jitter of 1e-6 in the solves. The equations do not, and this code does not add jitter. The returned diagonal variance is not clamped. The paper does not clamp it.

A sentence in the method section writes V = X W_V^T. Eq. (1) writes V = X W_V with W_V in R^{D x D_V}. `project_qkv` follows Eq. (1).
