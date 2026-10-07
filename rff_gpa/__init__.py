"""Random feature Gaussian process attention.

Equations follow arXiv:2610.08578. RFF-CGP is not implemented.
"""

from rff_gpa.attention import (
    exact_gp_attention,
    gram_from_features,
    rff_gp_attention,
    woodbury_inverse,
)
from rff_gpa.elbo import (
    init_train_params,
    linear_head_log_prob,
    mc_elbo,
    negative_mc_elbo,
    sample_frequencies,
    sample_predictive,
    train_step,
)
from rff_gpa.features import (
    project_qkv,
    rbf_kernel,
    rff_features,
    sample_rbf_frequencies,
)

__all__ = [
    "exact_gp_attention",
    "gram_from_features",
    "init_train_params",
    "linear_head_log_prob",
    "mc_elbo",
    "negative_mc_elbo",
    "project_qkv",
    "rbf_kernel",
    "rff_features",
    "rff_gp_attention",
    "sample_frequencies",
    "sample_predictive",
    "sample_rbf_frequencies",
    "train_step",
    "woodbury_inverse",
]
