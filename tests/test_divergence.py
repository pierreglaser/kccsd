import jax
import jax.numpy as jnp
import pytest
from jax import random
from kwgflows.pytypes import Array, Scalar
from kwgflows.rkhs.kernels import gaussian_kernel
from numpyro import distributions as np_distributions

from calibration.conditional_models.gaussian_models import (
    GaussianConditionalModel,
    kernelized_generalized_fisher_divergence_gaussian_model,
)
from calibration.kernels import (
    discretized_generalized_fisher_divergence,
    generalized_fisher_divergence_gaussian_model,
    kernelized_discretized_generalized_fisher_divergence,
)


@pytest.mark.parametrize("mu", [-1.4, 0.1, 2.5, 11.1])
@pytest.mark.parametrize("sigma", [0.3, 3.4, 5.5])
def test_gfd_p_equal_to_q(mu: float, sigma: float):
    """Check that the approximate generalized Fisher divergence is zero if p == q with a simple example."""
    kernelized_divergence = kernelized_discretized_generalized_fisher_divergence.create(
        np_distributions.Normal(0.8, 1.2),
        random.PRNGKey(1234),
        num_samples=1_000,
        squared=True,
        ground_space_kernel=gaussian_kernel.create(sigma=1.0),
    )
    # generalized Fisher divergence with a standard normal distribution as base measure
    divergence = discretized_generalized_fisher_divergence.create(
        np_distributions.Normal(0.8, 1.2), random.PRNGKey(1234), 1_000
    )

    # define log density of a normal distribution
    p = np_distributions.Normal(mu, sigma)
    grad_log_prob = jax.grad(p.log_prob)

    assert divergence(grad_log_prob, grad_log_prob) == 0
    assert kernelized_divergence(grad_log_prob, grad_log_prob) == 0


@pytest.mark.parametrize(
    "base_dist_mu,base_dist_sigma,mu,sigma",
    [
        (jnp.arange(2), 1.0, jnp.arange(2)[::-1], 1.0),
        (jnp.arange(5)[::-1], 2.0, jnp.arange(5)[::-1], 2.0),
    ],
)
def test_gfd_gaussian_model_p_equals_q(
    base_dist_mu: Array, base_dist_sigma: Scalar, mu: Array, sigma: Scalar
):
    # Check that the Gaussian generalized Fisher divergence is zero if p == q
    # with a simple example
    base_dist = np_distributions.Normal(base_dist_mu, base_dist_sigma)  # type: ignore
    div = generalized_fisher_divergence_gaussian_model(base_dist)
    p = GaussianConditionalModel(mu, sigma)
    d = div(p, p)
    assert d == 0, f"Expected d = 0, but got {d}."

    div_k = kernelized_generalized_fisher_divergence_gaussian_model(base_dist)
    p = GaussianConditionalModel(mu, sigma)
    d = div_k(p, p)
    assert d == 0, f"Expected d = 0, but got {d}."


@pytest.mark.parametrize(
    "base_dist_mu,base_dist_sigma,log_p_mu,log_p_sigma,log_q_mu,log_q_sigma",
    [
        (jnp.arange(2), 1.0, jnp.arange(4, 6), 0.5, jnp.arange(7, 9), 5.0),
        (jnp.arange(3, 6), 2.0, jnp.arange(7, 10), 2.0, jnp.arange(4, 7), 3.0),
    ],
)
def test_gfd_gaussian_model(
    base_dist_mu: Array,
    base_dist_sigma: Scalar,
    log_p_mu: Array,
    log_p_sigma: Scalar,
    log_q_mu: Array,
    log_q_sigma: Scalar,
):
    base_dist = np_distributions.Normal(base_dist_mu, base_dist_sigma)  # type: ignore
    div = generalized_fisher_divergence_gaussian_model.create(base_dist)
    discretized_div = discretized_generalized_fisher_divergence.create(
        base_dist,
        random.PRNGKey(1234),
        1000_000,
    )

    # Check that the discretized version of this divergence approximately equals
    # the value computed using the closed-form expression avaliable in the case of
    # Gaussian models
    p = GaussianConditionalModel(log_p_mu, log_p_sigma)
    q = GaussianConditionalModel(log_q_mu, log_q_sigma)
    assert jnp.allclose(discretized_div(p.score, q.score), div(p, q), rtol=1e-3)

    # Test that the squared version of the generalized Fisher divergence is the
    # square of the non-squared version
    discretized_div_sq = discretized_generalized_fisher_divergence.create(
        base_dist, random.PRNGKey(1234), 1000, squared=True
    )

    discretized_div_nonsq = discretized_generalized_fisher_divergence.create(
        base_dist, random.PRNGKey(1234), 1000, squared=False
    )
    assert jnp.allclose(
        jnp.sqrt(discretized_div_sq(p.score, q.score)),
        discretized_div_nonsq(p.score, q.score),
    )

    div_sq = generalized_fisher_divergence_gaussian_model.create(
        base_dist, squared=True
    )
    div_nonsq = generalized_fisher_divergence_gaussian_model.create(
        base_dist, squared=False
    )
    assert jnp.allclose(jnp.sqrt(div_sq(p, q)), div_nonsq(p, q))


@pytest.mark.parametrize(
    "base_dist_mu,base_dist_sigma,log_p_mu,log_p_sigma,log_q_mu,log_q_sigma",
    [
        (jnp.arange(3), 0.5, jnp.array([1, 3, 5]), 3.0, jnp.arange(2, 5), 4.0),
    ],
)
def test_kernelized_gfd_gaussian_model(
    base_dist_mu: Array,
    base_dist_sigma: Scalar,
    log_p_mu: Array,
    log_p_sigma: Scalar,
    log_q_mu: Array,
    log_q_sigma: Scalar,
):
    base_dist = np_distributions.Normal(base_dist_mu, base_dist_sigma)  # type: ignore
    div = kernelized_generalized_fisher_divergence_gaussian_model.create(
        base_dist, gaussian_kernel.create(sigma=0.8)
    )
    discretized_div = kernelized_discretized_generalized_fisher_divergence.create(
        base_dist,
        random.PRNGKey(1234),
        gaussian_kernel.create(sigma=0.8),
        8000,
    )

    # Check that the discretized version of this divergence approximately equals
    # the value computed using the closed-form expression avaliable in the case of
    # Gaussian models
    p = GaussianConditionalModel(log_p_mu, log_p_sigma)
    q = GaussianConditionalModel(log_q_mu, log_q_sigma)
    ddiv_val = discretized_div(p.score, q.score)
    div_val = div(p, q)
    assert jnp.allclose(ddiv_val, div_val, rtol=1e-3, atol=1e-2)

    # Test that the squared version of the generalized Fisher divergence is the
    # square of the non-squared version
    discretized_div_sq = discretized_generalized_fisher_divergence.create(
        base_dist, random.PRNGKey(1234), 1000, squared=True
    )

    discretized_div_nonsq = discretized_generalized_fisher_divergence.create(
        base_dist, random.PRNGKey(1234), 1000, squared=False
    )
    assert jnp.allclose(
        jnp.sqrt(discretized_div_sq(p.score, q.score)),
        discretized_div_nonsq(p.score, q.score),
    )

    div_sq = generalized_fisher_divergence_gaussian_model.create(
        base_dist, squared=True
    )
    div_nonsq = generalized_fisher_divergence_gaussian_model.create(
        base_dist, squared=False
    )
    assert jnp.allclose(jnp.sqrt(div_sq(p, q)), div_nonsq(p, q))
