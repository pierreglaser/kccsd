import logging
from typing import Any, Tuple, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
import pytest
from jax import jit, random
from kwgflows.pytypes import Array, PRNGKeyArray, Scalar
from kwgflows.rkhs.kernels import base_kernel, gaussian_kernel

from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.kernels import (
    BM,
    BM2,
    DiscreteExpFisherKernel,
    DiscreteExpKernelizedFisherKernel,
    DiscreteExpMMDKernel,
    ExpFisherKernel,
    ExpKernelizedFisherKernel,
    ExpMMDGaussianKernel,
    ExpMMDKernel,
    GaussianExpFisherKernel,
    GaussianExpKernelizedFisherKernel,
    GaussianExpWassersteinKernel,
    kernel_on_models,
)


def test_expwassersteinkernel():
    """Check correctness of the ExpWassersteinKernel with simple examples."""

    # Define kernel with random length scale
    key = jax.random.PRNGKey(1234)
    key, subkey = jax.random.split(key)
    sigma = cast(float, jax.random.exponential(subkey))
    kernel = GaussianExpWassersteinKernel.create(sigma=sigma)

    # Univariate Gaussian distributions
    key, subkey1, subkey2 = jax.random.split(key, 3)
    mu1 = jax.random.normal(subkey1, (1,))
    sigma1 = cast(float, jax.random.exponential(subkey2))
    dist1 = GaussianConditionalModel(mu=mu1, sigma=sigma1)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    mu2 = jax.random.normal(subkey1, (1,))
    sigma2 = cast(float, jax.random.exponential(subkey2))
    dist2 = GaussianConditionalModel(mu=mu2, sigma=sigma2)

    expected = jnp.exp(
        -((mu1[0] - mu2[0]) ** 2 + (sigma1 - sigma2) ** 2) / (2 * sigma**2)
    )
    assert kernel(dist1, dist2) == pytest.approx(expected)

    # Check that results for multivariate isotropic Gaussian distributions
    # match product of distances between components
    dimension = 20
    key, subkey = jax.random.split(key)
    mu1 = jax.random.normal(subkey, (dimension,))
    dist1 = GaussianConditionalModel(mu=mu1, sigma=sigma1)

    key, subkey = jax.random.split(key)
    mu2 = jax.random.normal(subkey, (dimension,))
    dist2 = GaussianConditionalModel(mu=mu2, sigma=sigma2)

    def body(carry: float, mus_i: Tuple[Array, Array]):
        mu1_i, mu2_i = mus_i
        dist1_i = GaussianConditionalModel(mu=jnp.array([mu1_i]), sigma=sigma1)
        dist2_i = GaussianConditionalModel(mu=jnp.array([mu2_i]), sigma=sigma2)
        return (carry * kernel(dist1_i, dist2_i), carry)

    expected, _ = jax.lax.scan(body, 1.0, (mu1, mu2))
    assert kernel(dist1, dist2) == pytest.approx(expected)


@pytest.mark.parametrize(
    "base_dist_mu,base_dist_sigma,log_p_mu,log_p_sigma,log_q_mu,log_q_sigma,"
    "kernel_sigma",
    [
        (jnp.arange(4, 6), 0.2, jnp.arange(4, 6), 0.5, jnp.arange(5, 7), 0.7, 0.5),
        (jnp.arange(3, 6), 2.0, jnp.arange(7, 10), 2.0, jnp.arange(4, 7), 3.0, 1.5),
    ],
)
def test_gaussian_expfisher_kernel(
    base_dist_mu: Array,
    base_dist_sigma: Scalar,
    log_p_mu: Array,
    log_p_sigma: Scalar,
    log_q_mu: Array,
    log_q_sigma: Scalar,
    kernel_sigma: Scalar,
):
    # Check the correctness of the GaussianExpFisherKernel, available for gaussian
    # models only, and computed using a closed-form expression, by ensuring that the
    # discretized version of this kernel approximately equals its closed-form
    # expression on various examples.
    base_dist = np_distributions.Normal(base_dist_mu, base_dist_sigma)  # type: ignore

    discretized_expfisher_kernel = DiscreteExpFisherKernel.create(
        base_dist=base_dist,
        key=jax.random.PRNGKey(1234),
        num_samples=1000_000,
        sigma=kernel_sigma,
    )
    gaussian_expfisher_kernel = GaussianExpFisherKernel.create(
        base_dist=base_dist, sigma=kernel_sigma
    )

    p = GaussianConditionalModel(log_p_mu, log_p_sigma)
    q = GaussianConditionalModel(log_q_mu, log_q_sigma)

    val_discrete = discretized_expfisher_kernel(p, q)
    val_gaussian = gaussian_expfisher_kernel(p, q)
    assert jnp.allclose(val_discrete, val_gaussian, rtol=1e-3)


@pytest.mark.parametrize(
    "base_dist_mu,base_dist_sigma,log_p_mu,log_p_sigma,log_q_mu,log_q_sigma,"
    "kernel_sigma",
    [
        (
            jnp.arange(2),
            1.2,
            jnp.array([1.2, 2.5]),
            1.1,
            jnp.array([1.3, 1.8]),
            0.9,
            0.5,
        ),
        (jnp.arange(3, 6), 2.0, jnp.arange(7, 10), 2.0, jnp.arange(4, 7), 3.0, 1.5),
    ],
)
def test_gaussian_kernelized_expfisher_kernel(
    base_dist_mu: Array,
    base_dist_sigma: Scalar,
    log_p_mu: Array,
    log_p_sigma: Scalar,
    log_q_mu: Array,
    log_q_sigma: Scalar,
    kernel_sigma: Scalar,
):
    # Check the correctness of the GaussianExpFisherKernel, available for gaussian
    # models only, and computed using a closed-form expression, by ensuring that the
    # discretized version of this kernel approximately equals its closed-form
    # expression on various examples.
    base_dist = np_distributions.Normal(base_dist_mu, base_dist_sigma)  # type: ignore

    discretized_expfisher_kernel = DiscreteExpKernelizedFisherKernel.create(
        base_dist=base_dist,
        key=jax.random.PRNGKey(1234),
        num_samples=8000,
        sigma=kernel_sigma,
        ground_space_kernel=gaussian_kernel.create(sigma=0.8),
    )
    gaussian_expfisher_kernel = GaussianExpKernelizedFisherKernel.create(
        base_dist=base_dist,
        sigma=kernel_sigma,
        ground_space_kernel=gaussian_kernel.create(sigma=0.8),
    )

    p = GaussianConditionalModel(log_p_mu, log_p_sigma)
    q = GaussianConditionalModel(log_q_mu, log_q_sigma)

    val_discrete = discretized_expfisher_kernel(p, q)
    val_gaussian = gaussian_expfisher_kernel(p, q)
    assert jnp.allclose(val_discrete, val_gaussian, rtol=1e-2)


@pytest.mark.parametrize(
    "p_mu,p_sigma,q_mu,q_sigma,kernel_sigma",
    [
        (jnp.arange(4, 6), 0.5, jnp.arange(5, 7), 0.7, 0.5),
        (jnp.arange(7, 9), 2.0, jnp.arange(4, 6), 3.0, 1.5),
    ],
)
def test_approximation_rules(
    p_mu: Array,
    p_sigma: Scalar,
    q_mu: Array,
    q_sigma: Scalar,
    kernel_sigma: Scalar,
    capsys: Any,
    caplog: Any,
):
    # test that intractable kernels are well approximated by their specified
    # approximation by ensuring that the discrepancy is low when using a large
    # computational budget for the approximation.
    num_particles = 3000

    def test_approximation_rule(
        intr_kernel: kernel_on_models[BM, base_kernel[BM2], BM2],
        true_kernel: base_kernel[BM],
        P: BM,
        Q: BM,
        key: PRNGKeyArray,
        rtol: float = 1e-2,
    ):
        sk1, sk2, sk3 = random.split(key, 3)

        approx_kernel = intr_kernel.maybe_approximate_kernel(num_particles, sk1)
        approx_P = intr_kernel.maybe_approximate_input(P, num_particles, sk2)
        approx_Q = intr_kernel.maybe_approximate_input(Q, num_particles, sk3)

        # https://github.com/google/jax/issues/13554
        val_exact = jit(type(true_kernel).__call__)(true_kernel, P, Q)
        val_approx = jit(type(approx_kernel).__call__)(
            approx_kernel, approx_P, approx_Q
        )

        assert jnp.allclose(val_exact, val_approx, rtol=rtol)

    exp_mmd_gausian_k = ExpMMDKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.2), sigma=kernel_sigma
    )
    tractable_exp_mmd_gausian_k = ExpMMDGaussianKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.2), sigma=kernel_sigma
    )

    exp_fisher_gaussian_k = ExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=1.0),  # type: ignore
        sigma=kernel_sigma,
    )
    tractable_exp_fisher_gaussian_k = GaussianExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=1.0),  # type: ignore
        sigma=kernel_sigma,
    )
    tractable_exp_kernelized_fisher_gaussian_k = GaussianExpKernelizedFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=1.0),  # type: ignore
        sigma=kernel_sigma,
        ground_space_kernel=gaussian_kernel.create(sigma=0.8),
    )
    exp_kernelized_fisher_gaussian_k = ExpKernelizedFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=1.0),  # type: ignore
        sigma=kernel_sigma,
        ground_space_kernel=gaussian_kernel.create(sigma=0.8),
    )

    key = random.PRNGKey(1234)
    key, subkey = random.split(key)
    P = GaussianConditionalModel(mu=p_mu, sigma=p_sigma)
    Q = GaussianConditionalModel(mu=q_mu, sigma=q_sigma)

    key, sk1, sk2 = random.split(key, 3)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            exp_mmd_gausian_k, tractable_exp_mmd_gausian_k, P, Q, sk1
        )
    assert caplog.text == ""
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            exp_fisher_gaussian_k, tractable_exp_fisher_gaussian_k, P, Q, sk2
        )

    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            exp_kernelized_fisher_gaussian_k,
            tractable_exp_kernelized_fisher_gaussian_k,
            P,
            Q,
            sk2,
        )
    assert caplog.text == ""
    caplog.clear()

    key, sk1, sk2 = random.split(key, 3)
    P_discrete = GaussianConditionalModel(
        mu=jnp.ones((2,)), sigma=1.0
    ).non_mcmc_discretize(sk1, num_particles)
    Q_discrete = GaussianConditionalModel(
        mu=jnp.ones((2,)) * 2, sigma=1.0
    ).non_mcmc_discretize(sk2, num_particles)

    discrete_exp_mmd_k = DiscreteExpMMDKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.2),
        sigma=kernel_sigma,
    )

    # discrete_exp_mmd_k is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            discrete_exp_mmd_k,
            discrete_exp_mmd_k,
            P_discrete,
            Q_discrete,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    discrete_exp_fisher_k: DiscreteExpFisherKernel[
        GaussianConditionalModel
    ] = DiscreteExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=1.0),  # type: ignore
        sigma=0.5,
        key=subkey,
        num_samples=num_particles,
    )

    discrete_kernelized_exp_fisher_k: DiscreteExpKernelizedFisherKernel[
        GaussianConditionalModel
    ] = DiscreteExpKernelizedFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=1.0),  # type: ignore
        sigma=0.5,
        key=subkey,
        num_samples=num_particles,
        ground_space_kernel=gaussian_kernel.create(sigma=0.8),
    )
    # DiscreteExpFisherKernel is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            discrete_exp_fisher_k,
            discrete_exp_fisher_k,
            P,
            Q,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    # DiscreteKernelizedExpFisherKernel is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            discrete_kernelized_exp_fisher_k,
            discrete_kernelized_exp_fisher_k,
            P,
            Q,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    # DiscreteExpMMDKernel is  is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            discrete_exp_mmd_k,
            discrete_exp_mmd_k,
            P_discrete,
            Q_discrete,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    # GaussianExpFisherKernel is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            tractable_exp_fisher_gaussian_k,
            tractable_exp_fisher_gaussian_k,
            P,
            Q,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    # GaussianExpKernelizedFisherKernel is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            tractable_exp_kernelized_fisher_gaussian_k,
            tractable_exp_kernelized_fisher_gaussian_k,
            P,
            Q,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    # ExpMMDGaussianKernel is its own approximation.
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(
            tractable_exp_mmd_gausian_k,
            tractable_exp_mmd_gausian_k,
            P,
            Q,
            subkey,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    # GaussianExpWassersteinKernel is its own approximation.
    expw_kernel = GaussianExpWassersteinKernel.create(sigma=kernel_sigma)
    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_approximation_rule(expw_kernel, expw_kernel, P, Q, subkey, rtol=0.0)
    assert "discouraged" in caplog.text
    caplog.clear()

    # do not show warnings at the end of the test
    _ = capsys.readouterr()
