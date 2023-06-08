import math
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import random
from kwgflows.divergences.mmd import mmd as scalar_mmd
from kwgflows.pytypes import Scalar
from kwgflows.rkhs.kernels import gaussian_kernel

from calibration.conditional_models.gaussian_models import (
    GaussianConditionalModel,
    mmd_gaussian_kernel,
)


def mmd_closed_form(
    sigma_k: float,
    mu_p: jax.Array,
    mu_q: jax.Array,
    sigma_p: float,
    sigma_q: float,
    d: int,
):
    mmd_cf = (
        1 / jnp.sqrt((1 + (sigma_p**2 + sigma_p**2) / sigma_k**2) ** d)
        + 1 / jnp.sqrt((1 + (sigma_q**2 + sigma_q**2) / sigma_k**2) ** d)
        - 2
        * jnp.exp(
            -0.5
            * (jnp.sum((mu_p - mu_q) ** 2))
            / (sigma_k**2 + sigma_p**2 + sigma_q**2)
        )
        / jnp.sqrt((1 + (sigma_p**2 + sigma_q**2) / (sigma_k**2)) ** d)
    )
    return cast(Scalar, mmd_cf)


def test_expectation():
    # Gaussian kernel
    kernel = gaussian_kernel.create(sigma=1.0)

    # check dependence of expectation on dimension
    d1 = 1
    cm1 = GaussianConditionalModel(
        mu=jnp.zeros((d1,)),
        sigma=1.0,
    )

    d2 = 3
    cm2 = GaussianConditionalModel(
        mu=jnp.zeros((d2,)),
        sigma=1.0,
    )

    ret1 = cm1.analytical_expectation_of(kernel, jnp.zeros((d1,)))
    ret2 = cm2.analytical_expectation_of(kernel, jnp.zeros((d2,)))
    assert ret2 ** (1 / d2) == pytest.approx(ret1 ** (1 / d1))

    # check independence w.r.t mean difference
    d1 = 2

    cm1 = GaussianConditionalModel(
        mu=jnp.zeros((d1,)),
        sigma=1.0,
    )

    cm2 = GaussianConditionalModel(
        mu=jnp.ones((d1,)),
        sigma=1.0,
    )

    assert cm1.analytical_expectation_of(kernel, cm2.mu) == pytest.approx(
        cm2.analytical_expectation_of(kernel, cm1.mu),
    )

    # check bivariate expectation nice case where all means are 0
    d1 = 3
    s1 = 2
    s2 = 4

    k = gaussian_kernel(sigma=1.0)
    cm1 = GaussianConditionalModel(
        mu=jnp.zeros((d1,)),
        sigma=s1,
    )

    cm2 = GaussianConditionalModel(
        mu=jnp.zeros((d1,)),
        sigma=s2,
    )

    ret = cm1.bivariate_analytical_expectation_of(k, cm2)
    assert ret == pytest.approx(1 / math.sqrt((1 + s1**2 + s2**2) ** d1))


def test_mmd_gaussian_model():
    d = 2
    mu = jnp.zeros((d,))
    sigma_p = 1.0

    p = GaussianConditionalModel(
        mu=mu,
        sigma=sigma_p,
    )

    q = p
    kernel = gaussian_kernel.create(sigma=1.0)
    mmd = mmd_gaussian_kernel(kernel=kernel)
    assert mmd(p, q) == pytest.approx(0.0)

    r"""
    For two d-dimensional gaussian random variables with isotropic diagonal covariance
    matrices: $ p = \mathcal N(\mu_p, \sigma_p^2 I_d) $ and
    $ q = \mathcal N(\mu_q, \sigma_q^2 I_d)$, $MMD^2(p, q)$ is given by:
    $$
    \textrm{MMD}^2(p, q) =
        \frac{1}{{(1 + 2(\sigma_p)^2 / \sigma_k^2)}^d} +
        \frac{1}{{(1 + 2(\sigma_q)^2 / \sigma_k^2)}^d} -
        \frac{2}{{1 + (\sigma_p + \sigma_q)^2 / \sigma_k^2}^d} \times
        \exp (
            -0.5 * \frac{\| \mu_p - \mu_q \|^2}{\sigma_k^2 + \sigma_p^2 + \sigma_q^2}
        )
    $$

    In the following, we check that the computing the MMD between two such models
    returns this correct value. This test is useful because the current MMD computation
    method in such cases goes through other code paths (`analytical_expectation_of`),
    and can thus be seen as an end-to-end to such methods, which are used in other
    parts of the code base (such as SKCE U-statistics computations)
    """
    d = 4
    mu_p = jnp.zeros((d,))
    mu_q = jnp.zeros((d,)) + 2.0
    sigma_p = 2.0
    sigma_q = 10.0

    p = GaussianConditionalModel(
        mu=mu_p,
        sigma=sigma_p,
    )
    q = GaussianConditionalModel(
        mu=mu_q,
        sigma=sigma_q,
    )

    kernel = gaussian_kernel.create(sigma=2.0)
    mmd = mmd_gaussian_kernel(kernel=kernel)

    mmdval = mmd(p, q)

    dp = p.non_mcmc_discretize(random.PRNGKey(0), 3000)
    dq = q.non_mcmc_discretize(random.PRNGKey(0), 3000)

    mmd_d = scalar_mmd(kernel)(dp, dq)

    mmd_cf = mmd_closed_form(kernel.sigma, mu_p, mu_q, sigma_p, sigma_q, d)
    assert mmdval == pytest.approx(mmd_cf)
    assert mmdval == pytest.approx(mmd_d, abs=1e-1)


def test_score():
    d = 10
    key = random.PRNGKey(1234)
    key, subkey = random.split(key)
    mu = random.normal(subkey, (d,))
    sigma = 4.1
    p = GaussianConditionalModel(
        mu=mu,
        sigma=sigma,
    )
    key, subkey = random.split(key)
    y = random.normal(subkey, (d,))
    assert p.score(y) == pytest.approx(jax.grad(p)(y))
