import jax
import jax.numpy as jnp
from jax import random
from kwgflows.base import DiscreteProbability
from kwgflows.divergences.mmd import mmd as scalar_mmd
from kwgflows.rkhs.kernels import gaussian_kernel


def test_analytical_expectation():
    # End to end test ensuring that the (bivariate) expectations methods of
    # DiscreteProbability are working as expected.
    kernel = gaussian_kernel.create(sigma=1.0)

    # smmd uses RKHS rountines and not bivariate expectations under the hood
    smmd = scalar_mmd(kernel)

    X = random.normal(random.PRNGKey(0), (100, 2))
    Y = random.normal(random.PRNGKey(1), (100, 2))
    P = DiscreteProbability.from_samples(X)
    Q = DiscreteProbability.from_samples(Y)

    def mmd_sq(P: DiscreteProbability[jax.Array], Q: DiscreteProbability[jax.Array]):
        # a mmd implementation going through bivariate_expectations
        # code paths.
        kpp = P.bivariate_analytical_expectation_of(kernel, P)
        kqq = Q.bivariate_analytical_expectation_of(kernel, Q)
        kqp = Q.bivariate_analytical_expectation_of(kernel, P)
        mmd_sq = kpp + kqq - 2 * kqp
        return mmd_sq

    smmd_val = smmd(P, Q)
    gmmd_val = mmd_sq(P, Q)

    assert jnp.allclose(smmd_val, gmmd_val)
