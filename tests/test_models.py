import jax.numpy as jnp
from jax import jit, random, vmap
from numpyro import distributions as np_distributions

from calibration.conditional_models.base import DistributionModel
from calibration.conditional_models.gaussian_models import GaussianConditionalModel


def test_distribution_model():
    loc = jnp.ones((2,))
    scale = 0.1

    d = np_distributions.Normal(loc=loc, scale=scale).to_event(1)  # type: ignore
    dm = DistributionModel.create(d)
    gm = GaussianConditionalModel(mu=loc, sigma=scale)
    dm_from_gm = gm.as_distribution_model()

    key = random.PRNGKey(1234)
    x1 = gm.sample_from_conditional(key, 1)
    x2 = dm.sample_from_conditional(key, 1)
    x3 = dm_from_gm.sample_from_conditional(key, 1)

    assert jnp.allclose(x1, x2)
    assert jnp.allclose(x1, x3)

    lm_from_dm = dm.as_logdensity_model()
    assert lm_from_dm.dimension == 2

    lp1 = gm(x1)
    lp2 = dm(x1)
    lp3 = dm_from_gm(x1)
    lp4 = lm_from_dm(x1)

    assert jnp.allclose(lp1, lp2)
    assert jnp.allclose(lp1, lp3)
    assert jnp.allclose(lp1, lp4)

    # test robustness to vmap/jit
    N = 3
    key, sk1, sk2 = random.split(key, 3)
    Xs = random.normal(sk1, (N, 2))
    Ys = random.normal(sk2, (N, 2))
    dms = vmap(lambda x: DistributionModel.create(np_distributions.Normal(x)))(Xs)
    lps = vmap(lambda dm, y: dm(y))(dms, Ys)
    lps_jit = jit(vmap(lambda dm, y: dm(y)))(dms, Ys)

    for X, Y, lp, lp_jit in zip(Xs, Ys, lps, lps_jit):
        dm = DistributionModel.create(np_distributions.Normal(X))
        assert jnp.allclose(lp, dm(Y))
        assert jnp.allclose(lp_jit, dm(Y))
