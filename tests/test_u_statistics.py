import logging
from typing import Any, Tuple, cast

import jax.numpy as jnp
import numpyro.distributions as np_distributions
import pytest
from jax import random, tree_map
from jax_samplers.inference_algorithms.mcmc.base import MCMCConfig
from jax_samplers.kernels.mala import MALAConfig, MALAKernelFactory
from kwgflows.pytypes import Array, PRNGKeyArray
from kwgflows.rkhs.kernels import base_kernel, gaussian_kernel

from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.kernels import (
    BM,
    DiscreteExpFisherKernel,
    DiscreteExpMMDKernel,
    ExpFisherKernel,
    ExpMMDGaussianKernel,
    ExpMMDKernel,
    GaussianExpFisherKernel,
    GaussianExpWassersteinKernel,
    kernel_on_models,
)
from calibration.statistical_tests.kccsd import KCCSDOneSample_T, OneSampleKCCSDStatFn
from calibration.statistical_tests.kcsd import ConditionalLogDensity_T
from calibration.statistical_tests.skce import OneSampleSKCEUStatFn, SKCEOneSample_T
from calibration_benchmarks.data_generators.kccsd import (
    kccsd_data_generator_base,
    kccsd_gaussian_data_generator,
)
from calibration_benchmarks.data_generators.skce import skce_gaussian_data_generator


def test_approximate_skce_u_stat_fn(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
):
    # test that intractable u-statistics functions are well approximated by
    # their specified approximation by ensuring that the discrepancy is low
    # when using a large computational budget for the approximation.
    num_particles = 5000
    y_kernel = gaussian_kernel.create(sigma=0.5)
    mcmc_config = MCMCConfig(
        num_samples=10,  # not used
        kernel_factory=MALAKernelFactory(config=MALAConfig(step_size=0.1)),
        num_warmup_steps=100,
        adapt_step_size=True,
        target_accept_rate=0.5,
        init_using_log_l_mode=False,
        num_chains=100,  # automatically adapted
    )

    def test_u_stat_fn(
        intr_p_kernel: kernel_on_models[BM, base_kernel[Any], Any],
        data_p: SKCEOneSample_T[BM],
        data_q: SKCEOneSample_T[BM],
        val_exact: float,
        key: PRNGKeyArray,
        rtol: float = 5e-2,
    ):
        skce_u_stat_fn = OneSampleSKCEUStatFn(
            intr_p_kernel,
            y_kernel,
        )

        key, subkey = random.split(key)
        approx_state_P = skce_u_stat_fn.make_approximation_state(
            data_p, num_particles=num_particles, key=subkey, mcmc_config=mcmc_config
        )
        key, subkey = random.split(key)
        approx_state_Q = skce_u_stat_fn.make_approximation_state(
            data_q, num_particles=num_particles, key=subkey, mcmc_config=mcmc_config
        )

        key, subkey = random.split(key)
        approx_internals = skce_u_stat_fn.maybe_approximate_internals(
            num_particles, subkey
        )

        val_approx = skce_u_stat_fn.call_approximate(
            (data_p, approx_state_P), (data_q, approx_state_Q), approx_internals
        )
        assert jnp.allclose(val_exact, val_approx, rtol=rtol, atol=5e-3)

    intr_exp_mmd_gaussian_k = ExpMMDKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.5), sigma=0.5
    )
    exp_mmd_gaussian_k = ExpMMDGaussianKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.5), sigma=0.5
    )

    intr_exp_gaussian_fisher_k = ExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
    )
    exp_gaussian_kisher_k = GaussianExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
    )

    exp_w_gaussian_k = GaussianExpWassersteinKernel.create(
        sigma=0.5,
    )

    discrete_exp_mmd_k = DiscreteExpMMDKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.5), sigma=2.0
    )

    key = random.PRNGKey(1234)
    key, subkey = random.split(key)
    discrete_exp_fisher_k = DiscreteExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
        key=subkey,
        num_samples=num_particles,
    )

    key, subkey = random.split(key)
    data = skce_gaussian_data_generator().get_data(subkey, 2)
    data_p_tractable = cast(
        Tuple[GaussianConditionalModel, Array], tree_map(lambda x: x[0], data)
    )
    # data_q_tractable = cast(
    #     Tuple[GaussianConditionalModel, Array], tree_map(lambda x: x[1], data)
    # )
    data_q_tractable = cast(
        Tuple[GaussianConditionalModel, Array], tree_map(lambda x: x[0] + 2.0, data)
    )
    data_p = (data_p_tractable[0].as_distribution_model(), data_p_tractable[1])
    data_q = (data_q_tractable[0].as_distribution_model(), data_q_tractable[1])

    data_p_lm = (data_p[0].as_logdensity_model(), data_p[1])
    data_q_lm = (data_q[0].as_logdensity_model(), data_q[1])

    key, sk1, sk2 = random.split(key, 3)
    data_p_discrete = (
        data_p_tractable[0].non_mcmc_discretize(sk1, 10),
        data_p_tractable[1],
    )
    data_q_discrete = (
        data_q_tractable[0].non_mcmc_discretize(sk2, 10),
        data_q_tractable[1],
    )

    true_ustat_val_mmd = OneSampleSKCEUStatFn(exp_mmd_gaussian_k, y_kernel)(
        data_p_tractable, data_q_tractable
    )
    true_ustat_val_discrete_mmd = OneSampleSKCEUStatFn(discrete_exp_mmd_k, y_kernel)(
        data_p_discrete, data_q_discrete
    )

    true_ustat_val_fisher = OneSampleSKCEUStatFn(exp_gaussian_kisher_k, y_kernel)(
        data_p_tractable, data_q_tractable
    )

    true_ustat_val_discrete_fisher = OneSampleSKCEUStatFn(
        discrete_exp_fisher_k, y_kernel
    )(data_p_tractable, data_q_tractable)

    true_ustat_val_wasserstein = OneSampleSKCEUStatFn(exp_w_gaussian_k, y_kernel)(
        data_p_tractable, data_q_tractable
    )

    key, sk1, sk2 = random.split(key, 3)
    test_u_stat_fn(intr_exp_mmd_gaussian_k, data_p, data_q, true_ustat_val_mmd, sk1)
    test_u_stat_fn(
        intr_exp_gaussian_fisher_k,
        data_p_tractable,
        data_q_tractable,
        true_ustat_val_fisher,
        sk2,
    )
    test_u_stat_fn(
        intr_exp_gaussian_fisher_k,
        data_p_lm,
        data_q_lm,
        true_ustat_val_fisher,
        sk2,
    )
    # these kernels are their own approximations -> rtol=0.
    key, sk1, sk2, sk3, sk4 = random.split(key, 5)
    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            exp_mmd_gaussian_k,
            data_p_tractable,
            data_q_tractable,
            true_ustat_val_mmd,
            sk1,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            exp_gaussian_kisher_k,
            data_p_tractable,
            data_q_tractable,
            true_ustat_val_fisher,
            sk2,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            exp_w_gaussian_k,
            data_p_tractable,
            data_q_tractable,
            true_ustat_val_wasserstein,
            sk3,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            discrete_exp_fisher_k,
            data_p_tractable,
            data_q_tractable,
            true_ustat_val_discrete_fisher,
            sk4,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    key, sk1, sk2 = random.split(key, 3)

    key, subkey = random.split(key)
    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            discrete_exp_mmd_k,
            data_p_discrete,
            data_q_discrete,
            true_ustat_val_discrete_mmd,
            subkey,
            rtol=0.0,
        )
        assert "discouraged" in caplog.text
        caplog.clear()

    _ = capsys.readouterr()


def test_approximate_kccsd_u_stat_fn(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
):
    # test that intractable u-statistics functions are well approximated by
    # their specified approximation by ensuring that the discrepancy is low
    # when using a large computational budget for the approximation.
    num_particles = 3000
    key = random.PRNGKey(1234)
    y_kernel = gaussian_kernel.create(sigma=0.5)

    def test_u_stat_fn(
        intr_p_kernel: kernel_on_models[BM, base_kernel[Any], Any],
        data_p: KCCSDOneSample_T[BM],
        data_q: KCCSDOneSample_T[BM],
        true_ustat_val: float,
        c_score: ConditionalLogDensity_T[BM],
        key: PRNGKeyArray,
        rtol: float = 5e-2,
    ):
        intr_kccsd_u_stat_fn = OneSampleKCCSDStatFn(intr_p_kernel, y_kernel, c_score)

        key, subkey = random.split(key)
        approx_state_P = intr_kccsd_u_stat_fn.make_approximation_state(
            data_p, num_particles=num_particles, key=subkey
        )
        key, subkey = random.split(key)
        approx_state_Q = intr_kccsd_u_stat_fn.make_approximation_state(
            data_q, num_particles=num_particles, key=subkey
        )

        key, subkey = random.split(key)
        approx_internals = intr_kccsd_u_stat_fn.maybe_approximate_internals(
            num_particles, subkey
        )

        val_approx = intr_kccsd_u_stat_fn.call_approximate(
            (data_p_tractable, approx_state_P),
            (data_q, approx_state_Q),
            approx_internals,
        )
        assert jnp.allclose(true_ustat_val, val_approx, rtol=rtol)

    intr_exp_mmd_gaussian_k = ExpMMDKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.5), sigma=0.5
    )
    true_exp_mmd_gaussian_k = ExpMMDGaussianKernel.create(
        ground_space_kernel=gaussian_kernel.create(sigma=0.5), sigma=0.5
    )

    intr_exp_fisher_gaussian_k = ExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
    )
    true_exp_fisher_gaussian_k = GaussianExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
    )

    key = random.PRNGKey(1234)
    key, subkey = random.split(key)
    discrete_exp_fisher_k = DiscreteExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
        key=subkey,
        num_samples=num_particles,
    )
    true_exp_w_gaussian_kernel = GaussianExpWassersteinKernel.create(
        sigma=0.5,
    )

    data_generator: kccsd_data_generator_base[
        GaussianConditionalModel
    ] = kccsd_gaussian_data_generator()
    (models, Y, c_score) = data_generator.get_data(key, 2)

    data_p_tractable = (tree_map(lambda x: x[0], models), Y[0])
    data_q_tractable = (tree_map(lambda x: x[1], models), Y[1])

    data_p = (data_p_tractable[0].as_distribution_model(), data_p_tractable[1])
    data_q = (data_q_tractable[0].as_distribution_model(), data_q_tractable[1])

    true_ustat_val_mmd = OneSampleKCCSDStatFn(
        true_exp_mmd_gaussian_k, y_kernel, c_score
    )(data_p_tractable, data_q_tractable)

    true_usat_val_fisher = OneSampleKCCSDStatFn(
        true_exp_fisher_gaussian_k, y_kernel, c_score
    )(data_p_tractable, data_q_tractable)

    true_usat_val_wasserstein = OneSampleKCCSDStatFn(
        true_exp_w_gaussian_kernel, y_kernel, c_score
    )(data_p_tractable, data_q_tractable)

    true_ustat_val_discrete_fisher = OneSampleKCCSDStatFn(
        discrete_exp_fisher_k, y_kernel, c_score
    )(data_p_tractable, data_q_tractable)

    key, sk1, sk2, sk3 = random.split(key, 4)
    test_u_stat_fn(
        intr_exp_mmd_gaussian_k,
        data_p,
        data_q,
        true_ustat_val_mmd,
        c_score,
        sk1,
    )
    test_u_stat_fn(
        intr_exp_fisher_gaussian_k,
        data_p,
        data_q,
        true_usat_val_fisher,
        c_score,
        sk2,
    )

    # these kernels are their own approximations -> rtol=0.
    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            discrete_exp_fisher_k,
            data_p,
            data_q,
            true_ustat_val_discrete_fisher,
            c_score,
            sk3,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    key, sk1, sk2, sk3 = random.split(key, 4)
    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            true_exp_mmd_gaussian_k,
            data_p_tractable,
            data_q_tractable,
            true_ustat_val_mmd,
            c_score,
            sk1,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            true_exp_fisher_gaussian_k,
            data_p_tractable,
            data_q_tractable,
            true_usat_val_fisher,
            c_score,
            sk2,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        test_u_stat_fn(
            true_exp_w_gaussian_kernel,
            data_p_tractable,
            data_q_tractable,
            true_usat_val_wasserstein,
            c_score,
            sk3,
            rtol=0.0,
        )
    assert "discouraged" in caplog.text
    caplog.clear()

    _ = capsys.readouterr()
