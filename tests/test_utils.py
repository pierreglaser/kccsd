import logging
from typing import Generic, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from flax import struct
from kwgflows.divergences.mmd import mmd
from kwgflows.pytypes import Array, Scalar
from kwgflows.rkhs.kernels import (
    gaussian_kernel,
    imq_kernel,
    laplace_kernel,
    median_heuristic,
    negative_distance_kernel,
)
from tensorflow_probability.substrates import jax as tfp

from calibration import utils
from calibration.conditional_models.base import LogDensityModel
from calibration.conditional_models.gaussian_models import (
    GaussianConditionalModel,
    kernelized_generalized_fisher_divergence_gaussian_model,
    mmd_gaussian_kernel,
)
from calibration.kernels import (
    DiscreteExpFisherKernel,
    DiscreteExpMMDKernel,
    ExpFisherKernel,
    ExpMMDGaussianKernel,
    ExpMMDKernel,
    GaussianExpFisherKernel,
    GaussianExpKernelizedFisherKernel,
    GaussianExpWassersteinKernel,
    discretized_generalized_fisher_divergence,
    generalized_fisher_divergence_gaussian_model,
)
from calibration.statistical_tests.ksd import steinalized_kernel

T = TypeVar("T", bound=dist.Distribution)

# Use jit-compiled version
median_euclidean_gaussians = jax.jit(utils.median_euclidean_gaussians)


class DistributionModel(Generic[T], LogDensityModel):
    distribution: T = struct.field(pytree_node=False)

    def __call__(self, y: Array) -> Scalar:
        return cast(Scalar, self.distribution.to_event(1).log_prob(y))


@pytest.mark.parametrize("dim", [3, 4, 5])
def test_fill_diagonal(dim: int):
    """Check correctness and assertion errors of `utils.fill_diagonal` on simple examples."""

    # create a square matrix of normally distributed random variates
    key, subkey = jax.random.split(jax.random.PRNGKey(1234))
    X = jax.random.normal(subkey, (dim, dim))

    # create a non-square matrix of normally distributed random variates
    key, subkey = jax.random.split(key)
    X_nonsquare = jax.random.normal(subkey, (dim, dim + 1))

    # create a vector of normally distributed random variates
    key, subkey = jax.random.split(key)
    X_vector = jax.random.normal(subkey, (dim,))

    # create a three-dimensional array of normally dist
    key, subkey = jax.random.split(key)
    X_array = jax.random.normal(subkey, (10, dim, dim))

    # create vector of diagonal entries
    key, subkey = jax.random.split(key)
    val_vector = jax.random.normal(subkey, (dim,))

    # for different inputs of diagonal entries
    for val in [3.1, -2.6, 4, -7, val_vector]:
        # compare result with numpy
        Y = utils.fill_diagonal(X, val)
        Y_numpy = np.copy(X)
        np.fill_diagonal(Y_numpy, val)
        assert jnp.array_equal(Y, jnp.asarray(Y_numpy))

        # assertion error if X is a vector
        with pytest.raises(AssertionError):
            utils.fill_diagonal(X_vector, val)

        # assertion error if X is a three-dimensional array
        with pytest.raises(AssertionError):
            utils.fill_diagonal(X_array, val)

        # assertion error if X is a non-square matrix
        with pytest.raises(AssertionError):
            utils.fill_diagonal(X_nonsquare, val)


@pytest.mark.parametrize("N", [3, 5, 12])
def test_superdiags_indices(N: int):
    """Check correctness and assertion errors of `utils.superdiags_indices`."""

    # Indices of all upper-diagonal elements
    x_triu_1, y_triu_1 = np.triu_indices(N, 1)
    x_y_triu_1 = set(zip(x_triu_1, y_triu_1))

    # for all possible values of R
    for R in np.arange(1, N):
        # compute the indices
        x, y = utils.superdiags_indices(N, R)

        # check type and shape of the indices
        n = N * R - R * (R + 1) // 2
        for z in (x, y):
            assert isinstance(z, jax.Array)
            assert jnp.dtype(z) == jnp.dtype("int32")
            assert jnp.shape(z) == (n,)

        # check that indices are sorted
        x_y = list(zip(x, y))
        sorted_x_y = sorted(x_y)
        assert len(x_y) == n
        assert x_y == sorted_x_y

        # check that indices are correct
        x_triu_Rp1, y_triu_Rp1 = np.triu_indices(N, R + 1)
        x_y_triu_Rp1 = set(zip(x_triu_Rp1, y_triu_Rp1))
        x_y_triu_1_R = list(x_y_triu_1 - x_y_triu_Rp1)
        assert x_y == sorted(x_y_triu_1_R)

    # assertion error if R is not positive
    for R in [0, -1, -12]:
        with pytest.raises(AssertionError):
            utils.superdiags_indices(N, R)

    # assertion error if R is greater than or equal to N
    for R in [N, N + 1, N + 10]:
        with pytest.raises(AssertionError):
            utils.superdiags_indices(N, R)


def test_with_median_heuristic():
    """Check `median_heuristic` and `with_median_heuristic`."""
    # Generate test data
    dim = 10
    num_samples = 20
    default_sigma = 2.0
    key, subkey = jax.random.split(jax.random.PRNGKey(1234))
    X = jax.random.normal(subkey, (num_samples, dim))
    key, subkey = jax.random.split(key)
    sigmas = jax.random.exponential(subkey, (num_samples,))
    X_gaussian = jax.vmap(lambda x, sigma: GaussianConditionalModel(mu=x, sigma=sigma))(
        X, sigmas
    )
    particles = 10
    subkeys = jax.random.split(key, num_samples)
    X_discrete = jax.vmap(
        lambda x, key: x.non_mcmc_discretize(key, particles), in_axes=(0, 0)
    )(X_gaussian, subkeys)

    # Different subsets of indices
    pairwise_indices = jnp.triu_indices(n=num_samples, k=1)
    linear_indices = (
        jnp.arange(start=0, stop=num_samples - 1, step=2),
        jnp.arange(start=1, stop=num_samples, step=2),
    )

    jitted_median_heuristic = jax.jit(median_heuristic, static_argnums=(0,))
    jitted_median = jax.jit(jnp.median)
    jitted_median_median_heuristic_euclidean_gaussian = jax.jit(
        utils.median_median_heuristic_euclidean_gaussian
    )
    jitted_median_median_heuristic_discrete = jax.jit(
        utils.median_median_heuristic_discrete
    )

    for arg in ((pairwise_indices,), (linear_indices,), (None,), ()):
        # Extract "proper" indices
        if len(arg) == 0 or arg[0] is None:
            indices = pairwise_indices
        else:
            indices = arg[0]
        assert isinstance(indices, tuple)
        assert len(indices) == 2
        assert isinstance(indices[0], Array)
        assert isinstance(indices[1], Array)
        indices1, indices2 = indices

        # Kernels based on the Euclidean distances
        median_euclidean = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: jnp.linalg.norm(X[i] - X[j], ord=2),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        assert jitted_median_heuristic(
            lambda x, y: jnp.linalg.norm(x - y, ord=2), X, *arg
        ) == pytest.approx(median_euclidean)
        kernels_euclidean = [
            gaussian_kernel.create(sigma=default_sigma),
            imq_kernel.create(sigma=default_sigma),
            negative_distance_kernel.create(sigma=default_sigma),
        ]
        for kernel in kernels_euclidean:
            k = kernel.with_median_heuristic(X, *arg)
            assert k.sigma == pytest.approx(median_euclidean)

        # Kernels based on the Manhattan distance
        median_manhattan = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: jnp.linalg.norm(X[i] - X[j], ord=1), in_axes=(0, 0)
                )
            )(indices1, indices2)
        )

        assert jitted_median_heuristic(
            lambda x, y: jnp.linalg.norm(x - y, ord=1), X, *arg
        ) == pytest.approx(median_manhattan)
        kernel = laplace_kernel.create(sigma=default_sigma).with_median_heuristic(
            X, *arg
        )
        assert kernel.sigma == pytest.approx(median_manhattan)

        # Kernels based on the MMD
        # Discrete probabilities
        median_discrete_gaussian_mmd_groundspace = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: jitted_median_heuristic(
                        lambda x, y: jnp.linalg.norm(x - y),
                        jnp.concatenate(
                            [
                                jax.tree_map(lambda x: x[i], X_discrete).X,
                                jax.tree_map(lambda x: x[j], X_discrete).X,
                            ]
                        ),
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        for k in kernels_euclidean:
            assert median_discrete_gaussian_mmd_groundspace == pytest.approx(
                jitted_median_median_heuristic_discrete(k, X_discrete, *arg)
            )
        median_discrete_gaussian_mmd = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: mmd(
                        gaussian_kernel.create(
                            sigma=median_discrete_gaussian_mmd_groundspace
                        ),
                        squared=False,
                    )(
                        jax.tree_map(lambda x: x[i], X_discrete),
                        jax.tree_map(lambda x: x[j], X_discrete),
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        assert jitted_median_heuristic(
            jax.jit(
                mmd(
                    gaussian_kernel.create(
                        sigma=float(median_discrete_gaussian_mmd_groundspace)
                    ),
                    squared=False,
                )
            ),
            X_discrete,
            *arg,
        ) == pytest.approx(median_discrete_gaussian_mmd)
        k = DiscreteExpMMDKernel.create(
            ground_space_kernel=gaussian_kernel.create(sigma=default_sigma),
            sigma=default_sigma,
        ).with_median_heuristic(X_discrete, *arg)
        assert k.sigma == pytest.approx(median_discrete_gaussian_mmd)
        assert k.ground_space_kernel.sigma == pytest.approx(median_discrete_gaussian_mmd_groundspace)  # type: ignore

        # Gaussian distributions
        median_gaussian_mmd_groundspace = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: median_euclidean_gaussians(
                        jax.tree_map(lambda x: x[i], X_gaussian),
                        jax.tree_map(lambda x: x[j], X_gaussian),
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        assert median_gaussian_mmd_groundspace == pytest.approx(
            median_discrete_gaussian_mmd_groundspace, rel=1e-1
        )
        assert median_gaussian_mmd_groundspace == pytest.approx(
            jitted_median_median_heuristic_euclidean_gaussian(X_gaussian, *arg)
        )
        median_gaussian_mmd = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: mmd_gaussian_kernel(
                        gaussian_kernel.create(sigma=median_gaussian_mmd_groundspace),
                        squared=False,
                    )(
                        jax.tree_map(lambda x: x[i], X_gaussian),
                        jax.tree_map(lambda x: x[j], X_gaussian),
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        assert median_gaussian_mmd == pytest.approx(
            median_discrete_gaussian_mmd, rel=1e-1
        )
        assert jitted_median_heuristic(
            jax.jit(
                mmd_gaussian_kernel(
                    gaussian_kernel.create(
                        sigma=float(median_gaussian_mmd_groundspace)
                    ),
                    squared=False,
                )
            ),
            X_gaussian,
            *arg,
        ) == pytest.approx(median_gaussian_mmd)
        k = ExpMMDGaussianKernel.create(
            ground_space_kernel=gaussian_kernel.create(sigma=default_sigma),
            sigma=default_sigma,
        ).with_median_heuristic(X_gaussian, *arg)
        assert k.sigma == pytest.approx(median_gaussian_mmd)
        assert k.ground_space_kernel.sigma == pytest.approx(
            median_gaussian_mmd_groundspace
        )

        # Kernels based on the generalized Fisher divergence
        fisher_kernel = DiscreteExpFisherKernel.create(
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
            key=jax.random.PRNGKey(1234),
            sigma=default_sigma,
        )
        discrete_gfd = discretized_generalized_fisher_divergence.create(
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
            key=jax.random.PRNGKey(1234),
            squared=False,
        )
        median_approx_fisher = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: discrete_gfd(
                        jax.tree_map(lambda x: x[i], X_gaussian).score,
                        jax.tree_map(lambda x: x[j], X_gaussian).score,
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        # Wrapping in a lambda because of https://github.com/google/jax/issues/13554
        assert jitted_median_heuristic(
            lambda *args: discrete_gfd(*map(lambda x: x.score, args)), X_gaussian, *arg
        ) == pytest.approx(median_approx_fisher)
        k = fisher_kernel.with_median_heuristic(X_gaussian, *arg)
        assert k.sigma == pytest.approx(median_approx_fisher)

        # Fisher Kernel for Gaussian distributions
        gaussian_gfd = generalized_fisher_divergence_gaussian_model.create(
            dist.Normal(jnp.zeros((dim,))), squared=False  # type: ignore
        )
        median_gaussian_fisher = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: gaussian_gfd(
                        jax.tree_map(lambda x: x[i], X_gaussian),
                        jax.tree_map(lambda x: x[j], X_gaussian),
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        # Wrapping in a lambda because of https://github.com/google/jax/issues/13554
        assert jitted_median_heuristic(
            lambda *args: gaussian_gfd(*args), X_gaussian, *arg
        ) == pytest.approx(median_gaussian_fisher)
        k = GaussianExpFisherKernel.create(
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
            sigma=default_sigma,
        ).with_median_heuristic(X_gaussian, *arg)
        assert k.sigma == pytest.approx(median_gaussian_fisher)

        # Steinalized kernels
        kernels = [
            gaussian_kernel.create(sigma=default_sigma),
            laplace_kernel.create(sigma=default_sigma),
        ]
        for kernel in kernels:
            stein_kernel = steinalized_kernel.create(
                kernel=kernel, score_q=lambda x: 2 * x
            )
            k = stein_kernel.with_median_heuristic(X, *arg)
            assert k.kernel.sigma == pytest.approx(kernel.with_median_heuristic(X, *arg).sigma)  # type: ignore

        # Kernels based on the Wasserstein distance
        def wasserstein(P: GaussianConditionalModel, Q: GaussianConditionalModel):
            x = jnp.concatenate([P.mu, jnp.full_like(P.mu, P.sigma)])
            y = jnp.concatenate([Q.mu, jnp.full_like(Q.mu, Q.sigma)])
            return jnp.linalg.norm(x - y, ord=2)

        median_wasserstein = jitted_median(
            jnp.asarray(
                [
                    wasserstein(
                        jax.tree_map(lambda x: x[i], X_gaussian),
                        jax.tree_map(lambda x: x[j], X_gaussian),
                    )
                    for (i, j) in zip(indices1, indices2)
                ]
            ),
        )
        assert jitted_median_heuristic(wasserstein, X_gaussian, *arg) == pytest.approx(
            median_wasserstein
        )
        k = GaussianExpWassersteinKernel.create(
            sigma=default_sigma
        ).with_median_heuristic(X_gaussian, *arg)
        assert k.sigma == pytest.approx(median_wasserstein)

        # Kernelized Fisher Kernel for Gaussian distributions
        gaussian_kernelized_gfd = (
            kernelized_generalized_fisher_divergence_gaussian_model.create(
                ground_space_kernel=gaussian_kernel.create(
                    sigma=median_gaussian_mmd_groundspace
                ),
                base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
                squared=False,
            )
        )
        median_gaussian_kernelized_fisher = jitted_median(
            jax.jit(
                jax.vmap(
                    lambda i, j: gaussian_kernelized_gfd(
                        jax.tree_map(lambda x: x[i], X_gaussian),
                        jax.tree_map(lambda x: x[j], X_gaussian),
                    ),
                    in_axes=(0, 0),
                )
            )(indices1, indices2)
        )
        # Wrapping in a lambda because of https://github.com/google/jax/issues/13554
        assert jitted_median_heuristic(
            lambda *args: gaussian_kernelized_gfd(*args), X_gaussian, *arg
        ) == pytest.approx(median_gaussian_kernelized_fisher)
        k = GaussianExpKernelizedFisherKernel.create(
            ground_space_kernel=gaussian_kernel.create(sigma=default_sigma),
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
            sigma=default_sigma,
        ).with_median_heuristic(X_gaussian, *arg)
        assert k.sigma == pytest.approx(median_gaussian_kernelized_fisher)
        assert k.fd.ground_space_kernel.sigma == pytest.approx(
            median_gaussian_mmd_groundspace
        )

    # test that median heuristic does not return 0 when some data points are identical
    X_many_identical = jnp.concatenate(
        [
            jnp.zeros((num_samples - 1, dim)),
            jnp.ones((1, dim)),
        ],
        axis=0,
    )

    X_all_identical = jnp.zeros((num_samples, dim))

    pairwise_indices = jnp.tril_indices(n=num_samples, k=-1)
    linear_indices = (
        jnp.arange(start=0, stop=num_samples - 1, step=2),
        jnp.arange(start=1, stop=num_samples, step=2),
    )

    for X in [X_many_identical, X_all_identical]:
        for indices in pairwise_indices, linear_indices:
            median_euclidean = jitted_median(
                jax.jit(
                    jax.vmap(
                        lambda i, j: jnp.linalg.norm(X[i] - X[j], ord=2),
                        in_axes=(0, 0),
                    )
                )(*indices)
            )
            assert median_euclidean == 0

            median_heuristic_val = jitted_median_heuristic(
                lambda x, y: jnp.linalg.norm(x - y, ord=2), X, indices
            )
            assert median_heuristic_val > 0


def test_median_heuristic_fixed_point():
    # Ensure that if one updates a kernel with `with_median_heuristic`,
    # two times in a row, the updated bandwidth remains the same after
    # the first call.
    dim = 10
    num_samples = 5
    X = jax.random.normal(jax.random.PRNGKey(1234), (num_samples, dim))
    k = gaussian_kernel.create(sigma=2.0)

    k_med = k.with_median_heuristic(X)
    k_med_med = k_med.with_median_heuristic(X)

    assert k_med.sigma == k_med_med.sigma


def test_with_approximate_median_heuristic(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
):
    """Check `median_heuristic` and `with_median_heuristic`."""
    # Generate test data
    dim = 10
    num_samples = 5
    key, subkey = jax.random.split(jax.random.PRNGKey(1234))
    X = jax.random.normal(subkey, (num_samples, dim))
    sigmas = jnp.linspace(start=0.88, stop=1.0, num=num_samples)

    # Different subsets of indices
    pairwise_indices = jnp.triu_indices(n=num_samples, k=1)
    linear_indices = (
        jnp.arange(start=0, stop=num_samples - 1, step=2),
        jnp.arange(start=1, stop=num_samples, step=2),
    )

    for arg in ((pairwise_indices,), (linear_indices,), (None,), ()):
        # Gaussian distributions
        X_gaussian = jax.vmap(lambda x, s: GaussianConditionalModel(mu=x, sigma=s))(
            X, sigmas
        )

        k = ExpMMDGaussianKernel.create(ground_space_kernel=gaussian_kernel.create())
        k_true = k.with_median_heuristic(X_gaussian, *arg)

        with caplog.at_level(logging.WARNING):
            key, subkey = jax.random.split(key)
            k_true_app = k.with_approximate_median_heuristic(
                X_gaussian,
                1,
                subkey,
                *arg,
                mcmc_config=None,
            )
        assert jnp.allclose(k_true.sigma, k_true_app.sigma, rtol=0)
        assert "discouraged" in caplog.text
        caplog.clear()

        k_approx = ExpMMDKernel.create(ground_space_kernel=gaussian_kernel.create())
        key, subkey = jax.random.split(key)
        k_approx = jax.jit(
            type(k_approx).with_approximate_median_heuristic, static_argnums=(2,)
        )(k_approx, X_gaussian, 1000, subkey, *arg)
        assert jnp.allclose(k_true.sigma, k_approx.sigma, rtol=1e-1)

        k = GaussianExpFisherKernel.create(
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
        )
        k_true = k.with_median_heuristic(X_gaussian, *arg)

        with caplog.at_level(logging.WARNING):
            key, subkey = jax.random.split(key)
            k_true_app = k.with_approximate_median_heuristic(
                X_gaussian, 1, subkey, *arg, mcmc_config=None
            )
        assert jnp.allclose(k_true.sigma, k_true_app.sigma, rtol=0)
        assert "discouraged" in caplog.text
        caplog.clear()

        k_approx = ExpFisherKernel.create(
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
        )
        # few particles when discretizing -> rough approximation
        key, subkey = jax.random.split(key)
        k_approx1 = k_approx.with_approximate_median_heuristic(
            X_gaussian, 1, subkey, *arg, mcmc_config=None
        )
        with pytest.raises(AssertionError):
            assert jnp.allclose(k_true.sigma, k_approx1.sigma, rtol=1e-3)

        # use more particles -> better approximation
        key, subkey = jax.random.split(key)
        k_approx50000 = jax.jit(
            type(k_approx).with_approximate_median_heuristic, static_argnums=(2,)
        )(k_approx, X_gaussian, 50000, subkey, *arg)
        assert jnp.allclose(k_true.sigma, k_approx50000.sigma, rtol=1e-3)

        k = GaussianExpWassersteinKernel.create(sigma=2.0)
        k_true = k.with_median_heuristic(X_gaussian, *arg)

        # expwasserstein's approximation is itself -> values should be equal.
        with caplog.at_level(logging.WARNING):
            key, subkey = jax.random.split(key)
            k_approx = k.with_approximate_median_heuristic(
                X_gaussian, 1, subkey, *arg, mcmc_config=None
            )
        assert jnp.allclose(k_true.sigma, k_approx.sigma, rtol=0)
        assert "discouraged" in caplog.text
        caplog.clear()

        k = DiscreteExpMMDKernel.create(
            ground_space_kernel=gaussian_kernel.create(sigma=2.0), sigma=2.0
        )

        key, subkey = jax.random.split(key)
        X_discrete = jax.vmap(
            lambda x, k: cast(GaussianConditionalModel, x).non_mcmc_discretize(k, 10)
        )(X_gaussian, jax.random.split(subkey, num_samples))

        k_true = k.with_median_heuristic(X_discrete, *arg)

        with caplog.at_level(logging.WARNING):
            key, subkey = jax.random.split(key)
            k_true_app = k.with_approximate_median_heuristic(
                X_discrete, 1, subkey, *arg, mcmc_config=None
            )
        assert jnp.allclose(k_true.sigma, k_true_app.sigma, rtol=0)
        assert "discouraged" in caplog.text
        caplog.clear()

        k = DiscreteExpFisherKernel.create(
            base_dist=dist.Normal(jnp.zeros((dim,))),  # type: ignore
            key=jax.random.PRNGKey(1234),
            sigma=2.0,
        )
        k_true = k.with_median_heuristic(X_gaussian, *arg)
        with caplog.at_level(logging.WARNING):
            key, subkey = jax.random.split(key)
            k_true_app = k.with_approximate_median_heuristic(
                X_gaussian, 1, subkey, *arg, mcmc_config=None
            )
        assert jnp.allclose(k_true.sigma, k_true_app.sigma, rtol=0)
        assert "discouraged" in caplog.text
        caplog.clear()

        # do not show warnings at the end of the test
        _ = capsys.readouterr()


@pytest.mark.parametrize("d", [2, 5, 7])
def test_median_euclidean_distance_gaussians(d: int):
    # Sample two Gaussian models
    key = jax.random.PRNGKey(1234)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    p = GaussianConditionalModel(
        mu=jax.random.normal(subkey1, (d,)),
        sigma=float(jax.random.exponential(subkey2)),
    )
    key, subkey1, subkey2 = jax.random.split(key, 3)
    q = GaussianConditionalModel(
        mu=jax.random.normal(subkey1, (d,)),
        sigma=float(jax.random.exponential(subkey2)),
    )

    # Median heuristic based on samples from both distributions
    num_samples = 2_000
    subkey1, subkey2 = jax.random.split(key)
    X = p.sample_from_conditional(subkey1, num_samples)
    Y = q.sample_from_conditional(subkey2, num_samples)
    XY = jnp.concatenate([X, Y])
    median_samples = median_heuristic(
        lambda x, y: cast(float, jnp.sqrt(jnp.sum(jnp.square(x - y)))), XY
    )
    assert median_samples == pytest.approx(
        gaussian_kernel.create().with_median_heuristic(XY).sigma
    )

    # Median using the analytical derivations
    median = median_euclidean_gaussians(p, q)
    assert median == pytest.approx(median_samples, rel=1e-2)

    # For a single distribution, the pairwise distances should follow
    # a chi distribution with d degrees of freedom scaled by
    # sqrt(2) * standard deviation of the Gaussian
    median_chi = tfp.distributions.Chi(d).quantile(0.5)
    for dist in (p, q):
        median_same = median_euclidean_gaussians(dist, dist)
        assert median_same == pytest.approx(jnp.sqrt(2) * dist.sigma * median_chi)
