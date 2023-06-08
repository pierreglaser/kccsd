import os

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as np_distribution
import pytest
import torch
from jax import random
from numpyro.distributions import transforms as np_transforms
from pyro import distributions as pyro_distributions
from pyro.distributions import transforms as pyro_transforms
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist
from sbi_ebm.sbibm.tasks import JaxTask
from sbibm.tasks import get_task
from torch import distributions as torch_distributions

SBIBM_TASKS = (
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "lotka_volterra",
    "sir",
    "two_moons",
    "slcp",
    "slcp_distractors",
    "bernoulli_glm_raw",
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("task_name", SBIBM_TASKS)
def test_prior_simulator_equal_output(task_name: str):
    if task_name in ("lotka_volterra", "sir"):
        from distutils import spawn

        julia_exe = spawn.find_executable("julia")
        if julia_exe is None:
            print("skipping lotka volterra task because julia is not installed")
            return
        elif os.environ.get("CI", False):
            print("skipping lotka volterra task during CI jobs")
            return

    t = get_task(task_name)
    jt = JaxTask(t)

    torch.manual_seed(42)
    jp = jt.get_prior()
    theta_j = jp()
    x_j = jt.get_simulator()(theta_j)

    torch.manual_seed(42)
    p = t.get_prior()
    theta_t = p()
    x_t = t.get_simulator()(theta_t)

    assert jnp.allclose(theta_j, jnp.array(theta_t.detach().numpy()))
    assert jnp.allclose(x_j, jnp.array(x_t.detach().numpy()))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("task_name", SBIBM_TASKS)
def test_distribution_conversion(task_name: str):
    if task_name in ("lotka_volterra", "sir"):
        from distutils import spawn

        julia_exe = spawn.find_executable("julia")
        if julia_exe is None:
            print("skipping lotka volterra task because julia is not installed")
            return

        elif os.environ.get("CI", False):
            print("skipping lotka volterra task during CI jobs")
            return

    pyro_prior = get_task(task_name).get_prior_dist()
    assert isinstance(pyro_prior, pyro_distributions.Distribution)

    converted_prior = convert_dist(pyro_prior, implementation="numpyro")
    assert isinstance(converted_prior, np_distribution.Distribution)

    torch.manual_seed(42)

    samples = pyro_prior.sample(sample_shape=torch.Size((1000,)))

    pyro_log_ls = jnp.array(pyro_prior.log_prob(samples))
    converted_log_ls = converted_prior.log_prob(jnp.array(samples))

    assert jnp.allclose(pyro_log_ls, converted_log_ls, rtol=1e-5, atol=1e-5)

    from sbibm.algorithms.sbi.utils import wrap_prior_dist

    wrapped_pyro_prior = wrap_prior_dist(
        pyro_prior, get_task(task_name)._get_transforms()["parameters"]  # type: ignore
    )

    converted_wrapped_prior = convert_dist(wrapped_pyro_prior, implementation="numpyro")
    assert isinstance(converted_wrapped_prior, np_distribution.Distribution)

    torch.manual_seed(42)

    wrapped_samples = wrapped_pyro_prior.sample(sample_shape=torch.Size((1000,)))

    wrapped_pyro_log_ls = jnp.array(wrapped_pyro_prior.log_prob(wrapped_samples))
    wrapped_converted_log_ls = converted_wrapped_prior.log_prob(
        jnp.array(wrapped_samples)
    )

    assert jnp.allclose(
        wrapped_pyro_log_ls, wrapped_converted_log_ls, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("task_name", SBIBM_TASKS)
def test_transform_to_bijector_conversion(task_name: str):
    pyro_prior = get_task(task_name).get_prior_dist()
    assert isinstance(pyro_prior, pyro_distributions.Distribution)

    np_prior = convert_dist(pyro_prior, implementation="numpyro")

    if hasattr(np_prior, "support"):
        pyro_transform = pyro_transforms.biject_to(pyro_prior.support)
        np_transform: np_transforms.Transform = np_distribution.biject_to(
            np_prior.support  # type: ignore
        )

        np_transformed_dist = np_distribution.TransformedDistribution(
            np_prior, [np_transform.inv]
        )
        pyro_transformed_dist = torch_distributions.TransformedDistribution(
            pyro_prior, [pyro_transform.inv]
        )

        key = random.PRNGKey(0)
        np_samples = np_transformed_dist.sample(key, sample_shape=(10,))  # type: ignore
        pyro_samples = torch.from_numpy(np.array(np_samples))

        np_transformed_samples = np_transform(np_samples)
        pyro_transformed_samples = pyro_transform(pyro_samples)
        assert jnp.allclose(
            np_transformed_samples,
            jnp.array(pyro_transformed_samples),
            rtol=1e-4,
            atol=1e-6,
        )

        # Check that the chain rule of probability is well verified in the (transformed)
        # tfp distribution
        np_log_probs = np_transformed_dist.log_prob(np_samples)
        pyro_log_probs = pyro_transformed_dist.log_prob(pyro_samples)
        assert jnp.allclose(
            jnp.array(pyro_log_probs), np_log_probs, rtol=1e-4, atol=1e-6
        )
