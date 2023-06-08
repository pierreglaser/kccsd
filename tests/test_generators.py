import jax
import pytest
from jax import numpy as jnp

from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration_benchmarks.data_generators.base import (
    gaussian_data_generator,
)
from calibration_benchmarks.data_generators.kccsd import kccsd_gaussian_data_generator
from calibration_benchmarks.data_generators.skce import skce_gaussian_data_generator


@pytest.mark.parametrize("num_samples", [15, 24])
@pytest.mark.parametrize(
    "generator",
    [
        skce_gaussian_data_generator(
            dims=3, sigma=2.1, model_type=GaussianConditionalModel
        ),
        kccsd_gaussian_data_generator(
            dims=5, sigma=0.6, model_type=GaussianConditionalModel
        ),
    ],
    ids=["SKCE", "KCCSD"],
)
def test_gaussian_data_generator(
    generator: gaussian_data_generator[GaussianConditionalModel], num_samples: int
):
    key = jax.random.PRNGKey(1234)
    models, targets = generator.get_models_and_targets(key, num_samples)
    assert isinstance(models, GaussianConditionalModel)
    assert models.mu.shape == (num_samples, generator.d)
    assert jnp.all(models.sigma == generator.sigma)
    assert targets.shape == (num_samples, generator.d)
