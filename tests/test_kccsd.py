from typing import Any, Dict, Type

import pytest
from kwgflows.pytypes import Array
from kwgflows.rkhs.kernels import base_kernel, gaussian_kernel
from test_statistical_tests import BaseTestOneSampleTest
from typing_extensions import Unpack
import numpyro.distributions as np_distributions
import jax.numpy as jnp

from calibration.conditional_models.base import (
    ConcreteLogDensityModel,
    DiscretizedModel,
    DistributionModel,
)
from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.kernels import (
    ExpFisherKernel,
    ExpMMDKernel,
    GaussianExpWassersteinKernel,
)
from calibration.statistical_tests.kccsd import (
    KCCSDOneSample_T,
    KCCSDTestInput_T,
    kccsd_test,
)
from calibration_benchmarks.data_generators.kccsd import (
    kccsd_data_generator_base,
    kccsd_gaussian_data_generator,
    kccsd_heteroscedastic_gaussian_data_generator,
    kccsd_linear_gaussian_data_generator,
    kccsd_quadratic_gaussian_data_generator,
)


@pytest.fixture
def kernel():
    return gaussian_kernel.create()


class GaussianAnalyticalKCCSDOneSampleTest(
    BaseTestOneSampleTest[
        KCCSDOneSample_T[GaussianConditionalModel],
        Unpack[KCCSDTestInput_T[GaussianConditionalModel]],
    ]
):
    data_generator: kccsd_data_generator_base[GaussianConditionalModel]
    test_cls: Type[
        kccsd_test[GaussianConditionalModel, DiscretizedModel[Any]]
    ] = kccsd_test
    x_kernel = GaussianExpWassersteinKernel.create(sigma=1)

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {"x_kernel": self.x_kernel, "y_kernel": kernel}


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0GaussianAnalyticalKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_gaussian_data_generator(sigma=1.3, model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0FirstKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_gaussian_data_generator(shift_dim=1, model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0LGMKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_linear_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_linear_gaussian_data_generator(model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True, False))
class TestH1GaussianAnalyticalKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_gaussian_data_generator(delta=1e-1, model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True, False))
class TestH1FirstKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_gaussian_data_generator(
        delta=1e-1, shift_dim=1, model_type=GaussianConditionalModel
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1HGMKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_heteroscedastic_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_heteroscedastic_gaussian_data_generator(
        delta=1, model_type=GaussianConditionalModel
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1QGMKCCSDOneSampleTest(GaussianAnalyticalKCCSDOneSampleTest):
    data_generator: kccsd_quadratic_gaussian_data_generator[
        GaussianConditionalModel
    ] = kccsd_quadratic_gaussian_data_generator(
        delta=1, model_type=GaussianConditionalModel
    )


class GaussianApproximateKCCSDOneSampleTest(
    BaseTestOneSampleTest[
        KCCSDOneSample_T[DistributionModel],
        Unpack[KCCSDTestInput_T[DistributionModel]],
    ]
):
    data_generator: kccsd_gaussian_data_generator[DistributionModel]
    test_cls: Type[kccsd_test[DistributionModel, DiscretizedModel[Any]]] = kccsd_test
    x_kernel = ExpMMDKernel.create(
        sigma=1.0, ground_space_kernel=gaussian_kernel.create(sigma=1.0)
    )

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {
            "x_kernel": self.x_kernel,
            "y_kernel": kernel,
            "approximate": True,
            "approximation_num_particles": 2,
        }


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0GaussianApproximateKCCSDOneSampleTest(
    GaussianApproximateKCCSDOneSampleTest
):
    data_generator: kccsd_gaussian_data_generator[
        DistributionModel
    ] = kccsd_gaussian_data_generator(sigma=0.9, model_type=DistributionModel)


@pytest.mark.parametrize("median_heuristic", (True, False))
class TestH1GaussianApproximateKCCSDOneSampleTest(
    GaussianApproximateKCCSDOneSampleTest
):
    data_generator = kccsd_gaussian_data_generator(
        delta=1e-1, model_type=DistributionModel
    )


class GaussianApproximateMCMCKCCSDOneSampleTest(
    BaseTestOneSampleTest[
        KCCSDOneSample_T[ConcreteLogDensityModel],
        Unpack[KCCSDTestInput_T[ConcreteLogDensityModel]],
    ]
):
    data_generator: kccsd_gaussian_data_generator[
        ConcreteLogDensityModel
    ] = kccsd_gaussian_data_generator(delta=1e-1)
    test_cls: Type[
        kccsd_test[ConcreteLogDensityModel, DiscretizedModel[Any]]
    ] = kccsd_test
    x_kernel = ExpMMDKernel.create(sigma=1.0, ground_space_kernel=gaussian_kernel(1.0))

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {
            "x_kernel": self.x_kernel,
            "y_kernel": kernel,
            "approximate": True,
            "approximation_num_particles": 2,
            "approximation_mcmc_num_warmup_steps": 10,
        }


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0GaussianApproximateMCMCKCCSDOneSampleTest(
    GaussianApproximateMCMCKCCSDOneSampleTest
):
    data_generator: kccsd_gaussian_data_generator[
        ConcreteLogDensityModel
    ] = kccsd_gaussian_data_generator(sigma=1.2, model_type=ConcreteLogDensityModel)
    x_kernel = ExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1GaussianApproximateMCMCKCCSDOneSampleTest(
    GaussianApproximateMCMCKCCSDOneSampleTest
):
    data_generator: kccsd_gaussian_data_generator[
        ConcreteLogDensityModel
    ] = kccsd_gaussian_data_generator(
        delta=1e-1, sigma=1.2, model_type=ConcreteLogDensityModel
    )
