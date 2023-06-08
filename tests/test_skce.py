from typing import Any, Dict, Tuple, Type

import jax.numpy as jnp
import numpyro.distributions as np_distributions
import pytest
from kwgflows.pytypes import Array
from kwgflows.rkhs.kernels import base_kernel, gaussian_kernel
from test_statistical_tests import BaseTestOneSampleTest, scalar_kernels
from typing_extensions import Unpack

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
from calibration.statistical_tests.skce import (
    SKCEOneSample_T,
    SKCETestInput_T,
    skce_test,
)
from calibration_benchmarks.data_generators.skce import (
    skce_data_generator_base,
    skce_gaussian_data_generator,
    skce_heteroscedastic_gaussian_data_generator,
    skce_linear_gaussian_data_generator,
    skce_quadratic_gaussian_data_generator,
)


@pytest.fixture
def kernel():
    return gaussian_kernel.create()


class GaussianAnalyticalSKCEOneSampleTest(
    BaseTestOneSampleTest[
        SKCEOneSample_T[GaussianConditionalModel],
        Unpack[SKCETestInput_T[GaussianConditionalModel]],
    ]
):
    data_generator: skce_data_generator_base[GaussianConditionalModel]
    test_cls: Type[
        skce_test[GaussianConditionalModel, DiscretizedModel[Any]]
    ] = skce_test
    x_kernel = GaussianExpWassersteinKernel.create(sigma=1)

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {"x_kernel": self.x_kernel, "y_kernel": kernel}


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0GaussianAnalyticalTestSKCEOneSampleTest(
    GaussianAnalyticalSKCEOneSampleTest
):
    data_generator = skce_gaussian_data_generator(
        sigma=0.9, model_type=GaussianConditionalModel
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0FirstSKCEOneSampleTest(GaussianAnalyticalSKCEOneSampleTest):
    data_generator: skce_gaussian_data_generator[
        GaussianConditionalModel
    ] = skce_gaussian_data_generator(shift_dim=1, model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0LGMSKCEOneSampleTest(GaussianAnalyticalSKCEOneSampleTest):
    data_generator: skce_linear_gaussian_data_generator[
        GaussianConditionalModel
    ] = skce_linear_gaussian_data_generator(model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True, False))
class TestH1SKCEOneSampleTest(GaussianAnalyticalSKCEOneSampleTest):
    data_generator: skce_gaussian_data_generator[
        GaussianConditionalModel
    ] = skce_gaussian_data_generator(delta=1e-1, model_type=GaussianConditionalModel)


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1FirstSKCEOneSampleTest(GaussianAnalyticalSKCEOneSampleTest):
    data_generator: skce_gaussian_data_generator[
        GaussianConditionalModel
    ] = skce_gaussian_data_generator(
        delta=1e-1, shift_dim=1, model_type=GaussianConditionalModel
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1HGMSKCEOneSampleTest(GaussianAnalyticalSKCEOneSampleTest):
    data_generator: skce_heteroscedastic_gaussian_data_generator[
        GaussianConditionalModel
    ] = skce_heteroscedastic_gaussian_data_generator(
        delta=1, model_type=GaussianConditionalModel
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1QGMSKCEOneSampleTest(GaussianAnalyticalSKCEOneSampleTest):
    data_generator: skce_quadratic_gaussian_data_generator[
        GaussianConditionalModel
    ] = skce_quadratic_gaussian_data_generator(
        delta=1, model_type=GaussianConditionalModel
    )


class GaussianApproximateSKCEOneSampleTest(
    BaseTestOneSampleTest[
        SKCEOneSample_T[DistributionModel],
        Unpack[SKCETestInput_T[DistributionModel]],
    ]
):
    data_generator: skce_gaussian_data_generator[
        DistributionModel
    ] = skce_gaussian_data_generator(delta=1e-1)
    test_cls: Type[skce_test[DistributionModel, DiscretizedModel[Any]]] = skce_test
    x_kernel = ExpMMDKernel.create(sigma=1.0, ground_space_kernel=gaussian_kernel(1.0))

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {
            "x_kernel": self.x_kernel,
            "y_kernel": kernel,
            "approximate": True,
            "approximation_num_particles": 2,
        }

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_quadratic_test(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: Tuple[int, ...] = (200,),
        num_permutations: int = 200,
        num_tests: int = 100,
    ):
        return super().test_quadratic_test(
            kernel,
            median_heuristic,
            num_samples,
            num_permutations,
            num_tests,
        )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0GaussianApproximateSKCEOneSampleTest(GaussianApproximateSKCEOneSampleTest):
    data_generator: skce_gaussian_data_generator[
        DistributionModel
    ] = skce_gaussian_data_generator(sigma=1.2, model_type=DistributionModel)


@pytest.mark.parametrize("median_heuristic", (True, False))
class TestH1GaussianApproximateSKCEOneSampleTest(GaussianApproximateSKCEOneSampleTest):
    data_generator: skce_gaussian_data_generator[
        DistributionModel
    ] = skce_gaussian_data_generator(delta=1e-1, model_type=DistributionModel)


class GaussianApproximateMCMCSKCEOneSampleTest(
    BaseTestOneSampleTest[
        SKCEOneSample_T[ConcreteLogDensityModel],
        Unpack[SKCETestInput_T[ConcreteLogDensityModel]],
    ]
):
    data_generator: skce_gaussian_data_generator[
        ConcreteLogDensityModel
    ] = skce_gaussian_data_generator(delta=1e-1)
    test_cls: Type[
        skce_test[ConcreteLogDensityModel, DiscretizedModel[Any]]
    ] = skce_test
    x_kernel = ExpMMDKernel.create(sigma=1.0, ground_space_kernel=gaussian_kernel(1.0))

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {
            "x_kernel": self.x_kernel,
            "y_kernel": kernel,
            "approximate": True,
            "approximation_num_particles": 2,
            "approximation_mcmc_num_warmup_steps": 10,
        }

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_quadratic_test(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: Tuple[int, ...] = (200,),
        num_permutations: int = 200,
        num_tests: int = 100,
    ):
        return super().test_quadratic_test(
            kernel,
            median_heuristic,
            num_samples,
            num_permutations,
            num_tests,
        )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH0GaussianApproximateMCMCSKCEOneSampleTest(
    GaussianApproximateMCMCSKCEOneSampleTest
):
    data_generator: skce_gaussian_data_generator[
        ConcreteLogDensityModel
    ] = skce_gaussian_data_generator(sigma=1.2, model_type=ConcreteLogDensityModel)
    x_kernel = ExpFisherKernel.create(
        base_dist=np_distributions.Normal(loc=jnp.zeros((2,)), scale=0.5),  # type: ignore
        sigma=0.5,
    )


@pytest.mark.parametrize("median_heuristic", (True,))
class TestH1GaussianApproximateMCMCSKCEOneSampleTest(
    GaussianApproximateMCMCSKCEOneSampleTest
):
    data_generator: skce_gaussian_data_generator[
        ConcreteLogDensityModel
    ] = skce_gaussian_data_generator(
        delta=1e-1, sigma=1.2, model_type=ConcreteLogDensityModel
    )
