from typing import Tuple

from calibration.statistical_tests.skce import P, SKCETestInput_T
from jax import random
from typing_extensions import Unpack

from calibration_benchmarks.data_generators.base import (
    GM,
    calibration_test_data_generator_base,
    gaussian_data_generator,
    heteroscedastic_gaussian_data_generator,
    linear_gaussian_data_generator,
    quadratic_gaussian_data_generator,
)


class skce_data_generator_base(
    calibration_test_data_generator_base[Unpack[SKCETestInput_T[P]], P]
):
    def get_data(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[Unpack[SKCETestInput_T[P]]]:
        return self.get_models_and_targets(key, num_samples)


class skce_gaussian_data_generator(
    skce_data_generator_base[GM],
    gaussian_data_generator[Unpack[SKCETestInput_T[GM]], GM],
):
    pass


class skce_linear_gaussian_data_generator(
    skce_data_generator_base[GM],
    linear_gaussian_data_generator[Unpack[SKCETestInput_T[GM]], GM],
):
    pass


class skce_heteroscedastic_gaussian_data_generator(
    skce_data_generator_base[GM],
    heteroscedastic_gaussian_data_generator[Unpack[SKCETestInput_T[GM]], GM],
):
    pass


class skce_quadratic_gaussian_data_generator(
    skce_data_generator_base[GM],
    quadratic_gaussian_data_generator[Unpack[SKCETestInput_T[GM]], GM],
):
    pass
