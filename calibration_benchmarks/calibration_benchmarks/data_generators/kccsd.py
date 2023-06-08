from typing import Generic, Tuple, Type, TypeVar, Union

from calibration.conditional_models.base import (
    DiscretizedModel,
    LogDensityModel,
)
from calibration.statistical_tests.kccsd import KCCSDTestInput_T
from flax import struct
from jax import random
from kwgflows.pytypes import Array
from typing_extensions import Unpack

from calibration_benchmarks.data_generators.base import (
    GM,
    calibration_test_data_generator_base,
    gaussian_data_generator,
    heteroscedastic_gaussian_data_generator,
    linear_gaussian_data_generator,
    quadratic_gaussian_data_generator,
)

M = TypeVar("M", bound=LogDensityModel)


class calibrated_score(Generic[M], struct.PyTreeNode):
    model_cls: Type[M] = struct.field(pytree_node=False)

    def __call__(self, y: Array, log_p_x: M) -> Array:
        return log_p_x.score(y)


class kccsd_data_generator_base(
    calibration_test_data_generator_base[Unpack[KCCSDTestInput_T[M]], M]
):
    def add_score(
        self, conditional_models: M, Y: Array
    ) -> Tuple[Unpack[KCCSDTestInput_T[M]]]:
        score_y_given_px = calibrated_score(type(conditional_models))
        return (conditional_models, Y, score_y_given_px)

    def get_data(self, key: random.KeyArray, num_samples: int):
        return self.add_score(*self.get_models_and_targets(key, num_samples))


class kccsd_gaussian_data_generator(
    kccsd_data_generator_base[GM],
    gaussian_data_generator[Unpack[KCCSDTestInput_T[GM]], GM],
):
    pass


class kccsd_linear_gaussian_data_generator(
    kccsd_data_generator_base[GM],
    linear_gaussian_data_generator[Unpack[KCCSDTestInput_T[GM]], GM],
):
    pass


class kccsd_heteroscedastic_gaussian_data_generator(
    kccsd_data_generator_base[GM],
    heteroscedastic_gaussian_data_generator[Unpack[KCCSDTestInput_T[GM]], GM],
):
    pass


class kccsd_quadratic_gaussian_data_generator(
    kccsd_data_generator_base[GM],
    quadratic_gaussian_data_generator[Unpack[KCCSDTestInput_T[GM]], GM],
):
    pass
