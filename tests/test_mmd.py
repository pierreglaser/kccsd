from typing import Any, Dict, Type

import pytest
from jax_samplers.pytypes import Array
from kwgflows.rkhs.kernels import base_kernel
from test_statistical_tests import BaseTestOneSampleTest
from typing_extensions import Unpack

from calibration.statistical_tests.mmd import MMDOneSample_T, MMDTestInput_T, mmd_test
from calibration_benchmarks.data_generators.mmd import mmd_test_data_generator


@pytest.mark.parametrize(
    "median_heuristic",
    (
        True,
        False,
    ),
)
class MMDOneSampleTest(
    BaseTestOneSampleTest[MMDOneSample_T[Array], Unpack[MMDTestInput_T[Array]]]
):
    data_generator: mmd_test_data_generator
    test_cls: Type[mmd_test[Array]] = mmd_test

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {"kernel": kernel}


class TestH0MMDOneSampleTest(MMDOneSampleTest):
    data_generator: mmd_test_data_generator = mmd_test_data_generator()


class TestH1MMDOneSampleTest(MMDOneSampleTest):
    data_generator: mmd_test_data_generator = mmd_test_data_generator(delta=1e-1)
