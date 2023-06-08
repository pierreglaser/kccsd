from typing import Any, Dict, Type

import pytest
from kwgflows.pytypes import Array
from kwgflows.rkhs.kernels import base_kernel
from test_statistical_tests import BaseTestOneSampleTest
from typing_extensions import Unpack

from calibration.statistical_tests.kcsd import (
    KCSDOneSample_T,
    KCSDTestInput_T,
    kcsd_test,
)
from calibration_benchmarks.data_generators.kcsd import kcsd_test_data_generator


@pytest.mark.parametrize("median_heuristic", (True, False))
class KCSDOneSampleTest(
    BaseTestOneSampleTest[KCSDOneSample_T[Array], Unpack[KCSDTestInput_T[Array]]]
):
    data_generator: kcsd_test_data_generator
    test_cls: Type[kcsd_test[Array]] = kcsd_test

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {"x_kernel": kernel, "y_kernel": kernel}


class TestH0KCSDOneSampleTest(KCSDOneSampleTest):
    data_generator: kcsd_test_data_generator = kcsd_test_data_generator()


class TestH1KCSDOneSampleTest(KCSDOneSampleTest):
    data_generator: kcsd_test_data_generator = kcsd_test_data_generator(delta=1e-1)
