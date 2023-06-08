from typing import Any, Dict, Type

import pytest
from kwgflows.pytypes import Array
from kwgflows.rkhs.kernels import base_kernel
from test_statistical_tests import BaseTestOneSampleTest
from typing_extensions import Unpack

from calibration.statistical_tests.ksd import KSDOneSample_T, KSDTestInput_T, ksd_test
from calibration_benchmarks.data_generators.ksd import ksd_test_data_generator


@pytest.mark.parametrize("median_heuristic", (False,))
class KSDOneSampleTest(BaseTestOneSampleTest[KSDOneSample_T, Unpack[KSDTestInput_T]]):
    data_generator: ksd_test_data_generator
    test_cls: Type[ksd_test] = ksd_test

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        return {"kernel": kernel}


class TestH0KSDOneSampleTest(KSDOneSampleTest):
    data_generator: ksd_test_data_generator = ksd_test_data_generator()


class TestH1KSDOneSampleTest(KSDOneSampleTest):
    data_generator: ksd_test_data_generator = ksd_test_data_generator(delta=1e-1)
