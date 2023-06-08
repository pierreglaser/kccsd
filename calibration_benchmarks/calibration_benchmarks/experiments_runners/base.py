import logging
from functools import partial
from time import time
from typing import Any, Dict, Literal, Tuple, cast

import jax
import jax.numpy as jnp
from calibration.statistical_tests.base import (
    OneSample_T,
    OneSampleTest,
    TestInput_T,
    TestResults,
)
from flax import struct
from jax import jit, random, vmap
from jax.random import KeyArray
from jax_samplers.pytypes import Array
from kwgflows.pytypes import PRNGKeyArray
from kwgflows.rkhs.kernels import base_kernel
from typing_extensions import Unpack

from calibration_benchmarks.data_generators.base import data_generator

from .utils import get_logger


class BenchmarkResult(struct.PyTreeNode):
    time: float
    results: TestResults

    @property
    def reject_mean(self):
        return jnp.mean(self.results.result)


@partial(jit, static_argnums=(3,))
def _sample_and_test(
    key: KeyArray,
    t: OneSampleTest[OneSample_T, Unpack[TestInput_T]],
    d: data_generator[Unpack[TestInput_T]],
    num_samples: int,
) -> TestResults:
    subkey1, subkey2 = random.split(key)
    data = d.get_data(subkey1, num_samples)
    ret = t.__call__(*data, key=subkey2)
    return cast(TestResults, ret)


def distribute_test(
    t: OneSampleTest[OneSample_T, Unpack[TestInput_T]],
    data_generator: data_generator[Unpack[TestInput_T]],
    num_tests: int,
    num_samples: int,
    key: KeyArray,
    parallel_backend: Literal["vmap", "scan", "sequential"] = "scan",
    log_level: int = logging.INFO,
):
    logger = get_logger("calibration_benchmarks.experiments_runner", log_level)

    def test_manytimes(key: KeyArray) -> TestResults:
        subkeys = random.split(key, num_tests)
        if parallel_backend == "vmap":
            rets = vmap(_sample_and_test, in_axes=(0, None, None, None))(
                subkeys,
                t,
                data_generator,
                num_samples,
            )
        elif parallel_backend == "scan":

            def step_fn(_: Any, k: Array) -> Tuple[Any, TestResults]:
                ret = _sample_and_test(k, t, data_generator, num_samples)
                return None, ret

            logger.info("distributing tests via jax.lax.scan")
            # fori loop
            _, rets = jax.lax.scan(step_fn, None, subkeys)
        elif parallel_backend == "sequential":
            logger.info("running tests one by one")

            def step_fn(_: Any, k: Array) -> Tuple[Any, TestResults]:
                ret = _sample_and_test(k, t, data_generator, num_samples)
                return None, ret

            all_rets = []
            # jitted_step_fn = jit(step_fn)
            for i, k in enumerate(subkeys):
                logger.debug(f"running test {i}")
                _, ret = step_fn(None, k)
                all_rets.append(ret)

            rets = cast(TestResults, jax.tree_map(lambda *x: jnp.stack(x), *all_rets))
            logger.info("done.")
        else:
            raise ValueError(f"Unknown parallel_backend: {parallel_backend}")
        return rets

    key, subkey = random.split(key)
    # return jit(test_manytimes)(subkey)
    # don't jit for now as distributions methods other than `sequential` blow
    # up the memory usage, and jitting python loops has very long compilation
    # times. jit the inner function instead.
    return test_manytimes(subkey)


class ExperimentRunner(struct.PyTreeNode):
    parallel_backend: Literal["vmap", "scan", "sequential"] = "scan"
    log_level: int = struct.field(pytree_node=False, default=logging.INFO)

    def _test(
        self,
        t: OneSampleTest[OneSample_T, Unpack[TestInput_T]],
        generator: data_generator[Unpack[TestInput_T]],
        num_samples: int,
        num_tests: int,
        key: KeyArray,
    ):
        return distribute_test(
            t,
            generator,
            num_tests,
            num_samples,
            key,
            self.parallel_backend,
            log_level=self.log_level,
        )

    def make_kernel_kwargs(self, _: base_kernel[Array]) -> Dict[str, Any]:
        raise NotImplementedError

    def run_manytimes(
        self,
        key: PRNGKeyArray,
        generator: data_generator[Unpack[TestInput_T]],
        test: OneSampleTest[OneSample_T, Unpack[TestInput_T]],
        num_samples: int,
        num_tests: int,
    ):
        logger = get_logger("calibration_benchmarks.experiments_runner", self.log_level)
        logger.info(
            f"level: {test.alpha:<10} num permutations: {test.num_permutations:<10} "
            f"num tests: {num_tests:<10}"
        )

        t0 = time()
        ret = self._test(
            test,
            generator,
            num_samples,
            num_tests,
            key,
        )
        t = time() - t0
        return BenchmarkResult(time=t, results=ret)
