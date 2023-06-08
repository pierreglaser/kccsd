import time
from typing import Any, Dict, Generic, Literal, Tuple, Type, cast

import jax
import jax.numpy as jnp
import pytest
import scipy
from jax import jit, random, vmap
from jax.random import KeyArray
from jax_samplers.pytypes import Array
from kwgflows.pytypes import T
from kwgflows.rkhs.kernels import (
    base_kernel,
    energy_kernel,
    gaussian_kernel,
    imq_kernel,
    laplace_kernel,
)
from typing_extensions import Unpack

from calibration.statistical_tests.base import OneSample_T, OneSampleTest, TestInput_T
from calibration_benchmarks.data_generators.base import data_generator

scalar_kernels = [
    gaussian_kernel.create(),
    laplace_kernel.create(),
    imq_kernel.create(),
    energy_kernel.create(),
]


def _test(
    t: OneSampleTest[OneSample_T, Unpack[TestInput_T]],
    data_generator: data_generator[Unpack[TestInput_T]],
    num_samples: Tuple[int, ...],
    num_tests: int,
    key: KeyArray,
    use_vmap: bool = True,
):
    def _sample_and_test(ns: int, key: KeyArray) -> bool:
        key, subkey = random.split(key)
        data = data_generator.get_data(subkey, ns)
        key, subkey = random.split(key)
        ret = t(*data, key=subkey)
        return ret.result

    def get_reject_mean(ns: int, key: KeyArray) -> float:
        subkeys = random.split(key, num_tests)
        if use_vmap:
            rejects = cast(
                Array,
                vmap(_sample_and_test, in_axes=(None, 0))(ns, subkeys),
            )
            return cast(float, jnp.mean(rejects))
        else:

            def step_fn(i: int, acc_sum: float) -> float:
                reject = _sample_and_test(ns, subkeys[i])
                return acc_sum + reject

            # fori loop
            reject = cast(float, jax.lax.fori_loop(0, num_tests, step_fn, 0.0))
            return cast(float, reject / num_tests)

    reject_means = []
    h_prefix = "H0" if data_generator.h0 else "H1"
    for ns in num_samples:
        key, subkey = random.split(key)
        t0 = time.time()
        # reject_mean = get_reject_mean(ns, subkey)
        reject_mean = jit(get_reject_mean, static_argnums=(0,))(ns, subkey)
        t_test = time.time() - t0
        print(
            f"{h_prefix}: num_samples={ns:<10} reject_mean={reject_mean:<10.4f}"
            f"time={t_test:.4f}s"
        )
        reject_means.append(reject_mean)

    return reject_means


class BaseTestOneSampleTest(Generic[OneSample_T, Unpack[TestInput_T]]):
    data_generator: data_generator[Unpack[TestInput_T]]
    test_cls: Type[OneSampleTest[OneSample_T, Unpack[TestInput_T]]]

    def make_extra_test_kwargs(self, kernel: base_kernel[Array]) -> Dict[str, Any]:
        raise NotImplementedError

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_quadratic_test(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: Tuple[int, ...] = (500,),
        num_permutations: int = 500,
        num_tests: int = 100,
    ):
        level = 0.05
        key = random.PRNGKey(1234)
        print("\nThis test is currently only informative and cannot error out")
        print(
            f"level: {level:<10} num permutations: {num_permutations:<10} "
            f"num tests: {num_tests:<10} median heuristic: {median_heuristic}"
        )

        t = self.test_cls(
            alpha=level,
            num_permutations=num_permutations,
            median_heuristic=median_heuristic,
            **self.make_extra_test_kwargs(kernel),
        )

        # TODO: the test should be calibrated, and should probably error out if the
        # empirical type-I error is too far away from `level` if the null hypothesis is true
        # On the other hand, we cannot test for a power value.
        _test(t, self.data_generator, num_samples, num_tests, key, use_vmap=False)

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_linear_test(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: Tuple[int, ...] = (100, 1000),
        num_permutations: int = 500,
        num_tests: int = 100,
        R: int = 10,
    ):
        level = 0.05
        key = random.PRNGKey(0)
        print("\nThis test is currently only informative and cannot error out")
        print(
            f"level: {level:<10} num permutations: {num_permutations:<10} R: {R:<10}"
            f"num tests: {num_tests:<10} median heuristic: {median_heuristic}"
        )
        t = self.test_cls(
            alpha=level,
            num_permutations=num_permutations,
            R=R,
            median_heuristic=median_heuristic,
            **self.make_extra_test_kwargs(kernel),
        )
        # TODO: the test should be calibrated, and should probably error out if the
        # empirical type-I error is too far away from `level` if the null hypothesis is true
        # On the other hand, we cannot test for a power value.
        _test(t, self.data_generator, num_samples, num_tests, key, use_vmap=False)

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_linear_equals_quadratic_test(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: int = 20,
        num_permutations: int = 100,
        num_tests: int = 100,
    ):
        if not self.data_generator.h0:
            return

        level = 0.1
        key = random.PRNGKey(0)

        print(
            f"\nlevel: {level:<10} num samples: {num_samples:<10} "
            f"num permutations: {num_permutations:<10} num tests: {num_tests:<10} "
            f"median heuristic: {median_heuristic}"
        )

        tq = self.test_cls(
            alpha=level,
            num_permutations=num_permutations,
            median_heuristic=median_heuristic,
            **self.make_extra_test_kwargs(kernel),
        )
        tl = self.test_cls(
            alpha=level,
            num_permutations=num_permutations,
            R=num_samples - 1,
            median_heuristic=median_heuristic,
            **self.make_extra_test_kwargs(kernel),
        )

        mq = _test(
            tq, self.data_generator, (num_samples,), num_tests, key, use_vmap=True
        )
        ml = _test(
            tl, self.data_generator, (num_samples,), num_tests, key, use_vmap=True
        )

        assert jnp.allclose(mq[0], ml[0])

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_linear_noperm(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: Tuple[int, ...] = (100, 1000, 10000),
        num_tests: int = 100,
    ):
        level = 0.05
        key = random.PRNGKey(0)
        print("\nThis test is currently only informative and cannot error out")
        print(
            f"level: {level:<10} num tests: {num_tests:<10} median heuristic: {median_heuristic}"
        )
        t = self.test_cls(
            alpha=level,
            R=1,
            prefer_analytical_quantiles=True,
            median_heuristic=median_heuristic,
            **self.make_extra_test_kwargs(kernel),
        )

        # TODO: the test should be calibrated, and should probably error out if the
        # empirical type-I error is too far away from `level` if the null hypothesis is true
        # On the other hand, we cannot test for a power value.
        _test(t, self.data_generator, num_samples, num_tests, key, use_vmap=False)

    @pytest.mark.parametrize(
        "kernel",
        scalar_kernels[:1],
        ids=[type(k).__name__ for k in scalar_kernels][:1],
    )
    @pytest.mark.parametrize("type_", ["paired", "linear"])
    def test_unbiased_u_statistics(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        type_: Literal["paired", "linear"],
        num_samples: Tuple[int, ...] = (50, 100, 500),
        num_tests: int = 50,
    ):
        if not self.data_generator.h0:
            return

        level = 0.05
        print("\nThis test is currently only informative and cannot error out")
        print(
            f"type={type_:<15} num tests: {num_tests:<10} median heuristic: {median_heuristic}"
        )

        def _sample_and_test(
            ns: int, key: KeyArray, type_: Literal["paired", "linear"]
        ):
            key, subkey = random.split(key)
            data = self.data_generator.get_data(subkey, ns)

            if type_ == "paired":
                return self.test_cls(
                    median_heuristic=median_heuristic,
                    **self.make_extra_test_kwargs(kernel),
                ).unbiased_estimate(*data, key=key)
            elif type_ == "linear":
                return self.test_cls(
                    R=ns - 1,
                    median_heuristic=median_heuristic,
                    **self.make_extra_test_kwargs(kernel),
                ).unbiased_estimate(*data, key=key)
            else:
                raise NotImplementedError

        key = random.PRNGKey(0)

        mean_test_stats = []
        for ns in num_samples:
            key, subkey = random.split(key)
            subkeys = random.split(subkey, num=num_tests)
            t0 = time.time()
            test_stat = jit(
                vmap(_sample_and_test, in_axes=(None, 0, None)), static_argnums=(0, 2)
            )(ns, subkeys, type_)
            t_est = time.time() - t0
            mean_test_stat = jnp.mean(test_stat)
            mean_test_stats.append(mean_test_stat)
            print(
                f"num_samples={ns:<10} mean_test_stat={mean_test_stat.item():<15.4e}"
                f"time={t_est:.4f}s"
            )

    @pytest.mark.parametrize(
        "kernel", scalar_kernels[:1], ids=[type(k).__name__ for k in scalar_kernels][:1]
    )
    def test_div_empirical(
        self,
        kernel: base_kernel[Array],
        median_heuristic: bool,
        num_samples: Tuple[int, ...] = (10, 50, 100, 500),
        num_tests: int = 50,
    ):
        if not self.data_generator.h0:
            return

        t = self.test_cls(
            alpha=0.05,
            median_heuristic=median_heuristic,
            **self.make_extra_test_kwargs(kernel),
        )
        v_statistic = getattr(t, "v_statistic", None)
        if v_statistic is None:
            print("no v_statistic defined for this divergence, skipping test")
            return

        assert v_statistic is not None

        def _sample_compute_empirical_div(ns: int, key: KeyArray):
            data = self.data_generator.get_data(key, ns)
            return v_statistic(*data)

        key = random.PRNGKey(0)

        mean_empirical_divs = []
        for ns in num_samples:
            t0 = time.time()
            key, subkey = random.split(key)
            subkeys = random.split(subkey, num=num_tests)
            empirical_div = vmap(_sample_compute_empirical_div, in_axes=(None, 0))(
                ns, subkeys
            )
            mean_empircial_div = jnp.mean(empirical_div)
            mean_empirical_divs.append(mean_empircial_div)
            t_est = time.time() - t0
            print(f"num_samples={ns:<10} time={t_est:.4f}s")
        _, _, r_value, _, _ = scipy.stats.linregress(
            jnp.array(mean_empirical_divs), 1.0 / jnp.array(num_samples)
        )
        # r_value should be close to 1
        assert r_value > 0.99  # Should be enough to catch big regressions.
        print(f"r^2: {r_value**2:.4f}")
