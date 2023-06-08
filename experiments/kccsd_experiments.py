from jax import random
from kwgflows.rkhs.kernels import gaussian_kernel

from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.kernels import ExpMMDGaussianKernel
from calibration.statistical_tests.kcsd import kcsd_test
from calibration_benchmarks.data_generators.kccsd import kccsd_gaussian_data_generator
from calibration_benchmarks.experiments_runners.base import ExperimentRunner

if __name__ == "__main__":
    # define generator and tests
    h0_generator = kccsd_gaussian_data_generator(model_type=GaussianConditionalModel)
    h1_generator = kccsd_gaussian_data_generator(
        delta=1e-0, model_type=GaussianConditionalModel
    )

    l_test = kcsd_test(
        alpha=0.05,
        R=10,
        num_permutations=100,
        x_kernel=ExpMMDGaussianKernel(1.0, gaussian_kernel(1.0)),
        y_kernel=gaussian_kernel.create(sigma=1.0),
    )

    # the runner object will run repeateadly run h0 and h1 for a
    # given number of samples, distributing the computations using
    # the specified `parallel_backend`

    runner = ExperimentRunner(
        parallel_backend="sequential"
        # parallel_backend="vmap"
    )

    key, subkey = random.split(random.PRNGKey(1234))
    ret_h0 = runner.run_manytimes(
        subkey,
        h0_generator,
        l_test,
        num_samples=80,
        num_tests=500,
    )
    key, subkey = random.split(key)
    ret_h1 = runner.run_manytimes(
        subkey,
        h1_generator,
        l_test,
        num_samples=80,
        num_tests=500,
    )
    print(
        f"benchmarking h0 took time {ret_h0.time:<10.2f}, "
        f"h0 reject rate: {ret_h0.reject_mean:<10.4f} "
        f"h1 reject rate {ret_h1.reject_mean:<10.4f}"
    )
