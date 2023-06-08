import subprocess
import sys


def test_logging():
    # test the logging of the experiment runner class

    cmd_template = """if 1:
    import logging

    from jax import random
    from kwgflows.rkhs.kernels import gaussian_kernel
    from calibration.kernels import ExpMMDGaussianKernel
    from calibration.statistical_tests.skce import skce_test
    from calibration_benchmarks.data_generators.skce import skce_gaussian_data_generator
    from calibration_benchmarks.experiments_runners.base import ExperimentRunner

    if __name__ == "__main__":
        # define generators and tests
        h0_generator = skce_gaussian_data_generator(delta=0.0)

        l_test = skce_test(
            alpha=0.05,
            R=None,
            num_permutations=10,
            x_kernel=ExpMMDGaussianKernel(1.0, gaussian_kernel(1.0)),
            y_kernel=gaussian_kernel.create(sigma=1.0),
            approximation_num_particles=2,
        )

        runner = ExperimentRunner(parallel_backend="sequential", log_level={})

        key, subkey = random.split(random.PRNGKey(1234))
        ret_h0 = runner.run_manytimes(
            subkey,
            h0_generator,
            l_test,
            num_samples=50,
            num_tests=100,
        )
    """

    cmd_debug = cmd_template.format("logging.DEBUG")
    cmd_info = cmd_template.format("logging.INFO")

    # test that the debug logging works
    p = subprocess.Popen(
        [sys.executable, "-c", cmd_debug],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    p.wait()
    out, err = p.communicate()
    out, err = out.decode(), err.decode()
    assert p.returncode == 0
    assert out == ""
    for i in [1, 35, 53]:
        assert "running test {}".format(i) in err

    # test that the info logging works
    p = subprocess.Popen(
        [sys.executable, "-c", cmd_info], stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    p.wait()
    out, err = p.communicate()
    out, err = out.decode(), err.decode()
    for i in [1, 35, 53]:
        assert "running test {}".format(i) not in err
    assert "running tests one by one" in err
    assert p.returncode == 0
    assert out == ""
