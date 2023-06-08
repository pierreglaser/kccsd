# Calibration tests in Python

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Instructions

### Development environment

We provide an environment with all dependencies:

- Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html) (used for integration tests).

- In a terminal, navigate to this folder and run
  ```shell
  conda env create -f environment.yml
  ```
  This assumes that you have installed conda such that the `conda` command is available in your terminal.
  The command will install all required dependencies in a new environment.
  You can specify the location of the environment as well, e.g., you can create it in the (hidden) subfolder `.conda_env` with the command
  ```shell
  conda env create -p ./.conda_env/ -f environment.yml
  ```

### Support for NVidia GPUs

For machines with an NVidia GPU, we provide an environment with GPU support in the file `environment_gpu.yml`.
To install it, follow the instructions above using `environment_gpu.yml` instead of `environment.yml`.

### Activating and deactivating the environment

You can activate the conda environment in a terminal by running
```shell
conda activate calibration
```
or, if you created it in the folder `.conda_env`, by
```shell
conda activate ./.conda_env/
```
It can be deactivated again by running
```shell
conda deactivate
```

### Tests

If the conda environment is activated, you can run the tests with the command
```shell
python -m pytest -n auto ./tests
```

### Formatting

If the conda environment is activated, you can format the code with the command
```shell
python -m black --verbose ./calibration ./tests
```
