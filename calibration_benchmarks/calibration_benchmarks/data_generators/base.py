from abc import ABCMeta, abstractmethod
from typing import Generic, Optional, Tuple, Type, TypeVar, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import random, vmap
from kwgflows.pytypes import Array
from typing_extensions import Unpack, Self

from calibration.conditional_models.base import (
    ConcreteLogDensityModel,
    DistributionModel,
    M,
)
from calibration.conditional_models.gaussian_models import GaussianConditionalModel
from calibration.statistical_tests.base import TestInput_T


class data_generator(
    Generic[Unpack[TestInput_T]], struct.PyTreeNode, metaclass=ABCMeta
):
    @property
    @abstractmethod
    def h0(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_data(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[Unpack[TestInput_T]]:
        raise NotImplementedError


class data_delta_generator(data_generator[Unpack[TestInput_T]]):
    delta: float = 0.0

    @property
    def h0(self) -> bool:
        return self.delta == 0.0


class calibration_test_data_generator_base(
    data_generator[Unpack[TestInput_T]], Generic[Unpack[TestInput_T], M]
):
    @abstractmethod
    def get_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[M, Array]:
        raise NotImplementedError


GM = TypeVar("GM", GaussianConditionalModel, DistributionModel, ConcreteLogDensityModel)


class gaussian_data_generator_base(
    calibration_test_data_generator_base[Unpack[TestInput_T], GM],
    data_delta_generator[Unpack[TestInput_T]],
):
    model_type: Type[GM] = struct.field(
        pytree_node=False, default=GaussianConditionalModel
    )

    @property
    @abstractmethod
    def d(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_gaussian_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[GaussianConditionalModel, Array]:
        raise NotImplementedError

    def get_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[GM, Array]:
        gaussian_models, Y = self.get_gaussian_models_and_targets(key, num_samples)
        if issubclass(self.model_type, GaussianConditionalModel):
            return (cast(GM, gaussian_models), Y)
        elif issubclass(self.model_type, DistributionModel):
            return (
                cast(
                    GM,
                    vmap(
                        lambda y: cast(
                            GaussianConditionalModel, y
                        ).as_distribution_model()
                    )(gaussian_models),
                ),
                Y,
            )
        elif issubclass(self.model_type, ConcreteLogDensityModel):
            return (
                cast(
                    GM,
                    vmap(
                        lambda y: cast(GaussianConditionalModel, y)
                        .as_distribution_model()
                        .as_logdensity_model()
                    )(gaussian_models),
                ),
                Y,
            )
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} not supported for gaussian data generator"
            )


class gaussian_data_generator(gaussian_data_generator_base[Unpack[TestInput_T], GM]):
    dims: int = struct.field(pytree_node=False, default=2)
    shift_dim: Optional[int] = None
    sigma: float = 1.0

    @property
    def d(self) -> int:
        return self.dims

    def get_observation_models(
        self, key: random.KeyArray, num_samples: int
    ) -> GaussianConditionalModel:
        key, subkey = random.split(key)
        conditional_models_means = random.normal(subkey, (num_samples, self.d))
        return vmap(
            lambda x: GaussianConditionalModel(
                mu=x,
                sigma=self.sigma,
            )
        )(conditional_models_means)

    def get_gaussian_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[GaussianConditionalModel, Array]:
        # Sample observation models and targets
        key, subkey = random.split(key)
        observation_models = self.get_observation_models(subkey, num_samples)
        key, subkey = random.split(key)
        subkeys = random.split(subkey, num_samples)
        Y = vmap(lambda x, k: x.sample_from_conditional(k, 1)[0], in_axes=(0, 0))(
            observation_models, subkeys
        )

        # Disturb the models by shifting with delta
        def shift(model: GaussianConditionalModel):
            if self.shift_dim is None:
                mu = model.mu + self.delta
            else:
                mu = model.mu.at[self.shift_dim].add(self.delta)
            return GaussianConditionalModel(mu=mu, sigma=model.sigma)

        conditional_models = vmap(shift)(observation_models)

        return (conditional_models, Y)


# Linear Gaussian Model (LGM) from Jitkrittum et al., 2020
# The original (calibrated) model is recovered with delta = 0
class linear_gaussian_data_generator(
    gaussian_data_generator_base[Unpack[TestInput_T], GM]
):
    r"""
    Class that can be used to generate randomly sampled datasets of iid pairs of
    Gaussian distributions :math:`p^i` and observations :math:`y^i`
    according to the following hierarchical model:

    ..math..:
    \begin{aligned}
      Z &\sim \mathcal{N}(0, I_d), \\
      Y \mid Z &\sim \mathcal{N}(\sum_{i=1}^d i Z_i, 1), \\
      P &= \mathcal{N}(\delta + \sum_{i=1}^d i Z_i, 1).
    \end{aligned}

    We have :math:`\operatorname{law}(Y | P) = P` if and only if `:math:`\delta = 0`:

    Let :math:`z, z' \in \mathbb{R}^d` and define
    :math:`p = \mathcal{N}(\delta + \sum_{i=1}^d i z_i, 1)` and
    :math:`p' = \mathcal{N}(\delta + \sum_{i=1}^d i z'_i, 1)`.
    We see that :math:`p = p'` if and only if :math:`\sum_{i=1}^d i z_i = \sum_{i=1}^d i z'_i`, or equivalently if
    :math:`\operatorname{law}(Y | Z = z) = \operatorname{law}(Y | Z = z')`.
    Thus we obtain
    
    ..math..:
      \operatorname{law}(Y | P = p) = \operatorname{law}(Y | Z = z) = \mathcal{N}(\sum_{i=1}^d i z_i, 1).

    Therefore :math:`\operatorname{law}(Y | P = p) = p` if :math:`\delta = 0`, and

    ..math..:
      \operatorname{law}(Y | P = p) \neq p

    if :math:`\delta \neq 0.
    """

    z_dim: int = struct.field(pytree_node=False, default=5)

    @property
    def d(self) -> int:
        return 1

    def get_gaussian_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[GaussianConditionalModel, Array]:
        key, subkey = random.split(key)
        zs = random.normal(subkey, (num_samples, self.z_dim))
        key, subkey = random.split(key)
        ms = (zs @ jnp.arange(1, self.z_dim + 1)).reshape(-1, 1)
        Y = ms + random.normal(subkey, (num_samples, 1))

        # Define the Gaussian models
        conditional_models = jax.vmap(
            lambda m: GaussianConditionalModel(mu=m + self.delta, sigma=1)
        )(ms)

        return (conditional_models, Y)


# Heteroscedastic Gaussian Model (HGM) from Jitkrittum et al., 2020
# The original (uncalibrated) model is recovered with delta = 1
# The model is miscalibrated locally close to c if delta > 0
class heteroscedastic_gaussian_data_generator(
    gaussian_data_generator_base[Unpack[TestInput_T], GM]
):
    r"""
    Class that can be used to generate randomly sampled datasets of iid pairs of
    Gaussian distributions :math:`p^i` and observations :math:`y^i`
    according to the following hierarchical model:

    ..math..:
    \begin{aligned}
      Z &\sim \mathcal{N}(0, I_d), \\
      Y \mid Z &\sim \mathcal{N}(\sum_{i=1}^d Z_i, 1), \\
      P &= \mathcal{N}(\sum_{i=1}^d Z_i, 1 + 10 \delta \exp{(- \|Z - c\|^2/(2 0.8^2))}).
    \end{aligned}

    We have :math:`\operatorname{law}(Y | P) = P` if and only if `:math:`\delta = 0`:
    Let :math:`z, z' \in \mathbb{R}^d` and define
    
    ..math..:
      p = \mathcal{N}(\sum_i z_i, 1 + 10 \delta \exp{(- \|z - c\|^2/(2 0.8^2))})

    and

    ..math..:
      p' = \mathcal{N}(\sum_i z'_i, 1 + 10 \delta \exp{(- \|z' - c\|^2/(2 0.8^2))}).

    We see that :math:`p = p'` implies :math:`\sum_i z_i = \sum_i z'_i`, and hence
    :math:`\operatorname{law}(Y | Z = z) = \operatorname{law}(Y | Z = z')`.
    Thus we obtain
    
    ..math..:
      \operatorname{law}(Y | P = p) = \operatorname{law}(Y | Z = z) = \mathcal{N}(\sum_i z_i, 1).

    Therefore :math:`\operatorname{law}(Y | P = p) = p` if :math:`\delta = 0`, and

    ..math..:
      \operatorname{law}(Y | P = p) \neq p

    if :math:`\delta \neq 0.
    """

    z_dim: int = struct.field(pytree_node=False, default=3)
    c: float = 2 / 3

    @property
    def d(self) -> int:
        return 1

    def get_gaussian_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[GaussianConditionalModel, Array]:
        # Sample targets
        key, subkey = random.split(key)
        zs = random.normal(subkey, (num_samples, self.z_dim))
        key, subkey = random.split(key)
        sum_zs = jnp.sum(zs, axis=-1, keepdims=True)
        Y = sum_zs + random.normal(subkey, (num_samples, 1))

        # Define the Gaussian models
        def conditional_sigma(z: Array) -> float:
            a = -jnp.sum(jnp.square((z - self.c) / 0.8)) / 2
            log_sigma2 = jax.nn.softplus(jnp.log(10 * self.delta) + a)
            return cast(float, jnp.exp(log_sigma2 / 2))

        conditional_models = vmap(
            lambda z, sum_z: GaussianConditionalModel(
                mu=sum_z,
                sigma=conditional_sigma(z),
            ),
            in_axes=(0, 0),
        )(zs, sum_zs)

        return (conditional_models, Y)


# Quadratic Gaussian Model (QGM) from Jitkrittum et al., 2020
# The original (miscalibrated) model is recovered with delta = 1
class quadratic_gaussian_data_generator(
    gaussian_data_generator_base[Unpack[TestInput_T], GM]
):
    r"""
    Class that can be used to generate randomly sampled datasets of iid pairs of
    Gaussian distributions :math:`p^i` and observations :math:`y^i`
    according to the following hierarchical model:

    ..math..:
    \begin{aligned}
      Z &\sim \operatorname{Uniform}(-2, 2), \\
      Y \mid Z &\sim \mathcal{N}(0.1 Z^2 + Z + 1, 1), \\
      P &= \mathcal{N}(0.1 (1 - \delta) Z^2 + Z + 1, 1).
    \end{aligned}

    For :math:`\delta = 0`, we have :math:`\operatorname{law}(Y | P) = P`.
    Moreover, for :math:`\delta = 1`, we have
    
    ..math..:
      \operatorname{law}(Y | P) \neq P \qquad \text{almost surely}.

    More generally, let :math:`p = \mathcal{N}(0.1 (1 - \delta) z^2 + z + 1, 1)`
    for some :math:`z \in [-2, 2]`.
    Let :math:`z' \in [-2, 2]` and define
    :math:`p' = \mathcal{N}(0.1 (1 - \delta) {z'}^2 + z' + 1, 1)`.
    We know that :math:`p = p'` if and only if

    ..math..:
      0.1 (\delta - 1) (z - z') (z + z') = z - z',
    
    i.e., if :math:`z' = z` or :math:`0.1 (\delta - 1) (z + z') = 1`.
    Thus for :math:`\delta = 1`, :math:`p = p'` if and only if :math:`z' = z`,
    and for :math:`\delta \neq 1`, :math:`p = p'` if and only if :math:`z' = z` or
    :math:`z' = 10 / (\delta - 1) - z`.
    
    Hence for :math:`\delta = 1`, we have

    ..math..:
      \operatorname{law}(Y | P = p) = \operatorname{law}(Y | Z = z) = \mathcal{N}(0.1 z^2 + z + 1, 1) \neq p.
    
    For :math:`\delta \neq 1`, let :math:`z' := 10 / (\delta - 1) - z`.
    We have

    ..math..:
    \begin{split}
      \mathbb{P}(Y \in A | P = p) &\propto \mathbb{P}(Y \in A | Z = z) \mathbb{P}(Z = z) + \mathbb{P}(Y \in A | Z = z') \mathbb{P}(Z = z') \\
      &\propto \begin{cases}
      \int_A \mathcal{N}(y; 0.1 z^2 + z + 1, 1) \,\mathrm{d}y & \text{if } z' \notin [-2, 2],\\
      \int_A \mathcal{N}(y; 0.1 z^2 + z + 1, 1) + \mathcal{N}(y; 0.1 z' + z' + 1, 1) \,\mathrm{d}y & \text{otherwise}.
      \end{cases}
    \end{split}

    This implies that :math:`\operatorname{law}(Y | P = p) = p` if :math:`\delta = 0`, and

    ..math..:
      \operatorname{law}(Y | P = p) \neq p

    if :math:`\delta \neq 0.
    """

    z_dim: int = struct.field(pytree_node=False, default=1)

    @property
    def d(self) -> int:
        return 1

    def get_gaussian_models_and_targets(
        self, key: random.KeyArray, num_samples: int
    ) -> Tuple[GaussianConditionalModel, Array]:
        # Sample targets
        key, subkey = random.split(key)
        zs = random.uniform(subkey, (num_samples, self.z_dim), minval=-2, maxval=2)
        key, subkey = random.split(key)
        sum_zs = jnp.sum(zs, axis=-1, keepdims=True)
        Y = 0.1 * sum_zs**2 + sum_zs + 1 + random.normal(subkey, (num_samples, 1))

        # Define the Gaussian models
        conditional_means = 0.1 * (1 - self.delta) * sum_zs**2 + sum_zs + 1
        conditional_models = vmap(lambda mu: GaussianConditionalModel(mu=mu, sigma=1))(
            conditional_means
        )

        return (conditional_models, Y)
