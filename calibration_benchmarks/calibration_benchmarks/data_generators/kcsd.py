from typing import cast

import jax.numpy as jnp
from flax import struct
from jax import grad, random, vmap
from kwgflows.pytypes import Array
from numpyro import distributions as np_distributions
from typing_extensions import Unpack

from calibration.statistical_tests.kcsd import KCSDTestInput_T
from calibration_benchmarks.data_generators.base import data_delta_generator


class kcsd_test_data_generator(data_delta_generator[Unpack[KCSDTestInput_T[Array]]]):
    d: int = struct.field(pytree_node=False, default=2)

    @property
    def P_x(self) -> np_distributions.Distribution:
        return np_distributions.Normal(
            jnp.zeros(self.d), jnp.ones(self.d)  # type: ignore
        ).to_event(1)

    def P_y_factory(self, x: Array) -> np_distributions.Distribution:
        return np_distributions.Normal(x, jnp.ones_like(x)).to_event(1)  # type: ignore

    def Q_y_factory(self, x: Array) -> np_distributions.Distribution:
        return np_distributions.Normal(
            x + self.delta, jnp.ones_like(x)  # type: ignore
        ).to_event(1)

    def conditional_score_q(self, y: Array, x: Array) -> Array:
        return grad(self.Q_y_factory(x).log_prob)(y)

    def get_data(
        self, key: random.KeyArray, num_samples: int
    ) -> KCSDTestInput_T[Array]:
        key, subkey = random.split(key)
        X = cast(Array, self.P_x.sample(subkey, (num_samples,)))

        key, subkey = random.split(key)
        subkeys = random.split(subkey, num_samples)
        Y = vmap(lambda x, k: self.P_y_factory(x).sample(k), in_axes=(0, 0))(X, subkeys)

        return (X, Y, self.conditional_score_q)
