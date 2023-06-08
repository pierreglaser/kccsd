import jax.numpy as jnp
from flax import struct
from jax import grad, random
from numpyro import distributions as np_distributions
from typing_extensions import Unpack

from calibration.statistical_tests.ksd import KSDTestInput_T
from calibration_benchmarks.data_generators.base import data_delta_generator


class ksd_test_data_generator(data_delta_generator[Unpack[KSDTestInput_T]]):
    d: int = struct.field(pytree_node=False, default=2)

    @property
    def P(self) -> np_distributions.Distribution:
        return np_distributions.Normal(
            jnp.zeros(self.d), jnp.ones(self.d)  # type: ignore
        ).to_event(1)

    @property
    def Q(self) -> np_distributions.Distribution:
        return np_distributions.Normal(
            jnp.zeros(self.d) + self.delta, jnp.ones(self.d)  # type: ignore
        ).to_event(1)

    def get_data(self, key: random.KeyArray, num_samples: int) -> KSDTestInput_T:
        key, subkey = random.split(key)
        X = self.P.sample(subkey, (num_samples,))
        score_q = grad(self.Q.log_prob)
        return (X, score_q)
