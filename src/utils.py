import numpy as np
from scipy.stats import truncnorm  # type: ignore


def truncated_z_sample(batch_size: int, z_dim: int, truncation: float = 0.5, seed: int | None = None):
    state = np.random.RandomState(seed) if seed is not None else None
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values
