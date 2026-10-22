from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    source_time: float
    receive_time: float
    state: np.ndarray

    @property
    def age(self) -> float:
        return self.receive_time - self.source_time


class DelayBuffer:
    def __init__(self, initial: np.ndarray, horizon: float = 5.0):
        self.samples = deque([(0.0, np.asarray(initial, dtype=float).copy())])
        self.horizon = horizon
        self.last_stamp = -np.inf

    def push(self, time: float, value: np.ndarray) -> None:
        if time < self.samples[-1][0]:
            raise ValueError("Source timestamps must be monotonic")
        self.samples.append((time, np.asarray(value, dtype=float).copy()))
        while len(self.samples) > 2 and self.samples[1][0] < time - self.horizon:
            self.samples.popleft()

    def observe(self, time: float, delay: float, monotonic_delivery: bool = True) -> Observation:
        if delay < 0:
            raise ValueError("Delay cannot be negative")
        target = time - delay
        if monotonic_delivery:
            target = max(target, self.last_stamp)
        sample = self.samples[0]
        for candidate in reversed(self.samples):
            if candidate[0] <= target + 1e-10:
                sample = candidate
                break
        self.last_stamp = sample[0]
        return Observation(sample[0], time, sample[1].copy())
