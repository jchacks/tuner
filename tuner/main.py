from dataclasses import dataclass
from typing import Generic, TypeVar, List, Generator, Optional
from pathlib import Path

import yaml
import itertools
import hashlib
import copy


T = TypeVar("T")


@dataclass
class Linear:
    start: float
    stop: float
    interval: float

    def __call__(self, x) -> float:
        return self.start + x * self.interval

    def __iter__(self):
        count = 0
        while (val := self(count)) <= self.stop:
            yield val
            count += 1

    def __len__(self):
        return (self.stop - self.start) // self.interval


@dataclass
class ExponentialDecay:
    start: float
    stop: float
    halflife: float
    round: Optional[int] = None

    def __call__(self, x) -> float:
        val = self.start * 0.5 ** (x / self.halflife)
        if self.round:
            val = round(val, self.round)
        return val

    def __iter__(self):
        count = 0
        while (val := self(count)) >= self.stop:
            yield val
            count += 1


@dataclass
class Choice(Generic[T]):
    values: List[T]

    def __iter__(self) -> Generator[T, None, None]:
        yield from self.values

    def __len__(self):
        return len(self.values)


static = {
    "iterations": 8_000,
    "test_end_date": "",
    "test_start_date": "2022-01-01",
    "train_end_date": "",
    "train_start_date": "",
    "num_stocks": 800,
    "warmup_steps": 0,
    "beta_2": 0.999,
    "clip_value": 1.0,
    "dropout_rate": 0.5,
    "reg_rates.activity_reg": 1e-5,
    "reg_rates.kernel_reg": 1e-4,
    "reg_rates.bias_reg": 0.0,
    "loss_scales.reg_loss_scale": 0.0,
    "loss_scales.cash_loss_scale": 0.0,
    "loss_scales.critic_loss_scale": 0.0,
    "loss_scales.dist_loss_scale": 1.0,
    "loss_scales.negative_loss_scale": 0.0,
    "loss_scales.weight_decay": 1e-2,
    "entropy_loss_scale": 0,
    "cash_return": 1.0,
}
values = {
    "batch_size": Choice[int]([16, 32, 64, 128, 256, 512]),
    "initial_learning_rate": ExponentialDecay(1e-3, 1e-4, 3),
    "learning_rate_decay": Choice[float]([0.99, 0.975]),
    # "dropout": Linear(0.0, 0.6, 0.1),
    # "loss_scales.weight_decay": ExponentialDecay(1e-1, 1e-5, 3),
}


identifier = hashlib.md5(str(values).encode()).hexdigest()

path = Path.cwd() / "configs" / identifier
path.mkdir(parents=True, exist_ok=True)
keys, values = zip(*[(key, tuple(val)) for key, val in values.items()])


def update(data: dict, key: str, val):
    parts = key.split(".", 1)
    if len(parts) > 1:
        key, rem = parts
        val = update(data.get(key, {}), rem, val)
    data.update({key: val})
    return data


configs = list(itertools.product(*values))
print("Generated", len(configs))
for i, entry in enumerate(itertools.product(*values)):
    data = {}
    static = copy.deepcopy(static)
    for k, v in static.items():
        update(data, k, v)
    for key, val in zip(keys, entry):
        update(data, key, val)

    file_path = path / f"config_{i}.yaml"
    config = yaml.dump(data, Dumper=yaml.Dumper)
    print(f"Running with config:\n{config}")
