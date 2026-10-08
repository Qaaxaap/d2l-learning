"""A1 参考实现。做完再看。

对应 exercises/a1/code.md 的 T1-T5。
"""

import time

import torch
import torch.nn as nn


class Counter:
    def __init__(self, start=0):
        self.value = start
        self.calls = 0

    def __call__(self, step=1):
        self.value += step
        self.calls += 1
        return self.value

    def __len__(self):
        return self.calls

    def __repr__(self):
        return f"Counter(value={self.value}, calls={self.calls})"


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if not isinstance(other, Vec2):
            return NotImplemented
        return Vec2(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        if not isinstance(other, Vec2):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"


def batch_indices(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield list(range(i, min(i + batch_size, n)))


class ScaledShift(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, X):
        return X * self.scale + self.bias


class Timer:
    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self._t0
        return False
