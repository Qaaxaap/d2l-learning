"""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        # 没有调用父类的__init__, 加上 super().__init__()即可。
        self.weight = torch.randn(in_dim, out_dim) # 没有用 nn.Parameter 登记, 用它包裹即可。
        self.bias = torch.zeros(out_dim) # 同上

    def forward(self, X):
        return X @ self.weight + self.bias 


def batches(data=[]): # 仅在定义时初始化一次, 多次调用会串上次的data, 应该用 None + 判断。
    for i in range(len(data)):
        return data[i] # 应该是 yield, 因为 return 的话该函数仅仅输出 data 的第一个元素, 应该是一个能遍历元素的生成器才对, 换成 yield 即可。


net = Net(3, 1)
print(len(list(net.parameters())))
print(net(torch.randn(2, 3)))
"""

import time

import torch
import torch.nn as nn
class Counter:
    """用法：
        c = Counter()         # 从 0 开始
        c()                   # 返回 1，调用次数变 1
        c(5)                  # 返回 6，调用次数变 2
        len(c)                # 返回 2
        repr(c)               # 类似 "Counter(value=6, calls=2)"
        Counter(start=10)     # 从 10 开始
    """
    def __init__(self, start=0):
        self.value = start
        self.calls = 0
    def __call__(self, step=1):
        self.calls += 1
        self.value += step
        return self.value
    def __len__(self):
        return self.calls
    def __repr__(self):
        return f"Counter(value={self.value}, calls={self.calls})"
class Vec2:
    """二维向量，需要支持：
        Vec2(1, 2) + Vec2(3, 4)   -> Vec2(4, 6)
        Vec2(1, 2) == Vec2(1, 2)  -> True
        print(Vec2(1, 2))         -> 类似 "Vec2(1, 2)"
        v.x, v.y                  -> 属性可读
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, vb):
        vc = Vec2(self.x + vb.x, self.y + vb.y)
        return vc 
    def __eq__(self, vb):
        if (type(self) != type(vb)):
            return NotImplemented
        return bool((self.x == vb.x) and (self.y == vb.y))
    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"
def batch_indices(n: int, batch_size: int):
    """把 0 到 n-1 这 n 个索引按 batch_size 分批，逐批产出。"""
    end = 0
    while (end < n):
        ret = list(range(end, min(end + batch_size, n)))
        end += batch_size
        yield ret
class ScaledShift(nn.Module):
    """把输入逐元素乘一个可学习的系数，再逐元素加一个可学习的偏置。"""
    def __init__(self, size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size,))
        self.bias = nn.Parameter(torch.zeros(size,)) 
    def forward(self, X):
        return X * self.scale + self.bias
class Timer:
    """用法：
        t = Timer()
        with t:
            time.sleep(0.05)
        print(t.elapsed)     # 大约 0.05
    """
    def __init__(self):
        self.start = 0
        self.elapsed = 0
    def __enter__(self):
        self.start= time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        return False