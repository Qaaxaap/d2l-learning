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
        return None
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
    
