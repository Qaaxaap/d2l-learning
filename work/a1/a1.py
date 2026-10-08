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