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
        return None
    def __call__()