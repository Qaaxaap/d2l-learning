# A1 Python 最小子集

你写过 Rust 和 C++，Python 基本没碰过。这份讲义按"从 C++/Rust 过来会撞到什么"来组织，
不按语法手册的顺序。

目标是两件事：看懂 d2l 的代码；能自己把 B06 那个线性回归写出来。
不追求会写正经的 Python 工程。

每一节都配能直接跑的片段。看完一章就敲一遍跑一遍，别只读。

---

## 0. 先校准六件事

你带着 C++/Rust 的直觉读 Python 代码，会在这六处判断错。先把它们拧过来。

### 0.1 变量没有类型，值才有

```python
x = 3
x = "现在 x 是字符串"
x = [1, 2, 3]
```

C++ 里 `int x = 3; x = "abc";` 编译不过。Rust 里 `let x = 3; x = "abc";` 也编译不过。
Python 里合法，因为 `x` 只是一个名字，绑到哪个对象上都行。

代价是**没有编译期类型检查**。你写错的地方全在运行时炸，而且炸的位置可能离出错的位置很远。
所以 Python 调试靠跑，不靠看。

### 0.2 缩进是语法

```python
if x > 0:
    print("正数")
    print("还是正数")
print("无论如何都执行")
```

没有大括号，冒号加缩进就是块。缩进混用空格和 Tab 会直接报 `IndentationError`。
统一用四个空格。

### 0.3 赋值不复制

```python
a = [1, 2, 3]
b = a          # 不是复制
b.append(4)
print(a)       # [1, 2, 3, 4]
```

`b = a` 让两个名字指向同一个对象。C++ 里对应 `auto& b = a;`，Rust 里对应 `let b = &a;`。
要复制得写 `a.copy()`。

Python 没有所有权和借用检查，全是引用，靠引用计数回收。所以对象在哪都能改，
函数里改传进来的列表，调用方看得见。

### 0.4 没有 `const`、没有 `mut`

任何对象的属性随时可改。想表达"这是常量"只能靠全大写命名（`MAX_SIZE = 100`），语言不拦你。

### 0.5 一切都是对象，函数也是

```python
def square(x):
    return x * x

f = square          # 函数可以赋给变量
print(f(3))         # 9
print(type(square)) # <class 'function'>
```

把函数当参数传、当返回值返回，在 Python 里是常规操作。`nn.Module` 的 hook 机制就靠这个。

### 0.6 `None` 相当于空指针，但不是 0

```python
y = None
if y is None:
    print("还没赋值")
```

判断用 `is None`，不用 `== None`。

---

## 1. 语法骨架

### 1.1 基本类型

```python
n = 42              # int，任意精度，不会溢出
f = 3.14            # float，就是 C 的 double
s = "文本"           # str，不可变
b = True            # bool
nothing = None      # 空
```

没有 `char`、没有 `unsigned`、没有固定宽度整数。需要 `float32` 这类精度是张量的事（B02 讲），
不是 Python 数字的事。

`3 / 2` 得 `1.5`；要整除写 `3 // 2` 得 `1`。

### 1.2 四个容器

```python
lst = [1, 2, 3]                  # list，可变，类似 std::vector
tup = (1, 2, 3)                  # tuple，不可变
d = {"lr": 0.03, "epochs": 3}    # dict，哈希表，类似 std::unordered_map
st = {1, 2, 3}                   # set
```

取值：

```python
lst[0]        # 1，下标从 0 开始
lst[-1]       # 3，负数是倒数
lst[0:2]      # [1, 2]，切片，左闭右开
d["lr"]       # 0.03
d.get("x", 0) # 取不到给默认值，不抛异常
```

注意 `lst[0:2]` 对 list 返回**新列表**。张量的切片不是这样，B02 会专门讲。

### 1.3 控制流

```python
if x > 0:
    ...
elif x == 0:
    ...
else:
    ...
```

```python
for i in range(3):              # 0 1 2
    print(i)

for i, v in enumerate(["a", "b"]):   # 要下标时
    print(i, v)

for a, b in zip([1, 2], [3, 4]):     # 两个序列并排
    print(a, b)

while n < 3:
    n += 1                      # 没有 n++，写 n += 1
```

Python 的 `for` 就是 Rust 的 `for x in iter`，它遍历的是**迭代器**，不是 C 风格的三段式。
这一点在第 5 节会展开，因为你可以给自己的类实现迭代。

### 1.4 推导式

生成列表、字典、集合的紧凑写法：

```python
squares = [i * i for i in range(5)]              # [0, 1, 4, 9, 16]
evens   = [i for i in range(10) if i % 2 == 0]   # [0, 2, 4, 6, 8]
lengths = {w: len(w) for w in ["ab", "cde"]}     # {'ab': 2, 'cde': 3}
```

Rust 里对应 `(0..5).map(|i| i * i).collect::<Vec<_>>()`。读代码时见到它别慌，
就是"循环加收集"。

### 1.5 字符串

```python
name = "fashion"
epoch = 3
print(f"第 {epoch} 轮，数据集 {name}")   # f-string，最常用
print(f"loss = {loss:.3f}")              # 保留三位小数
print(f"{x=}")                           # 调试用，打印 x=3
```

多行字符串用三个引号，docstring 也用它：

```python
def f(x):
    """这个函数的说明。运行时它只是被忽略的字符串。"""
    return x
```

---

## 2. 函数

### 2.1 基本形态

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

没有返回类型声明，没有参数类型声明。现代代码会写类型注解，但**注解不参与运行**：

```python
def train(net, X: torch.Tensor, lr: float = 0.03) -> float:
    ...
```

`X: torch.Tensor` 传错了不会有任何报错。读代码时把注解当注释看。

### 2.2 默认值与可变参数

```python
def train(net, lr=0.03, num_epochs=3, **kwargs):
    print(lr, num_epochs, kwargs)

train(None)                              # 0.03 3 {}
train(None, 0.1, device="cuda")          # 0.1 3 {'device': 'cuda'}
```

`**kwargs` 把多余的关键字参数收成一个 dict，d2l 里大量用它把参数往下传。
对应的还有 `*args`，收多余的位置参数成 tuple。

### 2.3 返回多个值

```python
def two():
    return 1, 2

a, b = two()      # 解包
```

实际上返回的是一个 tuple，`a, b = ...` 是解包。d2l 里
`train_iter, test_iter = load_data_fashion_mnist(32)` 就是这个。

### 2.4 坑：可变默认参数

```python
def bad(x, acc=[]):     # 错
    acc.append(x)
    return acc

print(bad(1))   # [1]
print(bad(2))   # [1, 2]  ← 不是 [2]
```

`[]` 在**函数定义时**创建一次，所有调用共享。C++ 里默认参数是每次调用求值的，这里不是。
正确写法：

```python
def good(x, acc=None):
    if acc is None:
        acc = []
    acc.append(x)
    return acc
```

---

## 3. 类

### 3.1 最简形态

```python
class Model:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        return self.w * x + self.b


m = Model(2.0, 1.0)
print(m.forward(3.0))
```

对照着看：

| Python | C++ | Rust |
|---|---|---|
| `class Model:` | `class Model {` | `struct Model {` + `impl Model {` |
| `def __init__(self, ...)` | `Model(...)` 构造函数 | `fn new(...) -> Self` |
| `self` | `this`（隐式） | `self`（显式） |
| `self.w = w` | `this->w = w;` | `self.w = w;` |

`self` 必须写在参数列表第一个，调用时不用传。这一点和 Rust 的 `self` 最像。

### 3.2 继承与 `super()`

```python
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, X):
        return X * self.weight + self.bias
```

`class MyLayer(nn.Module)` 表示继承 `nn.Module`，对应 C++ 的 `: public nn::Module`。
`super().__init__()` 调用父类构造函数，对应 C++ 的基类初始化列表。

**`super().__init__()` 漏掉会出怪事**：`nn.Module` 的构造函数负责建内部登记表，
不调它，后面 `self.weight = nn.Parameter(...)` 无处登记，`parameters()` 返回空，
优化器拿到空列表，训练时参数一动不动，而且**不报错**。这条在 B06 会踩。

### 3.3 `nn.Parameter` 的作用

```python
self.weight = torch.randn(3)                  # 只是普通属性，优化器看不见
self.weight = nn.Parameter(torch.randn(3))    # 登记成参数，会被收集、搬运、保存
```

为什么赋个值就能"登记"？因为 `nn.Module` 重写了 `__setattr__`（见第 4 节），
在属性赋值时拦截并分类。这是 Python 特有的做法，C++/Rust 里没有对应物。

### 3.4 调用模型用 `net(X)`，不用 `net.forward(X)`

```python
y = net(X)          # 正确
y = net.forward(X)  # 能跑，但绕过了框架的 hook
```

`net(X)` 会走 `nn.Module.__call__`，里面依次触发 forward pre-hook、`forward`、forward hook。
`torch.compile`、梯度检查点、调试工具都挂在这层 hook 上。

---

## 4. dunder 到底是什么

这一节是这份讲义的重点。你说不知道 `__xx__` 是什么，那正好从这里建立直觉。

### 4.1 你在别的语言里已经见过它

C++：

```cpp
struct Vec { double x, y; };
Vec operator+(const Vec& a, const Vec& b) { return {a.x + b.x, a.y + b.y}; }
```

Rust：

```rust
impl Add for Vec {
    type Output = Vec;
    fn add(self, o: Vec) -> Vec { Vec { x: self.x + o.x, y: self.y + o.y } }
}
```

Python：

```python
class Vec:
    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)
```

三个是同一件事：**让自定义类型支持语言内置的运算符**。

`__add__` 就是 `operator+`，就是 `Add::add`。这类双下划线包起来的名字叫 dunder
（double underscore）。它们是一组**钩子**：你不直接调用它们，语言在特定场合替你调。

你写 `v + w`，解释器去找 `Vec.__add__`；你写 `len(v)`，解释器去找 `Vec.__len__`。

### 4.2 协议表

看懂 d2l 代码需要认识这些。第三列是你熟悉语言里的对应物。

| dunder | 什么时候被自动调用 | C++ / Rust 对应 |
|---|---|---|
| `__init__` | `Foo(...)` 构造对象 | 构造函数 / `new()` |
| `__call__` | `foo(...)` 把对象当函数调 | `operator()` / `Fn::call` |
| `__len__` | `len(foo)` | `size()` |
| `__getitem__` | `foo[i]`、`foo[a:b]` | `operator[]` / `Index` |
| `__setitem__` | `foo[i] = v` | `operator[]` 赋值 |
| `__iter__` | `for x in foo` | `begin()/end()` / `IntoIterator` |
| `__next__` | 迭代器取下一个 | `Iterator::next` |
| `__add__` `__mul__` | `a + b`、`a * b` | `operator+` / `Add` |
| `__eq__` | `a == b` | `operator==` / `PartialEq` |
| `__repr__` | `print(foo)`、调试显示 | `operator<<` / `Debug` |
| `__enter__` `__exit__` | `with foo:` 进入和退出 | RAII 析构 |
| `__getattr__` | 访问不存在的属性时 | 没有对应物 |
| `__get__` `__set__` | 属性被读/写时（描述符协议），`@property` 就是用它实现的 | 没有对应物 |
| `__setattr__` | 给属性赋值时 | 没有对应物 |

前十一行你都能用 C++/Rust 的直觉理解。最后两行是 Python 特有的，也是 `nn.Module` 的魔法来源。

### 4.3 三个你会反复遇到的

**`__init__` 与 `__call__` 的区别**：

```python
class Multiplier:
    def __init__(self, k):        # 创建对象时调一次
        self.k = k

    def __call__(self, x):        # 每次把对象当函数用时调
        return x * self.k

double = Multiplier(2)   # __init__
print(double(5))         # __call__，输出 10
```

这就是"装饰器"和"可调用对象"的实现基础（第 7 节）。

**`__getitem__` 与 `__len__`**：

```python
class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

ds = Dataset([10, 20, 30])
print(len(ds))       # 3
print(ds[1])         # 20
for v in ds:         # 有 __getitem__ 和 __len__ 就能被 for 遍历
    print(v)
```

`torch.utils.data.Dataset` 就是靠这两个方法工作的，B07 会用到。

**`__setattr__` 与 `__getattr__`**：`nn.Module` 重写了 `__setattr__`，
所以 `self.weight = nn.Parameter(...)` 这一句不只是赋值，还会被拦截、分类、登记。
这也解释了一个现象：给模块赋一个 `nn.Module` 类型的属性，它会自动出现在 `children()` 里。

### 4.4 `__eq__` 遇到别的类型该返回什么

```python
Vec2(1, 2) == "abc"
```

直接返回 `False` 看起来能用，但更正确的是返回一个特殊的常量 `NotImplemented`：

```python
def __eq__(self, other):
    if not isinstance(other, Vec2):
        return NotImplemented
    return (self.x, self.y) == (other.x, other.y)
```

`NotImplemented` **不是** `False`。它的意思是"我不认识对面这个东西，你（解释器）去问问对面"。
解释器收到它之后会去试 `other.__eq__(self)`（反射比较），两边都说不认识，
才落到默认行为——同一个对象才相等，于是最终得到 `False`。

区别在于：直接返回 `False` 会堵死对面类的机会。以后如果你写了另一个类，
它声明"我能和 `Vec2` 比较"，`Vec2` 这边一口回绝，对面就永远没机会参与。

（`NotImplemented` 和 `NotImplementedError` 是两个东西，后者是个异常，用途不同。）

### 4.5 你不需要会写 dunder

除了 `__init__` 一定要会写，其余的在 d2l 学习期间**只需要认识**。
你会在 d2l 的代码里看到 `def forward(self, X)`，那是普通方法不是 dunder；
但你会看到 `nn.Module` 帮你处理了 `__call__`，所以调用时写成 `net(X)`。

判断方法：看到 `__xxx__` 就知道它不是给你直接调的，去查它对应哪个语法结构。

---

## 5. 迭代器与生成器

### 5.1 迭代的协议

`for x in foo` 背后做两件事：调 `foo.__iter__()` 拿到迭代器，然后反复调它的 `__next__()`，
直到抛 `StopIteration`。

Rust 的 `Iterator::next` 返回 `Option`，Python 用抛异常表示结束，是同一件事的两种表达。

### 5.2 生成器：写迭代器不用写类

```python
def count_up(n):
    i = 0
    while i < n:
        yield i          # 交出一个值，然后暂停在这里
        i += 1

for v in count_up(3):
    print(v)             # 0 1 2
```

`yield` 出现，这个函数就变成生成器函数。调用它**一行都不执行**，只返回一个生成器对象；
第一次迭代才跑到第一个 `yield`，然后停住，下次迭代从停的地方继续。

C++ 里要写一个 `begin()/end()` 加迭代器类，Rust 里要 impl 一个 Iterator，
Python 里一个 `yield` 就够了。这是 Python 最省事的地方。

### 5.3 后面会用到

B06 会要你写一个"每次吐一批数据"的函数，那时用 `yield` 实现。这里先记一个坑：

把 `yield` 写成 `return`，函数只返回第一批就结束，`for` 循环跑一次就停，而且不报错。
这类错误最难查，因为程序看起来是在正常跑的。

判断该用哪个：要"一次给一个、可以边算边给"就用 `yield`；要"算完一次性给"就用 `return`。

## 6. `with`

```python
with torch.no_grad():
    y = net(X)
```

`with` 管理的对象要实现 `__enter__` 和 `__exit__`。进入块时调前者，离开块时调后者
（抛异常也调）。C++ 里对应 RAII：构造时获取资源，析构时释放。

所以 `torch.no_grad()` 进入时关掉梯度记录，退出时自动恢复，哪怕块里炸了也恢复。

自己写一个：

```python
class Timer:
    def __enter__(self):
        import time
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"耗时 {time.time() - self.t0:.3f} 秒")
        return False      # 返回 False 表示不吞掉异常

with Timer():
    sum(range(10**6))
```

d2l 里用 `with` 的地方主要是 `torch.no_grad()`、打开文件、计时。

---

## 7. 装饰器

`@` 开头的一行是装饰器。它做的事很简单：**把下面那个函数传给一个函数，用返回值替换它**。

```python
def logged(fn):
    def wrapper(*args, **kwargs):
        print(f"调用 {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper

@logged
def add(a, b):
    return a + b

add(1, 2)      # 打印"调用 add"，然后返回 3
```

`@logged` 等价于 `def add(...)` 之后写 `add = logged(add)`。就这么多，没有别的魔法。

### 你会遇到的几个

| 装饰器 | 作用 |
|---|---|
| `@property` | 让方法能当属性读，写 `foo.shape` 而不是 `foo.shape()` |
| `@staticmethod` | 不需要 `self` 的方法 |
| `@torch.no_grad()` | 整个函数在关闭梯度记录的情况下运行 |
| `@d2l.add_to_class(...)` | d2l 专用的，给已有的类动态挂方法 |

```python
class Foo:
    @property
    def shape(self):
        return (2, 3)

f = Foo()
print(f.shape)      # (2, 3)，不写括号
```

你不需要会写装饰器，认识这几个就行。

---

## 8. 会咬人的坑

**缩进混用**：空格和 Tab 混着用会报 `IndentationError`，且报错信息难懂。统一四空格。

**`is` 与 `==`**：`==` 比值，`is` 比是不是同一个对象。判断 `None` 用 `is`。

**可变默认参数**：见 2.4。

**浅拷贝**：

```python
a = [[1, 2], [3, 4]]
b = a.copy()        # 只复制外层
b[0].append(99)
print(a)            # [[1, 2, 99], [3, 4]]  内层还是共享的
```

需要彻底独立得用 `copy.deepcopy(a)`。张量同理：`x2 = x` 不复制数据，`x.clone()` 才复制。

**整数除法**：`/` 是浮点除，`//` 是整除。

**作用域**：函数里赋值就是局部变量，想改外层的要 `global`（少用）。

**类型注解不检查**：见 2.1。

**`print` 调试张量**：张量 `print` 出来是截断的，元素多了显示省略号。要完整看得用
`.tolist()` 或者设 `torch.set_printoptions(threshold=float('inf'))`。

---

## 自测

关掉讲义答。答不上来的回去看对应小节。

1. `a = [1, 2]; b = a; b.append(3)` 之后 `a` 是什么？为什么？在 Rust 里这行代码会怎样？
2. `def f(x, acc=[])` 和 `def f(x, acc=None)` 有什么区别？为什么前者是 bug？
3. `self` 是什么？Rust 里的 `self` 和它像在哪，C++ 里对应什么？
4. 什么是 dunder？用 C++ 或 Rust 的术语解释 `__add__` 是什么。
5. `__init__` 和 `__call__` 分别在什么时候被调用？
6. `nn.Module` 子类里漏掉 `super().__init__()` 会怎样？为什么不报错？
7. `self.w = torch.randn(3)` 和 `self.w = nn.Parameter(torch.randn(3))` 有什么区别？
8. `net(X)` 和 `net.forward(X)` 有什么区别？
9. `yield` 是什么？`data_iter(...)` 被调用后立刻发生了什么？
10. `@property` 装饰器做了什么？它和第 4 节的哪个 dunder 有关？
11. `with torch.no_grad():` 块里抛异常了，梯度记录状态会怎样？为什么？

<details>
<summary>做完再看：答案</summary>

1. `[1, 2, 3]`。`b = a` 让两个名字指向同一个列表，没有复制。Rust 里 `let b = a;` 会发生移动，
   `a` 之后不可用；`let b = &a;` 才是共享引用，但借用检查器会拦住"同时存在可变借用"的情况。
   Python 两样都没有，全靠自觉。
2. 默认值 `[]` 只在函数定义时创建一次，所有调用共享，会越积越多。用 `None` 当默认值，
   函数内部再新建。
3. `self` 是当前对象的引用，和 Rust 的 `self` 最像（显式写在参数列表里）；
   C++ 里对应隐式的 `this`。调用时不用传。
4. dunder 是双下划线包起来的名字，由解释器在特定语法场合自动调用。`__add__` 就是 C++ 的
   `operator+`，Rust 的 `impl Add`。
5. `__init__` 在 `Foo(...)` 创建对象时调一次；`__call__` 在把对象当函数调用 `foo(...)` 时每次调。
6. `parameters()` 返回空，优化器拿到空列表，训练时参数不更新，且不报错。
   `nn.Module.__init__` 负责建内部登记表，漏掉就没地方登记。
7. 前者是普通属性，不会被 `parameters()` 收集，优化器看不见；后者会被登记，
   能被优化器更新、被 `.to("cuda")` 搬走、被 `state_dict()` 保存。
8. `net(X)` 走 `nn.Module.__call__`，会触发 forward pre-hook、`forward`、forward hook。
   直接调 `forward` 绕过 hook，`torch.compile`、梯度检查点之类的功能会失效。
9. `yield` 把函数变成生成器。调用 `data_iter(...)` 时函数体一行都不执行，只返回生成器对象；
   第一次 `for` 迭代才执行到第一个 `yield`。
10. 让方法能当属性读。它和 `__get__` 有关（描述符协议），`@property` 本身就是用描述符实现的。
11. 恢复。`with` 的 `__exit__` 在异常时同样执行。

</details>
