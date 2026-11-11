# B26 计算性能

前面 25 章关心「模型对不对」。本章关心「跑得快不快、吃得下多少显存」。

同样一个模型，写法不同可以差几倍速度。本章讲四件事：
代码为什么慢、`torch.compile` 怎么帮上忙、混合精度怎么省显存、多张卡怎么用。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 一段代码为什么慢 | 无，本节补充 | 无，本节补充 |
| 2 命令式与图式执行 | 8.1 命令式和符号式混合编程 | [12.1](https://zh.d2l.ai/chapter_computational-performance/hybridize.html) |
| 3 `torch.compile` | 8.1 的 `hybridize` | 12.1 的同一处 |
| 4 异步执行与 CUDA 流 | 8.2 异步计算 | [12.2](https://zh.d2l.ai/chapter_computational-performance/async-computation.html) |
| 5 混合精度 | 无，第一版没有 | 无，第二版也没有 |
| 6 多 GPU | 8.3 自动并行、8.4 多 GPU 计算 | [12.3](https://zh.d2l.ai/chapter_computational-performance/auto-parallelism.html)、[12.4](https://zh.d2l.ai/chapter_computational-performance/multiple-gpus.html) |
| 7 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**这一章与书差异最大。**

第一，书上讲的是 MXNet 的机制：符号式编程、前后端异步、自动并行。
这些概念在 torch 里对应别的东西。

第二，书上的 `hybridize` 在 torch 里对应 **`torch.compile`**。
两者的思路相同（把 Python 循环换成编译好的图），用法与限制不同。

第三，**混合精度书上完全没有**。这是实践里最省显存、最快的一招，本章补上。

**解锁点**：这一章没有要手写的核心算法。任务是测量与对比，代码题以「跑通并解释数字」为主。

## 1 一段代码为什么慢

### 1.1 三种开销

一个训练步的时间不是全花在矩阵乘上。它由三部分组成。

| 开销 | 来源 | 典型占比 |
|---|---|---|
| **Python 开销** | 每次算子调用都要走 Python 解释器 | 小模型上可占 50% 往上 |
| **GPU 计算** | 矩阵乘、卷积 | 大模型上占主要 |
| **显存搬运** | 数据在 CPU 与 GPU 之间、HBM 与缓存之间搬 | 大小取决于实现 |

**Python 开销为什么能占那么大**。一个算子调用要做这些事：

1. Python 解释器解析 `torch.matmul(a, b)`。
2. 检查参数类型与形状。
3. 构造一个 C++ 的 Tensor 对象。
4. 把任务提交到 GPU 队列。

每一步都是微秒级。一次矩阵乘在 GPU 上可能只要 5 微秒。
层数多、每层小的时候，Python 的开销反而超过计算。

### 1.2 本章要解决的三个问题

- **Python 开销**：用 `torch.compile` 把 Python 循环变成编译好的图（第 2、3 节）。
- **等待 GPU**：理解 CUDA 的异步执行，避免写出「每步都在等」的代码（第 4 节）。
- **显存放不下**：用混合精度与多卡（第 5、6 节）。

## 2 命令式与图式执行

### 2.1 命令式是什么

**命令式**（eager，也叫动态图）：代码写到哪就执行到哪。

```python
a = torch.ones(3)
b = torch.ones(3)
c = a + b        # 这一行执行完，c 已经有值
d = c * 2        # 这一行执行完，d 已经有值
```

torch 默认就是命令式。好处是调试方便：随便 `print`，随便加 `if`。

代价是每一行都要走 Python 解释器。`if`、`for` 这些控制流在 GPU 上没法并行，
只能一行一行过。

### 2.2 图式是什么

**图式**（graph，也叫静态图）：先定义一整个计算流程，再一起执行。

```
定义阶段：
  a = placeholders()
  b = placeholders()
  c = a + b
  d = c * 2

执行阶段：
  喂入 a、b 的值 → 得到 d
```

框架可以在这张图上做优化：

- **算子融合**：把 `c = a + b` 与 `d = c * 2` 合成一个核，少一次显存读写。
- **内存复用**：中间的 `c` 用完立刻回收。
- **减少 Python 调用**：整张图一次提交。

### 2.3 两者的取舍

| | 命令式 | 图式 |
|---|---|---|
| 调试 | 方便 | 需要专门的工具 |
| 控制流 | 随便写 | 要特殊处理 |
| 速度 | 慢（Python 开销） | 快 |
| 首次运行 | 立即 | 要编译 |

**实际项目里的做法**：默认用命令式写，训练时把模型包一层编译。

### 2.4 书上的 hybridize

书上用 `hybridize()` 在两个模式之间切换。第一次前向会记录计算图，
之后的前向直接跑编译好的图。

**torch 的对应是 `torch.compile`**。用法见下一节。

两者思路相同，差别在实现：`hybridize` 要求代码里不能有普通的 Python 控制流，
`torch.compile` 能处理一部分控制流。

## 3 torch.compile

### 3.1 怎么用

一行包住模型：

```python
model = MyModel()
model = torch.compile(model)
```

之后正常训练。第一次前向会花时间编译（几秒到几分钟），第二次之后加速。

也可以只编译某个函数：

```python
@torch.compile
def train_step(model, x, y):
    ...
```

### 3.2 有效果的三种情况

**情况一：小算子多。** 模型由几百个小算子组成，Python 开销占大头。
编译后算子被融合，提升明显。

**情况二：固定的输入形状。** 编译按具体形状优化。形状固定时效果好。

**情况三：重复执行。** 编译的开销是一次性的。训练几千步时它摊得很薄。

### 3.3 没效果的三种写法

**情况一：模型已经在 GPU 上跑满。** 计算是瓶颈时，减少 Python 开销没用。

**情况二：输入形状每次都变。** 形状变一次要重新编译一次，反而更慢。

**情况三：代码里有大量控制流。** `for` 循环依赖张量值的、`if` 判断张量值的，
会触发**图中断**（graph break），性能收益被切成碎片。

### 3.4 图中断

**图中断**：编译时遇到无法跟踪的操作，把图切成两段，前一段编译、后一段走 Python。

常见的触发点：

- 打印张量值：`print(x)`
- 把张量转成 Python 数：`x.item()`、`float(x)`
- 依赖张量值的 `if`：`if x.sum() > 0:`
- 不支持的第三方库调用

**怎么发现**：设环境变量 `TORCH_LOGS="graph_breaks"`，编译时会打印每次中断的原因。

**实践含义**：训练循环里不要写 `print(loss)` 或 `loss.item()`。那会把编译的收益吃掉。
需要记录时，用 `loss.detach()` 累积，每若干步打印一次。

### 3.5 编译的代价

- **首次编译慢**。小模型几秒，大模型可能几分钟。
- **编译缓存占显存**。
- **出错时的错误信息难读**。它会指向生成的代码而不是你的源码。

**建议**：先确认命令式版本正确，再加编译。不要一上手就编译。

## 4 异步执行与 CUDA 流

### 4.1 GPU 上的执行是异步的

Python 调用一个 GPU 算子时，**不会等它算完**。它只是把任务放进 GPU 的队列，
立刻返回。

```python
a = torch.randn(1000, 1000, device="cuda")
b = a @ a          # 提交任务，立刻返回
print("完成了")     # 这行可能在矩阵乘之前就执行
```

**CPU 与 GPU 是两条流水线**。CPU 提交任务，GPU 按队列执行。
这样 CPU 不必空等，可以继续提交下一个任务。

### 4.2 什么时候会同步

大部分情况下 CPU 不需要等。但下面这些操作会强制等待：

| 操作 | 为什么同步 |
|---|---|
| `x.item()` | 要把值拿到 CPU，必须等算完 |
| `print(x)` | 同 `item()` |
| `float(x)`、`int(x)` | 同 `item()` |
| `x.cpu()` | 数据要拷回，要等 |
| `torch.cuda.synchronize()` | 显式等待全部完成 |

**这也是 `print(loss)` 拖慢训练的第二个原因**（第一个是图中断）。
每个 batch 同步一次，GPU 的队列就被清空一次。

### 4.3 怎么正确测时间

**错误做法**：用 `time.time()` 包住一个 GPU 算子。

```python
t0 = time.time()
c = a @ b
print(time.time() - t0)   # 测到的是提交任务的时间，不是计算时间
```

测到的是提交耗时，通常是几微秒。

**正确做法**：先同步，再计时。

```python
torch.cuda.synchronize()
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
print(time.time() - t0)
```

如果要测很多次，用 `torch.cuda.Event`：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# 要测的代码
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))    # 毫秒
```

`Event` 记录在 GPU 的时间线上，不受 CPU 提交延迟影响。

### 4.4 数据加载的非阻塞传输

从 CPU 拷数据到 GPU 默认是同步的。要异步，两步：

1. 用 `DataLoader(..., pin_memory=True)`。它把数据放进**锁页内存**（pinned memory），
   GPU 能直接访问这种内存。
2. 拷到 GPU 时加 `non_blocking=True`。

```python
x = x.to("cuda", non_blocking=True)
```

**这样数据搬运与计算能重叠**。GPU 在算上一个 batch 时，CPU 已经在准备下一个。

**锁页内存的代价**：占用物理内存，不能换页。所以只对需要的数据用。

## 5 混合精度

### 5.1 两种低精度格式

默认训练用 fp32（32 位浮点）。混合精度用 fp16 或 bf16，各占 2 字节。

| 格式 | 位数 | 指数位 | 尾数位 | 最大范围 | 最小规格数 |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ |
| fp16 | 16 | 5 | 10 | 65504 | $6\times10^{-8}$ |
| bf16 | 16 | 8 | 7 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ |

两者的取舍：

- **fp16** 精度高（10 位尾数），范围小。超过 65504 就变成 `inf`。
- **bf16** 范围与 fp32 相同，精度低（7 位尾数）。

**推荐 bf16**。范围不会溢出，训练更稳。Ampere 之后的 GPU 都支持。

### 5.2 省了什么

显存：

- 参数、梯度、激活值都减半。
- 一个 1 亿参数的模型，fp32 要 400 MB 存参数，混合精度下占 200 MB。

速度：

- 同一次矩阵乘的吞吐翻倍（Tensor Core 对 fp16 的吞吐远高于 fp32）。
- 显存带宽减半，搬运更快。

### 5.3 AMP 的用法

torch 的混合精度叫 **AMP**（automatic mixed precision）。它不把所有计算都换成 fp16，
而是有选择地换。

```python
scaler = torch.amp.GradScaler("cuda")

for x, y in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**`autocast` 做了什么**：进入这个上下文后，部分算子自动用低精度。

| 用低精度 | 保持 fp32 |
|---|---|
| 矩阵乘、卷积 | 损失函数、softmax |
| 逐元素运算 | 归一化（LayerNorm、BatchNorm）|
| 大部分前向 | 参数更新 |

**为什么要有保留**：数值范围敏感的算子在小精度下会溢出或下溢。

### 5.4 GradScaler 干什么

**问题**：fp16 的梯度下溢。

梯度常常很小，比如 $10^{-8}$。fp16 能表示的最小规格数是 $6\times10^{-8}$，
比它小的值直接变成 0。梯度变成 0，参数就不更新了。

**做法**：反向传播前把损失乘一个放大系数，得到放大后的梯度，再缩回去。

1. `scaler.scale(loss)`：损失乘上 scale（初始 65536）。
2. `backward()`：得到放大 65536 倍的梯度。
3. `scaler.step(optimizer)`：更新前把梯度除回原大小。
4. `scaler.update()`：根据这次有没有出现 `inf`，动态调整 scale。

**每步自动调整**：出现 `inf` 时缩小 scale，连续成功时增大 scale。

**用 bf16 时还要 GradScaler 吗**：不需要。bf16 的范围与 fp32 相同，不会下溢。
用 bf16 时把 scaler 换成空操作即可。

### 5.5 什么时候不该用

- **数值稳定性要求高的任务**。例如小批次上的归一化、极小概率的 softmax。
- **老 GPU**。Pascal（GTX 10 系列）之后的卡才支持 fp16 加速很明显的实现。
- **调试阶段**。混合精度会掩盖数值问题，先用 fp32 确认模型对。

## 6 多 GPU

### 6.1 两种并行

| | 数据并行 | 模型并行 |
|---|---|---|
| 切分什么 | 切**数据**，每个 GPU 一份完整模型 | 切**模型**，每张卡放一部分 |
| 适合 | 模型放得进单卡 | 模型放不进单卡 |
| 通信量 | 每步同步梯度 | 层之间的中间结果 |

**实际用得多的是数据并行**。模型并行要手工切分，麻烦。

### 6.2 DataParallel

最省事的写法，一行包住：

```python
model = nn.DataParallel(model)
```

**它怎么工作**：

1. 把这一批数据按 GPU 数量切分。
2. 每个 GPU 复制一份模型，各算自己那份。
3. 把各 GPU 的梯度汇总求平均。
4. 每张卡更新自己的参数。

**问题**：每步都要在 GPU 之间汇总梯度，开销大。而且第一个 GPU 要负责汇总，
负载不均。

**性能通常不如 DDP**。单机多卡时也不推荐。

### 6.3 DistributedDataParallel

**DDP** 是推荐做法。它用多进程，每个 GPU 一个进程。

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

**与 DP 的差别**：每个进程只跑一个 GPU，没有 GIL 瓶颈，
梯度用 ring-allreduce 算法同步（带宽利用更好）。

### 6.4 各项该选哪个

| 场景 | 选择 |
|---|---|
| 单卡 | 不并行 |
| 单机多卡 | DDP |
| 多机多卡 | DDP |
| 模型放不进单卡 | 模型并行 + DDP |

**注意 DDP 的细节**：

- 每个 epoch 要 `sampler.set_epoch(epoch)`，否则每轮的数据顺序相同。
- 只在 rank 0 上保存 checkpoint 与打印。
- 学习率要按 GPU 数量放大（线性缩放规则）。

### 6.5 显存怎么算

训练时显存由四部分组成：

| 部分 | 大小（每参数） |
|---|---|
| 参数 | 4 字节（fp32）|
| 梯度 | 4 字节 |
| 优化器状态（Adam）| 8 字节（动量 + 二阶矩）|
| 激活值 | 取决于层数与批量 |

**总账**：Adam 训练时，每个参数约 16 字节（fp32）+ 激活值。
一个 1 亿参数的模型光参数与优化器状态就要 1.6 GB。

**省显存的顺序**：

1. 减小批量。
2. 混合精度（参数与激活减半）。
3. 梯度累积（小批量多次，等效大批量）。
4. 梯度检查点（用计算换显存）。

## 7 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 用 `time.time()` 测 GPU 算子 | 测出来只有几微秒 | 没有同步（4.3）|
| 训练循环里 `print(loss)` | 速度慢一半 | 每步同步，还触发图中断（4.2、3.4）|
| `torch.compile` 包住整个文件 | 错误信息看不懂 | 先确认命令式版本正确（3.5）|
| 输入形状每步变 | 编译后更慢 | 每换个形状重编译（3.3）|
| 混合精度忘了 `GradScaler` | 梯度变成 0，不收敛 | fp16 下溢（5.4）|
| 用 bf16 还配 `GradScaler` | 不影响结果，多一层开销 | bf16 不需要缩放（5.4）|
| 用 `DataParallel` 上多机 | 不支持 | DP 只支持单机（6.2）|
| DDP 忘了 `set_epoch` | 每轮数据顺序相同 | sampler 需要重新打乱（6.4）|
| DDP 每个 rank 都打印 | 输出重复 N 遍 | 只在 rank 0 上输出（6.4）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.compile(model)` | 编译模型 | 3.1 |
| `torch.cuda.synchronize()` | 等待 GPU 全部完成 | 4.2 |
| `torch.cuda.Event(enable_timing=True)` | 精确计时 | 4.3 |
| `torch.amp.GradScaler("cuda")` | 梯度缩放 | 5.3 |
| `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` | 混合精度上下文 | 5.3 |
| `scaler.scale(loss).backward()` | 缩放后反传 | 5.3 |
| `scaler.step(optimizer)` | 缩回后更新 | 5.3 |
| `scaler.update()` | 调整缩放系数 | 5.4 |
| `nn.DataParallel(model)` | 单机多卡，简单写法 | 6.2 |
| `DistributedDataParallel(model, device_ids=[rank])` | 推荐的多卡方案 | 6.3 |
| `dist.init_process_group("nccl")` | 初始化 DDP | 6.3 |
| `sampler.set_epoch(epoch)` | 让每轮打乱不同 | 6.4 |
| `x.to("cuda", non_blocking=True)` | 非阻塞传输 | 4.4 |
| `DataLoader(..., pin_memory=True)` | 锁页内存 | 4.4 |

## 自测题

1. 写出训练一个 step 的三种开销，并说明 Python 开销为什么能占大头。
2. 对比命令式与图式执行在调试、控制流、速度三项上的差别。
3. 说明 `torch.compile` 有效果的三种情况与没效果的三种情况。
4. 写出图中断的定义与三个常见触发点，并说明怎么发现它。
5. 说明 GPU 算子为什么是异步的，并列出四个会强制同步的操作。
6. 说明用 `time.time()` 测 GPU 算子为什么测不准，并写出正确做法。
7. 对比 fp16 与 bf16 在位数、范围、精度上的差别，并说明推荐哪个。
8. 说明 fp16 训练为什么需要 `GradScaler`。
9. 写出数据并行与模型并行的差别，并说明各自适合的场景。
10. 对比 `DataParallel` 与 `DistributedDataParallel` 的性能与原理。
11. 写出训练显存的四个组成部分，以及每个参数字节数的数量级。

## 答案（做完再看）

**1.** Python 开销、GPU 计算、显存搬运。

Python 开销大的原因：每次算子调用都要经过解释器解析、类型检查、构造 C++ 对象、
提交任务。这些小模型上每步只有几微秒，而层数一多，累积开销超过计算本身。

**2.**

| | 命令式 | 图式 |
|---|---|---|
| 调试 | 方便 | 需要专门工具 |
| 控制流 | 随便写 | 要特殊处理 |
| 速度 | 慢 | 快 |

**3.** 有效果：小算子多、输入形状固定、重复执行。

没效果：模型已经跑满 GPU、输入形状每步变、代码里控制流多。

**4.** 图中断是编译时遇到无法跟踪的操作，把图切成两段。

常见触发点：`print(x)`、`x.item()`、依赖张量值的 `if`。

发现方法：设 `TORCH_LOGS="graph_breaks"`，编译时会打印每次中断的原因。

**5.** CPU 只负责把任务放进 GPU 队列，不等结果就返回，这样两条流水线能并行。

会同步的四个操作：`x.item()`、`print(x)`、`x.cpu()`、`torch.cuda.synchronize()`。

**6.** `time.time()` 只包住了「提交任务」那几微秒，GPU 还没开始算。

正确做法：前后各加一次 `torch.cuda.synchronize()`。
测多次时用 `torch.cuda.Event`。

**7.**

| 格式 | 尾数位 | 最大范围 | 最小规格数 |
|---|---|---|---|
| fp16 | 10 | 65504 | $6\times10^{-8}$ |
| bf16 | 7 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ |

推荐 bf16，范围与 fp32 相同，不会溢出，训练更稳。

**8.** fp16 的最小规格数是 $6\times10^{-8}$，比它小的梯度变成 0。
`GradScaler` 先乘一个系数把梯度放大，更新前再缩回去。

**9.** 数据并行切数据，每个 GPU 一份完整模型，适合模型放得进单卡。
模型并行切模型，每张卡放一部分，适合模型放不进单卡。

**10.** DP 用单进程多线程，有 GIL 瓶颈，第一个 GPU 负责汇总，负载不均。
DDP 用多进程，每 GPU 一个进程，梯度用 ring-allreduce 同步，带宽利用更好。
单机多卡也应该用 DDP。

**11.** 参数（4 字节）、梯度（4 字节）、优化器状态（Adam 是 8 字节）、激活值。
Adam 训练时每参数约 16 字节，再加上激活值。
