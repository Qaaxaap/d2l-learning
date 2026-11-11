# B26 代码题

把你的实现写在 `work/b26/b26.py` 里，函数名与签名必须和下面一致。

验收：`just check b26`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

本章不实现算法，做四件事：估算显存、正确计时、写混合精度的训练步、选并行方案。

**验收标准里三类都有**：精确（字节数）、范围（计时值）、趋势（损失下降）。
题面写明了每类对应哪条。

---

## T1 显存估算

```python
def model_memory_bytes(num_params: int, precision: str = "fp32",
                       optimizer: str = "adam") -> dict:
    """估算训练时各部分的字节数。

    参数：
        num_params: 参数个数。
        precision: "fp32" 时每个数 4 字节，"fp16" 与 "bf16" 时 2 字节。
        optimizer: "adam" 有动量与二阶矩两份状态；
                   "sgd_momentum" 只有动量一份；
                   "sgd" 没有优化器状态。

    返回 dict，含四个键：params、grads、optimizer_states、total。
    """
```

| 检查 | 期望 | 类型 |
|---|---|---|
| 1 万参数 fp32 + Adam | params 40000、grads 40000、optimizer_states 80000、total 160000 | 精确 |
| 1 万参数 bf16 + Adam | params 与 optimizer_states 都减半 | 精确 |
| 1 万参数 fp32 + sgd | optimizer_states 为 0、total 为 80000 | 精确 |

## T2 精确计时

```python
def measure_time(fn, iters: int = 5, warmup: int = 3) -> float:
    """测 fn 的平均执行时间，单位毫秒。

    fn 不接受参数、不返回值。
    先跑 warmup 次预热，再测 iters 次取平均。
    """
```

**两个要求**：

1. 测的是 GPU 执行时间，不是提交任务的时间。
2. 每次测之前要预热。第一次执行会包含初始化开销，不能算进去。

| 检查 | 期望 | 类型 |
|---|---|---|
| 2000×2000 fp32 矩阵乘 | 结果在 0.05 到 5000 毫秒之间 | 范围 |
| 同一操作测两次 | 两次的比值不超过 10 | 范围 |

**测到微秒级就是错的**。那说明没同步，测到的只是提交任务的时间。

## T3 混合精度的训练步

```python
def train_step_amp(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   optimizer: torch.optim.Optimizer, scaler,
                   criterion: nn.Module, device_type: str = "cpu") -> float:
    """用 AMP 做一步训练，返回标量损失。

    用 bfloat16。步骤见讲义 5.3 节。
    """
```

**要做的顺序**（讲义 5.3）：

1. 清梯度。
2. 在 `autocast` 上下文里前向、算损失。
3. `scaler.scale(loss).backward()`。
4. `scaler.step(optimizer)`。
5. `scaler.update()`。

返回的损失是 `loss.item()`。

| 检查 | 期望 | 类型 |
|---|---|---|
| 返回类型与数值 | `float`，不是 nan | 精确 |
| 固定数据上训练 30 步 | 末步损失低于首步的 0.8 倍 | 趋势 |
| 一步之后 | 参数被更新过 | 精确 |

**设备**：`x`、`y`、`model` 都在同一个设备上，由调用方保证。
`device_type` 是字符串，`"cuda"` 或 `"cpu"`，要传给 `autocast` 与 `GradScaler`。

## T4 并行方案选择

```python
def choose_parallel(num_params: int, num_gpus: int,
                    model_fits_single: bool) -> str:
    """返回并行方案名。

    单卡返回 "none"。
    多卡且模型放得进单卡返回 "ddp"。
    模型放不进单卡返回 "model_parallel"。
    """
```

| 检查 | 期望 | 类型 |
|---|---|---|
| `(1e6, 1, True)` | `"none"` | 精确 |
| `(1e6, 4, True)` | `"ddp"` | 精确 |
| `(1e10, 4, False)` | `"model_parallel"` | 精确 |

## T5 改错

下面这段是「训练一个 epoch 并计时」的写法。**三处**问题。

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
    return time.time() - t0
```

三处分别指：**计时方式**、**混合精度的配套**、**循环里的打印**。

三段提示：第一处测到的是提交任务的时间；第二处会让 fp16 的梯度下溢成 0；
第三处每步都强制同步 GPU，还会触发图中断。
