"""B26 参考答案。做完再看。

跑一遍：
    python3 solutions/b26/b26.py
或者拿它验断言：
    python3 tools/check_b26.py solutions/b26/b26.py
"""

import time

import torch
import torch.nn as nn


# ---------- T1 显存估算 ----------

def model_memory_bytes(num_params: int, precision: str = "fp32",
                       optimizer: str = "adam") -> dict:
    """估算训练时各部分的字节数。

    返回 dict，键为 params、grads、optimizer_states、total。
    精度 fp32 时每个数占 4 字节，fp16 与 bf16 占 2 字节。
    adam 的优化器状态是动量和二阶矩两份，各与参数同大小。
    sgd 无动量时没有优化器状态。
    """
    bytes_per = 4 if precision == "fp32" else 2
    params = num_params * bytes_per
    grads = num_params * bytes_per
    if optimizer == "adam":
        opt = num_params * bytes_per * 2
    elif optimizer == "sgd_momentum":
        opt = num_params * bytes_per
    else:
        opt = 0
    return {
        "params": params,
        "grads": grads,
        "optimizer_states": opt,
        "total": params + grads + opt,
    }


# ---------- T2 精确计时 ----------

def measure_time(fn, iters: int = 5, warmup: int = 3) -> float:
    """测 fn 在 GPU 上的平均执行时间，单位毫秒。

    fn 不接受参数、不返回值。测之前先预热，测的时候要同步。
    """
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


# ---------- T3 混合精度训练步 ----------

def train_step_amp(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   optimizer: torch.optim.Optimizer, scaler,
                   criterion: nn.Module, device_type: str = "cpu") -> float:
    """用 AMP 做一步训练，返回标量损失。

    autocast 用 bfloat16。用 scaler.scale 包住 loss 再反传。
    """
    optimizer.zero_grad()
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


class TinyNet(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- T4 并行方案选择 ----------

def choose_parallel(num_params: int, num_gpus: int,
                    model_fits_single: bool) -> str:
    """返回并行方案名。

    单卡返回 "none"，多卡且模型放得进单卡返回 "ddp"，
    模型放不进单卡返回 "model_parallel"。
    """
    if num_gpus <= 1:
        return "none"
    if not model_fits_single:
        return "model_parallel"
    return "ddp"


"""T5 改错

第 1 处：用 `time.time()` 直接包住 GPU 算子
    测到的是提交任务的时间，不是计算时间，因为 GPU 是异步的。
    改法：前后各加一次 `torch.cuda.synchronize()`，或用 `torch.cuda.Event`。

第 2 处：混合精度忘了 `scaler`
    直接 `loss.backward()` 时 fp16 的梯度会下溢成 0，参数不更新。
    改法：`scaler.scale(loss).backward()`，再 `scaler.step(optimizer)`、
    `scaler.update()`。

第 3 处：训练循环里每一步都 `print(loss)`
    `print` 会把张量拿到 CPU，强制同步，GPU 队列被清空。
    同时它会触发图中断，编译的收益被吃掉。
    改法：累积到列表里，每若干步打印一次，打印前用 `.detach().item()`。
"""


if __name__ == "__main__":
    m = model_memory_bytes(100_000_000, "fp32", "adam")
    print("T1 1 亿参数 fp32 + Adam 总字节:", m["total"], "=", m["total"] / 2**30, "GiB")

    x = torch.randn(2000, 2000)
    print("T2 CPU 上 2000x2000 矩阵乘:", round(measure_time(lambda: x @ x), 3), "ms")

    torch.manual_seed(0)
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu")
    xb = torch.randn(16, 8)
    yb = torch.randint(0, 3, (16,))
    loss = train_step_amp(model, xb, yb, opt, scaler, nn.CrossEntropyLoss(), "cpu")
    print("T3 AMP 一步 loss:", round(loss, 4))

    print("T4 选择:", choose_parallel(1_000_000, 4, True),
          choose_parallel(10_000_000_000, 4, False),
          choose_parallel(1_000_000, 1, True))
