"""B26 验收断言。用法: just check b26

读取 work/b26/b26.py 并验证 exercises/b26/code.md 里的 T1-T4。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b26" / "b26.py"

_failures: list[str] = []
_passed: list[str] = []


def case(name: str):
    def deco(fn):
        try:
            fn()
        except AssertionError as exc:
            _failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            _failures.append(f"{name}: 意外异常 {type(exc).__name__}: {exc}")
        else:
            _passed.append(name)
        return fn

    return deco


class TinyNet(nn.Module):
    """测试用的小模型，不依赖学习者的实现。"""

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_user_module():
    if not USER_FILE.exists():
        print(f"找不到 {USER_FILE}")
        print("先照着 exercises/b26/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b26_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @case("T1 fp32 + Adam 的字节数")
    def _t1a():
        m = mod.model_memory_bytes(10_000, "fp32", "adam")
        assert m["params"] == 40_000, f"参数应为 40000 字节，得到 {m['params']}"
        assert m["grads"] == 40_000, f"梯度应为 40000 字节，得到 {m['grads']}"
        assert m["optimizer_states"] == 80_000, (
            f"Adam 状态应为 80000 字节（两份），得到 {m['optimizer_states']}"
        )
        assert m["total"] == 160_000, f"总计应为 160000，得到 {m['total']}"

    @case("T1 bf16 减半")
    def _t1b():
        a = mod.model_memory_bytes(10_000, "fp32", "adam")
        b = mod.model_memory_bytes(10_000, "bf16", "adam")
        assert b["params"] == a["params"] // 2, (
            f"bf16 参数应减半，得到 {b['params']}"
        )
        assert b["optimizer_states"] == a["optimizer_states"] // 2, (
            f"bf16 优化器状态应减半，得到 {b['optimizer_states']}"
        )

    @case("T1 sgd 无动量没有优化器状态")
    def _t1c():
        m = mod.model_memory_bytes(10_000, "fp32", "sgd")
        assert m["optimizer_states"] == 0, (
            f"无动量 SGD 不该有优化器状态，得到 {m['optimizer_states']}"
        )
        assert m["total"] == 80_000, f"总计应为 80000，得到 {m['total']}"

    @case("T2 计时不准会暴露")
    def _t2a():
        size = 2000
        x = torch.randn(size, size, device=device)
        t = mod.measure_time(lambda: x @ x, iters=3, warmup=1)
        # 2000x2000 fp32 矩阵乘约 0.5 到几十毫秒。没同步会测到微秒级。
        assert 0.05 < t < 5000, (
            f"测得 {t:.6f} ms。2000x2000 矩阵乘约 0.5 到几十毫秒，"
            "测到微秒级说明没同步，测到的只是提交任务的时间"
        )

    @case("T2 计时结果稳定")
    def _t2b():
        size = 1500
        x = torch.randn(size, size, device=device)
        t1 = mod.measure_time(lambda: x @ x, iters=3, warmup=1)
        t2 = mod.measure_time(lambda: x @ x, iters=3, warmup=1)
        assert t1 > 0 and t2 > 0, "两次计时都应为正数"
        ratio = max(t1, t2) / min(t1, t2)
        assert ratio < 10, (
            f"两次测量差 {ratio:.1f} 倍（{t1:.4f} vs {t2:.4f}），"
            "计时不稳定。检查是否每次都重新创建了随机张量"
        )

    @case("T3 AMP 训练步能跑")
    def _t3a():
        torch.manual_seed(0)
        model = TinyNet().to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(device)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(16, 8, device=device)
        y = torch.randint(0, 3, (16,), device=device)
        loss = mod.train_step_amp(model, x, y, opt, scaler, crit, device)
        assert isinstance(loss, float), f"应返回 float，得到 {type(loss)}"
        assert loss == loss, "loss 是 nan"

    @case("T3 AMP 训练能降损失")
    def _t3b():
        torch.manual_seed(0)
        model = TinyNet(out_dim=2).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        scaler = torch.amp.GradScaler(device)
        crit = nn.CrossEntropyLoss()
        # 固定数据上过拟合，损失应当明显下降
        torch.manual_seed(1)
        x = torch.randn(32, 8, device=device)
        y = (x[:, 0] > 0).long()
        losses = []
        for _ in range(30):
            losses.append(mod.train_step_amp(model, x, y, opt, scaler, crit, device))
        assert losses[-1] < losses[0] * 0.8, (
            f"30 步后损失没降下来：{losses[0]:.4f} → {losses[-1]:.4f}。"
            "检查 scaler 的三个调用是否齐全"
        )

    @case("T3 参数被更新")
    def _t3c():
        torch.manual_seed(0)
        model = TinyNet().to(device)
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(device)
        x = torch.randn(16, 8, device=device)
        y = torch.randint(0, 3, (16,), device=device)
        mod.train_step_amp(model, x, y, opt, scaler, nn.CrossEntropyLoss(), device)
        changed = any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
        )
        assert changed, "参数没有被更新，检查 optimizer.step 是否执行"

    @case("T4 并行方案选择")
    def _t4():
        assert mod.choose_parallel(1_000_000, 1, True) == "none", "单卡应为 none"
        assert mod.choose_parallel(1_000_000, 4, True) == "ddp", "多卡且放得下应为 ddp"
        assert mod.choose_parallel(10_000_000_000, 4, False) == "model_parallel", (
            "放不下单卡时应为 model_parallel"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
