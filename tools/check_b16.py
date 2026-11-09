"""B16 验收断言。用法: just check b16

读取 work/b16/b16.py 并验证 exercises/b16/code.md 里的 T1-T3。
T4 是纸面题，这里只提醒人工检查。

T3 用 Fashion-MNIST 的一个小子集训练 2 个 epoch（约 1 秒），
断言只验趋势与下限，不追求高精度。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b16" / "b16.py"
DATA_DIR = REPO / "data"

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


def load_user_module():
    if not USER_FILE.exists():
        print(f"找不到 {USER_FILE}")
        print("先照着 exercises/b16/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b16_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @case("T1 LeNet 的前向形状")
    def _t1a():
        net = mod.make_lenet()
        x = torch.randn(4, 1, 28, 28)
        out = net(x)
        assert tuple(out.shape) == (4, 10), (
            f"输入 (4,1,28,28) 应输出 (4,10)，得到 {tuple(out.shape)}。"
            "按形状链算：28→24→12→8→4→16*4*4=256"
        )
        net3 = mod.make_lenet(in_channels=3, num_classes=5)
        out3 = net3(torch.randn(2, 3, 28, 28))
        assert tuple(out3.shape) == (2, 5), (
            f"in_channels=3, num_classes=5 时应输出 (2,5)，得到 {tuple(out3.shape)}"
        )

    @case("T1 LeNet 的参数量")
    def _t1b():
        net = mod.make_lenet()
        n = sum(p.numel() for p in net.parameters())
        assert n == 44426, (
            f"参数量应为 44426，得到 {n}。核对公式：\n"
            "  卷积1 6*1*5*5+6 = 156\n  卷积2 16*6*5*5+16 = 2416\n"
            "  线性1 120*256+120 = 30840\n  线性2 84*120+84 = 10164\n"
            "  线性3 10*84+10 = 850"
        )

    @case("T1 结构与激活")
    def _t1c():
        net = mod.make_lenet()
        types = [type(m) for m in net.modules()]
        assert nn.Flatten in types, "缺少展平层，四维张量进不了 nn.Linear"
        assert nn.Sigmoid not in types, (
            "不该用 sigmoid——这一章要求 ReLU。sigmoid 会让梯度消失（讲义 4.5 节）"
        )
        n_relu = sum(1 for m in net.modules() if isinstance(m, nn.ReLU))
        assert n_relu == 4, (
            f"应当有 4 个 nn.ReLU（两个卷积后、两个全连接后），得到 {n_relu} 个"
        )
        # 漏了 ReLU 的卷积输出会有负值，池化取最大值会把它带过来
        pools = [m for m in net.modules() if isinstance(m, nn.MaxPool2d)]
        assert len(pools) == 2, f"应当有两个最大池化层，得到 {len(pools)} 个"
        outs = []
        handles = [m.register_forward_hook(
            lambda mod_, i, o: outs.append(o.detach())) for m in pools]
        try:
            with torch.no_grad():
                net(torch.randn(2, 1, 28, 28))
        finally:
            for h in handles:
                h.remove()
        for i, out in enumerate(outs):
            assert (out >= 0).all(), (
                f"第 {i + 1} 个池化的输出出现负值（最小 {out.min().item():.4f}）。"
                "卷积后要接 ReLU，池化取最大值会保留符号"
            )
        keys = list(net.state_dict().keys())
        # 带参数的层有 5 个：2 个卷积 + 3 个线性，各贡献 weight 与 bias
        assert len(keys) == 10, (
            f"state_dict 应有 10 个键（5 个带参数的层各 2 个），得到 {len(keys)}：{keys}"
        )

    @case("T2 评估：全猜同一类")
    def _t2a():
        class Const(nn.Module):
            def forward(self, x):
                out = torch.zeros(x.size(0), 10, device=x.device)
                out[:, 0] = 1.0
                return out

        y = torch.tensor([0, 0, 0, 1, 2, 3])
        X = torch.randn(6, 1, 28, 28)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)
        acc = mod.evaluate(Const(), loader, device)
        assert abs(acc - 0.5) < 1e-6, (
            f"全猜 0 类时准确率应为 3/6 = 0.5，得到 {acc}"
        )

    @case("T2 评估：完美模型")
    def _t2b():
        class Perfect(nn.Module):
            def __init__(self, ys):
                super().__init__()
                self.ys = ys

            def forward(self, x):
                out = torch.zeros(x.size(0), 10, device=x.device)
                idx = self.ys[: x.size(0)].long().to(x.device)
                out.scatter_(1, idx.view(-1, 1), 1.0)
                self.ys = self.ys[x.size(0):]
                return out

        y = torch.arange(10)
        X = torch.randn(10, 1, 28, 28)
        loader = DataLoader(TensorDataset(X, y), batch_size=5)
        net = Perfect(torch.arange(10))
        acc = mod.evaluate(net, loader, device)
        assert abs(acc - 1.0) < 1e-6, f"完美模型准确率应为 1.0，得到 {acc}"

    @case("T2 评估：空 loader 与 eval 模式")
    def _t2c():
        y = torch.tensor([0, 1])
        X = torch.randn(2, 1, 28, 28)
        empty = DataLoader(TensorDataset(X[:0], y[:0]), batch_size=2)
        net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        acc = mod.evaluate(net, empty, device)
        assert acc == 0.0, f"空 loader 应返回 0.0，得到 {acc}"
        full = DataLoader(TensorDataset(X, y), batch_size=2)
        mod.evaluate(net, full, device)
        assert not net.training, "evaluate 之后模型应当处于 eval 模式"

    @case("T3 训练：结构与趋势")
    def _t3():
        from torchvision import datasets, transforms

        tf = transforms.ToTensor()
        train_set = datasets.FashionMNIST(DATA_DIR, True, transform=tf, download=True)
        test_set = datasets.FashionMNIST(DATA_DIR, False, transform=tf, download=True)
        # 取小子集，控制断言耗时（约 1 秒）
        sub_train = torch.utils.data.Subset(train_set, range(8000))
        sub_test = torch.utils.data.Subset(test_set, range(2000))
        train_iter = DataLoader(sub_train, batch_size=128, shuffle=True)
        test_iter = DataLoader(sub_test, batch_size=256)

        torch.manual_seed(0)
        net = mod.make_lenet()
        before = [p.detach().clone() for p in net.parameters()]
        hist = mod.train_lenet(net, train_iter, test_iter, 3, 0.1, device)

        for key in ("train_loss", "train_acc", "test_acc"):
            assert key in hist, f"返回字典缺 {key} 键"
            assert len(hist[key]) == 3, (
                f"{key} 的长度应为 num_epochs=3，得到 {len(hist[key])}"
            )
        assert hist["train_loss"][-1] < hist["train_loss"][0] * 0.9, (
            f"训练损失应当明显下降，得到 {hist['train_loss']}"
        )
        for key in ("train_acc", "test_acc"):
            for v in hist[key]:
                assert 0.0 <= v <= 1.0, f"{key} 应当在 [0,1]，得到 {v}"
        assert hist["test_acc"][-1] > 0.5, (
            f"8000 张、3 epoch 后测试精度应超过 0.5，得到 {hist['test_acc'][-1]:.4f}"
        )
        changed = any(
            not torch.equal(b, p.detach().cpu())
            for b, p in zip(before, net.parameters())
        )
        assert changed, "训练后参数应当变化"

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT4 是纸面题，人工检查。")
        return 1

    print("T1-T3 全部通过。")
    print("T4 是纸面题，等我看你标的三处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
