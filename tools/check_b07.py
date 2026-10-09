"""B07 验收断言。用法: just check b07

读取 work/b07/b07.py 并验证 exercises/b07/code.md 里的 T1-T6。
T7 是纸面题，这里只提醒人工检查。

数据用子集（训练 10000、测试 2000）跑，保证断言几秒内结束；
精度阈值按这个规模定，全量数据会更高（线性模型约 0.84）。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b07" / "b07.py"
DATA_ROOT = REPO / "data"
N_TRAIN, N_TEST, BATCH = 10000, 2000, 256

_failures: list[str] = []
_passed: list[str] = []


def case(name: str):
    def deco(fn):
        try:
            fn()
        except AssertionError as exc:
            _failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001  用户代码可能抛任何东西
            _failures.append(f"{name}: 意外异常 {type(exc).__name__}: {exc}")
        else:
            _passed.append(name)
        return fn

    return deco


def load_user_module():
    if not USER_FILE.exists():
        print(f"找不到 {USER_FILE}")
        print("先照着 exercises/b07/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b07_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_iters: dict = {}


def get_iters():
    """数据只读一次，断言之间复用。"""
    if not _iters:
        tf = transforms.ToTensor()
        tr = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=tf)
        g = torch.Generator().manual_seed(0)
        _iters["train"] = DataLoader(
            Subset(tr, range(N_TRAIN)), batch_size=BATCH, shuffle=True, generator=g
        )
        _iters["test"] = DataLoader(Subset(te, range(N_TEST)), batch_size=BATCH)
        _iters["n_test"] = N_TEST
    return _iters["train"], _iters["test"]


def main() -> int:
    mod = load_user_module()

    @case("T1 softmax 数值稳定")
    def _t1a():
        got = mod.softmax(torch.tensor([[1000.0, 0.0]]))
        assert not torch.isnan(got).any(), (
            "出现 nan。检查是不是直接写了 exp(o)/exp(o).sum()，那会上溢"
        )
        assert torch.allclose(got, torch.tensor([[1.0, 0.0]]), atol=1e-6), (
            f"logits 差 1000 时结果应是 [[1, 0]]，得到 {got.tolist()}"
        )

    @case("T1 softmax 归一化")
    def _t1b():
        uniform = mod.softmax(torch.zeros(3, 4))
        assert torch.allclose(uniform, torch.full((3, 4), 0.25), atol=1e-6), (
            f"全零 logits 应得到均匀分布 0.25，得到 {uniform.tolist()}"
        )
        r = mod.softmax(torch.randn(5, 10))
        assert torch.allclose(r.sum(dim=1), torch.ones(5), atol=1e-5), (
            f"每行之和应为 1，得到 {r.sum(dim=1).tolist()}"
        )

    @case("T1 交叉熵")
    def _t1c():
        got = mod.cross_entropy(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0]))
        assert abs(got.item() - 0.41703) < 1e-4, (
            f"单样本交叉熵应约 0.41703，得到 {got.item():.6f}"
        )
        torch.manual_seed(0)
        logits = torch.randn(8, 5)
        y = torch.randint(0, 5, (8,))
        mine = mod.cross_entropy(logits, y).item()
        ref = nn.CrossEntropyLoss(reduction="sum")(logits, y).item()
        assert abs(mine - ref) < 1e-4, (
            f"与 nn.CrossEntropyLoss(reduction='sum') 应当一致（{ref:.6f}），得到 {mine:.6f}。"
            "这里的约定是返回和，不是平均"
        )

    @case("T2 精度")
    def _t2():
        assert mod.accuracy(torch.tensor([[2.0, 1.0], [0.1, 3.0]]), torch.tensor([0, 1])) == 1.0
        assert mod.accuracy(torch.tensor([[1.0, 2.0], [3.0, 0.1]]), torch.tensor([0, 1])) == 0.0
        got = mod.accuracy(torch.zeros(4, 3), torch.tensor([0, 1, 2, 0]))
        assert abs(got - 0.5) < 1e-9, f"应得到 0.5，得到 {got}"

    @case("T3 随机模型的准确率")
    def _t3():
        _, test_iter = get_iters()
        W, b = mod.make_net()
        acc = mod.evaluate_accuracy(lambda X: X @ W + b, test_iter)
        assert acc < 0.3, (
            f"未训练的模型应当接近随机水平，得到 {acc:.4f}。"
            "随机初始化下 argmax 会集中在某一类，所以具体数值由那一类的占比决定，不必是 0.1"
        )

    @case("T4 从零实现训练")
    def _t4():
        train_iter, test_iter = get_iters()
        res = mod.train_from_scratch(train_iter, test_iter, num_epochs=5)
        for key in ("train_loss", "train_acc", "test_acc"):
            assert key in res, f"返回的字典缺 {key} 键"
            assert len(res[key]) == 5, f"{key} 应有 5 项，得到 {len(res[key])}"
        assert res["train_loss"][-1] < res["train_loss"][0], (
            f"训练损失应逐轮下降，得到 {[round(v, 4) for v in res['train_loss']]}"
        )
        assert res["test_acc"][-1] > 0.75, (
            f"5 轮后测试精度应超过 0.75，得到 {res['test_acc'][-1]:.4f}。"
            "明显偏低时先检查：损失是不是返回了和、sgd 里有没有除以批量大小、"
            "两者都除会把学习率压小几百倍"
        )

    @case("T5 混淆矩阵")
    def _t5():
        _, test_iter = get_iters()
        W, b = mod.make_net()
        net = lambda X: X @ W + b  # noqa: E731
        cm = mod.confusion_matrix(net, test_iter)
        assert tuple(cm.shape) == (10, 10), f"形状应为 (10, 10)，得到 {tuple(cm.shape)}"
        assert cm.sum().item() == N_TEST, (
            f"元素之和应等于样本数 {N_TEST}，得到 {cm.sum().item()}"
        )
        acc = mod.evaluate_accuracy(net, test_iter)
        diag_ratio = cm.diag().sum().item() / cm.sum().item()
        assert abs(diag_ratio - acc) < 1e-6, (
            f"对角线占比（{diag_ratio:.4f}）应与准确率（{acc:.4f}）一致"
        )

    @case("T6 简洁实现训练")
    def _t6():
        train_iter, test_iter = get_iters()
        res = mod.train_concise(train_iter, test_iter, num_epochs=5)
        for key in ("train_loss", "train_acc", "test_acc"):
            assert key in res, f"返回的字典缺 {key} 键"
        assert res["test_acc"][-1] > 0.75, (
            f"5 轮后测试精度应超过 0.75，得到 {res['test_acc'][-1]:.4f}。"
            "偏低时先检查有没有在损失前又做了一次 softmax"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT7 是纸面题，人工检查。")
        return 1

    print("T1-T6 全部通过。")
    print("T7 是纸面题，等我看你标的三处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
