"""B08 验收断言。用法: just check b08

读取 work/b08/b08.py 并验证 exercises/b08/code.md 里的 T1-T5。
T6 是纸面题，这里只提醒人工检查。

这一章用全量数据：MLP 的参数量是 20 万，用 1 万样本训练会不稳定
（讲义 4.5 节有实测）。两次训练各约 12 秒。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b08" / "b08.py"
DATA_ROOT = REPO / "data"
BATCH = 256
PARAMS_FULL = 203530
PARAMS_LINEAR = 7850

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
        print("先照着 exercises/b08/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b08_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_iters: dict = {}


def get_iters():
    if not _iters:
        tf = transforms.ToTensor()
        tr = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=tf)
        g = torch.Generator().manual_seed(0)
        _iters["train"] = DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g)
        _iters["test"] = DataLoader(te, batch_size=BATCH)
    return _iters["train"], _iters["test"]


def check_history(res, tag):
    """T4 与 T5 共用的历史检查。"""
    keys = ("train_loss", "train_acc", "test_acc")
    for key in keys:
        assert key in res, f"{tag} 返回的字典缺 {key} 键"
        assert len(res[key]) == 10, f"{tag} 的 {key} 应有 10 项，得到 {len(res[key])}"
    ids = [id(res[k]) for k in keys]
    assert len(set(ids)) == 3, (
        f"{tag} 的三个键指向了同一个列表。`d['a'] = d['b'] = d['c'] = []` 这种链式赋值"
        "会让它们共享一个对象"
    )
    for key in ("train_acc", "test_acc"):
        assert all(0.0 <= v <= 1.0 for v in res[key]), (
            f"{tag} 的 {key} 里出现了不在 0 到 1 之间的值 {[round(v, 4) for v in res[key]]}，"
            "检查是不是把损失填进了准确率"
        )
    # 第一轮的平均损失：未训练时单批约 log(10)=2.3，但一轮有 234 批，
    # 跑完第一轮平均值已降到 0.86 左右。区间用来抓量级错误（例如多除了两次样本数）
    # 与"根本没训练"两种情况。
    assert 0.3 < res["train_loss"][0] < 2.0, (
        f"{tag} 第一轮的平均损失应在 0.3 到 2.0 之间（实测约 0.86），"
        f"得到 {res['train_loss'][0]:.4f}。偏小检查是不是多除了样本数，"
        "偏大检查是不是没更新参数"
    )
    assert res["train_loss"][-1] < res["train_loss"][0] * 0.8, (
        f"{tag} 的损失应明显下降，得到 {[round(v, 4) for v in res['train_loss']]}。"
        "只降一点点通常说明参数没更新，或者忘了加激活函数"
    )
    assert res["test_acc"][-1] > 0.82, (
        f"{tag} 10 轮后测试精度应超过 0.82，得到 {res['test_acc'][-1]:.4f}。"
        "明显偏低时按顺序查：有没有加激活函数、ReLU 是不是放在输出层之后、"
        "参数初始化是不是全零"
    )


def main() -> int:
    mod = load_user_module()

    @case("T1 激活函数")
    def _t1():
        got = mod.relu(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))
        want = torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0])
        assert torch.allclose(got, want), f"relu 应得到 {want.tolist()}，得到 {got.tolist()}"
        g = mod.relu_grad(torch.tensor([-2.0, 0.0, 3.0]))
        want_g = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(g, want_g), (
            f"relu_grad 应得到 {want_g.tolist()}，得到 {g.tolist()}。x=0 处取 0"
        )
        two_d = mod.relu(torch.tensor([[-1.0, 2.0], [3.0, -4.0]]))
        assert tuple(two_d.shape) == (2, 2), "relu 应当逐元素处理，形状不变"

    @case("T2 参数与模型")
    def _t2():
        W1, b1, W2, b2 = mod.init_params()
        shapes = [(784, 256), (256,), (256, 10), (10,)]
        got = [tuple(t.shape) for t in (W1, b1, W2, b2)]
        assert got == shapes, f"四个参数的形状应为 {shapes}，得到 {got}"
        assert all(t.requires_grad for t in (W1, b1, W2, b2)), "四个参数都要开 requires_grad"
        W1b, b1b, W2b, b2b = mod.init_params()
        assert torch.allclose(W1, W1b) and torch.allclose(W2, W2b), "同一个 seed 应给出相同结果"
        out = mod.mlp_forward(torch.randn(4, 1, 28, 28), W1, b1, W2, b2)
        assert tuple(out.shape) == (4, 10), f"输出形状应为 (4, 10)，得到 {tuple(out.shape)}"
        assert (out < 0).any(), (
            "输出里应当有负数。全是非负说明过了 ReLU 或 softmax，"
            "而这里只该返回 logits"
        )

    @case("T3 参数量")
    def _t3():
        net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)
        )
        got = mod.count_params(net)
        assert got == PARAMS_FULL, f"MLP 的参数个数应为 {PARAMS_FULL}，得到 {got}"
        got2 = mod.count_params(nn.Linear(784, 10))
        assert got2 == PARAMS_LINEAR, f"单层模型的参数个数应为 {PARAMS_LINEAR}，得到 {got2}"

    @case("T4 从零实现训练")
    def _t4():
        train_iter, test_iter = get_iters()
        res = mod.train_from_scratch(train_iter, test_iter, num_epochs=10)
        check_history(res, "T4")

    @case("T5 简洁实现训练")
    def _t5():
        train_iter, test_iter = get_iters()
        res = mod.train_concise(train_iter, test_iter, num_epochs=10)
        check_history(res, "T5")

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT6 是纸面题，人工检查。")
        return 1

    print("T1-T5 全部通过。")
    print("T6 是纸面题，等我看你标的三处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
