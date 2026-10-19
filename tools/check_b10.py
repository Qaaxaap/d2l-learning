"""B10 验收断言。用法: just check b10

读取 work/b10/b10.py 并验证 exercises/b10/code.md 里的 T1-T5。
T6 是纸面题，这里只提醒人工检查。

两个实验：高维线性回归（小数据，很快）与 Fashion-MNIST 上的丢弃法（5 轮，约 10 秒）。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b10" / "b10.py"
DATA_ROOT = REPO / "data"
BATCH = 256

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
        print("先照着 exercises/b10/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b10_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fashion: dict = {}


def get_fashion_iters():
    if not _fashion:
        tf = transforms.ToTensor()
        tr = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=tf)
        g = torch.Generator().manual_seed(0)
        _fashion["train"] = DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g)
        _fashion["test"] = DataLoader(te, batch_size=BATCH)
    return _fashion["train"], _fashion["test"]


def get_high_dim_iters(mod):
    """用学习者自己造的 T1 数据做迭代器，这样 T3 与 T1 一致。"""
    x_tr, y_tr, x_te, y_te = mod.make_high_dim()
    g = torch.Generator().manual_seed(0)
    train = DataLoader(TensorDataset(x_tr, y_tr), batch_size=5, shuffle=True, generator=g)
    test = DataLoader(TensorDataset(x_te, y_te), batch_size=100)
    return train, test


def main() -> int:
    mod = load_user_module()

    @case("T1 高维数据")
    def _t1():
        out = mod.make_high_dim()
        assert len(out) == 4, f"应返回四个张量，得到 {len(out)} 个"
        names = ("x_train", "y_train", "x_test", "y_test")
        shapes = [(20, 200), (20, 1), (100, 200), (100, 1)]
        for name, t, want in zip(names, out, shapes):
            assert tuple(t.shape) == want, f"{name} 形状应为 {want}，得到 {tuple(t.shape)}"
            assert t.dtype == torch.float32, f"{name} 的 dtype 应为 float32，得到 {t.dtype}"
        again = mod.make_high_dim()
        for t, u in zip(out, again):
            assert torch.allclose(t, u), "同一个 seed 应给出相同结果"
        y_std = out[1].std().item()
        assert 0.05 < y_std < 1.5, (
            f"y 的标准差应在 0.05 到 1.5 之间（200 维各 0.01，约 0.14），得到 {y_std:.4f}。"
            "检查系数是不是 0.01 而不是 0.1"
        )

    @case("T2 惩罚项")
    def _t2():
        got = mod.l2_penalty(torch.tensor([3.0, 4.0]), 1.0)
        assert torch.allclose(got, torch.tensor(12.5)), (
            f"l2_penalty([3,4], 1.0) 应为 12.5，得到 {got.item() if got.numel() == 1 else got}"
        )
        assert torch.allclose(mod.l2_penalty(torch.tensor([3.0, 4.0]), 0.0), torch.tensor(0.0)), (
            "lam=0 时惩罚项应为 0"
        )
        assert torch.allclose(mod.l2_penalty(torch.zeros(5), 2.0), torch.tensor(0.0)), (
            "全零权重的惩罚项应为 0"
        )

    @case("T3 高维训练：lam=0 会过拟合")
    def _t3a():
        train_iter, test_iter = get_high_dim_iters(mod)
        res = mod.train_high_dim(train_iter, test_iter, lam=0.0, num_epochs=3)
        for key in ("train_loss", "test_loss"):
            assert key in res, f"返回的字典缺 {key} 键"
            assert len(res[key]) == 3, f"{key} 应有 3 项，得到 {len(res[key])}"
        assert id(res["train_loss"]) != id(res["test_loss"]), "两个键不能指向同一个列表"
        tr, te = res["train_loss"][-1], res["test_loss"][-1]
        assert tr < 0.05, (
            f"20 个样本、200 维特征，不加正则化时训练损失应低于 0.05（过拟合），得到 {tr:.4f}。"
            "偏大检查：特征维度、迭代轮数、记录的是不是平均损失"
        )
        assert te > tr * 3, (
            f"测试损失应明显高于训练损失（这就是过拟合），得到训练 {tr:.4f}、测试 {te:.4f}。"
            "两者接近说明没造出过拟合的场景"
        )

    @case("T3 高维训练：加惩罚项后测试损失下降")
    def _t3b():
        train_iter, test_iter = get_high_dim_iters(mod)
        no_reg = mod.train_high_dim(train_iter, test_iter, lam=0.0, num_epochs=3)
        reg = mod.train_high_dim(train_iter, test_iter, lam=3.0, num_epochs=3)
        te_no, te_reg = no_reg["test_loss"][-1], reg["test_loss"][-1]
        assert te_reg < te_no, (
            f"加惩罚项后测试损失应更低，得到 lam=0 时 {te_no:.4f}、lam=3 时 {te_reg:.4f}。"
            "检查惩罚项是不是真的加进了训练损失"
        )
        tr_no, tr_reg = no_reg["train_loss"][-1], reg["train_loss"][-1]
        assert tr_reg > tr_no, (
            f"加惩罚项后训练损失应当更高（正则化换来了泛化），得到 lam=0 时 {tr_no:.4f}、"
            f"lam=3 时 {tr_reg:.4f}。两者相同说明惩罚项没进训练损失"
        )

    @case("T4 丢弃函数：边界")
    def _t4a():
        x = torch.randn(64, 32)
        same = mod.dropout_layer(x, 0.0)
        assert torch.allclose(same, x), "dropout=0 时应原样返回"
        zero = mod.dropout_layer(x, 1.0)
        assert tuple(zero.shape) == tuple(x.shape), "dropout=1 时形状应与输入相同"
        assert zero.abs().sum().item() == 0.0, "dropout=1 时应返回全零"

    @case("T4 丢弃函数：随机性与缩放")
    def _t4b():
        torch.manual_seed(0)
        x = torch.ones(2000, 5)
        out = mod.dropout_layer(x, 0.5)
        assert tuple(out.shape) == tuple(x.shape), f"形状应不变，得到 {tuple(out.shape)}"
        zero_ratio = (out == 0).float().mean().item()
        assert 0.4 < zero_ratio < 0.6, (
            f"dropout=0.5 时约一半元素应为 0，实测比例 {zero_ratio:.3f}。"
            "检查掩码是不是按 dropout 的概率生成的"
        )
        # 留下的元素应被放大到 1/(1-p) = 2，所以整体均值仍接近 1
        mean = out.mean().item()
        assert 0.8 < mean < 1.2, (
            f"经过 1/(1-p) 缩放后整体均值应接近输入的 1，得到 {mean:.3f}。"
            "偏小说明忘了除以 (1 - dropout)"
        )
        a = mod.dropout_layer(x, 0.5)
        b = mod.dropout_layer(x, 0.5)
        assert not torch.allclose(a, b), "两次调用结果应当不同（确实是随机的）"
        assert torch.allclose(x, torch.ones(2000, 5)), "不应修改输入张量本身"

    @case("T5a make_dropout_net 的结构")
    def _t5a():
        net = mod.make_dropout_net()
        kinds = [type(m).__name__ for m in net]
        want = ["Flatten", "Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]
        assert kinds == want, f"层的顺序应为 {want}，得到 {kinds}"
        p1, p2 = net[3].p, net[6].p
        assert abs(p1 - 0.2) < 1e-9, f"第一个丢弃层的丢弃概率应为 0.2，得到 {p1}"
        assert abs(p2 - 0.5) < 1e-9, f"第二个丢弃层的丢弃概率应为 0.5，得到 {p2}"
        out = net(torch.randn(2, 1, 28, 28))
        assert tuple(out.shape) == (2, 10), f"输出形状应为 (2, 10)，得到 {tuple(out.shape)}"

    @case("T5b evaluate 不建图且两次结果相同")
    def _t5b():
        # 断言自己造模型，不依赖 make_dropout_net
        torch.manual_seed(0)
        net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10), nn.Dropout(0.5))
        _, test_iter = get_fashion_iters()
        a = mod.evaluate(net, test_iter)
        b = mod.evaluate(net, test_iter)
        assert a == b, (
            f"同一个模型两次评估结果应完全相同，得到 {a:.6f} 与 {b:.6f}。"
            "两次不同说明评估时丢弃层还在随机丢弃——评估前要切到推理模式"
        )
        assert 0.0 <= a <= 1.0, f"评估结果应是 0 到 1 的比例，得到 {a}"

    @case("T5c evaluate 用完恢复原来的模式")
    def _t5c():
        torch.manual_seed(0)
        net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10), nn.Dropout(0.5))
        _, test_iter = get_fashion_iters()
        # 从训练模式开始，评估完应当回到训练模式
        net.train()
        mod.evaluate(net, test_iter)
        assert net.training is True, (
            "模型原本处于训练模式，评估之后应当恢复成训练模式，"
            "否则下一轮训练不会丢弃"
        )
        # 从推理模式开始，评估完应当仍是推理模式
        net.eval()
        mod.evaluate(net, test_iter)
        assert net.training is False, (
            "模型原本处于推理模式，评估之后被切成了训练模式——"
            "应当先记下原状态，用完按原状态恢复"
        )

    @case("T5d train_dropout 的完整流程")
    def _t5d():
        train_iter, test_iter = get_fashion_iters()
        res = mod.train_dropout(train_iter, test_iter, use_dropout=True, num_epochs=5)
        keys = ("train_loss", "train_acc", "test_acc")
        for key in keys:
            assert key in res, f"返回的字典缺 {key} 键"
            assert len(res[key]) == 5, f"{key} 应有 5 项，得到 {len(res[key])}"
        ids = [id(res[k]) for k in keys]
        assert len(set(ids)) == 3, "三个键指向了同一个列表"
        for key in ("train_acc", "test_acc"):
            assert all(0.0 <= v <= 1.0 for v in res[key]), f"{key} 里出现了不在 0 到 1 之间的值"
        assert res["test_acc"][-1] > 0.75, (
            f"5 轮后测试精度应超过 0.75，得到 {res['test_acc'][-1]:.4f}。"
            "偏低时按顺序查：丢弃层的顺序对不对、评估时有没有切 eval、"
            "损失是不是被 weight_decay 影响了"
        )

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
