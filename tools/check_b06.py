"""B06 验收断言。用法: just check b06

读取 work/b06/b06.py 并验证 exercises/b06/code.md 里的 T1-T6。
T7 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
import types
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# 默认查学习者的实现；传一个路径参数可以查别的文件（自测用）
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b06" / "b06.py"

_failures: list[str] = []
_passed: list[str] = []

TRUE_W = torch.tensor([2.0, -3.4])
TRUE_B = 4.2


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
        print("先照着 exercises/b06/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b06_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 生成数据集")
    def _t1():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        assert tuple(X.shape) == (1000, 2), f"X 形状应为 (1000, 2)，得到 {tuple(X.shape)}"
        assert tuple(y.shape) == (1000, 1), f"y 形状应为 (1000, 1)，得到 {tuple(y.shape)}"
        X2, y2 = mod.make_data(1000, TRUE_W, TRUE_B)
        assert torch.allclose(X, X2), "同一个 seed 必须给出相同的 X"
        assert torch.allclose(y, y2), "同一个 seed 必须给出相同的 y"
        clean = X @ TRUE_W.reshape(-1, 1) + TRUE_B
        noise_std = (y - clean).std().item()
        assert 0.005 < noise_std < 0.02, (
            f"噪声的标准差应接近 0.01，实测 {noise_std:.4f}，检查是否漏加或多加了噪声"
        )

    @case("T2 分批的形状与数量")
    def _t2():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        batches = list(mod.data_iter(10, X, y))
        assert len(batches) == 100, f"1000 个样本按 10 分批应有 100 批，得到 {len(batches)}"
        bx, by = batches[0]
        assert tuple(bx.shape) == (10, 2), f"特征批量形状应为 (10, 2)，得到 {tuple(bx.shape)}"
        assert tuple(by.shape) == (10, 1), f"标签批量形状应为 (10, 1)，得到 {tuple(by.shape)}"

    @case("T2 不重不漏")
    def _t2_cover():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        allx = torch.cat([bx for bx, _ in mod.data_iter(10, X, y)], dim=0)
        assert allx.shape[0] == 1000, f"拼起来应有 1000 行，得到 {allx.shape[0]}"
        assert torch.allclose(allx.sort(dim=0).values, X.sort(dim=0).values), (
            "每一批拼起来应当恰好覆盖原数据，不重不漏"
        )

    @case("T2 最后一批可以不满")
    def _t2_tail():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        batches = list(mod.data_iter(3, X, y))
        assert len(batches) == 334, f"按 3 分批应有 334 批，得到 {len(batches)}"
        assert batches[-1][0].shape[0] == 1, (
            f"最后一批应剩 1 个样本，得到 {batches[-1][0].shape[0]}"
        )

    @case("T2 是生成器且可复现")
    def _t2_gen():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        g = mod.data_iter(10, X, y)
        assert isinstance(g, types.GeneratorType), (
            f"应当返回生成器（用 yield），得到 {type(g).__name__}"
        )
        first = [bx.clone() for bx, _ in mod.data_iter(10, X, y)]
        second = [bx.clone() for bx, _ in mod.data_iter(10, X, y)]
        assert all(torch.allclose(a, b) for a, b in zip(first, second)), (
            "同一个 seed 应当给出相同的切分顺序"
        )

    @case("T3 模型与损失")
    def _t3():
        out = mod.linreg(torch.ones(3, 2), torch.tensor([[1.0], [2.0]]), torch.tensor([0.5]))
        assert tuple(out.shape) == (3, 1), f"输出形状应为 (3, 1)，得到 {tuple(out.shape)}"
        assert torch.allclose(out, torch.full((3, 1), 3.5)), (
            f"每行应为 1+2+0.5=3.5，得到 {out.reshape(-1).tolist()}"
        )
        loss = mod.squared_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 4.0]))
        want = torch.tensor([0.125, 2.0])
        assert torch.allclose(loss, want), (
            f"平方损失应为 {want.tolist()}，得到 {loss.tolist()}。注意要除以 2、且不取平均"
        )
        loss2 = mod.squared_loss(torch.ones(2, 1), torch.zeros(2))
        assert tuple(loss2.shape) == (2, 1), (
            f"y 形状为 (2,) 时应先变形成 y_hat 的形状，得到 {tuple(loss2.shape)}"
        )

    @case("T4 sgd 更新与清零")
    def _t4():
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        (p * p).sum().backward()  # grad = 2p = [2, 4]
        mod.sgd([p], 0.1, 2)
        want = torch.tensor([1.0 - 0.1, 2.0 - 0.2])
        assert torch.allclose(p, want), (
            f"梯度是 [2,4]，除以批量 2 再乘 lr 0.1，p 应变成 {want.tolist()}，得到 {p.tolist()}"
        )
        assert p.grad is not None and torch.allclose(p.grad, torch.zeros(2)), (
            f"更新后梯度应当清零，得到 {p.grad}"
        )
        before = p.detach().clone()
        mod.sgd([p], 0.1, 2)
        assert torch.allclose(p.detach(), before), (
            "梯度已清零，再调一次 sgd 不应当改变参数"
        )

    @case("T5 从零实现训练")
    def _t5():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        w, b, losses = mod.train_scratch(X, y, lr=0.03, num_epochs=3, batch_size=10)
        assert len(losses) == 3, f"losses 应有 3 项，得到 {len(losses)}"
        assert losses[-1] < losses[0], (
            f"损失应当逐轮下降，得到 {[round(v, 4) for v in losses]}"
        )
        got_w = w.detach().reshape(-1)
        assert got_w.shape[0] == 2, f"w 应有 2 个元素，得到 {got_w.shape[0]}"
        assert torch.allclose(got_w, TRUE_W, atol=0.05), (
            f"学到的 w 应接近 [2, -3.4]，得到 {[round(v, 3) for v in got_w.tolist()]}。"
            "3 轮不够接近时先检查 sgd 有没有除以批量大小"
        )
        got_b = b.detach().reshape(-1)[0].item()
        assert abs(got_b - TRUE_B) < 0.05, (
            f"学到的 b 应接近 4.2，得到 {got_b:.3f}"
        )

    @case("T6 简洁实现训练")
    def _t6():
        X, y = mod.make_data(1000, TRUE_W, TRUE_B)
        net, losses = mod.train_concise(X, y, lr=0.03, num_epochs=3, batch_size=10)
        # 不假设 net 的具体结构，从 parameters() 里按元素个数认人
        params = list(net.parameters())
        w_params = [p for p in params if p.numel() == TRUE_W.numel()]
        b_params = [p for p in params if p.numel() == 1]
        assert w_params and b_params, (
            f"net 里应有 2 个元素的权重与 1 个元素的偏置，实际参数的形状是 "
            f"{[tuple(p.shape) for p in params]}"
        )
        got_w = w_params[0].detach().reshape(-1)
        assert torch.allclose(got_w, TRUE_W, atol=0.05), (
            f"学到的权重应接近 [2, -3.4]，得到 {[round(v, 3) for v in got_w.tolist()]}"
        )
        got_b = b_params[0].detach().reshape(-1)[0].item()
        assert abs(got_b - TRUE_B) < 0.05, f"学到的偏置应接近 4.2，得到 {got_b:.3f}"

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
