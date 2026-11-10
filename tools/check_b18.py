"""B18 验收断言。用法: just check b18

读取 work/b18/b18.py 并验证 exercises/b18/code.md 里的 T1-T4。

全部是小张量，一秒内跑完。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b18" / "b18.py"

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
        print("先照着 exercises/b18/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b18_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 BN 训练：输出被标准化")
    def _t1a():
        torch.manual_seed(0)
        X = torch.randn(8, 5) * 3 + 10          # 明显偏离均值 0 方差 1
        gamma = torch.ones(5)
        beta = torch.zeros(5)
        mm = torch.zeros(5)
        mv = torch.ones(5)
        Y, _, _ = mod.batch_norm(X, gamma, beta, mm, mv, 1e-5, 0.1, True)
        assert tuple(Y.shape) == (8, 5), f"形状应不变，得到 {tuple(Y.shape)}"
        assert Y.mean(dim=0).abs().max().item() < 1e-5, (
            f"训练模式下每列均值应接近 0，得到 {Y.mean(dim=0).tolist()}"
        )
        assert (Y.var(dim=0, unbiased=False) - 1).abs().max().item() < 1e-4, (
            f"训练模式下每列方差应接近 1，得到 {Y.var(dim=0, unbiased=False).tolist()}"
        )

    @case("T1 BN 训练：gamma 与 beta 生效")
    def _t1b():
        torch.manual_seed(0)
        X = torch.randn(16, 4) * 2 + 5
        gamma = torch.full((4,), 2.0)
        beta = torch.full((4,), 3.0)
        Y, _, _ = mod.batch_norm(X, gamma, beta, torch.zeros(4), torch.ones(4),
                                 1e-5, 0.1, True)
        assert (Y.mean(dim=0) - 3).abs().max().item() < 1e-5, (
            f"beta=3 时均值应为 3，得到 {Y.mean(dim=0).tolist()}"
        )
        assert (Y.var(dim=0, unbiased=False) - 4).abs().max().item() < 1e-4, (
            f"gamma=2 时方差应为 4，得到 {Y.var(dim=0, unbiased=False).tolist()}"
        )

    @case("T1 BN 卷积输入：按通道统计且 keepdim")
    def _t1c():
        torch.manual_seed(0)
        X = torch.randn(4, 3, 6, 6) * 5 + 2
        # 卷积的 BN 参数形状是 (1, C, 1, 1)，与 keepdim 后的统计量一致
        shape = (1, 3, 1, 1)
        Y, _, _ = mod.batch_norm(X, torch.ones(shape), torch.zeros(shape),
                                 torch.zeros(shape), torch.ones(shape),
                                 1e-5, 0.1, True)
        assert tuple(Y.shape) == (4, 3, 6, 6), f"形状应不变，得到 {tuple(Y.shape)}"
        for c in range(3):
            m = Y[:, c].mean().item()
            v = Y[:, c].var(unbiased=False).item()
            assert abs(m) < 1e-5, f"通道 {c} 的均值应接近 0，得到 {m}"
            assert abs(v - 1) < 1e-4, f"通道 {c} 的方差应接近 1，得到 {v}"

    @case("T1 BN 更新移动平均")
    def _t1d():
        torch.manual_seed(0)
        X = torch.full((4, 2), 10.0) + torch.randn(4, 2) * 0.01
        mm = torch.zeros(2)
        mv = torch.ones(2)
        _, mm2, mv2 = mod.batch_norm(X, torch.ones(2), torch.zeros(2), mm, mv,
                                     1e-5, 0.1, True)
        batch_mean = X.mean(dim=0)
        want = 0.1 * 0 + 0.9 * batch_mean
        assert torch.allclose(mm2, want, atol=1e-5), (
            f"moving_mean 应更新为 momentum*旧值 + (1-momentum)*当前批均值，"
            f"期望 {want.tolist()}，得到 {mm2.tolist()}"
        )
        assert not torch.allclose(mv2, torch.ones(2)), "moving_var 应当被更新"

    @case("T1 BN 预测模式：用移动统计量")
    def _t1e():
        X = torch.tensor([[5.0, 7.0]])
        mm = torch.tensor([5.0, 7.0])
        mv = torch.tensor([4.0, 9.0])
        Y, mm2, mv2 = mod.batch_norm(X, torch.ones(2), torch.zeros(2), mm, mv,
                                     1e-5, 0.1, False)
        assert torch.allclose(Y, torch.zeros(1, 2), atol=1e-4), (
            f"输入恰好等于移动均值时应输出 0，得到 {Y.tolist()}"
        )
        assert torch.allclose(mm2, mm) and torch.allclose(mv2, mv), (
            "预测模式下不应更新移动统计量"
        )

    @case("T2 BatchNorm 层：参数与 buffer")
    def _t2a():
        bn = mod.BatchNorm(3, 4)
        params = dict(bn.named_parameters())
        assert set(params.keys()) == {"gamma", "beta"}, (
            f"可训练参数应只有 gamma 与 beta，得到 {list(params.keys())}"
        )
        for name, p in params.items():
            assert isinstance(p, nn.Parameter), f"{name} 必须是 nn.Parameter"
        keys = set(bn.state_dict().keys())
        assert {"gamma", "beta", "moving_mean", "moving_var"} <= keys, (
            f"state_dict 应含四个键，得到 {sorted(keys)}。"
            "moving 两个要用 register_buffer，否则不会被保存"
        )

    @case("T2 BatchNorm 层：两种模式")
    def _t2b():
        torch.manual_seed(0)
        bn = mod.BatchNorm(4, 2)
        X = torch.randn(8, 4) * 3 + 5
        bn.train()
        Y1 = bn(X)
        assert Y1.mean(dim=0).abs().max().item() < 1e-5, "训练模式下应标准化"
        bn.eval()
        Y2 = bn(X)
        assert not torch.allclose(Y1, Y2), (
            "eval 模式应改用移动统计量，输出不该与训练模式相同"
        )
        # 同一个样本单独送进去，结果应与整批送进去时那一行相同
        Y3 = bn(X[:1])
        assert torch.allclose(Y2[:1], Y3, atol=1e-5), (
            "eval 模式下单个样本的输出不该依赖同批其他样本"
        )

    @case("T3 Residual 形状")
    def _t3a():
        torch.manual_seed(0)
        X = torch.randn(4, 3, 6, 6)
        blk = mod.Residual(3, 3)
        out = blk(X)
        assert tuple(out.shape) == (4, 3, 6, 6), (
            f"通道不变时输出形状应等于输入，得到 {tuple(out.shape)}"
        )
        blk2 = mod.Residual(3, 6, use_1x1conv=True, strides=2)
        out2 = blk2(X)
        assert tuple(out2.shape) == (4, 6, 3, 3), (
            f"通道变 6、步幅 2 时应输出 (4,6,3,3)，得到 {tuple(out2.shape)}"
        )

    @case("T3 Residual 的结构")
    def _t3b():
        # BN 用 nn.BatchNorm2d 或 T2 自己写的都算，别限死实现方式
        bn_types = (nn.BatchNorm2d,)
        if hasattr(mod, "BatchNorm"):
            bn_types = (nn.BatchNorm2d, mod.BatchNorm)

        blk = mod.Residual(3, 4)
        convs = [m for m in blk.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 2, f"不用 1×1 时应有 2 个卷积，得到 {len(convs)}"
        bns = [m for m in blk.modules() if isinstance(m, bn_types)]
        assert len(bns) == 2, (
            f"应有 2 个 BN（nn.BatchNorm2d 或自己写的 BatchNorm 都算），"
            f"得到 {len(bns)}"
        )
        blk2 = mod.Residual(3, 4, use_1x1conv=True)
        convs2 = [m for m in blk2.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs2) == 3, f"用 1×1 时应有 3 个卷积，得到 {len(convs2)}"

    @case("T3 Residual 末尾的 ReLU")
    def _t3c():
        torch.manual_seed(0)
        # 不用 1×1 时输入输出通道必须相同，否则跳跃连接加不起来
        for use_1x1, strides, cin, cout in [
            (False, 1, 3, 3), (True, 2, 3, 6), (False, 1, 5, 5),
        ]:
            blk = mod.Residual(cin, cout, use_1x1conv=use_1x1, strides=strides)
            blk.eval()
            X = torch.randn(4, cin, 6, 6) * 3
            Y = blk(X)
            assert (Y >= 0).all(), (
                f"Residual({cin},{cout},use_1x1conv={use_1x1},strides={strides}) "
                f"的输出出现负值（最小 {Y.min().item():.4f}）。"
                "跳跃连接相加之后要过 ReLU"
            )

    @case("T4 DenseBlock 通道增长")
    def _t4a():
        torch.manual_seed(0)
        X = torch.randn(4, 3, 8, 8)
        blk = mod.DenseBlock(2, 10, in_channels=3)
        out = blk(X)
        assert tuple(out.shape) == (4, 23, 8, 8), (
            f"3 + 2*10 = 23 通道，得到 {tuple(out.shape)}"
        )
        blk3 = mod.DenseBlock(4, 32, in_channels=64)
        out3 = blk3(torch.randn(2, 64, 4, 4))
        assert tuple(out3.shape) == (2, 64 + 4 * 32, 4, 4), (
            f"64 + 4*32 = 192 通道，得到 {tuple(out3.shape)}"
        )

    @case("T4 transition_block")
    def _t4a2():
        blk = mod.transition_block(10, in_channels=23)
        out = blk(torch.randn(4, 23, 8, 8))
        assert tuple(out.shape) == (4, 10, 4, 4), (
            f"通道 23→10、高宽减半，应输出 (4,10,4,4)，得到 {tuple(out.shape)}"
        )
        convs = [m for m in blk.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 1 and convs[0].kernel_size == (1, 1), (
            "过渡层应当用 1×1 卷积改通道数"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1

    print("T1-T4 全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
