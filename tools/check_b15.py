"""B15 验收断言。用法: just check b15

读取 work/b15/b15.py 并验证 exercises/b15/code.md 里的 T1-T6。
T7 是纸面题，这里只提醒人工检查。

整个脚本一两秒跑完（全是小张量上的循环）。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b15" / "b15.py"

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
        print("先照着 exercises/b15/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b15_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 corr2d 与书上的例子")
    def _t1a():
        X = torch.tensor([[0.0, 1, 2], [3, 4, 5], [6, 7, 8]])
        K = torch.tensor([[0.0, 1], [2, 3]])
        out = mod.corr2d(X, K)
        want = torch.tensor([[19.0, 25.0], [37.0, 43.0]])
        assert tuple(out.shape) == (2, 2), f"输出应为 2x2，得到 {tuple(out.shape)}"
        assert torch.allclose(out, want), (
            f"应为 {want.tolist()}，得到 {out.tolist()}。"
            "第一个元素是 0*0+1*1+3*2+4*3=19，检查窗口切片与'按元素乘再求和'"
        )

    @case("T1 corr2d 与 nn.Conv2d 交叉验证")
    def _t1b():
        torch.manual_seed(0)
        X = torch.randn(9, 11)
        K = torch.randn(3, 4)
        mine = mod.corr2d(X, K)
        conv = nn.Conv2d(1, 1, (3, 4), bias=False)
        with torch.no_grad():
            conv.weight.copy_(K.reshape(1, 1, 3, 4))
        ref = conv(X.reshape(1, 1, 9, 11)).squeeze(0).squeeze(0)
        assert tuple(mine.shape) == tuple(ref.shape), (
            f"形状应与 nn.Conv2d 一致：{tuple(ref.shape)}，得到 {tuple(mine.shape)}"
        )
        assert torch.allclose(mine, ref, atol=1e-4), (
            f"与 nn.Conv2d 的最大差 {abs(mine - ref).max().item():.6f}，应接近 0"
        )
        # 不修改输入
        assert X.shape == (9, 11), "不应修改传入的 X"

    @case("T2 输出形状公式")
    def _t2a():
        # p_h / p_w 按讲义的定义是"两侧一共填充"。nn.Conv2d(padding=1) 等于这里的 p=2
        cases = [
            ((8, 8, 3, 3), (6, 6)),
            ((8, 8, 3, 3, 2, 2), (8, 8)),
            ((8, 8, 3, 3, 2, 2, 2, 2), (4, 4)),
            ((8, 8, 5, 3, 4, 2), (8, 8)),
            ((7, 7, 3, 3), (5, 5)),
        ]
        for args, want in cases:
            got = tuple(mod.conv_out_shape(*args))
            assert got == want, (
                f"conv_out_shape{args} 应为 {want}，得到 {got}。"
                "公式：(n - k + p) // s + 1"
            )

    @case("T2 与 nn.Conv2d 实测一致")
    def _t2b():
        # nn.Conv2d 的 padding 是"每侧"，所以公式里要乘 2
        for n, k, p, s in [(8, 3, 0, 1), (8, 3, 1, 1), (8, 3, 1, 2), (9, 4, 2, 3), (11, 5, 0, 2)]:
            conv = nn.Conv2d(1, 1, k, padding=p, stride=s)
            with torch.no_grad():
                real = tuple(conv(torch.zeros(1, 1, n, n)).shape[2:])
            got = tuple(mod.conv_out_shape(n, n, k, k, 2 * p, 2 * p, s, s))
            assert got == real, (
                f"n={n} k={k} p={p} s={s}：公式给出 {got}，nn.Conv2d 实测 {real}"
            )

    @case("T3 多输入通道")
    def _t3a():
        X = torch.tensor([[[0.0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]])
        K = torch.tensor([[[0.0, 1], [2, 3]], [[1.0, 2], [3, 4]]])
        out = mod.corr2d_multi_in(X, K)
        want = torch.tensor([[56.0, 72.0], [104.0, 120.0]])
        assert out.dim() == 2, f"输出应是二维，得到 {out.dim()} 维"
        assert torch.allclose(out, want), (
            f"应为 {want.tolist()}，得到 {out.tolist()}。"
            "第一个元素 = (1*1+2*2+4*3+5*4) + (0*0+1*1+3*2+4*3) = 37+19 = 56"
        )

    @case("T3 单通道时与 corr2d 一致")
    def _t3b():
        torch.manual_seed(0)
        x = torch.randn(5, 6)
        k = torch.randn(2, 3)
        a = mod.corr2d_multi_in(x.reshape(1, 5, 6), k.reshape(1, 2, 3))
        b = mod.corr2d(x, k)
        assert torch.allclose(a, b, atol=1e-6), "单通道时两者应当完全相同"

    @case("T4 多输出通道")
    def _t4a():
        X = torch.tensor([[[0.0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]])
        K2 = torch.tensor([[[0.0, 1], [2, 3]], [[1.0, 2], [3, 4]]])
        K = torch.stack([K2, K2 + 1, K2 + 2])
        out = mod.corr2d_multi_in_out(X, K)
        assert tuple(out.shape) == (3, 2, 2), (
            f"3 个输出通道、2x2 输出，应为 (3,2,2)，得到 {tuple(out.shape)}"
        )
        first = mod.corr2d_multi_in(X, K2)
        assert torch.allclose(out[0], first, atol=1e-6), (
            "第 0 个输出通道应当与 corr2d_multi_in(X, K2) 相同"
        )

    @case("T4 输出形状随通道数变化")
    def _t4b():
        torch.manual_seed(0)
        X = torch.randn(3, 6, 7)
        for c_o in (1, 2, 5):
            K = torch.randn(c_o, 3, 2, 3)
            out = mod.corr2d_multi_in_out(X, K)
            assert tuple(out.shape) == (c_o, 5, 5), (
                f"c_o={c_o}：应为 ({c_o}, 5, 5)，得到 {tuple(out.shape)}"
            )

    @case("T5 1x1 卷积与普通卷积等价")
    def _t5():
        torch.manual_seed(0)
        X = torch.randn(3, 4, 5)
        K = torch.randn(2, 3, 1, 1)
        a = mod.corr2d_multi_in_out_1x1(X, K)
        b = mod.corr2d_multi_in_out(X, K)
        assert tuple(a.shape) == (2, 4, 5), (
            f"输出应为 (c_o, h, w) = (2,4,5)，得到 {tuple(a.shape)}"
        )
        assert torch.allclose(a, b, atol=1e-5), (
            f"1x1 卷积与普通卷积的最大差 {abs(a-b).max().item():.6f}，应接近 0"
        )

    @case("T6 池化与书上的例子")
    def _t6a():
        X = torch.tensor([[0.0, 1, 2], [3, 4, 5], [6, 7, 8]])
        mx = mod.pool2d(X, (2, 2), "max")
        av = mod.pool2d(X, (2, 2), "avg")
        assert torch.allclose(mx, torch.tensor([[4.0, 5.0], [7.0, 8.0]])), (
            f"max 池化应为 [[4,5],[7,8]]，得到 {mx.tolist()}"
        )
        assert torch.allclose(av, torch.tensor([[2.0, 3.0], [5.0, 6.0]])), (
            f"avg 池化应为 [[2,3],[5,6]]，得到 {av.tolist()}"
        )

    @case("T6 与 nn.MaxPool2d 交叉验证")
    def _t6b():
        torch.manual_seed(0)
        X = torch.randn(8, 8)
        mine = mod.pool2d(X, (3, 3), "max")
        # 手写版是逐格滑（步幅 1）；nn.MaxPool2d 的默认步幅等于窗口大小，所以要显式给 stride=1
        ref = nn.MaxPool2d(3, stride=1)(X.reshape(1, 1, 8, 8)).squeeze(0).squeeze(0)
        assert tuple(mine.shape) == tuple(ref.shape), (
            f"形状应与 nn.MaxPool2d 一致：{tuple(ref.shape)}，得到 {tuple(mine.shape)}"
        )
        assert torch.allclose(mine, ref), "与 nn.MaxPool2d 的结果应当一致"

    @case("T6 非方形窗口")
    def _t6c():
        torch.manual_seed(0)
        X = torch.randn(7, 9)
        out = mod.pool2d(X, (2, 3), "max")
        assert tuple(out.shape) == (6, 7), (
            f"7x9 输入配 (2,3) 窗口应为 (6,7)，得到 {tuple(out.shape)}"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT7 是纸面题，人工检查。")
        return 1

    print("T1-T6 全部通过。")
    print("T7 是纸面题，等我看你标的两处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
