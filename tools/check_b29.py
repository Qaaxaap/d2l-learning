"""B29 验收断言。用法: just check b29"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b29" / "b29.py"

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
        print("先照着 exercises/b29/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b29_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 超长时截断")
    def _t1a():
        got = mod.truncate_pad([1, 2, 3, 4, 5], 3, 0)
        assert got == [1, 2, 3], f"应截断成 [1,2,3]，得到 {got}"

    @case("T1 不足时填充")
    def _t1b():
        got = mod.truncate_pad([1, 2], 5, 0)
        assert got == [1, 2, 0, 0, 0], f"应填充成 [1,2,0,0,0]，得到 {got}"

    @case("T1 长度刚好时不改")
    def _t1c():
        line = [1, 2, 3]
        got = mod.truncate_pad(line, 3, 0)
        assert got == [1, 2, 3], f"长度刚好应原样返回，得到 {got}"

    @case("T1 顺序不能反")
    def _t1d():
        # 先填充再截断会得到 [1,2,0]，顺序错了
        got = mod.truncate_pad([1, 2, 3, 4], 3, 0)
        assert got[0] == 1 and got[-1] == 3, (
            f"截断要保留前 3 个，得到 {got}。顺序反了会变成填充分号"
        )

    @case("T2 一维卷积的输出长度")
    def _t2():
        assert mod.conv1d_output_len(5, 2) == 4, (
            f"5-2+1=4，得到 {mod.conv1d_output_len(5, 2)}"
        )
        assert mod.conv1d_output_len(16, 3, 2, 1) == 8, (
            f"(16+2-3)//2+1=8，得到 {mod.conv1d_output_len(16, 3, 2, 1)}"
        )
        assert mod.conv1d_output_len(100, 5, 1, 2) == 100, (
            f"填充 2、核 5 时长度不变，得到 {mod.conv1d_output_len(100, 5, 1, 2)}"
        )

    @case("T3 时序最大池化")
    def _t3a():
        x = torch.tensor([[[1.0, 5.0, 3.0], [2.0, 0.5, 8.0]]])   # (1, 2, 3)
        got = mod.time_max_pool(x)
        want = torch.tensor([[5.0, 8.0]])
        assert torch.allclose(got, want), f"应为 {want.tolist()}，得到 {got.tolist()}"

    @case("T3 形状与求导")
    def _t3b():
        x = torch.randn(4, 6, 10, requires_grad=True)
        y = mod.time_max_pool(x)
        assert tuple(y.shape) == (4, 6), (
            f"形状应为 (4,6)，得到 {tuple(y.shape)}"
        )
        y.sum().backward()
        # 最大池化的梯度只落在被选中的位置上，每个通道一行
        per_channel = (x.grad.abs().sum(dim=2) > 0).sum(dim=1)
        assert torch.equal(per_channel, torch.full((4,), 6)), (
            "每个通道应当只有一个位置拿到梯度"
        )

    @case("T4 文本对的特征")
    def _t4a():
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 1.0]])
        got = mod.pair_features(a, b)
        # 依次是 a、b、|a-b|、a*b
        want = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 3.0, 2.0]])
        assert torch.allclose(got, want), f"应为 {want.tolist()}，得到 {got.tolist()}"

    @case("T4 做差要取绝对值")
    def _t4b():
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[5.0, 0.0]])
        got = mod.pair_features(a, b)
        # 第三段是 |a-b|，不该有负数
        assert got[0, 4] == 4.0, (
            f"|1-5| 应为 4，得到 {got[0, 4]}。做差要取绝对值"
        )
        assert got[0, 5] == 0.0, f"|0-0| 应为 0，得到 {got[0, 5]}"

    @case("T4 输出形状")
    def _t4c():
        a = torch.randn(8, 16)
        b = torch.randn(8, 16)
        got = mod.pair_features(a, b)
        assert tuple(got.shape) == (8, 64), (
            f"四段拼接后应为 (8,64)，得到 {tuple(got.shape)}"
        )

    @case("T5 分类头的形状")
    def _t5():
        emb = torch.randn(3, 16)
        head = mod.build_classifier(emb, 5)
        assert isinstance(head, nn.Linear), f"应返回 nn.Linear，得到 {type(head)}"
        assert head.in_features == 16, (
            f"输入维应为 16，得到 {head.in_features}"
        )
        assert head.out_features == 5, (
            f"输出维应为 5，得到 {head.out_features}"
        )
        assert head(emb).shape == (3, 5), "前向输出形状应为 (3,5)"

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
