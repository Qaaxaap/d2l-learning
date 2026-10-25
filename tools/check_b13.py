"""B13 验收断言。用法: just check b13

读取 work/b13/b13.py 并验证 exercises/b13/code.md 里的 T1-T5。
T6 是纸面题，这里只提醒人工检查。

整个脚本几秒钟跑完（不训练）。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b13" / "b13.py"

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
        print("先照着 exercises/b13/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b13_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 自定义 MLP 的结构")
    def _t1a():
        net = mod.MLP()
        assert isinstance(net, nn.Module), "MLP 应当是 nn.Module 的子类"
        names = [n for n, _ in net.named_parameters()]
        want = [
            "hidden1.weight", "hidden1.bias",
            "hidden2.weight", "hidden2.bias",
            "out.weight", "out.bias",
        ]
        assert names == want, f"参数名应为 {want}，得到 {names}"
        assert len(list(net.parameters())) == 6, (
            f"应有 6 个参数张量，得到 {len(list(net.parameters()))}"
        )

    @case("T1 前向的形状")
    def _t1b():
        torch.manual_seed(0)
        net = mod.MLP()
        out = net(torch.randn(2, 20))
        assert tuple(out.shape) == (2, 3), f"前向应得到 (2, 3)，得到 {tuple(out.shape)}"

    @case("T2 共享参数：只有两份参数")
    def _t2a():
        net = mod.SharedMLP()
        names = [n for n, _ in net.named_parameters()]
        assert sorted(names) == ["shared.bias", "shared.weight"], (
            f"应只有 shared.weight 与 shared.bias 两个参数，得到 {names}。"
            "参数个数多于 2 说明用了两个不同的层，不是共享"
        )

    @case("T2 共享参数：权重置零后输出全零")
    def _t2b():
        torch.manual_seed(0)
        net = mod.SharedMLP()
        with torch.no_grad():
            net.shared.weight.zero_()
            net.shared.bias.zero_()
        out = net(torch.randn(4, 8))
        assert out.abs().max().item() == 0, (
            "把 shared 的权重与偏置都置零后，输出应当全零。"
            "不为零说明前向里有别的层或别的运算"
        )
        # 前向形状
        with torch.no_grad():
            nn.init.normal_(net.shared.weight)
            nn.init.normal_(net.shared.bias)
        assert tuple(net(torch.randn(4, 8)).shape) == (4, 8)

    @case("T3 参数统计")
    def _t3():
        assert mod.count_by_kind(mod.MLP()) == (3, 3), (
            f"MLP 应为 (3, 3)，得到 {mod.count_by_kind(mod.MLP())}"
        )
        assert mod.count_by_kind(mod.SharedMLP()) == (1, 1), (
            f"SharedMLP 应为 (1, 1)，得到 {mod.count_by_kind(mod.SharedMLP())}"
        )
        want_mlp = 20 * 64 + 64 + 64 * 32 + 32 + 32 * 3 + 3
        assert mod.count_params(mod.MLP()) == want_mlp, (
            f"MLP 参数量应为 {want_mlp}，得到 {mod.count_params(mod.MLP())}。"
            "检查是不是把张量个数当成了元素总数"
        )
        assert mod.count_params(mod.SharedMLP()) == 8 * 8 + 8, (
            f"SharedMLP 参数量应为 72，得到 {mod.count_params(mod.SharedMLP())}"
        )

    @case("T4 自定义初始化的分布")
    def _t4a():
        t = torch.empty(20000)
        out = mod.half_zero_init(t)
        assert out is t, "应当原地改并返回同一个张量"
        zero_ratio = (out == 0).float().mean().item()
        assert 0.45 < zero_ratio < 0.55, (
            f"0 的比例应约 0.5，得到 {zero_ratio:.3f}。"
            "检查是不是先 uniform(-10,10) 再把 |值|<5 的置零"
        )
        nonzero = out[out != 0]
        assert nonzero.abs().min().item() >= 5, (
            f"非零元素的绝对值应至少 5，得到最小值 {nonzero.abs().min().item():.4f}"
        )
        assert nonzero.abs().max().item() <= 10, (
            f"非零元素的绝对值应不超过 10，得到最大值 {nonzero.abs().max().item():.4f}"
        )

    @case("T4 对需要梯度的叶子张量也能用")
    def _t4b():
        t = torch.empty(100)
        t.requires_grad_(True)
        out = mod.half_zero_init(t)
        assert out is t, "应当返回同一个张量"
        assert out.requires_grad, "不该把 requires_grad 弄丢"
        assert (out == 0).float().mean().item() > 0.3, "分布不对"

    @case("T5 ModuleList 堆层")
    def _t5a():
        net = mod.StackedMLP(20, 16, 3, 5)
        n = len(list(net.parameters()))
        assert n == 8, (
            f"StackedMLP(20, 16, 3, 5) 应有 8 个参数张量"
            f"（first 2 + middle 2*2 + out 2），得到 {n}。"
            "个数不对检查：middle 有没有用 ModuleList、层数是不是 num_hidden-1"
        )
        out = net(torch.randn(2, 20))
        assert tuple(out.shape) == (2, 5), f"前向应得到 (2, 5)，得到 {tuple(out.shape)}"

    @case("T5 隐藏层个数随参数变化")
    def _t5b():
        for num_hidden in (1, 2, 5):
            net = mod.StackedMLP(10, 8, num_hidden, 4)
            want = 2 * (1 + (num_hidden - 1) + 1)
            got = len(list(net.parameters()))
            assert got == want, (
                f"num_hidden={num_hidden} 时应有 {want} 个参数张量，得到 {got}"
            )
            assert tuple(net(torch.randn(3, 10)).shape) == (3, 4)

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
