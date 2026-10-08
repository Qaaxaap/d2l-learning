"""A1 验收断言。用法: just check a1

读取 work/a1/a1.py 并验证 exercises/a1/code.md 的 T1-T5。
T6 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "a1" / "a1.py"

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
        print("先照着 exercises/a1/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("a1_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 计数器")
    def _t1():
        c = mod.Counter()
        assert c() == 1, f"第一次调用应返回 1，得到 {c()!r}"
        assert c(5) == 6, "第二次传 5，应返回 6"
        assert len(c) == 2, f"len(c) 应是调用次数 2，得到 {len(c)!r}"
        text = repr(c)
        assert "6" in text and "2" in text, f"repr 里应含当前值和调用次数，得到 {text!r}"
        c2 = mod.Counter(start=10)
        assert c2() == 11, f"start=10 时第一次调用应返回 11，得到 {c2()!r}"

    @case("T2 运算符")
    def _t2():
        v = mod.Vec2(1, 2)
        got = v + mod.Vec2(3, 4)
        assert (got.x, got.y) == (4, 6), f"加法结果应是 (4, 6)，得到 ({got.x}, {got.y})"
        assert mod.Vec2(1, 2) == mod.Vec2(1, 2), "相同坐标应判等"
        assert not (mod.Vec2(1, 2) == mod.Vec2(1, 3)), "不同坐标不应判等"
        assert (mod.Vec2(1, 2) == "abc") is False, "和别的类型比较应返回 False，不该抛异常"
        assert "1" in repr(v) and "2" in repr(v), f"repr 应含坐标，得到 {repr(v)!r}"

    @case("T3 生成器")
    def _t3():
        assert isinstance(mod.batch_indices(5, 2), types.GeneratorType), (
            "batch_indices 必须是生成器函数（用 yield），不能返回列表"
        )
        assert list(mod.batch_indices(5, 2)) == [[0, 1], [2, 3], [4]], (
            f"n=5, batch_size=2 应产出 [[0,1],[2,3],[4]]，得到 {list(mod.batch_indices(5, 2))!r}"
        )
        assert list(mod.batch_indices(4, 2)) == [[0, 1], [2, 3]], "整除时不应多出空的一批"

    @case("T4 继承 nn.Module")
    def _t4():
        net = mod.ScaledLinear(3, 2, scale=0.5)
        assert isinstance(net, nn.Module), "ScaledLinear 必须继承 nn.Module"
        params = list(net.parameters())
        assert len(params) == 2, (
            f"parameters() 应有 2 个张量，得到 {len(params)} 个——"
            "多半是漏了 super().__init__() 或者没用 nn.Parameter 包起来"
        )
        shapes = sorted(tuple(p.shape) for p in params)
        assert shapes == [(2,), (3, 2)], f"参数形状应是 (3,2) 与 (2,)，得到 {shapes}"
        bias = [p for p in params if tuple(p.shape) == (2,)][0]
        assert torch.allclose(bias, torch.zeros(2)), "偏置应初始化为全零"
        X = torch.randn(4, 3)
        y = net(X)
        assert tuple(y.shape) == (4, 2), f"forward 输出形状应是 (4, 2)，得到 {tuple(y.shape)}"
        y.sum().backward()
        w = [p for p in params if tuple(p.shape) == (3, 2)][0]
        assert w.grad is not None, "参数没拿到梯度，检查权重是不是没包成 nn.Parameter"

    @case("T5 上下文管理器")
    def _t5():
        t = mod.Timer()
        with t:
            time.sleep(0.05)
        assert hasattr(t, "elapsed"), "退出 with 之后应该有 elapsed 属性"
        assert 0.04 <= t.elapsed <= 0.5, f"elapsed 应约为 0.05，得到 {t.elapsed!r}"

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT6 是纸面题，人工检查。")
        return 1

    print("T1-T5 全部通过。")
    print("T6 是纸面题，等我看你标的四处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
