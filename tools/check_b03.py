"""B03 验收断言。用法: just check b03

读取 work/b03/b03.py 并验证 exercises/b03/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# 默认查学习者的实现；传一个路径参数可以查别的文件（自测用）
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b03" / "b03.py"

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
        print("先照着 exercises/b03/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b03_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 按轴求和的形状")
    def _t1():
        cases = [
            ((3, 4), 0, False, (4,)),
            ((3, 4), 0, True, (1, 4)),
            ((3, 4), 1, False, (3,)),
            ((3, 4), 1, True, (3, 1)),
            ((2, 3, 4), 1, False, (2, 4)),
            ((2, 3, 4), 1, True, (2, 1, 4)),
            ((5,), 0, False, ()),
            ((5,), 0, True, (1,)),
            ((2, 3, 4), -1, False, (2, 3)),
            ((2, 3, 4), -1, True, (2, 3, 1)),
        ]
        for shape, dim, keepdim, want in cases:
            got = mod.reduce_shape(shape, dim, keepdim)
            assert type(got) is tuple, (
                f"reduce_shape({shape}, {dim}, {keepdim}) 应返回普通 tuple，"
                f"得到 {type(got).__name__}"
            )
            assert got == want, (
                f"reduce_shape({shape}, {dim}, {keepdim}) 应为 {want}，得到 {got}"
            )

    @case("T1 与 torch 实际行为一致")
    def _t1_cross():
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for dim in (0, 1, 2, -1, -2):
            for keepdim in (False, True):
                want = tuple(x.sum(dim=dim, keepdim=keepdim).shape)
                got = mod.reduce_shape(tuple(x.shape), dim, keepdim)
                assert got == want, (
                    f"reduce_shape({tuple(x.shape)}, {dim}, {keepdim}) 与 torch 的实际结果不符："
                    f"你给 {got}，torch 给 {want}"
                )

    @case("T2 选乘法函数")
    def _t2():
        cases = [
            ((3,), (3,), "dot"),
            ((2, 3), (3,), "mv"),
            ((2, 3), (3, 4), "mm"),
            ((2, 3, 4), (4, 5), "matmul"),
            ((2, 3, 4), (2, 4, 5), "matmul"),
        ]
        fn_of = {
            "dot": torch.dot,
            "mv": torch.mv,
            "mm": torch.mm,
            "matmul": torch.matmul,
        }
        for sa, sb, want in cases:
            a, b = torch.ones(sa), torch.ones(sb)
            got = mod.matmul_kind(a, b)
            assert got == want, f"形状 {sa} 与 {sb} 应当用 {want}，得到 {got!r}"
            assert got in fn_of, f"返回值应当是 'dot' / 'mv' / 'mm' / 'matmul' 之一，得到 {got!r}"
            fn_of[got](a, b)  # 选出来的函数要真能用，否则抛异常

    @case("T3 手写 L2 范数")
    def _t3():
        for data in ([3.0, 4.0], [0.0] * 5, [1.0] * 6, [-3.0, -4.0]):
            x = torch.tensor(data)
            got = mod.l2_norm(x)
            want = torch.linalg.vector_norm(x)
            assert torch.allclose(got, want), (
                f"l2_norm({data}) 应为 {want.item():.6f}，得到 {got}"
            )
        X = torch.ones(2, 3)
        want = torch.tensor(6.0**0.5)
        got = mod.l2_norm(X)
        assert torch.allclose(got, want), (
            f"多维输入要先展平：l2_norm(2x3 全 1) 应为 {want.item():.6f}，得到 {got}"
        )
        assert mod.l2_norm(torch.tensor([3.0, 4.0])).dim() == 0, (
            "返回值应当是 0 维张量，不是 Python 浮点数也不是一维张量"
        )

    @case("T4 按行归一化")
    def _t4():
        X = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0], [0.0, -2.0]])
        backup = X.clone()
        Y = mod.normalize_rows(X)

        assert torch.equal(X, backup), "不能修改 X"
        assert Y.shape == X.shape, f"形状应与输入相同，得到 {tuple(Y.shape)}"
        assert Y.dtype == X.dtype, f"dtype 应与输入相同，得到 {Y.dtype}"
        assert not torch.isnan(Y).any(), "结果里出现了 nan，零行没有被单独处理"

        assert torch.allclose(Y[0], torch.tensor([0.6, 0.8])), (
            f"第 0 行应为 [0.6, 0.8]，得到 {Y[0].tolist()}"
        )
        assert torch.allclose(Y[1], torch.tensor([1.0, 0.0])), (
            f"第 1 行应为 [1.0, 0.0]，得到 {Y[1].tolist()}"
        )
        assert torch.allclose(Y[2], torch.zeros(2)), (
            f"第 2 行全是 0，结果应保持全 0，得到 {Y[2].tolist()}"
        )
        assert torch.allclose(Y[3], torch.tensor([0.0, -1.0])), (
            f"第 3 行应为 [0.0, -1.0]，得到 {Y[3].tolist()}"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT5 是纸面题，人工检查。")
        return 1

    print("T1-T4 全部通过。")
    print("T5 是纸面题，等我看你标的三个 bug。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
