"""B04 验收断言。用法: just check b04

读取 work/b04/b04.py 并验证 exercises/b04/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import torch

# T2 会故意访问非叶子的 .grad，torch 为此发一条很长的警告，这里压掉
warnings.filterwarnings("ignore", message=".*not a leaf Tensor.*")

REPO = Path(__file__).resolve().parent.parent
# 默认查学习者的实现；传一个路径参数可以查别的文件（自测用）
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b04" / "b04.py"

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
        print("先照着 exercises/b04/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b04_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 二次型的梯度")
    def _t1():
        I3 = torch.eye(3)
        got = mod.grad_of_quadratic(I3, torch.ones(3))
        assert tuple(got.shape) == (3,), f"返回值形状应为 (3,)，得到 {tuple(got.shape)}"
        assert torch.allclose(got, torch.tensor([2.0, 2.0, 2.0])), (
            f"单位阵下 x 全 1 时梯度应为 [2,2,2]，得到 {got.tolist()}"
        )
        got = mod.grad_of_quadratic(I3, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(got, torch.tensor([2.0, 4.0, 6.0])), (
            f"单位阵下应为 2x，得到 {got.tolist()}"
        )
        # 非对称，对着 (A + A.T) @ x 比
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x = torch.tensor([1.0, -1.0])
        want = (A + A.T) @ x
        got = mod.grad_of_quadratic(A, x)
        assert torch.allclose(got, want), (
            f"非对称时梯度应为 (A + A.T) @ x = {want.tolist()}，得到 {got.tolist()}"
        )

    @case("T2 叶子与非叶子")
    def _t2():
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        rep = mod.grad_report(x)
        assert isinstance(rep, dict), f"应返回 dict，得到 {type(rep).__name__}"
        for key in ("x_grad", "y_grad", "x_is_leaf", "y_is_leaf"):
            assert key in rep, f"报告缺少键 {key!r}，实际有 {list(rep)}"
        assert rep["x_is_leaf"] is True, "x 是叶子"
        assert rep["y_is_leaf"] is False, "y 是中间结果，不是叶子"
        assert rep["y_grad"] is None, "非叶子的 .grad 默认是 None"
        assert rep["x_grad"] is not None, "x 反传后应当有梯度"
        assert torch.allclose(rep["x_grad"], torch.tensor([2.0, 2.0])), (
            f"y = (x * 2).sum() 对 x 的梯度应为 [2,2]，得到 {rep['x_grad'].tolist()}"
        )

    @case("T3 连续反传两次")
    def _t3():
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        got = mod.grad_after_two_backwards(x)
        want = torch.tensor([0.0, 4.0, 8.0, 12.0])
        assert torch.allclose(got, want), (
            f"两次反传后梯度累加应为 2 * 2x = {want.tolist()}，得到 {got.tolist()}。"
            "如果得到 [0,2,4,6] 说明只反传了一次；"
            "如果中途清零过，同样只剩一份"
        )

    @case("T4 一次参数更新")
    def _t4():
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        w = torch.tensor([0.0, 0.0], requires_grad=True)
        mod.train_step(w, x, 0.5)
        assert torch.allclose(w, torch.tensor([-1.0, -1.0])), (
            f"梯度是每列之和 [2,2]，w 应变成 [-1,-1]，得到 {w.tolist()}"
        )
        assert w.is_leaf, "更新后 w 必须仍是叶子，说明用了 no_grad 或原地写法"
        assert w.requires_grad, "更新后 w 的 requires_grad 应保持 True"

    @case("T4 连续两步不带残留")
    def _t4_twice():
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        w = torch.tensor([0.0, 0.0], requires_grad=True)
        mod.train_step(w, x, 0.5)
        mod.train_step(w, x, 0.5)
        assert torch.allclose(w, torch.tensor([-2.0, -2.0])), (
            f"两步之后 w 应为 [-2,-2]，得到 {w.tolist()}。"
            "多半是上一轮的梯度没清，第二轮用了叠加后的 [4,4]"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT5 是纸面题，人工检查。")
        return 1

    print("T1-T4 全部通过。")
    print("T5 是纸面题，等我看你标的三处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
