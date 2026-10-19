"""B11 验收断言。用法: just check b11

读取 work/b11/b11.py 并验证 exercises/b11/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。

T1 的梯度用 autograd 算一份参考值——不是为了让学习者调 autograd，
而是断言需要一个独立于他实现的对照。
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b11" / "b11.py"

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
        print("先照着 exercises/b11/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b11_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(seed: int = 0):
    """造一组固定的输入与参数。"""
    g = torch.Generator().manual_seed(seed)
    d, h, q = 5, 4, 3
    return {
        "x": torch.randn(d, 1, generator=g),
        "W1": torch.randn(h, d, generator=g),
        "b1": torch.randn(h, 1, generator=g),
        "W2": torch.randn(q, h, generator=g),
        "b2": torch.randn(q, 1, generator=g),
        "y": torch.randn(q, 1, generator=g),
    }


def reference(name: str, params: dict, lam: float):
    """用 autograd 算一份参考值，仅供断言对照。

    只对四个参数求梯度——x 与 y 是数据，不参与求导。
    """
    vals = {k: v.clone().requires_grad_() for k, v in params.items() if k not in ("x", "y")}
    x, y = params["x"], params["y"]
    z = vals["W1"] @ x + vals["b1"]
    h = torch.relu(z)
    o = vals["W2"] @ h + vals["b2"]
    loss = 0.5 * ((o - y) ** 2).sum()
    reg = lam / 2 * ((vals["W1"] ** 2).sum() + (vals["W2"] ** 2).sum())
    J = loss + reg
    J.backward()
    return J.detach(), {k: v.grad.clone() for k, v in vals.items()}


def main() -> int:
    mod = load_user_module()

    @case("T1 手写反向传播：目标函数与四个梯度")
    def _t1a():
        params = make_case()
        J_ref, grads_ref = reference("t1", params, lam=0.1)
        J, grads = mod.forward_backward(
            params["x"], params["W1"], params["b1"], params["W2"], params["b2"],
            params["y"], lam=0.1,
        )
        assert isinstance(grads, dict), f"第二个返回值应是字典，得到 {type(grads).__name__}"
        assert torch.allclose(J, J_ref), f"目标函数应为 {J_ref.item():.6f}，得到 {J.item():.6f}"
        for key in ("W1", "b1", "W2", "b2"):
            assert key in grads, f"字典缺 {key} 键"
            got = grads[key]
            want = grads_ref[key]
            assert tuple(got.shape) == tuple(want.shape), (
                f"{key} 的梯度形状应为 {tuple(want.shape)}，得到 {tuple(got.shape)}。"
                "形状不对通常是转置或 keepdim 的问题"
            )
            assert torch.allclose(got, want, atol=1e-6), (
                f"{key} 的梯度与参考值不符。\n"
                f"  你的: {got.flatten()[:4].tolist()}\n"
                f"  参考: {want.flatten()[:4].tolist()}"
            )

    @case("T1 用了 autograd 吗")
    def _t1b():
        src = USER_FILE.read_text(encoding="utf-8")
        # 只检查 forward_backward 这一个函数体，别处可以有 autograd
        start = src.find("def forward_backward")
        assert start != -1, "找不到 forward_backward 的定义"
        end = src.find("\ndef ", start + 1)
        body = src[start:end if end != -1 else len(src)]
        for bad in (".backward(", "requires_grad", "torch.autograd"):
            assert bad not in body, (
                f"forward_backward 里出现了 {bad!r}。这一题要手推链式法则，"
                "不能用 autograd"
            )

    @case("T1 正则项只进权重")
    def _t1c():
        params = make_case()
        x, W1, b1, W2, b2, y = (
            params["x"], params["W1"], params["b1"], params["W2"], params["b2"], params["y"],
        )
        _, g_no = mod.forward_backward(x, W1, b1, W2, b2, y, lam=0.0)
        _, g_reg = mod.forward_backward(x, W1, b1, W2, b2, y, lam=0.5)
        for key in ("W1", "W2"):
            diff = g_reg[key] - g_no[key]
            want = 0.5 * params[key]
            assert torch.allclose(diff, want, atol=1e-6), (
                f"lam 从 0 变到 0.5 时，{key} 的梯度应多出 0.5*{key}，实际不符"
            )
        for key in ("b1", "b2"):
            assert torch.allclose(g_reg[key], g_no[key], atol=1e-6), (
                f"{key} 是偏置，正则项不该作用到它，但梯度变了"
            )

    @case("T2 数值梯度")
    def _t2():
        x = torch.tensor([[1.0], [2.0], [3.0]])
        got = mod.numeric_grad(lambda t: (t**2).sum(), x)
        want = torch.tensor([[2.0], [4.0], [6.0]])
        assert tuple(got.shape) == tuple(x.shape), f"形状应与输入相同，得到 {tuple(got.shape)}"
        assert torch.allclose(got, want, atol=1e-4), f"应为 {want.flatten().tolist()}，得到 {got.flatten().tolist()}"
        ones = mod.numeric_grad(lambda t: t.sum(), x)
        assert torch.allclose(ones, torch.ones_like(x), atol=1e-4), "f=sum 的梯度应全 1"
        assert torch.allclose(x, torch.tensor([[1.0], [2.0], [3.0]])), "不应修改传入的 x"

    @case("T3 梯度检查")
    def _t3():
        a = torch.randn(4, 3)
        assert mod.grad_check(a, a.clone()) is True, "完全相同的两组梯度应返回 True"
        assert mod.grad_check(a, a + 1e-6) is True, "差 1e-6 量级应返回 True"
        assert mod.grad_check(a, a * 1.01) is False, "相差 1% 应返回 False"
        z = torch.zeros(3)
        assert mod.grad_check(z, z.clone()) is True, "两组全零应返回 True"

    @case("T4 Xavier 与 He 初始化")
    def _t4():
        w = mod.xavier_uniform(100, 50)
        assert tuple(w.shape) == (100, 50), f"形状应为 (100, 50)，得到 {tuple(w.shape)}"
        bound = (6 / 150) ** 0.5
        assert w.abs().max().item() <= bound + 1e-6, (
            f"Xavier 的取值范围应在 ±{bound:.4f} 以内，得到最大值 {w.abs().max().item():.4f}"
        )
        assert w.abs().max().item() > bound * 0.9, (
            "取值范围远小于上界，检查是不是用了别的分布"
        )
        w2 = mod.xavier_uniform(100, 50)
        assert torch.allclose(w, w2), "同一个 seed 应给出相同结果"

        h = mod.he_uniform(100, 50)
        hb = (6 / 100) ** 0.5
        assert h.abs().max().item() <= hb + 1e-6, (
            f"He 的取值范围应在 ±{hb:.4f} 以内，得到最大值 {h.abs().max().item():.4f}"
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
