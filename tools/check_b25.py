"""B25 验收断言。用法: just check b25

读取 work/b25/b25.py 并验证 exercises/b25/code.md 里的 T1-T4。
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b25" / "b25.py"

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
        print("先照着 exercises/b25/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b25_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grad_seq(step: int, size: int) -> torch.Tensor:
    """固定的梯度序列：每步不同，能暴露累积与修正的差别。"""
    g = torch.ones(size) * (step + 1)
    g[::2] *= -1
    return g


def main() -> int:
    mod = load_user_module()

    @case("T1 动量：与 torch.optim.SGD 一致")
    def _t1a():
        torch.manual_seed(0)
        size = 5
        p_ref = torch.randn(size, requires_grad=True)
        p_mine = p_ref.detach().clone()
        vel = [torch.zeros(size)]
        opt = torch.optim.SGD([p_ref], lr=0.1, momentum=0.9)
        for step in range(10):
            g = grad_seq(step, size)
            p_ref.grad = g.clone()
            opt.step()
            opt.zero_grad()
            mod.sgd_momentum_step([p_mine], [g], vel, 0.1, 0.9)
        assert torch.allclose(p_ref.detach(), p_mine, atol=1e-5), (
            f"10 步后与 torch.optim.SGD(momentum=0.9) 的最大差为 "
            f"{(p_ref.detach() - p_mine).abs().max().item():.6f}。"
            "注意 torch 的约定是 buf = momentum*buf + g、p -= lr*buf"
        )

    @case("T1 动量：momentum=0 时等于普通 SGD")
    def _t1b():
        torch.manual_seed(0)
        size = 4
        p_ref = torch.randn(size, requires_grad=True)
        p_mine = p_ref.detach().clone()
        vel = [torch.zeros(size)]
        opt = torch.optim.SGD([p_ref], lr=0.2, momentum=0.0)
        for step in range(5):
            g = grad_seq(step, size)
            p_ref.grad = g.clone()
            opt.step()
            opt.zero_grad()
            mod.sgd_momentum_step([p_mine], [g], vel, 0.2, 0.0)
        assert torch.allclose(p_ref.detach(), p_mine, atol=1e-6), (
            "momentum=0 时应当退化为普通 SGD"
        )

    @case("T2 AdaGrad：与 torch.optim.Adagrad 一致")
    def _t2():
        torch.manual_seed(0)
        size = 5
        p_ref = torch.randn(size, requires_grad=True)
        p_mine = p_ref.detach().clone()
        s = [torch.zeros(size)]
        opt = torch.optim.Adagrad([p_ref], lr=0.3, eps=1e-10)
        for step in range(10):
            g = grad_seq(step, size)
            p_ref.grad = g.clone()
            opt.step()
            opt.zero_grad()
            mod.adagrad_step([p_mine], [g], s, 0.3, eps=1e-10)
        assert torch.allclose(p_ref.detach(), p_mine, atol=1e-5), (
            f"10 步后与 torch.optim.Adagrad 的最大差为 "
            f"{(p_ref.detach() - p_mine).abs().max().item():.6f}"
        )

    @case("T2 AdaGrad：学习率随累加衰减")
    def _t3a():
        torch.manual_seed(0)
        size = 3
        p = torch.randn(size)
        s = [torch.zeros(size)]
        g = torch.ones(size)
        deltas = []
        for _ in range(5):
            before = p.clone()
            mod.adagrad_step([p], [g], s, 0.1, eps=1e-10)
            deltas.append((p - before).abs().mean().item())
        for i in range(1, 5):
            assert deltas[i] < deltas[i - 1], (
                f"同样的梯度下更新量应当逐步变小（学习率被 s 衰减），"
                f"得到 {[round(d, 6) for d in deltas]}"
            )

    @case("T3 RMSProp：与 torch.optim.RMSprop 一致")
    def _t3():
        torch.manual_seed(0)
        size = 5
        p_ref = torch.randn(size, requires_grad=True)
        p_mine = p_ref.detach().clone()
        s = [torch.zeros(size)]
        opt = torch.optim.RMSprop([p_ref], lr=0.05, alpha=0.9, eps=1e-10)
        for step in range(10):
            g = grad_seq(step, size)
            p_ref.grad = g.clone()
            opt.step()
            opt.zero_grad()
            mod.rmsprop_step([p_mine], [g], s, 0.05, gamma=0.9, eps=1e-10)
        assert torch.allclose(p_ref.detach(), p_mine, atol=1e-5), (
            f"10 步后与 torch.optim.RMSprop 的最大差为 "
            f"{(p_ref.detach() - p_mine).abs().max().item():.6f}。"
            "注意 torch 里衰减系数叫 alpha"
        )

    @case("T3 RMSProp：学习率不单调下降")
    def _t3b():
        torch.manual_seed(0)
        size = 3
        p = torch.randn(size)
        s = [torch.zeros(size)]
        g = torch.ones(size)
        deltas = []
        for _ in range(60):
            before = p.clone()
            mod.rmsprop_step([p], [g], s, 0.1, gamma=0.9, eps=1e-10)
            deltas.append((p - before).abs().mean().item())
        # s 从 0 爬到稳态 g²，更新量会先降后稳。关键是后段不再下降，
        # 不像 AdaGrad 那样一路降到 0。
        late = deltas[50:]
        assert max(late) - min(late) < 2e-3, (
            f"后 10 步的更新量应当趋于稳定（s 收敛到 g² 后不再变），"
            f"得到 {[round(d, 6) for d in late]}"
        )
        assert deltas[-1] > 0.05, (
            f"末步更新量不该趋近 0（稳态是 lr/sqrt(g²) = 0.1），得到 {deltas[-1]:.6f}。"
            "如果一路降到 0，说明 s 写成了累加而不是移动平均"
        )

    @case("T4 Adam：与 torch.optim.Adam 一致")
    def _t4a():
        torch.manual_seed(0)
        size = 5
        p_ref = torch.randn(size, requires_grad=True)
        p_mine = p_ref.detach().clone()
        v = [torch.zeros(size)]
        s = [torch.zeros(size)]
        opt = torch.optim.Adam([p_ref], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
        for step in range(1, 11):
            g = grad_seq(step, size)
            p_ref.grad = g.clone()
            opt.step()
            opt.zero_grad()
            mod.adam_step([p_mine], [g], v, s, 0.05, t=step)
        assert torch.allclose(p_ref.detach(), p_mine, atol=1e-5), (
            f"10 步后与 torch.optim.Adam 的最大差为 "
            f"{(p_ref.detach() - p_mine).abs().max().item():.6f}"
        )

    @case("T4 Adam：第一步的更新量约为 lr")
    def _t4b():
        torch.manual_seed(0)
        size = 4
        p = torch.randn(size)
        v = [torch.zeros(size)]
        s = [torch.zeros(size)]
        g = torch.tensor([2.0, -0.5, 10.0, 0.01])
        before = p.clone()
        mod.adam_step([p], [g], v, s, 0.1, t=1)
        delta = (p - before).abs()
        assert torch.allclose(delta, torch.full_like(delta, 0.1), atol=2e-3), (
            f"第一步的更新量应当约为 lr（偏差修正的效果），得到 {delta.tolist()}。"
            "不修的话更新量只有 0.1*lr"
        )

    @case("T4 Adam：状态形状与参数一致")
    def _t4c():
        torch.manual_seed(0)
        p = torch.randn(3, 4)
        v = [torch.zeros(3, 4)]
        s = [torch.zeros(3, 4)]
        mod.adam_step([p], [torch.randn(3, 4)], v, s, 0.1, t=1)
        assert tuple(v[0].shape) == (3, 4), f"v 形状应为 (3,4)，得到 {tuple(v[0].shape)}"
        assert tuple(s[0].shape) == (3, 4), f"s 形状应为 (3,4)，得到 {tuple(s[0].shape)}"
        assert v[0].abs().sum().item() > 0, "v 应当被更新了"
        assert s[0].abs().sum().item() > 0, "s 应当被更新了"

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
