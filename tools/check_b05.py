"""B05 验收断言。用法: just check b05

读取 work/b05/b05.py 并验证 exercises/b05/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# 默认查学习者的实现；传一个路径参数可以查别的文件（自测用）
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b05" / "b05.py"

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
        print("先照着 exercises/b05/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b05_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 掷骰子估计概率")
    def _t1():
        p = mod.roll_and_estimate(6000, seed=0)
        assert tuple(p.shape) == (6,), f"返回值形状应为 (6,)，得到 {tuple(p.shape)}"
        assert abs(p.sum().item() - 1.0) < 1e-6, (
            f"六个频率之和应为 1，得到 {p.sum().item()}"
        )
        assert ((p - 1 / 6).abs() < 0.03).all(), (
            f"6000 次之后每个面的频率应当都在 1/6 附近，得到 {p.tolist()}"
        )
        p2 = mod.roll_and_estimate(6000, seed=0)
        assert torch.allclose(p, p2), "同一个 seed 必须给出完全相同的结果"
        p3 = mod.roll_and_estimate(6000, seed=1)
        assert not torch.allclose(p, p3), "不同 seed 应当给出不同结果，检查 seed 是否真的传进了生成器"

    @case("T1 小样本波动更大")
    def _t1_small():
        small = mod.roll_and_estimate(60, seed=0)
        assert abs(small.sum().item() - 1.0) < 1e-6, "频率之和仍应为 1"
        big = mod.roll_and_estimate(60000, seed=0)
        dev_small = (small - 1 / 6).abs().max().item()
        dev_big = (big - 1 / 6).abs().max().item()
        assert dev_small > dev_big, (
            f"60 次的最大偏差（{dev_small:.4f}）应当大于 60000 次的（{dev_big:.4f}），"
            "这是大数定律的直接体现"
        )

    @case("T2 期望与方差")
    def _t2():
        m, v = mod.mean_var(torch.arange(1.0, 7.0), torch.full((6,), 1 / 6))
        assert abs(m.item() - 3.5) < 1e-5, f"骰子的期望应为 3.5，得到 {m.item()}"
        assert abs(v.item() - 35 / 12) < 1e-4, (
            f"骰子的方差应为 35/12 ≈ 2.9167，得到 {v.item()}"
        )
        m, v = mod.mean_var(torch.tensor([0.0, 1.0]), torch.tensor([0.3, 0.7]))
        assert abs(m.item() - 0.7) < 1e-6, f"期望应为 0.7，得到 {m.item()}"
        assert abs(v.item() - 0.21) < 1e-6, f"方差应为 0.21，得到 {v.item()}"
        m, v = mod.mean_var(torch.tensor([2.0, 2.0, 2.0]), torch.tensor([0.2, 0.3, 0.5]))
        assert abs(m.item() - 2.0) < 1e-6, f"常量随机变量的期望应为 2，得到 {m.item()}"
        assert abs(v.item()) < 1e-9, f"常量随机变量的方差应为 0，得到 {v.item()}"

    @case("T3 贝叶斯公式")
    def _t3():
        got = mod.posterior(1.0, 0.01, 0.0015)
        assert isinstance(got, float), f"应当返回 Python 浮点数，得到 {type(got).__name__}"
        assert abs(got - 0.1306) < 1e-3, (
            f"HIV 例子第一次检测的后验应约 0.1306，得到 {got:.6f}"
        )
        got2 = mod.posterior(1.0, 0.01, 0.5)
        assert abs(got2 - 0.9901) < 1e-3, (
            f"先验取一半时后验应约 0.9901，得到 {got2:.6f}"
        )
        got3 = mod.posterior(1.0, 0.0, 0.0015)
        assert abs(got3 - 1.0) < 1e-9, f"假阳性率为 0 时后验应为 1，得到 {got3}"

    @case("T4 累积均值")
    def _t4():
        c = mod.cumulative_means(1000, seed=0)
        assert tuple(c.shape) == (1000,), f"返回形状应为 (1000,)，得到 {tuple(c.shape)}"
        assert 1.0 <= c[0].item() <= 6.0, (
            f"第一个值是第一次掷的结果，应在 1 到 6 之间，得到 {c[0].item()}"
        )
        assert abs(c[-1].item() - 3.5) < 0.3, (
            f"1000 次的平均应接近 3.5，得到 {c[-1].item():.4f}"
        )
        tail_std = c[-100:].std().item()
        assert tail_std < 0.2, (
            f"序列后段应当稳定下来（标准差小于 0.2），得到 {tail_std:.4f}"
        )
        c1 = mod.cumulative_means(1, seed=0)
        assert tuple(c1.shape) == (1,), f"n_max=1 时形状应为 (1,)，得到 {tuple(c1.shape)}"

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
