"""B20 验收断言。用法: just check b20

读取 work/b20/b20.py 并验证 exercises/b20/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。

T4 训练一个小语料 10 个 epoch，几秒钟完成。
"""

import importlib.util
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b20" / "b20.py"

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
        print("先照着 exercises/b20/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b20_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @case("T1 one-hot")
    def _t1a():
        X = torch.tensor([[0, 1, 2], [2, 1, 0]])
        out = mod.to_onehot(X, 3)
        assert isinstance(out, list) or isinstance(out, tuple), (
            f"应返回列表（每个时间步一个张量），得到 {type(out).__name__}"
        )
        assert len(out) == 3, f"num_steps=3 时应返回 3 个张量，得到 {len(out)}"
        for t, step in enumerate(out):
            assert tuple(step.shape) == (2, 3), (
                f"第 {t} 步形状应为 (batch, vocab) = (2,3)，得到 {tuple(step.shape)}"
            )
            assert torch.allclose(step.sum(dim=1), torch.ones(2)), (
                "每个 one-hot 向量的元素和应当是 1"
            )
        # 第 0 步：第一行索引 0，第二行索引 2
        assert out[0][0, 0].item() == 1.0 and out[0][1, 2].item() == 1.0, (
            f"第 0 步应当把索引 0 与 2 置 1，得到 {out[0].tolist()}"
        )

    @case("T2 init_rnn_state 与 rnn_forward")
    def _t2a():
        torch.manual_seed(0)
        vocab, h, batch, T = 5, 4, 3, 2
        params = (
            torch.randn(vocab, h, device=device), torch.randn(h, h, device=device),
            torch.randn(h, device=device), torch.randn(h, vocab, device=device),
            torch.randn(vocab, device=device),
        )
        state = mod.init_rnn_state(batch, h, device)
        assert isinstance(state, tuple) and len(state) == 1, (
            f"状态应当是含一个张量的元组，得到 {type(state).__name__}"
        )
        assert tuple(state[0].shape) == (batch, h), (
            f"初始状态形状应为 ({batch},{h})，得到 {tuple(state[0].shape)}"
        )
        assert state[0].abs().sum().item() == 0, "初始隐藏状态应当全 0"

        X = torch.randint(0, vocab, (batch, T))
        inputs = [torch.nn.functional.one_hot(X[:, t], vocab).float().to(device)
                  for t in range(T)]
        outputs, new_state = mod.rnn_forward(inputs, state, params)

        W_xh, W_hh, b_h, W_hq, b_q = params
        H = state[0]
        want_out, want_hs = [], []
        for t in range(T):
            H = torch.tanh(inputs[t] @ W_xh + H @ W_hh + b_h)
            want_out.append(H @ W_hq + b_q)
            want_hs.append(H.clone())
        assert len(outputs) == T, f"应当有 {T} 个输出，得到 {len(outputs)}"
        for t in range(T):
            assert tuple(outputs[t].shape) == (batch, vocab), (
                f"第 {t} 步输出形状应为 (batch, vocab)，得到 {tuple(outputs[t].shape)}"
            )
            assert torch.allclose(outputs[t], want_out[t], atol=1e-6), (
                f"第 {t} 步输出与手算的最大差为 "
                f"{(outputs[t] - want_out[t]).abs().max().item():.6f}"
            )
        assert torch.allclose(new_state[0], want_hs[-1], atol=1e-6), (
            "返回的最终隐藏状态应当是最后一个时间步的 H"
        )

    @case("T3 梯度裁剪：超阈值")
    def _t3a():
        p1 = nn.Parameter(torch.zeros(3))
        p2 = nn.Parameter(torch.zeros(2))
        p1.grad = torch.tensor([3.0, 4.0, 0.0])      # 范数 5
        p2.grad = torch.tensor([0.0, 0.0])
        theta = 1.0
        before = mod.grad_clipping([p1, p2], theta)
        norm = torch.sqrt((p1.grad ** 2).sum() + (p2.grad ** 2).sum())
        assert abs(norm.item() - theta) < 1e-5, (
            f"裁剪后梯度范数应当等于 {theta}，得到 {norm.item():.6f}"
        )
        # 方向不能变：p1 原来的方向是 (3,4,0)/5
        want_dir = torch.tensor([3.0, 4.0, 0.0]) / 5
        got_dir = p1.grad / p1.grad.norm()
        assert torch.allclose(got_dir, want_dir, atol=1e-6), (
            f"裁剪应当只改长度不改方向。原方向 {want_dir.tolist()}，"
            f"得到 {got_dir.tolist()}"
        )
        assert abs(before - 5.0) < 1e-5, (
            f"应当返回裁剪前的范数 5.0，得到 {before}"
        )
        # 参数本身不能被改
        assert p1.abs().sum().item() == 0, (
            "裁剪改的是梯度，不是参数值——检查是不是写成了 p.data *= ..."
        )

    @case("T3 梯度裁剪：不超阈值不变")
    def _t3b():
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([0.3, 0.4])            # 范数 0.5 < 1.0
        mod.grad_clipping([p], 1.0)
        assert torch.allclose(p.grad, torch.tensor([0.3, 0.4])), (
            f"范数未超阈值时梯度不该变，得到 {p.grad.tolist()}"
        )

    @case("T4 训练：困惑度下降")
    def _t4():
        torch.manual_seed(0)
        random.seed(0)
        corpus = "abcd" * 200
        c2i = {ch: i for i, ch in enumerate(sorted(set(corpus)))}
        idx = [c2i[ch] for ch in corpus]
        ppl = mod.train_rnn(idx, 4, 32, 10, 0.5, 8, 8, device, use_random=True)

        assert isinstance(ppl, (list, tuple)), (
            f"应返回每个 epoch 的困惑度列表，得到 {type(ppl).__name__}"
        )
        assert len(ppl) == 10, f"10 个 epoch 应返回 10 个数，得到 {len(ppl)}"
        for v in ppl:
            assert v > 0, f"困惑度应当为正，得到 {v}"
        assert ppl[0] > 1.5, (
            f"初始困惑度应当接近词表大小 4（瞎猜水平），得到 {ppl[0]:.3f}。"
            "如果一开始就很小，可能是在用标签当输入（数据泄漏）"
        )
        assert ppl[-1] < 1.5, (
            f"10 个 epoch 后困惑度应当明显低于基线 4，得到 {ppl[-1]:.3f}。"
            "检查：标签是否右移、是否裁剪了梯度、损失函数是否传对了 logits"
        )

    @case("T4 训练：相邻采样也能跑")
    def _t4b():
        torch.manual_seed(0)
        random.seed(0)
        corpus = "abcd" * 200
        c2i = {ch: i for i, ch in enumerate(sorted(set(corpus)))}
        idx = [c2i[ch] for ch in corpus]
        ppl = mod.train_rnn(idx, 4, 32, 10, 0.5, 8, 8, device, use_random=False)
        assert len(ppl) == 10, f"应返回 10 个数，得到 {len(ppl)}"
        assert ppl[-1] < 1.5, (
            f"相邻采样 10 epoch 后困惑度应当明显下降，得到 {ppl[-1]:.3f}。"
            "检查隐藏状态有没有在每次读新批前 detach"
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
