"""B19 验收断言。用法: just check b19

读取 work/b19/b19.py 并验证 exercises/b19/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。

全部是小张量，一秒内跑完。
"""

import importlib.util
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b19" / "b19.py"

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
        print("先照着 exercises/b19/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b19_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 build_vocab 与 to_indices")
    def _t1a():
        text = "abcaabbc"
        c2i, i2c = mod.build_vocab(text)
        assert set(c2i.keys()) == {"a", "b", "c"}, (
            f"词表应只含 a、b、c，得到 {sorted(c2i.keys())}"
        )
        assert len(c2i) == len(i2c) == 3, f"词表大小应为 3，得到 {len(c2i)}"
        for ch, i in c2i.items():
            assert i2c[i] == ch, f"两个映射不互逆：{ch} -> {i} -> {i2c[i]}"
        idx = mod.to_indices(text, c2i)
        assert len(idx) == len(text), f"索引个数应等于文本长度 {len(text)}，得到 {len(idx)}"
        assert "".join(i2c[i] for i in idx) == text, "转回字符应当得到原文"

    @case("T1 词表顺序固定")
    def _t1b():
        c2i1, _ = mod.build_vocab("bcabca")
        c2i2, _ = mod.build_vocab("bcabca")
        assert c2i1 == c2i2, "两次建词表的映射应当一致（用 sorted 而不是裸 set）"
        assert c2i1 == {"a": 0, "b": 1, "c": 2}, (
            f"按字符排序后 a、b、c 应是 0、1、2，得到 {c2i1}"
        )

    @case("T2 随机采样：形状与右移")
    def _t2a():
        torch.manual_seed(0)
        random.seed(0)
        seq = list(range(30))
        batches = list(mod.data_iter_random(seq, batch_size=2, num_steps=6))
        assert len(batches) == 2, (
            f"30 个元素、num_steps=6、batch=2 时应有 (29//6)//2 = 2 个批量，"
            f"得到 {len(batches)}"
        )
        for X, Y in batches:
            assert tuple(X.shape) == (2, 6), f"X 形状应为 (2,6)，得到 {tuple(X.shape)}"
            assert tuple(Y.shape) == (2, 6), f"Y 形状应为 (2,6)，得到 {tuple(Y.shape)}"
            assert torch.equal(Y, X + 1), (
                f"标签应当是输入右移一位（对 0..29 的连续序列等于 X+1）。\n"
                f"X={X.tolist()}\nY={Y.tolist()}"
            )

    @case("T2 随机采样：起点是随机的")
    def _t2b():
        seq = list(range(100))
        random.seed(1)
        first = [X[0, 0].item() for X, _ in mod.data_iter_random(seq, 2, 5)]
        random.seed(2)
        second = [X[0, 0].item() for X, _ in mod.data_iter_random(seq, 2, 5)]
        assert first != second, (
            "两次调用（不同随机种子）的起点顺序应当不同。"
            "检查是不是忘了 random.shuffle"
        )

    @case("T3 相邻采样：形状与右移")
    def _t3a():
        seq = list(range(30))
        batches = list(mod.data_iter_consecutive(seq, batch_size=2, num_steps=6))
        assert len(batches) == 2, (
            f"30 个元素切 2 条各 15，num_steps=6 时应有 (15-1)//6 = 2 个批量，"
            f"得到 {len(batches)}"
        )
        for X, Y in batches:
            assert tuple(X.shape) == (2, 6), f"X 形状应为 (2,6)，得到 {tuple(X.shape)}"
            assert tuple(Y.shape) == (2, 6), f"Y 形状应为 (2,6)，得到 {tuple(Y.shape)}"
            assert torch.equal(Y, X + 1), (
                f"标签应当是输入右移一位。X={X.tolist()} Y={Y.tolist()}"
            )

    @case("T3 相邻采样：批量之间连续")
    def _t3b():
        seq = list(range(30))
        batches = list(mod.data_iter_consecutive(seq, batch_size=2, num_steps=6))
        X1 = batches[0][0]
        X2 = batches[1][0]
        # 第 0 条序列：0..14，批量 0 取 0..5，批量 1 接着取 6..11
        assert X2[0, 0].item() == X1[0, -1].item() + 1, (
            f"相邻批量的片段应当相接。第一批最后 {X1[0, -1].item()}，"
            f"第二批第一 {X2[0, 0].item()}"
        )
        # 第 1 条序列从 15 开始
        assert X1[1, 0].item() == 15, (
            f"第 1 条序列应从 15 开始（30 个元素切 2 条），得到 {X1[1, 0].item()}"
        )

    @case("T4 rnn_step 与手算一致")
    def _t4a():
        torch.manual_seed(0)
        d, h, n = 3, 4, 2
        X = torch.randn(n, d)
        H = torch.randn(n, h)
        W_xh = torch.randn(d, h)
        W_hh = torch.randn(h, h)
        b_h = torch.randn(h)
        got = mod.rnn_step(X, H, W_xh, W_hh, b_h)
        want = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        assert tuple(got.shape) == (n, h), (
            f"输出形状应为 ({n},{h})，得到 {tuple(got.shape)}"
        )
        assert torch.allclose(got, want, atol=1e-6), (
            f"与 tanh(X@W_xh + H@W_hh + b_h) 的最大差为 "
            f"{(got - want).abs().max().item():.6f}"
        )

    @case("T4 rnn_step 依赖上一步隐藏状态")
    def _t4b():
        torch.manual_seed(0)
        d, h, n = 2, 3, 1
        X = torch.randn(n, d)
        W_xh = torch.randn(d, h)
        W_hh = torch.randn(h, h)
        b_h = torch.zeros(h)
        H_a = torch.zeros(n, h)
        H_b = torch.ones(n, h)
        out_a = mod.rnn_step(X, H_a, W_xh, W_hh, b_h)
        out_b = mod.rnn_step(X, H_b, W_xh, W_hh, b_h)
        assert not torch.allclose(out_a, out_b), (
            "同样的输入、不同的上一步隐藏状态，输出应当不同——"
            "检查是不是漏了 H @ W_hh 这一项"
        )

    @case("T4 rnn_forward 与逐步循环一致")
    def _t4c():
        torch.manual_seed(0)
        d, h, n, T = 3, 4, 2, 5
        Xs = torch.randn(T, n, d)
        H0 = torch.zeros(n, h)
        W_xh = torch.randn(d, h)
        W_hh = torch.randn(h, h)
        b_h = torch.randn(h)

        got = mod.rnn_forward(Xs, H0, W_xh, W_hh, b_h)
        assert tuple(got.shape) == (T, n, h), (
            f"输出形状应为 (seq_len, batch, h) = ({T},{n},{h})，得到 {tuple(got.shape)}"
        )
        H = H0
        want = []
        for t in range(T):
            H = torch.tanh(Xs[t] @ W_xh + H @ W_hh + b_h)
            want.append(H.clone())
        want = torch.stack(want)
        assert torch.allclose(got, want, atol=1e-6), (
            f"与逐步循环的最大差为 {(got - want).abs().max().item():.6f}"
        )

    @case("T4 rnn_forward 与 nn.RNN 一致")
    def _t4d():
        torch.manual_seed(0)
        d, h, n, T = 3, 4, 2, 5
        Xs = torch.randn(T, n, d)
        H0 = torch.zeros(n, h)
        W_xh = torch.randn(d, h)
        W_hh = torch.randn(h, h)
        b_h = torch.randn(h)

        got = mod.rnn_forward(Xs, H0, W_xh, W_hh, b_h)

        rnn = nn.RNN(d, h, batch_first=True, nonlinearity="tanh")
        with torch.no_grad():
            rnn.weight_ih_l0.copy_(W_xh.T)     # nn.RNN 的权重是 (h, d)
            rnn.weight_hh_l0.copy_(W_hh.T)
            rnn.bias_ih_l0.copy_(b_h)
            rnn.bias_hh_l0.zero_()
            out, _ = rnn(Xs.transpose(0, 1), H0.unsqueeze(0))
        want = out.transpose(0, 1)
        assert torch.allclose(got, want, atol=1e-5), (
            f"与 nn.RNN 的最大差为 {(got - want).abs().max().item():.6f}。"
            "检查 tanh 有没有用、参数顺序对不对"
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
