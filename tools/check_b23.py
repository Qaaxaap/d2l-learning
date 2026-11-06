"""B23 验收断言。用法: just check b23

读取 work/b23/b23.py 并验证 exercises/b23/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b23" / "b23.py"

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
        print("先照着 exercises/b23/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b23_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 masked_softmax：不掩蔽")
    def _t1a():
        torch.manual_seed(0)
        X = torch.rand(2, 2, 4)
        out = mod.masked_softmax(X, None)
        want = F.softmax(X, dim=-1)
        assert torch.allclose(out, want, atol=1e-6), (
            "valid_lens 为 None 时应当等价于普通 softmax"
        )

    @case("T1 masked_softmax：一维 valid_lens")
    def _t1b():
        torch.manual_seed(0)
        X = torch.rand(2, 2, 4)
        out = mod.masked_softmax(X, torch.tensor([2, 3]))
        assert tuple(out.shape) == (2, 2, 4), f"形状应不变，得到 {tuple(out.shape)}"
        # 第 0 个样本所有 query 的有效长度都是 2
        assert torch.allclose(out[0, :, 2:], torch.zeros(2, 2), atol=1e-6), (
            f"第 0 个样本第 2 位之后应当被掩蔽成 0，得到 {out[0].tolist()}"
        )
        assert torch.allclose(out[0, :, :2].sum(-1), torch.ones(2), atol=1e-6), (
            "有效位置上的权重和应当是 1"
        )
        # 第 1 个样本是 3
        assert torch.allclose(out[1, :, 3:], torch.zeros(2, 1), atol=1e-6), (
            "第 1 个样本第 3 位之后应当被掩蔽"
        )
        assert torch.allclose(out[1, :, :3].sum(-1), torch.ones(2), atol=1e-6), (
            "有效位置上的权重和应当是 1"
        )

    @case("T1 masked_softmax：二维 valid_lens")
    def _t1c():
        torch.manual_seed(0)
        X = torch.rand(2, 2, 4)
        out = mod.masked_softmax(X, torch.tensor([[1, 3], [2, 4]]))
        assert torch.allclose(out[0, 0, 1:], torch.zeros(3), atol=1e-6), (
            f"第 0 个样本第 0 个 query 有效长度 1，之后应为 0，得到 {out[0, 0].tolist()}"
        )
        assert abs(out[0, 1, 3].item()) < 1e-6, "第 0 个样本第 1 个 query 有效长度 3"
        assert torch.allclose(out[1, 0, 2:], torch.zeros(2), atol=1e-6), (
            "第 1 个样本第 0 个 query 有效长度 2"
        )

    @case("T1 masked_softmax：不用 -inf")
    def _t1d():
        torch.manual_seed(0)
        X = torch.randn(1, 1, 4, requires_grad=True)
        out = mod.masked_softmax(X, torch.tensor([2]))
        out.sum().backward()
        assert torch.isfinite(X.grad).all(), (
            "梯度里出现 nan 或 inf。掩蔽值应当用有限的 -1e6，不能用 -inf"
        )

    @case("T2 加性注意力")
    def _t2a():
        torch.manual_seed(0)
        b, nq, nk, q, k, v, h = 2, 3, 4, 5, 6, 7, 8
        attn = mod.AdditiveAttention(key_size=k, query_size=q, num_hiddens=h)
        queries = torch.randn(b, nq, q)
        keys = torch.randn(b, nk, k)
        values = torch.randn(b, nk, v)
        out = attn(queries, keys, values)
        assert tuple(out.shape) == (b, nq, v), (
            f"输出形状应为 (batch, n_q, v) = ({b},{nq},{v})，得到 {tuple(out.shape)}"
        )
        w = attn.attention_weights
        assert tuple(w.shape) == (b, nq, nk), (
            f"注意力权重形状应为 ({b},{nq},{nk})，得到 {tuple(w.shape)}"
        )
        assert torch.allclose(w.sum(-1), torch.ones(b, nq), atol=1e-5), (
            "每个 query 的权重和应当是 1"
        )

    @case("T2 加性注意力：与手算一致")
    def _t2b():
        torch.manual_seed(0)
        b, nq, nk, q, k, v, h = 1, 2, 3, 4, 4, 5, 3
        attn = mod.AdditiveAttention(key_size=k, query_size=q, num_hiddens=h)
        queries = torch.randn(b, nq, q)
        keys = torch.randn(b, nk, k)
        values = torch.randn(b, nk, v)
        out = attn(queries, keys, values)

        with torch.no_grad():
            qq = attn.W_q(queries)
            kk = attn.W_k(keys)
            feats = torch.tanh(qq.unsqueeze(2) + kk.unsqueeze(1))
            scores = attn.w_v(feats).squeeze(-1)
            w = F.softmax(scores, dim=-1)
            want = torch.bmm(w, values)
        assert torch.allclose(out, want, atol=1e-5), (
            f"与手算的最大差为 {(out - want).abs().max().item():.6f}"
        )

    @case("T3 缩放点积注意力")
    def _t3a():
        torch.manual_seed(0)
        b, nq, nk, d, v = 2, 3, 4, 5, 6
        attn = mod.DotProductAttention()
        queries = torch.randn(b, nq, d)
        keys = torch.randn(b, nk, d)
        values = torch.randn(b, nk, v)
        out = attn(queries, keys, values)
        assert tuple(out.shape) == (b, nq, v), (
            f"输出形状应为 ({b},{nq},{v})，得到 {tuple(out.shape)}"
        )
        with torch.no_grad():
            scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
            w = F.softmax(scores, dim=-1)
            want = torch.bmm(w, values)
        assert torch.allclose(out, want, atol=1e-5), (
            f"与手算的最大差为 {(out - want).abs().max().item():.6f}。"
            "检查有没有除以 sqrt(d)"
        )

    @case("T3 缩放的作用")
    def _t3b():
        torch.manual_seed(0)
        b, n, d = 1, 2, 2
        # 把 d 设大，看权重是否仍然平滑（不接近 one-hot）
        d = 64
        attn = mod.DotProductAttention()
        queries = torch.randn(b, n, d) * 1.0
        keys = torch.randn(b, n, d) * 1.0
        values = torch.randn(b, n, 1)
        attn(queries, keys, values)
        w = attn.attention_weights
        assert w.max().item() < 0.99, (
            f"d={d} 时权重最大值 {w.max().item():.4f} 太接近 1。"
            "检查是否忘了除以 sqrt(d)"
        )

    @case("T4 Nadaraya-Watson：形状与手算")
    def _t4a():
        torch.manual_seed(0)
        b, nq, nk = 1, 3, 5
        queries = torch.randn(b, nq)
        keys = torch.randn(b, nk)
        values = torch.randn(b, nk)
        out, weights = mod.nadaraya_watson(queries, keys, values, w=1.0)
        assert tuple(out.shape) == (b, nq), (
            f"输出形状应为 ({b},{nq})，得到 {tuple(out.shape)}"
        )
        dist = (queries.unsqueeze(2) - keys.unsqueeze(1)) ** 2
        want_w = F.softmax(-0.5 * dist, dim=-1)
        want = (want_w * values.unsqueeze(1)).sum(-1)
        assert torch.allclose(weights, want_w, atol=1e-5), "权重与手算不一致"
        assert torch.allclose(out, want, atol=1e-5), "输出与手算不一致"

    @case("T4 w 变大时更关注最近的点")
    def _t4b():
        queries = torch.tensor([[1.0]])
        keys = torch.tensor([[0.0, 1.0, 2.0]])
        values = torch.tensor([[0.0, 10.0, 20.0]])
        _, w_small = mod.nadaraya_watson(queries, keys, values, w=1.0)
        _, w_big = mod.nadaraya_watson(queries, keys, values, w=10.0)
        # w 大时权重集中到与查询重合的那个位置（下标 1）
        assert w_big[0, 0, 1].item() > 0.99, (
            f"w=10 时中间那个点的权重应当接近 1，得到 {w_big[0, 0].tolist()}"
        )
        assert w_small[0, 0, 1].item() < w_big[0, 0, 1].item(), (
            "w 变大时权重应当更集中"
        )
        # w 很小时趋近均匀
        _, w_flat = mod.nadaraya_watson(queries, keys, values, w=0.0)
        assert torch.allclose(w_flat, torch.full_like(w_flat, 1 / 3), atol=1e-5), (
            f"w=0 时权重应当均匀（每个 1/3），得到 {w_flat[0, 0].tolist()}"
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
