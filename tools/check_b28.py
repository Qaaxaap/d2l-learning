"""B28 验收断言。用法: just check b28"""

import importlib.util
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b28" / "b28.py"

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
        print("先照着 exercises/b28/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b28_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 单对 log 概率与手算一致")
    def _t1a():
        torch.manual_seed(0)
        V, d = 5, 3
        vecs = torch.randn(V, d)
        c = torch.randn(d)
        got = mod.skipgram_log_prob(c, 2, vecs)
        scores = vecs @ c
        want = float(scores[2] - torch.logsumexp(scores, dim=0))
        assert abs(got - want) < 1e-5, (
            f"应为 {want:.6f}，得到 {got:.6f}。"
            "log P = u_o·v_c - logsumexp(所有 u_i·v_c)"
        )

    @case("T1 log 概率都是负数")
    def _t1b():
        torch.manual_seed(0)
        V, d = 6, 4
        vecs = torch.randn(V, d)
        c = torch.randn(d)
        for idx in range(V):
            v = mod.skipgram_log_prob(c, idx, vecs)
            assert v <= 0, f"概率不超过 1，log 概率应为负数，下标 {idx} 得到 {v}"

    @case("T1 一批损失等于逐个的平均")
    def _t1c():
        torch.manual_seed(0)
        V, d, n = 5, 3, 4
        vecs = torch.randn(V, d)
        cs = torch.randn(n, d)
        idx = torch.tensor([0, 1, 2, 3])
        got = mod.skipgram_loss(cs, idx, vecs)
        singles = [mod.skipgram_log_prob(cs[i], int(idx[i]), vecs) for i in range(n)]
        want = -sum(singles) / n
        assert abs(got - want) < 1e-5, (
            f"一批损失应为 {-want:.6f} 平均后的 {want:.6f}，得到 {got:.6f}"
        )

    @case("T2 负采样损失：正样本内积大时更小")
    def _t2():
        torch.manual_seed(0)
        d, n, K = 4, 3, 2
        ce = torch.randn(n, d)
        # 正样本用与中心词同向的向量，损失应当较小
        po_same = ce.clone()
        ne_opp = -ce.unsqueeze(1).expand(n, K, d).clone()
        l_good = mod.negative_sampling_loss(ce, po_same, ne_opp)
        po_opp = -ce.clone()
        ne_same = ce.unsqueeze(1).expand(n, K, d).clone()
        l_bad = mod.negative_sampling_loss(ce, po_opp, ne_same)
        assert l_good < l_bad, (
            f"正样本同向、负样本反向时损失应更小，得到 {l_good:.4f} 与 {l_bad:.4f}"
        )

    @case("T2 负采样损失：与手算一致")
    def _t2b():
        torch.manual_seed(0)
        d, n, K = 3, 2, 2
        ce = torch.randn(n, d)
        po = torch.randn(n, d)
        ne = torch.randn(n, K, d)
        got = mod.negative_sampling_loss(ce, po, ne)
        want = 0.0
        for i in range(n):
            pos = float((ce[i] * po[i]).sum())
            s = -math.log(1 / (1 + math.exp(-pos)))
            for k in range(K):
                neg = float((ce[i] * ne[i, k]).sum())
                s -= math.log(1 / (1 + math.exp(neg)))
            want += s
        want /= n
        assert abs(got - want) < 1e-4, f"应为 {want:.6f}，得到 {got:.6f}"

    @case("T3 共现矩阵：小语料")
    def _t3a():
        # 语料 0 1 2 1 0，窗口 1。逐个位置数一遍：
        # t=0(c=0) 只见 1；t=1(c=1) 见 0 与 2；t=2(c=2) 见 1 与 1；
        # t=3(c=1) 见 2 与 0；t=4(c=0) 只见 1。
        X = mod.cooccurrence([0, 1, 2, 1, 0], vocab_size=3, window=1)
        want = torch.tensor([[0.0, 2.0, 0.0],
                             [2.0, 0.0, 2.0],
                             [0.0, 2.0, 0.0]])
        assert torch.allclose(X, want), f"共现矩阵应为\n{want}\n得到\n{X}"

    @case("T3 共现矩阵：对称")
    def _t3b():
        X = mod.cooccurrence([0, 1, 2, 1, 0], vocab_size=3, window=1)
        assert torch.allclose(X, X.T), (
            "窗口是双向的，x[i,j] 应等于 x[j,i]。"
            "中心词 i 的窗口里有 j，等价于 j 的窗口里有 i"
        )

    @case("T4 余弦相似度")
    def _t4():
        a = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
        b = torch.tensor([[2.0, 0.0], [1.0, -1.0], [0.0, 3.0]])
        got = mod.cosine_similarity(a, b)
        want = torch.tensor([1.0, 0.0, 1.0])
        assert torch.allclose(got, want, atol=1e-5), (
            f"应为 {want.tolist()}，得到 {got.tolist()}"
        )

    @case("T4 余弦与长度无关")
    def _t4b():
        a = torch.tensor([[1.0, 2.0]])
        b1 = torch.tensor([[1.0, 2.0]])
        b2 = torch.tensor([[100.0, 200.0]])
        v1 = mod.cosine_similarity(a, b1).item()
        v2 = mod.cosine_similarity(a, b2).item()
        assert abs(v1 - 1.0) < 1e-5 and abs(v2 - 1.0) < 1e-5, (
            f"方向相同时余弦应都是 1，得到 {v1} 与 {v2}"
        )

    @case("T5 类比排除输入词")
    def _t5():
        torch.manual_seed(0)
        V, d = 8, 4
        vecs = torch.randn(V, d)
        got = mod.analogy(0, 1, 2, vecs)
        assert got not in (0, 1, 2), f"结果不该是输入词之一，得到 {got}"
        assert 0 <= got < V, f"结果应在词表范围内，得到 {got}"

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
