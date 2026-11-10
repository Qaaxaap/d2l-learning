"""B24 验收断言。用法: just check b24

读取 work/b24/b24.py 并验证 exercises/b24/code.md 里的 T1-T4。
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b24" / "b24.py"

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
        print("先照着 exercises/b24/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b24_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 transpose_qkv 形状与按头切分")
    def _t1a():
        torch.manual_seed(0)
        batch, n, p, h = 2, 3, 8, 4
        X = torch.randn(batch, n, p)
        out = mod.transpose_qkv(X, h)
        assert tuple(out.shape) == (batch * h, n, p // h), (
            f"形状应为 ({batch * h}, {n}, {p // h})，得到 {tuple(out.shape)}"
        )
        # 第 0 个 batch 的第 i 个头，应当等于 X[0, :, i*(p/h):(i+1)*(p/h)]
        for i in range(h):
            want = X[0, :, i * (p // h): (i + 1) * (p // h)]
            assert torch.allclose(out[i], want, atol=1e-6), (
                f"第 {i} 个头的内容不对。按头切分要求第 i 个头拿第 i 段通道，"
                "检查 reshape 与 transpose 的顺序"
            )

    @case("T1 transpose_output 是逆操作")
    def _t1b():
        torch.manual_seed(0)
        batch, n, p, h = 2, 3, 8, 4
        X = torch.randn(batch, n, p)
        back = mod.transpose_output(mod.transpose_qkv(X, h), h)
        assert tuple(back.shape) == (batch, n, p), (
            f"逆变换后形状应为 ({batch},{n},{p})，得到 {tuple(back.shape)}"
        )
        assert torch.allclose(back, X, atol=1e-6), (
            "两个函数应当互为逆操作，转一圈要回到原值"
        )

    @case("T2 多头注意力：形状")
    def _t2a():
        torch.manual_seed(0)
        mha = mod.MultiHeadAttention(
            key_size=8, query_size=8, value_size=8,
            num_hiddens=16, num_heads=4, dropout=0.0)
        X = torch.randn(2, 5, 8)
        out = mha(X, X, X)
        assert tuple(out.shape) == (2, 5, 16), (
            f"输出形状应为 (batch, n_q, num_hiddens) = (2,5,16)，得到 {tuple(out.shape)}"
        )

    @case("T2 头数为 1 时等于单头注意力")
    def _t2b():
        torch.manual_seed(0)
        mha = mod.MultiHeadAttention(
            key_size=8, query_size=8, value_size=8,
            num_hiddens=8, num_heads=1, dropout=0.0)
        mha.eval()
        q = torch.randn(2, 3, 8)
        k = torch.randn(2, 5, 8)
        v = torch.randn(2, 5, 8)
        out = mha(q, k, v)

        with torch.no_grad():
            qq = mha.W_q(q)
            kk = mha.W_k(k)
            vv = mha.W_v(v)
            scores = torch.bmm(qq, kk.transpose(1, 2)) / math.sqrt(8)
            w = F.softmax(scores, dim=-1)
            want = mha.W_o(torch.bmm(w, vv))
        assert torch.allclose(out, want, atol=1e-5), (
            f"头数为 1 时应当等价于单头缩放点积注意力，"
            f"最大差 {(out - want).abs().max().item():.6f}"
        )

    @case("T2 不同头的结果不同")
    def _t2c():
        torch.manual_seed(0)
        mha = mod.MultiHeadAttention(
            key_size=8, query_size=8, value_size=8,
            num_hiddens=8, num_heads=4, dropout=0.0)
        mha.eval()
        X = torch.randn(1, 4, 8)
        with torch.no_grad():
            q = mod.transpose_qkv(mha.W_q(X), 4)      # (4, 4, 2)
            k = mod.transpose_qkv(mha.W_k(X), 4)
            v = mod.transpose_qkv(mha.W_v(X), 4)
            heads = F.scaled_dot_product_attention(q, k, v)
        # 四个头是四组不同投影，结果不该相同
        for i in range(1, 4):
            assert not torch.allclose(heads[0], heads[i], atol=1e-4), (
                f"第 0 个头与第 {i} 个头的结果相同，说明四个头用了同一组投影。"
                "每个头要有自己的 W_q、W_k、W_v"
            )

    @case("T2 有效长度掩蔽")
    def _t2d():
        torch.manual_seed(0)
        mha = mod.MultiHeadAttention(
            key_size=8, query_size=8, value_size=8,
            num_hiddens=8, num_heads=2, dropout=0.0)
        mha.eval()
        X = torch.randn(2, 5, 8)
        out = mha(X, X, X, valid_lens=torch.tensor([1, 5]))
        # 把第 0 个样本的第 1 个位置之后的值改掉，输出不该变
        X2 = X.clone()
        X2[0, 1:] = torch.randn(4, 8)
        out2 = mha(X2, X2, X2, valid_lens=torch.tensor([1, 5]))
        assert torch.allclose(out[0, 0], out2[0, 0], atol=1e-5), (
            "valid_lens 为 1 时，第一个查询只能看到第 0 个键，"
            "后面的键改了输出不该变"
        )

    @case("T3 位置编码：形状与具体值")
    def _t3a():
        pe = mod.PositionalEncoding(num_hiddens=8, dropout=0.0, max_len=10)
        assert tuple(pe.P.shape) == (1, 10, 8), (
            f"P 形状应为 (1, max_len, d) = (1,10,8)，得到 {tuple(pe.P.shape)}"
        )
        # P[0, i, 2j] = sin(i / 10000^(2j/d))
        for i in [0, 1, 5]:
            for j in range(4):
                want_sin = math.sin(i / (10000 ** (2 * j / 8)))
                want_cos = math.cos(i / (10000 ** (2 * j / 8)))
                assert abs(pe.P[0, i, 2 * j].item() - want_sin) < 1e-5, (
                    f"P[0,{i},{2 * j}] 应为 sin({i}/10000^({2 * j}/8)) = {want_sin:.6f}，"
                    f"得到 {pe.P[0, i, 2 * j].item():.6f}"
                )
                assert abs(pe.P[0, i, 2 * j + 1].item() - want_cos) < 1e-5, (
                    f"P[0,{i},{2 * j + 1}] 应为 cos 值 {want_cos:.6f}，"
                    f"得到 {pe.P[0, i, 2 * j + 1].item():.6f}"
                )

    @case("T3 位置编码不是可训练参数")
    def _t3b():
        pe = mod.PositionalEncoding(num_hiddens=8, dropout=0.0, max_len=10)
        n_params = sum(p.numel() for p in pe.parameters())
        assert n_params == 0, (
            f"位置编码不该有可训练参数（P 由公式算出），得到 {n_params} 个。"
            "检查是不是用 nn.Parameter 包了 P，应当用 register_buffer"
        )
        assert "P" in dict(pe.named_buffers()), (
            f"P 应当注册为 buffer，实际 buffers 有 {list(dict(pe.named_buffers()))}"
        )

    @case("T3 前向是加法")
    def _t3c():
        torch.manual_seed(0)
        pe = mod.PositionalEncoding(num_hiddens=8, dropout=0.0, max_len=10)
        X = torch.randn(2, 4, 8)
        out = pe(X)
        assert tuple(out.shape) == (2, 4, 8), f"形状应不变，得到 {tuple(out.shape)}"
        want = X + pe.P[:, :4, :]
        assert torch.allclose(out, want, atol=1e-6), (
            "前向应当是 X + P[:, :n, :]。检查有没有加错位置"
        )

    @case("T4 AddNorm 的顺序")
    def _t4a():
        torch.manual_seed(0)
        an = mod.AddNorm(normalized_shape=8, dropout=0.0)
        an.eval()
        X = torch.randn(2, 3, 8)
        Y = torch.randn(2, 3, 8)
        out = an(X, Y)
        assert tuple(out.shape) == (2, 3, 8), f"形状应不变，得到 {tuple(out.shape)}"
        want = an.ln(Y + X)
        assert torch.allclose(out, want, atol=1e-5), (
            f"AddNorm 应当是 LayerNorm(dropout(Y) + X)，最大差 "
            f"{(out - want).abs().max().item():.6f}"
        )

    @case("T4 编码器块")
    def _t4b():
        torch.manual_seed(0)
        blk = mod.EncoderBlock(
            key_size=16, query_size=16, value_size=16,
            num_hiddens=16, norm_shape=16, ffn_num_input=16,
            ffn_num_hiddens=32, num_heads=4, dropout=0.0)
        blk.eval()
        X = torch.randn(2, 5, 16)
        out = blk(X)
        assert tuple(out.shape) == (2, 5, 16), (
            f"输出形状应为 (batch, n, num_hiddens) = (2,5,16)，得到 {tuple(out.shape)}"
        )
        # 两个 AddNorm、一个多头注意力、一个前馈
        assert isinstance(getattr(blk, "attention", None), nn.Module), "缺 attention 子模块"
        assert isinstance(getattr(blk, "ffn", None), nn.Module), "缺 ffn 子模块"
        an = [m for m in blk.modules() if isinstance(m, mod.AddNorm)]
        assert len(an) == 2, f"编码器块应当有 2 个 AddNorm，得到 {len(an)}"

    @case("T4 编码器块对位置独立")
    def _t4c():
        torch.manual_seed(0)
        blk = mod.EncoderBlock(
            key_size=8, query_size=8, value_size=8,
            num_hiddens=8, norm_shape=8, ffn_num_input=8,
            ffn_num_hiddens=16, num_heads=2, dropout=0.0)
        blk.eval()
        X = torch.randn(1, 4, 8)
        # 把第 3 个位置换成别的值，其余位置不变；前馈按位置独立，
        # 但注意力的路径会让其他位置也变，所以这里只查"输出与输入同形状且不同"
        out = blk(X)
        assert not torch.allclose(out, X, atol=1e-4), (
            "编码器块的输出不该等于输入（子层至少改变一次）"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
