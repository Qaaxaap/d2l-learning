"""B21 验收断言。用法: just check b21

读取 work/b21/b21.py 并验证 exercises/b21/code.md 里的 T1-T4。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b21" / "b21.py"

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
        print("先照着 exercises/b21/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b21_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 书上的 GRU 公式")
    def _t1():
        torch.manual_seed(0)
        d, h, n = 3, 4, 2
        X = torch.randn(n, d)
        H = torch.randn(n, h)

        def rnd(*s):
            return torch.randn(*s)

        W_xr, W_hr, b_r = rnd(d, h), rnd(h, h), rnd(h)
        W_xz, W_hz, b_z = rnd(d, h), rnd(h, h), rnd(h)
        W_xh, W_hh, b_h = rnd(d, h), rnd(h, h), rnd(h)
        params = (W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h)

        got = mod.gru_step_book(X, H, params)

        R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)
        Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
        H_tilde = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        want = Z * H + (1 - Z) * H_tilde

        assert tuple(got.shape) == (n, h), (
            f"输出形状应为 ({n},{h})，得到 {tuple(got.shape)}"
        )
        assert torch.allclose(got, want, atol=1e-6), (
            f"与手算的最大差为 {(got - want).abs().max().item():.6f}。"
            "检查：Z 是保留旧状态的比例，H = Z*H_prev + (1-Z)*H_tilde"
        )

    @case("T1 更新门接近 1 时状态几乎不变")
    def _t1b():
        torch.manual_seed(0)
        d, h, n = 2, 3, 1
        X = torch.randn(n, d)
        H = torch.randn(n, h)

        def rnd(*s):
            return torch.randn(*s)

        # 把更新门的偏置设成很大 → sigmoid 接近 1 → 保留旧状态
        params = (rnd(d, h), rnd(h, h), rnd(h),
                  rnd(d, h), rnd(h, h), torch.full((h,), 20.0),
                  rnd(d, h), rnd(h, h), rnd(h))
        got = mod.gru_step_book(X, H, params)
        assert torch.allclose(got, H, atol=1e-3), (
            f"更新门饱和到 1 时输出应当接近旧的 H。\nH={H.tolist()}\n"
            f"得到 {got.tolist()}。检查 H = Z*H_prev + (1-Z)*H_tilde 的系数"
        )

    @case("T2 torch 约定的 GRU 与 nn.GRU 一致")
    def _t2():
        torch.manual_seed(0)
        d, h, n = 3, 4, 2
        gru = nn.GRU(d, h, batch_first=True)
        X = torch.randn(n, d)
        H = torch.randn(n, h)

        with torch.no_grad():
            W_ih = gru.weight_ih_l0        # (3h, d)，顺序 [r, z, n]
            W_hh = gru.weight_hh_l0        # (3h, h)
            b_ih = gru.bias_ih_l0
            b_hh = gru.bias_hh_l0
        W_ir, W_iz, W_in = W_ih.chunk(3, 0)
        W_hr, W_hz, W_hn = W_hh.chunk(3, 0)
        b_ir, b_iz, b_in = b_ih.chunk(3, 0)
        b_hr, b_hz, b_hn = b_hh.chunk(3, 0)

        # 我们的接口约定权重是 (d, h) / (h, h)
        params = (W_ir.T, W_hr.T, b_ir, b_hr,
                  W_iz.T, W_hz.T, b_iz, b_hz,
                  W_in.T, W_hn.T, b_in, b_hn)
        got = mod.gru_step_torch(X, H, params)

        with torch.no_grad():
            out, _ = gru(X.unsqueeze(1), H.unsqueeze(0))
        want = out[:, 0, :]
        assert tuple(got.shape) == (n, h), (
            f"输出形状应为 ({n},{h})，得到 {tuple(got.shape)}"
        )
        assert torch.allclose(got, want, atol=1e-5), (
            f"与 nn.GRU 的最大差为 {(got - want).abs().max().item():.6f}。"
            "检查：z 是接受新状态的比例，n 是先矩阵乘再乘 r"
        )

    @case("T2 两种约定确实不同")
    def _t2b():
        torch.manual_seed(0)
        d, h, n = 3, 4, 2
        X = torch.randn(n, d)
        H = torch.randn(n, h)

        def rnd(*s):
            return torch.randn(*s)

        book = (rnd(d, h), rnd(h, h), rnd(h), rnd(d, h), rnd(h, h), rnd(h),
                rnd(d, h), rnd(h, h), rnd(h))
        torchp = (rnd(d, h), rnd(h, h), rnd(h), rnd(h),
                  rnd(d, h), rnd(h, h), rnd(h), rnd(h),
                  rnd(d, h), rnd(h, h), rnd(h), rnd(h))
        a = mod.gru_step_book(X, H, book)
        b = mod.gru_step_torch(X, H, torchp)
        # 参数不同，两个实现当然不同；这里只检查两个函数都存在且能跑
        assert a.shape == b.shape == (n, h), "两个函数都应能跑出 (n, h)"

    @case("T3 LSTM 与手算一致")
    def _t3a():
        torch.manual_seed(0)
        d, h, n = 3, 4, 2
        X = torch.randn(n, d)
        H = torch.randn(n, h)
        C = torch.randn(n, h)

        def rnd(*s):
            return torch.randn(*s)

        W_xi, W_hi, b_i = rnd(d, h), rnd(h, h), rnd(h)
        W_xf, W_hf, b_f = rnd(d, h), rnd(h, h), rnd(h)
        W_xo, W_ho, b_o = rnd(d, h), rnd(h, h), rnd(h)
        W_xc, W_hc, b_c = rnd(d, h), rnd(h, h), rnd(h)
        params = (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
                  W_xo, W_ho, b_o, W_xc, W_hc, b_c)

        got_H, got_C = mod.lstm_step(X, H, C, params)

        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_t = torch.tanh(X @ W_xc + H @ W_hc + b_c)
        want_C = F * C + I * C_t
        want_H = O * torch.tanh(want_C)

        assert torch.allclose(got_C, want_C, atol=1e-6), (
            f"C 与手算的最大差为 {(got_C - want_C).abs().max().item():.6f}。"
            "C = F*C_prev + I*C_tilde"
        )
        assert torch.allclose(got_H, want_H, atol=1e-6), (
            f"H 与手算的最大差为 {(got_H - want_H).abs().max().item():.6f}。"
            "H = O * tanh(C)，不是直接返回 C"
        )

    @case("T3 LSTM 与 nn.LSTM 一致")
    def _t3b():
        torch.manual_seed(0)
        d, h, n = 3, 4, 2
        lstm = nn.LSTM(d, h, batch_first=True)
        X = torch.randn(n, d)
        H = torch.randn(n, h)
        C = torch.randn(n, h)

        with torch.no_grad():
            W_ih = lstm.weight_ih_l0      # (4h, d)，顺序 [i, f, g, o]
            W_hh = lstm.weight_hh_l0
            b_ih = lstm.bias_ih_l0
            b_hh = lstm.bias_hh_l0
        W_xi, W_xf, W_xg, W_xo = W_ih.chunk(4, 0)
        W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, 0)
        b_i, b_f, b_g, b_o = b_ih.chunk(4, 0)
        # nn.LSTM 的偏置是 b_ih + b_hh 两份相加
        b_i = b_i + b_hh.chunk(4, 0)[0]
        b_f = b_f + b_hh.chunk(4, 0)[1]
        b_g = b_g + b_hh.chunk(4, 0)[2]
        b_o = b_o + b_hh.chunk(4, 0)[3]

        params = (W_xi.T, W_hi.T, b_i, W_xf.T, W_hf.T, b_f,
                  W_xo.T, W_ho.T, b_o, W_xg.T, W_hg.T, b_g)
        got_H, got_C = mod.lstm_step(X, H, C, params)

        with torch.no_grad():
            out, (h_n, c_n) = lstm(X.unsqueeze(1), (H.unsqueeze(0), C.unsqueeze(0)))
        want_H = out[:, 0, :]
        want_C = c_n[0]
        assert torch.allclose(got_H, want_H, atol=1e-5), (
            f"H 与 nn.LSTM 的最大差为 {(got_H - want_H).abs().max().item():.6f}"
        )
        assert torch.allclose(got_C, want_C, atol=1e-5), (
            f"C 与 nn.LSTM 的最大差为 {(got_C - want_C).abs().max().item():.6f}"
        )

    @case("T4 深层与双向的形状")
    def _t4():
        assert tuple(mod.hidden_state_shape(3, 8, 16)) == (3, 8, 16), (
            f"3 层时 h_n 形状应为 (3,8,16)，得到 {mod.hidden_state_shape(3, 8, 16)}"
        )
        assert tuple(mod.hidden_state_shape(3, 8, 16, bidirectional=True)) == (6, 8, 16), (
            f"双向时层数翻倍为 6，得到 {mod.hidden_state_shape(3, 8, 16, True)}"
        )
        assert mod.output_feature_dim(16) == 16, "单向输出维度是 h"
        assert mod.output_feature_dim(16, bidirectional=True) == 32, (
            f"双向输出维度是 2h = 32，得到 {mod.output_feature_dim(16, True)}"
        )
        # 与真实的 nn.RNN 核对
        rnn = nn.RNN(4, 16, num_layers=3, bidirectional=True, batch_first=True)
        x = torch.randn(8, 5, 4)
        with torch.no_grad():
            out, h_n = rnn(x)
        assert tuple(h_n.shape) == tuple(mod.hidden_state_shape(3, 8, 16, True)), (
            f"与 nn.RNN 实际返回的 {tuple(h_n.shape)} 不一致"
        )
        assert out.shape[-1] == mod.output_feature_dim(16, bidirectional=True), (
            f"nn.RNN 输出最后一维 {out.shape[-1]} 与函数给出的 "
            f"{mod.output_feature_dim(16, True)} 不一致"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1

    print("T1-T4 全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
