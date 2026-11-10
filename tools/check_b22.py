"""B22 验收断言。用法: just check b22

读取 work/b22/b22.py 并验证 exercises/b22/code.md 里的 T1-T4。
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b22" / "b22.py"

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
        print("先照着 exercises/b22/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b22_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 编码器")
    def _t1():
        torch.manual_seed(0)
        V, e, h, n, T = 10, 8, 16, 3, 5
        enc = mod.Encoder(V, e, h)
        assert hasattr(enc, "embedding"), "编码器应当有 embedding 层"
        X = torch.randint(0, V, (T, n))
        outputs, state = enc(X)
        assert tuple(outputs.shape) == (T, n, h), (
            f"outputs 形状应为 (seq_len, batch, h) = ({T},{n},{h})，"
            f"得到 {tuple(outputs.shape)}。注意默认 seq_len 在前，不加 batch_first"
        )
        assert tuple(state.shape) == (1, n, h), (
            f"state 形状应为 (num_layers, batch, h) = (1,{n},{h})，"
            f"得到 {tuple(state.shape)}"
        )
        # state 应当是最后一步的隐藏状态
        assert torch.allclose(state[-1], outputs[-1], atol=1e-6), (
            "单层时 state 的最后一项应当等于 outputs 的最后一步"
        )

    @case("T2 解码器")
    def _t2a():
        torch.manual_seed(0)
        V, e, h, n, T = 12, 8, 16, 3, 4
        dec = mod.Decoder(V, e, h)
        enc_state = torch.randn(1, n, h)
        logits, state = dec(torch.randint(0, V, (T, n)), dec.init_state(enc_state))
        assert tuple(logits.shape) == (T, n, V), (
            f"logits 形状应为 (seq_len, batch, V) = ({T},{n},{V})，"
            f"得到 {tuple(logits.shape)}"
        )
        assert tuple(state.shape) == (1, n, h), (
            f"state 形状应为 (1,{n},{h})，得到 {tuple(state.shape)}"
        )

    @case("T2 init_state 要 detach")
    def _t2b():
        torch.manual_seed(0)
        dec = mod.Decoder(12, 8, 16)
        enc_state = torch.randn(1, 3, 16, requires_grad=True)
        state = dec.init_state(enc_state)
        assert not state.requires_grad, (
            "init_state 应当返回 detach 过的状态。"
            "不 detach 的话编码器的计算图会连进解码器的训练，显存翻倍"
        )
        assert torch.allclose(state, enc_state, atol=1e-6), (
            "detach 只切梯度，数值应当不变"
        )

    @case("T3 seq2seq 串起来")
    def _t3():
        torch.manual_seed(0)
        Vx, Vy, e, h, n = 10, 12, 8, 16, 3
        enc = mod.Encoder(Vx, e, h)
        dec = mod.Decoder(Vy, e, h)
        model = mod.Seq2Seq(enc, dec)
        enc_X = torch.randint(0, Vx, (5, n))
        dec_X = torch.randint(0, Vy, (4, n))
        logits, state = model(enc_X, dec_X)
        assert tuple(logits.shape) == (4, n, Vy), (
            f"logits 形状应为 (dec_len, batch, Vy) = (4,{n},{Vy})，"
            f"得到 {tuple(logits.shape)}"
        )
        # 与手动串一遍对比
        with torch.no_grad():
            _, s = enc(enc_X)
            ref_logits, _ = dec(dec_X, dec.init_state(s))
        assert torch.allclose(logits, ref_logits, atol=1e-6), (
            "Seq2Seq 的输出应当等于「编码器 → init_state → 解码器」手动串起来的结果"
        )

    @case("T4 束搜索：beam=1 等于贪心")
    def _t4a():
        A, B, C, EOS, BOS = 0, 1, 2, 3, 9
        probs = {
            (BOS,): {A: 0.5, C: 0.4, B: 0.1},
            (BOS, A): {B: 0.4, C: 0.3, EOS: 0.3},
            (BOS, C): {B: 0.6, EOS: 0.3, C: 0.1},
            (BOS, A, B): {C: 0.4, EOS: 0.6},
            (BOS, C, B): {EOS: 0.6},
            (BOS, A, B, C): {EOS: 0.6},
        }

        def step_fn(token, state):
            hist = state + (token,)
            dist = probs.get(hist, {EOS: 1.0})
            lp = torch.full((4,), -1e9)
            for k, v in dist.items():
                lp[k] = math.log(v)
            return lp, hist

        toks, score = mod.beam_search(step_fn, (), 1, 6, BOS, EOS)
        assert list(toks) == [A, B], (
            f"束宽 1 应当等价于贪心，得到 [A, B]（即 [0, 1]），实际 {toks}"
        )

    @case("T4 束搜索：beam=2 找到更优序列")
    def _t4b():
        A, B, C, EOS, BOS = 0, 1, 2, 3, 9
        probs = {
            (BOS,): {A: 0.5, C: 0.4, B: 0.1},
            (BOS, A): {B: 0.4, C: 0.3, EOS: 0.3},
            (BOS, C): {B: 0.6, EOS: 0.3, C: 0.1},
            (BOS, A, B): {C: 0.4, EOS: 0.6},
            (BOS, C, B): {EOS: 0.6},
            (BOS, A, B, C): {EOS: 0.6},
        }

        def step_fn(token, state):
            hist = state + (token,)
            dist = probs.get(hist, {EOS: 1.0})
            lp = torch.full((4,), -1e9)
            for k, v in dist.items():
                lp[k] = math.log(v)
            return lp, hist

        greedy_toks, greedy_score = mod.beam_search(step_fn, (), 1, 6, BOS, EOS)
        beam_toks, beam_score = mod.beam_search(step_fn, (), 2, 6, BOS, EOS)
        assert list(beam_toks) == [C, B], (
            f"束宽 2 应当选到 [C, B]（即 [2, 1]），实际 {beam_toks}。"
            "检查：每步是否扩展了全部词表、是否按累积对数概率排序、"
            "是否做了长度惩罚"
        )
        assert beam_score > greedy_score, (
            f"束搜索的分数应当高于贪心。贪心 {greedy_score:.4f}，束搜索 {beam_score:.4f}"
        )

    @case("T4 束搜索：遇到 eos 就停止扩展")
    def _t4c():
        EOS, BOS = 1, 0
        # 每一步 EOS 都是 0.9、普通 token 只有 0.1：立刻结束的那条分数最高
        def step_fn(token, state):
            lp = torch.tensor([math.log(0.1), math.log(0.9)])
            return lp, state + (token,)

        toks, score = mod.beam_search(step_fn, (), 2, 5, BOS, EOS)
        assert list(toks) == [], (
            f"第一步就选 eos 时，返回的序列应当是空列表，实际 {toks}，分数 {score:.4f}"
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
