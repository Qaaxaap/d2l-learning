"""B14 验收断言。用法: just check b14

读取 work/b14/b14.py 并验证 exercises/b14/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。

T4 需要可用的 CUDA 设备；没有时会跳过那一项并说明。
整个脚本一两秒跑完（不训练）。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b14" / "b14.py"
OUT_DIR = REPO / "work" / "b14"

_failures: list[str] = []
_passed: list[str] = []
_skipped: list[str] = []


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
        print("先照着 exercises/b14/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b14_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 CenteredLayer 不含参数")
    def _t1a():
        layer = mod.CenteredLayer()
        assert isinstance(layer, nn.Module), "应当是 nn.Module 的子类"
        assert list(layer.parameters()) == [], (
            f"不该有参数，得到 {[n for n, _ in layer.named_parameters()]}"
        )

    @case("T1 CenteredLayer 的输出")
    def _t1b():
        layer = mod.CenteredLayer()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = layer(x)
        assert tuple(out.shape) == (2, 2), f"形状应不变，得到 {tuple(out.shape)}"
        assert abs(out.mean().item()) < 1e-6, (
            f"输出全元素均值应为 0，得到 {out.mean().item():.6f}"
        )
        # (1+2+3+4)/4 = 2.5
        want = torch.tensor([[-1.5, -0.5], [0.5, 1.5]])
        assert torch.allclose(out, want, atol=1e-6), (
            f"应为 {want.tolist()}，得到 {out.tolist()}。"
            "检查是不是减了“全部元素的均值”而不是“每个样本自己的均值”"
        )

    @case("T2 MyDense 的参数")
    def _t2a():
        dense = mod.MyDense(5, 3)
        params = dict(dense.named_parameters())
        assert set(params.keys()) == {"weight", "bias"}, (
            f"参数名应为 weight 与 bias，得到 {list(params.keys())}"
        )
        assert tuple(params["weight"].shape) == (5, 3), (
            f"weight 形状应为 (in_units, units) = (5, 3)，得到 {tuple(params['weight'].shape)}。"
            "写成 (units, in_units) 是把 nn.Linear 的约定套过来了"
        )
        assert tuple(params["bias"].shape) == (3,), (
            f"bias 形状应为 (units,)，得到 {tuple(params['bias'].shape)}"
        )
        for name, p in params.items():
            assert isinstance(p, nn.Parameter), (
                f"{name} 必须是 nn.Parameter，得到 {type(p).__name__}。"
                "普通张量属性不会被优化器看到"
            )

    @case("T2 MyDense 的前向与 state_dict")
    def _t2b():
        dense = mod.MyDense(5, 3)
        out = dense(torch.randn(4, 5))
        assert tuple(out.shape) == (4, 3), f"前向应得到 (4, 3)，得到 {tuple(out.shape)}"
        assert (out >= 0).all(), "最后有 ReLU，输出应当非负"
        keys = list(dense.state_dict().keys())
        assert keys == ["weight", "bias"], f"state_dict 键应为 ['weight', 'bias']，得到 {keys}"

    @case("T3 保存与加载")
    def _t3a():
        torch.manual_seed(0)
        net = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2), mod.CenteredLayer()
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "t3_test.pt"
        if path.exists():
            path.unlink()      # 先清掉旧产物
        out = mod.save_and_load(net, str(path))
        assert isinstance(out, dict), f"应返回字典，得到 {type(out).__name__}"
        for key in ("saved_keys", "loaded_ok", "max_diff"):
            assert key in out, f"字典缺 {key} 键"
        assert list(out["saved_keys"]) == list(net.state_dict().keys()), (
            "saved_keys 应与 net.state_dict() 的键一致"
        )
        assert out["loaded_ok"] is True, "加载应当成功"
        assert out["max_diff"] < 1e-6, (
            f"加载后参数与原模型的最大差应接近 0，得到 {out['max_diff']}"
        )
        assert path.exists(), f"应当把参数存到 {path}（相对仓库根是 work/b14/...）"

    @case("T3 make_twin 造出的参数确实不同")
    def _t3b():
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        twin = mod.make_twin(net)
        assert list(twin.state_dict().keys()) == list(net.state_dict().keys()), (
            "make_twin 造出的模型结构应当相同（state_dict 键一致）"
        )
        diffs = [
            (a - b).abs().max().item()
            for a, b in zip(net.parameters(), twin.parameters())
        ]
        assert max(diffs) > 1e-3, (
            "make_twin 的参数应当与原来不同（否则无法验证加载是否发生）"
        )

    @case("T3 形状不匹配时加载应当报错")
    def _t3c():
        net_a = nn.Sequential(nn.Linear(4, 8))
        net_b = nn.Sequential(nn.Linear(4, 16))     # 结构不同
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "t3_mismatch.pt"
        if path.exists():
            path.unlink()
        torch.save(net_a.state_dict(), str(path))
        try:
            net_b.load_state_dict(torch.load(str(path)))
        except RuntimeError:
            pass          # 预期报错
        else:
            raise AssertionError(
                "形状不匹配时 load_state_dict 应当报 RuntimeError，但它通过了——"
                "检查是不是用了 strict=False"
            )
        finally:
            if path.exists():
                path.unlink()

    @case("T4 to_device 的语义")
    def _t4a():
        x = torch.randn(3)
        same = mod.to_device(x, x.device)
        assert same is x, (
            "设备相同时 .to() 应返回同一个对象（不是副本）"
        )
        moved = mod.to_device(x, "cpu")
        assert moved.device.type == "cpu", f"设备应为 cpu，得到 {moved.device}"
        assert torch.allclose(moved, x), "搬设备不应改变数值"

    @case("T4 move_net 搬动全部参数")
    def _t4b():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        returned = mod.move_net(net, device)
        assert returned is net, (
            "nn.Module.to() 是原地操作，应返回同一个模型对象"
        )
        for name, p in net.named_parameters():
            # 比 .type：torch.device("cuda") 与 cuda:0 的 type 都是 "cuda"，
            # 但两者直接用 == 比较不相等（一个是默认设备、一个是具体设备）
            assert p.device.type == device.type, (
                f"{name} 的设备应为 {device.type}，得到 {p.device}"
            )

    @case("T4 有 GPU 时张量搬到 cuda")
    def _t4c():
        if not torch.cuda.is_available():
            _skipped.append("T4 有 GPU 时张量搬到 cuda（本机没有可用 CUDA 设备）")
            return
        x = torch.randn(3)
        moved = mod.to_device(x, "cuda")
        assert moved.device.type == "cuda", f"设备应为 cuda，得到 {moved.device}"
        assert x.device.type == "cpu", "原张量应当还在 cpu（.to() 不改原对象）"
        assert torch.allclose(moved.cpu(), x), "搬设备不应改变数值"

    print(f"通过 {len(_passed)} 项")
    if _skipped:
        print(f"跳过 {len(_skipped)} 项：")
        for item in _skipped:
            print(f"  - {item}")
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
