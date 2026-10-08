"""B02 验收断言。用法: just check b02

读取 work/b02/b02.py 并验证 exercises/b02/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# 默认查学习者的实现；传一个路径参数可以查别的文件（自测用）
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b02" / "b02.py"

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
        print("先照着 exercises/b02/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b02_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_csv() -> str:
    content = "num1,cat,num2\n1.0,a,10\n,b,20\n3.0,a,\n"
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    )
    handle.write(content)
    handle.close()
    return handle.name


def main() -> int:
    mod = load_user_module()

    @case("T1 基本属性")
    def _t1_basic():
        info = mod.tensor_info(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        assert type(info["shape"]) is tuple, (
            f"shape 必须是普通 tuple，得到 {type(info['shape']).__name__}"
        )
        assert info["shape"] == (3, 4), f"shape 应为 (3, 4)，得到 {info['shape']!r}"
        assert info["numel"] == 12, f"numel 应为 12，得到 {info['numel']!r}"
        assert info["dtype"] == "float32", f"dtype 应为 'float32'，得到 {info['dtype']!r}"
        assert info["device"] == "cpu", f"device 应为 'cpu'，得到 {info['device']!r}"
        assert info["is_contiguous"] is True, "这个张量是连续的"

    @case("T1 转置后不连续")
    def _t1_noncontig():
        info = mod.tensor_info(torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
        assert info["is_contiguous"] is False, "转置后的张量不连续"
        assert info["shape"] == (4, 3), f"shape 应为 (4, 3)，得到 {info['shape']!r}"

    @case("T2 不兼容时抛 ValueError 并指出第 0 维")
    def _t2_dim0():
        try:
            mod.safe_broadcast_add(torch.ones(3, 4), torch.ones(2, 4))
        except ValueError as exc:
            assert "0" in str(exc), f"报错要指出第一处不匹配的维度（第 0 维），实际: {exc}"
        else:
            raise AssertionError("(3,4) 与 (2,4) 不能广播，应当抛 ValueError")

    @case("T2 不兼容时指出第 1 维")
    def _t2_dim1():
        try:
            mod.safe_broadcast_add(torch.ones(3, 4), torch.ones(3, 5))
        except ValueError as exc:
            assert "1" in str(exc), f"报错要指出第一处不匹配的维度（第 1 维），实际: {exc}"
        else:
            raise AssertionError("(3,4) 与 (3,5) 不能广播，应当抛 ValueError")

    @case("T2 兼容时结果正确")
    def _t2_ok():
        out = mod.safe_broadcast_add(torch.ones(3, 1), torch.ones(1, 4))
        assert tuple(out.shape) == (3, 4), f"形状应为 (3, 4)，得到 {tuple(out.shape)}"
        out = mod.safe_broadcast_add(torch.ones(3, 4), torch.ones(4))
        assert tuple(out.shape) == (3, 4), f"一维向量按最后一维对齐，得到 {tuple(out.shape)}"
        out = mod.safe_broadcast_add(
            torch.tensor([[1.0, 2.0]]), torch.tensor([[10.0], [20.0]])
        )
        expect = torch.tensor([[11.0, 12.0], [21.0, 22.0]])
        assert torch.allclose(out, expect), f"数值不对，得到 {out.tolist()}"

    @case("T3 切片的视图语义")
    def _t3():
        assert mod.slice_is_view() is True, "切片是视图，改切片应当影响原张量"
        assert mod.clone_is_copy() is False, "clone 是副本，改副本不应影响原张量"

    @case("T4 CSV 读取与编码")
    def _t4():
        path = make_csv()
        feats, names, stats = mod.load_csv(path)
        assert isinstance(feats, torch.Tensor), "第一个返回值应是张量"
        assert feats.dtype == torch.float32, f"dtype 应为 float32，得到 {feats.dtype}"
        assert tuple(feats.shape) == (3, 4), (
            f"应有 3 行 4 列（num1, num2, cat 的两类），得到 {tuple(feats.shape)}"
        )
        assert len(names) == feats.shape[1], (
            f"特征名有 {len(names)} 个，张量有 {feats.shape[1]} 列，对不上"
        )
        idx = {n: i for i, n in enumerate(names)}

        def col(name: str) -> torch.Tensor:
            assert name in idx, f"特征名里找不到 {name!r}，实际有 {names}"
            return feats[:, idx[name]]

        assert torch.allclose(col("num1"), torch.tensor([1.0, 2.0, 3.0])), (
            f"num1 缺失值应填均值 2.0，得到 {col('num1').tolist()}"
        )
        assert torch.allclose(col("num2"), torch.tensor([10.0, 20.0, 15.0])), (
            f"num2 缺失值应填均值 15.0，得到 {col('num2').tolist()}"
        )
        hot = [n for n in names if n.startswith("cat_")]
        assert len(hot) == 2, f"cat 列应展成两列 one-hot，实际 {hot}"
        total = sum(col(name).sum().item() for name in hot)
        assert total == 3.0, (
            f"每行恰好一个 1，one-hot 各列之和应等于行数 3，得到 {total}"
        )
        assert isinstance(stats, dict), "第三个返回值应是 dict"
        assert "数值列" in stats, f"统计信息缺少 '数值列'，实际键 {list(stats)}"
        assert "类别列" in stats, f"统计信息缺少 '类别列'，实际键 {list(stats)}"
        assert stats["类别列"].get("cat") == {"a": 2, "b": 1}, (
            f"cat 列取值计数应为 {{'a': 2, 'b': 1}}，得到 {stats['类别列'].get('cat')}"
        )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT5 是纸面题，人工检查。")
        return 1

    print("T1-T4 全部通过。")
    print("T5 是纸面题，等我看你标的三个 bug。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
