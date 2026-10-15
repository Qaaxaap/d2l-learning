"""B09 验收断言。用法: just check b09

读取 work/b09/b09.py 并验证 exercises/b09/code.md 里的 T1-T6。
T7 是纸面题，这里只提醒人工检查。
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b09" / "b09.py"

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
        print("先照着 exercises/b09/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b09_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_user_module()

    @case("T1 生成数据")
    def _t1():
        out = mod.make_data()
        assert len(out) == 4, f"应返回四个张量，得到 {len(out)} 个"
        names = ("x_train", "y_train", "x_test", "y_test")
        for name, t in zip(names, out):
            assert tuple(t.shape) == (100, 1), f"{name} 形状应为 (100, 1)，得到 {tuple(t.shape)}"
            assert t.dtype == torch.float32, f"{name} 的 dtype 应为 float32，得到 {t.dtype}"
        again = mod.make_data()
        for t, u in zip(out, again):
            assert torch.allclose(t, u), "同一个 seed 应给出相同结果"
        # 标签的量级由公式决定：5.6x^3 那一项的方差是 5.6^2 * E[x^6] = 5.6^2 * 15，
        # 标准差约 21.7，加上另外两项与常数，整体在 25 上下。公式里系数抄错会明显偏小。
        y_std = out[1].std().item()
        assert 15.0 < y_std < 35.0, (
            f"标签的标准差应在 15 到 35 之间（实测约 25.8），得到 {y_std:.3f}。"
            "检查公式里各项的系数"
        )

    @case("T2 多项式特征")
    def _t2():
        got = mod.poly_features(torch.tensor([[2.0]]), 3)
        want = torch.tensor([[2.0, 4.0, 8.0]])
        assert tuple(got.shape) == (1, 3), f"形状应为 (1, 3)，得到 {tuple(got.shape)}"
        assert torch.allclose(got, want), f"应为 {want.tolist()}，得到 {got.tolist()}"
        x = torch.randn(5, 1)
        assert tuple(mod.poly_features(x, 1).shape) == (5, 1), "k=1 时形状应与输入相同"
        five = mod.poly_features(x, 5)
        assert torch.allclose(five[:, 2], x[:, 0] ** 3, atol=1e-5), "第 3 列应等于第 1 列的 3 次方"

    @case("T3 k=3 训练")
    def _t3():
        x_tr, y_tr, x_te, y_te = mod.make_data()
        res = mod.train_poly(x_tr, y_tr, x_te, y_te, 3, num_epochs=100)
        for key in ("train_loss", "test_loss"):
            assert key in res, f"返回的字典缺 {key} 键"
            assert len(res[key]) == 100, f"{key} 应有 100 项，得到 {len(res[key])}"
        assert id(res["train_loss"]) != id(res["test_loss"]), "两个键不能指向同一个列表"
        assert res["train_loss"][-1] < 0.5, (
            f"3 阶多项式跑 100 轮后训练损失应低于 0.5，得到 {res['train_loss'][-1]:.4f}。"
            "偏大检查：特征是不是造对了、记录的是不是平均损失、梯度有没有清零"
        )
        assert res["test_loss"][-1] < 1.0, (
            f"3 阶多项式的测试损失应低于 1.0，得到 {res['test_loss'][-1]:.4f}"
        )
        assert res["train_loss"][0] > res["train_loss"][-1], (
            f"训练损失应下降，得到首轮 {res['train_loss'][0]:.4f}、末轮 {res['train_loss'][-1]:.4f}"
        )

    @case("T3 欠拟合的量级")
    def _t3b():
        x_tr, y_tr, x_te, y_te = mod.make_data()
        res = mod.train_poly(x_tr, y_tr, x_te, y_te, 1, num_epochs=100)
        last = res["train_loss"][-1]
        assert last > 30.0, (
            f"线性模型拟合三次函数的训练损失应高于 30，得到 {last:.4f}。"
            "偏小说明阶数没用上，或者特征矩阵给错了"
        )

    @case("T4 画图")
    def _t4():
        x_tr, y_tr, x_te, y_te = mod.make_data()
        res = mod.train_poly(x_tr, y_tr, x_te, y_te, 3, num_epochs=20)
        plt.close("all")
        # 图写到学习者的目录而不是临时目录：这张图是 T4 的产物，
        # 断言只能查"文件存在、纵轴是对数"，画得对不对要靠他自己打开看。
        out_dir = REPO / "work" / "b09"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "curve.png"
        mod.plot_errors(res["train_loss"], res["test_loss"], str(path))
        assert path.exists(), f"应当把图存到 {path}，文件不存在"
        assert path.stat().st_size > 1000, "存出来的文件太小，可能是空图"
        fig = plt.gcf()
        axes = fig.get_axes()
        if axes:
            assert axes[0].get_yscale() == "log", (
                "纵轴应当是对数刻度。两条误差常差一两个数量级，"
                "线性刻度下小的那条会贴在轴上"
            )
        plt.close("all")
        print(f"      （T4 的图写在 {path.relative_to(REPO)}，自己打开看看两条曲线）")

    @case("T5 k 折切分")
    def _t5():
        parts = mod.k_fold_indices(10, 3)
        assert len(parts) == 3, f"应返回 3 份，得到 {len(parts)}"
        flat = sorted(i for p in parts for i in p)
        assert flat == list(range(10)), f"三份合起来应是 0 到 9 各一次，得到 {flat}"
        sizes = sorted(len(p) for p in parts)
        assert sizes == [3, 3, 4], f"各份大小应为 3、3、4 的某个排列，得到 {sizes}"
        again = mod.k_fold_indices(10, 3)
        assert [sorted(p) for p in parts] == [sorted(p) for p in again], "同一个 seed 应给出相同切分"
        five = mod.k_fold_indices(7, 5)
        sizes5 = sorted(len(p) for p in five)
        assert sizes5 == [1, 1, 1, 2, 2], f"7 个下标切 5 份应得到 1、1、1、2、2，得到 {sizes5}"

    @case("T6 诊断")
    def _t6():
        cases = [
            (([5.0, 4.0], [5.5, 4.5]), "underfitting"),
            (([0.01, 0.005], [0.02, 0.9]), "overfitting"),
            (([0.1, 0.02], [0.12, 0.03]), "good"),
            (([0.01], [5.0]), "overfitting"),
        ]
        for (train_c, test_c), want in cases:
            got = mod.diagnose(train_c, test_c)
            assert got == want, (
                f"diagnose({train_c}, {test_c}) 应返回 {want!r}，得到 {got!r}。"
                "判据按顺序：训练损失 > 0.5 判欠拟合；测试 > 训练 × 10 判过拟合；其余是 good"
            )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT7 是纸面题，人工检查。")
        return 1

    print("T1-T6 全部通过。")
    print("T7 是纸面题，等我看你标的三处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
