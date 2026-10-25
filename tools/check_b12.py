"""B12 验收断言。用法: just check b12

读取 work/b12/b12.py 并验证 exercises/b12/code.md 里的 T1-T6。
T7 是纸面题，这里只提醒人工检查。

数据在仓库的 data/ 目录下。整个脚本约 20 秒（训练用 20 轮，不是书上的 100 轮）。
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b12" / "b12.py"
DATA = REPO / "data"

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
        print("先照着 exercises/b12/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b12_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cache: dict = {}


def get_data():
    """读一次数据，多处复用。"""
    if not _cache:
        _cache["train"] = pd.read_csv(DATA / "kaggle_house_pred_train.csv")
        _cache["test"] = pd.read_csv(DATA / "kaggle_house_pred_test.csv")
    return _cache["train"], _cache["test"]


def main() -> int:
    mod = load_user_module()

    @case("T1 读数据")
    def _t1():
        out = mod.load_houses(str(DATA))
        assert len(out) == 2, f"应返回两个 DataFrame，得到 {len(out)} 个"
        tr, te = out
        assert isinstance(tr, pd.DataFrame), f"第一个应是 DataFrame，得到 {type(tr).__name__}"
        assert tuple(tr.shape) == (1460, 81), (
            f"训练集应为 (1460, 81)（Id + 79 特征 + 标签），得到 {tr.shape}"
        )
        assert tuple(te.shape) == (1459, 80), (
            f"测试集应为 (1459, 80)（Id + 79 特征），得到 {te.shape}"
        )
        assert "SalePrice" in tr.columns, "训练集应有 SalePrice 列"
        assert "SalePrice" not in te.columns, "测试集不该有 SalePrice 列"

    @case("T2 预处理的形状与 dtype")
    def _t2a():
        tr, te = get_data()
        out = mod.preprocess(tr, te)
        assert len(out) == 3, f"应返回三个张量，得到 {len(out)} 个"
        features, test_features, labels = out
        assert tuple(features.shape) == (1460, 330), (
            f"train_features 应为 (1460, 330)，得到 {tuple(features.shape)}。"
            "列数不对检查：有没有去掉 Id、有没有 one-hot、dummy_na 有没有开"
        )
        assert tuple(test_features.shape) == (1459, 330), (
            f"test_features 应为 (1459, 330)，得到 {tuple(test_features.shape)}"
        )
        assert tuple(labels.shape) == (1460, 1), f"train_labels 应为 (1460, 1)，得到 {tuple(labels.shape)}"
        for name, t in [("features", features), ("test_features", test_features), ("labels", labels)]:
            assert t.dtype == torch.float32, f"{name} 的 dtype 应为 float32，得到 {t.dtype}"

    @case("T2 预处理没有缺失值、统计量正确")
    def _t2b():
        tr, te = get_data()
        features, test_features, labels = mod.preprocess(tr, te)
        assert not torch.isnan(features).any(), "features 里还有 nan"
        assert not torch.isnan(test_features).any(), "test_features 里还有 nan"
        # 合并后的数值列标准化之后均值约 0、标准差约 1
        both = torch.cat([features, test_features], dim=0)
        mean = both.mean().item()
        assert abs(mean) < 0.35, (
            f"标准化的均值应接近 0，得到 {mean:.4f}。"
            "偏大检查：标准化用的是训练集自己的统计量、还是合并后的"
        )
        label_mean = labels.mean().item()
        assert 150_000 < label_mean < 250_000, (
            f"房价标签的均值应在 15 万到 25 万之间，得到 {label_mean:.0f}"
        )

    @case("T3 log_rmse")
    def _t3():
        labels = torch.tensor([[100_000.0], [200_000.0], [150_000.0]])
        same = mod.log_rmse(labels.clone(), labels)
        assert abs(same) < 1e-6, f"预测与标签相同时应为 0，得到 {same}"
        doubled = mod.log_rmse(labels * 2, labels)
        import math
        assert abs(doubled - math.log(2)) < 1e-5, (
            f"预测乘 2 时应为 log(2) ≈ {math.log(2):.6f}，得到 {doubled:.6f}"
        )
        # 含 0 与负数不能崩
        weird = torch.tensor([[0.0], [-5.0], [150_000.0]])
        val = mod.log_rmse(weird, labels)
        assert not (val != val), "结果不该是 nan"
        assert isinstance(val, float), f"应返回 float，得到 {type(val).__name__}"

    @case("T4 训练：曲线与趋势")
    def _t4a():
        tr, te = get_data()
        features, test_features, labels = mod.preprocess(tr, te)
        g = torch.Generator().manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 1))
        torch.manual_seed(0)
        train_ls, test_ls = mod.train(
            net, features, labels, features, labels,
            num_epochs=50, lr=5, weight_decay=0, batch_size=64,
        )
        assert len(train_ls) == 50, f"train_ls 应有 50 项，得到 {len(train_ls)}"
        assert len(test_ls) == 50, f"test_ls 应有 50 项，得到 {len(test_ls)}"
        assert train_ls[-1] < train_ls[0] * 0.8, (
            f"训练误差应明显下降：第一轮 {train_ls[0]:.4f}、最后一轮 {train_ls[-1]:.4f}。"
            "下降不明显检查：优化器是不是 Adam、学习率是不是用了、指标算得对不对"
        )
        assert 0.15 < train_ls[-1] < 0.35, (
            f"50 轮后训练误差应在 0.15 到 0.35 之间（实测约 0.20），得到 {train_ls[-1]:.4f}。"
            "远大于 0.35 说明没在学；远小于 0.15 检查指标是不是少开了根号"
        )

    @case("T4 训练：验证集传 None")
    def _t4b():
        tr, te = get_data()
        features, test_features, labels = mod.preprocess(tr, te)
        net = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 1))
        torch.manual_seed(0)
        train_ls, test_ls = mod.train(
            net, features, labels, None, None,
            num_epochs=3, lr=5, weight_decay=0, batch_size=64,
        )
        assert len(train_ls) == 3, f"train_ls 应有 3 项，得到 {len(train_ls)}"
        assert test_ls == [], f"没给验证集时第二个列表应为空，得到 {test_ls}"

    @case("T5 k 折切分")
    def _t5a():
        X = torch.arange(1460, dtype=torch.float32).reshape(-1, 1)
        y = torch.arange(1460, dtype=torch.float32).reshape(-1, 1)
        seen_valid, sizes = [], []
        for i in range(5):
            tx, ty, vx, vy = mod.get_k_fold_data(5, i, X, y)
            assert tx.shape[0] + vx.shape[0] == 1460, (
                f"第 {i} 折的训练与验证样本数之和应为 1460，得到 {tx.shape[0]} + {vx.shape[0]}"
            )
            assert tx.shape[0] > vx.shape[0], (
                f"第 {i} 折的训练集应比验证集大：k=5 时训练 4 份（1168）、验证 1 份（292），"
                f"得到训练 {tx.shape[0]}、验证 {vx.shape[0]}。"
                "如果反了，检查 get_k_fold_data 里第 i 份是拼进了训练集还是拿来当验证集"
            )
            assert tuple(tx.shape[1:]) == (1,), "特征的列数应保持为 1"
            sizes.append(vx.shape[0])
            seen_valid.extend(vx.flatten().tolist())
        assert max(sizes) - min(sizes) <= 1, (
            f"各折大小应相差不超过 1，得到 {sizes}"
        )
        assert len(set(seen_valid)) == 1460, (
            f"五折的验证集拼起来应覆盖全部 1460 个样本且不重复，实际覆盖 {len(set(seen_valid))} 个"
        )

    @case("T5 k 折的验证误差在合理范围")
    def _t5b():
        tr, te = get_data()
        features, test_features, labels = mod.preprocess(tr, te)
        torch.manual_seed(0)
        avg_train, avg_valid = mod.k_fold(
            5, features, labels, num_epochs=50, lr=5, weight_decay=0, batch_size=64
        )
        assert 0.18 < avg_valid < 0.35, (
            f"5 折平均验证误差应在 0.18 到 0.35 之间（50 轮实测约 0.27），得到 {avg_valid:.4f}。"
            "学习率与轮数是断言传进去的，不用怀疑超参数。查：训练集与验证集有没有反、"
            "train 有没有用上 lr、get_k_fold_data 每折是否返回 1168 个训练样本"
        )
        assert avg_valid > avg_train * 0.5, (
            f"验证误差 {avg_valid:.4f} 比训练误差 {avg_train:.4f} 小太多，"
            "检查是不是把训练集当验证集用了"
        )

    @case("T6 提交文件")
    def _t6():
        tr, te = get_data()
        features, test_features, labels = mod.preprocess(tr, te)
        torch.manual_seed(0)
        sub = mod.train_and_pred(
            features, test_features, labels, te,
            num_epochs=20, lr=5, weight_decay=0, batch_size=64,
        )
        assert isinstance(sub, pd.DataFrame), f"应返回 DataFrame，得到 {type(sub).__name__}"
        assert list(sub.columns) == ["Id", "SalePrice"], (
            f"列名应为 ['Id', 'SalePrice']，得到 {list(sub.columns)}"
        )
        assert len(sub) == 1459, f"行数应为 1459，得到 {len(sub)}"
        assert (sub["Id"].values == te["Id"].values).all(), "Id 列应与测试集一致"
        prices = sub["SalePrice"].values
        assert (prices > 0).all(), "预测价格应全为正"
        assert 50_000 < prices.mean() < 1_000_000, (
            f"预测价格均值应在 5 万到 100 万之间，得到 {prices.mean():.0f}"
        )
        out = REPO / "work" / "b12" / "submission.csv"
        assert out.exists(), f"应当存到 {out}"
        saved = pd.read_csv(out)
        assert list(saved.columns) == ["Id", "SalePrice"], (
            f"存下来的 CSV 列名应为 ['Id', 'SalePrice']，得到 {list(saved.columns)}。"
            "多一列通常是 to_csv 少了 index=False"
        )
        assert len(saved) == 1459, f"存下来的 CSV 应有 1459 行，得到 {len(saved)}"

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\nT7 是纸面题，人工检查。")
        return 1

    print("T1-T6 全部通过。")
    print("T7 是纸面题，等我看你标的四处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
