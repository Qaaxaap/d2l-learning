"""B17 验收断言。用法: just check b17

读取 work/b17/b17.py 并验证 exercises/b17/code.md 里的 T1-T4。
T5 是纸面题，这里只提醒人工检查。

四个网络都在 224×224 上跑一次前向，几秒钟完成（不训练）。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b17" / "b17.py"

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
        print("先照着 exercises/b17/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b17_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def main() -> int:
    mod = load_user_module()

    @case("T1 AlexNet 前向与参数量")
    def _t1a():
        net = mod.make_alexnet()
        out = net(torch.randn(1, 1, 224, 224))
        assert tuple(out.shape) == (1, 10), (
            f"(1,1,224,224) 应输出 (1,10)，得到 {tuple(out.shape)}"
        )
        n = count_params(net)
        assert n == 46764746, (
            f"参数量应为 46,764,746，得到 {n:,}。按形状链核对：\n"
            "  Conv 96×1×11×11 | 256×96×5×5 | 384×256×3×3 | 384×384×3×3 | 256×384×3×3\n"
            "  Linear 6400→4096 | 4096→4096 | 4096→10"
        )

    @case("T1 AlexNet 的结构")
    def _t1b():
        net = mod.make_alexnet()
        convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 5, f"应有 5 个卷积层，得到 {len(convs)}"
        assert convs[0].kernel_size == (11, 11), (
            f"第一个卷积核应为 11×11，得到 {convs[0].kernel_size}"
        )
        assert convs[0].stride == (4, 4), (
            f"第一个卷积步幅应为 4，得到 {convs[0].stride}"
        )
        drops = [m for m in net.modules() if isinstance(m, nn.Dropout)]
        assert len(drops) == 2, f"应有 2 个 Dropout，得到 {len(drops)}"

    @case("T2 vgg_block")
    def _t2a():
        blk = mod.vgg_block(2, 128)
        out = blk(torch.randn(1, 128, 56, 56))
        assert tuple(out.shape) == (1, 128, 28, 28), (
            f"vgg_block(2,128) 对 (1,128,56,56) 应输出 (1,128,28,28)，得到 {tuple(out.shape)}"
        )
        blk1 = mod.vgg_block(1, 64, in_channels=3)
        out1 = blk1(torch.randn(1, 3, 224, 224))
        assert tuple(out1.shape) == (1, 64, 112, 112), (
            f"vgg_block(1,64,in_channels=3) 对 (1,3,224,224) 应输出 (1,64,112,112)，"
            f"得到 {tuple(out1.shape)}"
        )

    @case("T2 VGG-11 前向与参数量")
    def _t2b():
        arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        net = mod.make_vgg(arch)
        out = net(torch.randn(1, 1, 224, 224))
        assert tuple(out.shape) == (1, 10), (
            f"(1,1,224,224) 应输出 (1,10)，得到 {tuple(out.shape)}"
        )
        n = count_params(net)
        assert n == 128806154, (
            f"参数量应为 128,806,154，得到 {n:,}。检查展平维度是不是 512*7*7=25088"
        )

    @case("T3 nin_block")
    def _t3a():
        blk = mod.nin_block(96, 11, 4, 0, in_channels=1)
        out = blk(torch.randn(1, 1, 224, 224))
        assert tuple(out.shape) == (1, 96, 54, 54), (
            f"nin_block(96,11,4,0) 对 (1,1,224,224) 应输出 (1,96,54,54)，得到 {tuple(out.shape)}"
        )

    @case("T3 NiN 前向、参数量与结构")
    def _t3b():
        net = mod.make_nin()
        out = net(torch.randn(1, 1, 224, 224))
        assert tuple(out.shape) == (1, 10), (
            f"(1,1,224,224) 应输出 (1,10)，得到 {tuple(out.shape)}"
        )
        n = count_params(net)
        assert n == 1992166, (
            f"参数量应为 1,992,166，得到 {n:,}。检查是不是漏了最后那个 1×1 块的通道数"
        )
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        assert not linears, f"NiN 不该有全连接层，得到 {len(linears)} 个"
        pools = [m for m in net.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
        assert pools, "缺少全局平均池化 AdaptiveAvgPool2d(1)"

    @case("T4 Inception 块")
    def _t4a():
        blk = mod.Inception(64, (96, 128), (16, 32), 32, in_channels=192)
        out = blk(torch.randn(1, 192, 28, 28))
        assert tuple(out.shape) == (1, 256, 28, 28), (
            f"Inception(64,(96,128),(16,32),32) 对 (1,192,28,28) 应输出 (1,256,28,28)，"
            f"得到 {tuple(out.shape)}。64+128+32+32 = 256"
        )

    @case("T4 Inception 四条线路的 ReLU")
    def _t4_relu():
        torch.manual_seed(0)
        block = mod.Inception(64, (96, 128), (16, 32), 32, 192)
        # 多造几个输入，避免随机数碰巧全为正
        for seed in range(3):
            torch.manual_seed(seed)
            x = torch.randn(2, 192, 28, 28)
            out = block(x)
            assert (out >= 0).all(), (
                f"第 {seed} 个输入上 Inception 的输出出现负值（最小 "
                f"{out.min().item():.4f}）。每条线路的卷积后都要接 ReLU，"
                "拼接前四个结果都应当非负"
            )
            # 分段查，报错时能指出是哪条线路漏了
            segs = [("线路 1", 0, 64), ("线路 2", 64, 192),
                    ("线路 3", 192, 224), ("线路 4", 224, 256)]
            for name, lo, hi in segs:
                seg = out[:, lo:hi]
                assert (seg >= 0).all(), (
                    f"{name}（通道 {lo}:{hi}）的输出出现负值（最小 "
                    f"{seg.min().item():.4f}），这条线路的卷积后漏了 ReLU"
                )

    @case("T4 GoogLeNet 里每个卷积后的 ReLU")
    def _t4_relu_net():
        torch.manual_seed(0)
        net = mod.make_googlenet()
        net.eval()

        pools = [m for m in net.modules() if isinstance(m, nn.MaxPool2d)]
        assert len(pools) >= 5, (
            f"应当至少有 5 个最大池化层（b1、b2 各一个，b3–b5 各一个），"
            f"找到 {len(pools)} 个"
        )
        pool_outs = []
        inc_outs = []
        handles = []
        for m in pools[:2]:
            handles.append(m.register_forward_hook(
                lambda mod_, i, o: pool_outs.append(o.detach())))
        for m in net.modules():
            if isinstance(m, mod.Inception):
                handles.append(m.register_forward_hook(
                    lambda mod_, i, o: inc_outs.append(o.detach())))
        try:
            with torch.no_grad():
                net(torch.randn(2, 1, 224, 224))
        finally:
            for h in handles:
                h.remove()

        assert len(pool_outs) == 2, (
            f"应当记录到 b1、b2 两个池化的输出，得到 {len(pool_outs)} 个"
        )
        for i, out in enumerate(pool_outs):
            assert (out >= 0).all(), (
                f"b{i + 1} 的池化输出出现负值（最小 {out.min().item():.4f}）。"
                f"b{i + 1} 的每个卷积后都要接 ReLU"
            )
        assert inc_outs, "没有找到任何 Inception 模块，检查 make_googlenet 是不是用了它"
        assert len(inc_outs) == 9, (
            f"应当有 9 个 Inception（b3 两个、b4 五个、b5 两个），"
            f"实际前向里出现 {len(inc_outs)} 个"
        )
        for i, out in enumerate(inc_outs):
            assert (out >= 0).all(), (
                f"第 {i + 1} 个 Inception 的输出出现负值（最小 "
                f"{out.min().item():.4f}），它内部的卷积后漏了 ReLU"
            )

    @case("T4 GoogLeNet 前向与参数量")
    def _t4b():
        net = mod.make_googlenet()
        out = net(torch.randn(1, 1, 224, 224))
        assert tuple(out.shape) == (1, 10), (
            f"(1,1,224,224) 应输出 (1,10)，得到 {tuple(out.shape)}"
        )
        n = count_params(net)
        assert n == 5977530, (
            f"参数量应为 5,977,530，得到 {n:,}"
        )
        pools = [m for m in net.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
        assert pools, "缺少全局平均池化"

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
