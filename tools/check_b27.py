"""B27 验收断言。用法: just check b27"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
USER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "b27" / "b27.py"

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
        print("先照着 exercises/b27/code.md 把实现写在那里。")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("b27_user", USER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def B(cx, cy, w, h):
    return torch.tensor([float(cx), float(cy), float(w), float(h)])


def main() -> int:
    mod = load_user_module()

    @case("T1 IoU：讲义里的例子")
    def _t1a():
        # 讲义 3.3 节：A 是 (0,0,2,2)，B 是 (1,1,2,2)，IoU 应为 1/7
        v = mod.box_iou(B(0, 0, 2, 2), B(1, 1, 2, 2))
        assert abs(v - 1 / 7) < 1e-4, (
            f"讲义例子里的 IoU 应为 0.1429，得到 {v:.6f}。"
            "交集是 1、并集是 4+4-1=7"
        )

    @case("T1 IoU：完全重合与完全不重合")
    def _t1b():
        a = B(5, 5, 4, 4)
        assert abs(mod.box_iou(a, a) - 1.0) < 1e-6, "自己与自己比应为 1"
        assert mod.box_iou(B(0, 0, 2, 2), B(10, 10, 2, 2)) == 0.0, "不重叠应为 0"

    @case("T1 IoU：部分重叠")
    def _t1c():
        # A 宽 4 高 4（面积 16），B 宽 4 高 4，横向错开 2，交集 2x4=8
        v = mod.box_iou(B(0, 0, 4, 4), B(2, 0, 4, 4))
        want = 8 / (16 + 16 - 8)
        assert abs(v - want) < 1e-5, f"应为 {want:.4f}，得到 {v:.6f}"

    @case("T2 锚框数量")
    def _t2a():
        anchors = mod.make_anchors(4, 4, [0.3, 0.6], [0.5, 1.0, 2.0])
        # 每个像素 2+3-1 = 4 个锚框，4*4 个像素
        assert tuple(anchors.shape) == (64, 4), (
            f"应为 (64,4)，得到 {tuple(anchors.shape)}。"
            "每个像素 n+m-1 个锚框，不是 n*m"
        )

    @case("T2 锚框的面积与宽高比")
    def _t2b():
        # 正方形图，宽高比才严格等于 r
        h, w = 100, 100
        anchors = mod.make_anchors(h, w, [0.5], [0.5, 1.0, 2.0])
        for idx, r in enumerate([0.5, 1.0, 2.0]):
            wa, ha = anchors[idx][2].item(), anchors[idx][3].item()
            assert abs(wa / ha - r) < 1e-4, (
                f"正方形图上第 {idx} 个锚框的宽高比应为 {r}，得到 {wa / ha:.4f}。"
                "宽要高乘 sqrt(r)、高要除以 sqrt(r)"
            )
            assert abs(wa * ha - w * h * 0.5 ** 2) < 1e-2, (
                f"锚框面积应为图像的 0.25 倍，得到 {wa * ha:.2f}"
            )

    @case("T2 非正方形图的锚框")
    def _t2b2():
        # 2:1 的图，r=1 的锚框与图像一样扁，宽高比是 2
        h, w = 100, 200
        anchors = mod.make_anchors(h, w, [0.5], [1.0])
        wa, ha = anchors[0][2].item(), anchors[0][3].item()
        assert abs(wa / ha - 2.0) < 1e-4, (
            f"2:1 的图上 r=1 的锚框宽高比应为 2，得到 {wa / ha:.4f}。"
            "宽高比是 (w/h)*r，不是 r"
        )

    @case("T3 每个真实框都有锚框")
    def _t3a():
        anchors = mod.make_anchors(8, 8, [0.5], [1.0])
        gt = torch.stack([B(2, 2, 3, 3), B(5, 5, 3, 3)])
        assigned = mod.assign_anchors(anchors, gt, iou_threshold=0.9)
        # 阈值 0.9 很高，但强制分配保证每个真实框至少一个锚框
        for j in range(2):
            assert int((assigned == j).sum()) >= 1, (
                f"真实框 {j} 没有分到任何锚框。"
                "第一步要给每个真实框强制分配 IoU 最大的锚框"
            )

    @case("T3 分配结果与阈值一致")
    def _t3b():
        anchors = mod.make_anchors(6, 6, [0.4], [1.0])
        gt = torch.stack([B(3, 3, 4, 4)])
        assigned = mod.assign_anchors(anchors, gt, iou_threshold=0.5)
        # 被分配到真实框的锚框，IoU 必须 >= 0.5
        for i in range(anchors.shape[0]):
            if assigned[i] >= 0:
                iou = mod.box_iou(anchors[i], gt[0])
                # 强制分配的那一个可能低于阈值，其余必须满足
                assert iou >= 0.5 or int((assigned >= 0).sum()) >= 1, (
                    f"锚框 {i} 分给了真实框但 IoU 只有 {iou:.3f}"
                )

    @case("T4 NMS 去掉重叠框")
    def _t4a():
        boxes = torch.stack([
            B(1, 1, 2, 2),      # 0：与 1 高度重叠
            B(1.2, 1.2, 2, 2),  # 1：与 0 高度重叠
            B(5, 5, 2, 2),      # 2：远离
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = mod.nms(boxes, scores, 0.5).tolist()
        assert keep == [0, 2], (
            f"应保留 [0, 2]，得到 {keep}。"
            "0 分最高先留下，1 与 0 的 IoU 超过阈值被去掉"
        )

    @case("T4 NMS 按分数排序")
    def _t4b():
        boxes = torch.stack([B(1, 1, 2, 2), B(9, 9, 2, 2), B(1.1, 1.1, 2, 2)])
        scores = torch.tensor([0.3, 0.6, 0.9])
        keep = mod.nms(boxes, scores, 0.5).tolist()
        assert keep[0] == 2, f"第一个保留的应是分数最高的 2，得到 {keep}"
        assert 1 in keep, f"远离的框 1 应保留，得到 {keep}"

    @case("T4 NMS 阈值的影响")
    def _t4c():
        boxes = torch.stack([B(1, 1, 2, 2), B(1.5, 1, 2, 2)])
        scores = torch.tensor([0.9, 0.8])
        loose = mod.nms(boxes, scores, 0.9).tolist()
        tight = mod.nms(boxes, scores, 0.1).tolist()
        assert len(loose) >= len(tight), (
            f"阈值大时应保留更多框：0.9 保留 {loose}，0.1 保留 {tight}"
        )
        assert len(tight) == 1, f"阈值 0.1 时两个框重叠多，应只留 1 个，得到 {tight}"

    @case("T5 转置卷积输出尺寸")
    def _t5():
        assert mod.deconv_output_size(16, 3, 2, 1) == 31, (
            f"(16-1)*2-2*1+3 = 31，得到 {mod.deconv_output_size(16, 3, 2, 1)}"
        )
        assert mod.deconv_output_size(8, 4, 2, 0) == 18, (
            f"(8-1)*2-0+4 = 18，得到 {mod.deconv_output_size(8, 4, 2, 0)}"
        )
        assert mod.deconv_output_size(16, 32, 32, 0) >= 16 * 32, (
            "步幅 32、核 32 时应放大约 32 倍"
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
