"""B27 参考答案。做完再看。"""

import torch


# ---------- T1 交并比 ----------

def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """两个框的交并比。

    框用中心格式 (cx, cy, w, h)，单位是像素。返回 Python float。
    """
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(ix2 - ix1, 0.0)
    inter_h = max(iy2 - iy1, 0.0)
    inter = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


# ---------- T2 生成锚框 ----------

def make_anchors(h: int, w: int, sizes: list[float],
                 ratios: list[float]) -> torch.Tensor:
    """在 (h, w) 的图上每个像素生成锚框。

    sizes 与 ratios 是相对比例。返回形状 (h*w*(len(sizes)+len(ratios)-1), 4)，
    每行是 (cx, cy, w_a, h_a)。
    """
    combos = [(sizes[0], r) for r in ratios]
    combos += [(s, ratios[0]) for s in sizes[1:]]
    out = []
    for cy in range(h):
        for cx in range(w):
            for s, r in combos:
                wa = w * s * (r ** 0.5)
                ha = h * s / (r ** 0.5)
                out.append([float(cx), float(cy), wa, ha])
    return torch.tensor(out, dtype=torch.float32)


# ---------- T3 标注锚框 ----------

def assign_anchors(anchors: torch.Tensor, gt_boxes: torch.Tensor,
                   iou_threshold: float = 0.5) -> torch.Tensor:
    """给每个锚框分配真实框的下标，-1 表示背景。

    两步：先给每个真实框强制分配 IoU 最大的锚框；再按阈值给其余锚框分配。
    """
    n_a, n_g = anchors.shape[0], gt_boxes.shape[0]
    ious = torch.zeros(n_a, n_g)
    for i in range(n_a):
        for j in range(n_g):
            ious[i, j] = box_iou(anchors[i], gt_boxes[j])

    assigned = torch.full((n_a,), -1, dtype=torch.long)

    # 第一步：每个真实框挑一个锚框
    best_anchor = ious.argmax(dim=0)          # (n_g,)
    for j in range(n_g):
        assigned[best_anchor[j]] = j

    # 第二步：剩下的按阈值分配
    best_iou, best_gt = ious.max(dim=1)
    for i in range(n_a):
        if assigned[i] == -1 and best_iou[i] >= iou_threshold:
            assigned[i] = best_gt[i]
    return assigned


# ---------- T4 非极大值抑制 ----------

def nms(boxes: torch.Tensor, scores: torch.Tensor,
        iou_threshold: float = 0.5) -> torch.Tensor:
    """返回保留下来的下标，按分数从高到低。

    boxes 形状 (n, 4)，中心格式。scores 形状 (n,)。
    """
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            if box_iou(boxes[i], boxes[j]) <= iou_threshold:
                rest.append(j)
        order = rest
    return torch.tensor(keep, dtype=torch.long)


# ---------- T5 转置卷积输出尺寸 ----------

def deconv_output_size(h_in: int, k: int, stride: int = 1,
                       padding: int = 0) -> int:
    """转置卷积的输出边长。"""
    return (h_in - 1) * stride - 2 * padding + k


"""T6 改错

第 1 处：`inter_w = ix2 - ix1` 没做截断
    两个框不重叠时 `ix2 - ix1` 是负数，面积算成负数。
    改法：`max(ix2 - ix1, 0)`，高同理。

第 2 处：`union` 算成了 `area_a + area_b`
    少了减交集。并集 = A + B − 交集。
    改法：`union = area_a + area_b - inter`。

第 3 处：`for j in order` 里拿 `i` 与 `j` 比时用了 `>=` 阈值又保留了
    逻辑写反了。要保留 IoU 小于等于阈值的框。
    改法：`if box_iou(boxes[i], boxes[j]) <= iou_threshold: rest.append(j)`。
"""


if __name__ == "__main__":
    a = torch.tensor([1.0, 1.0, 2.0, 2.0])
    b = torch.tensor([2.0, 2.0, 2.0, 2.0])
    print("T1 IoU:", round(box_iou(a, b), 4))

    an = make_anchors(4, 4, [0.3, 0.6], [0.5, 1.0, 2.0])
    print("T2 锚框数:", an.shape)

    gt = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    asg = assign_anchors(an, gt)
    print("T3 分到真实框的锚框数:", int((asg >= 0).sum()))

    boxes = torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.1, 1.1, 2.0, 2.0],
                          [5.0, 5.0, 2.0, 2.0]])
    sc = torch.tensor([0.9, 0.8, 0.7])
    print("T4 NMS 保留:", nms(boxes, sc).tolist())

    print("T5 转置卷积 16->", deconv_output_size(16, 3, 2, 1))
