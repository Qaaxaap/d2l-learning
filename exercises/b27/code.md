# B27 代码题

把你的实现写在 `work/b27/b27.py` 里，函数名与签名必须和下面一致。

验收：`just check b27`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

写目标检测与语义分割的五个核心组件：交并比、锚框生成、锚框标注、非极大值抑制、
转置卷积的输出尺寸。

**全部是纯 CPU 的数值计算**，没有训练、没有图像。断言一秒内跑完。

---

## T1 交并比

```python
def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """两个框的交并比。

    框用中心格式 (cx, cy, w, h)，单位是像素。返回 Python float。
    """
```

**计算三步**：

1. 把中心格式转成角点：`x1 = cx - w/2`、`x2 = cx + w/2`，`y` 同理。
2. 算交集：左上角取两个框的较大值，右下角取较小值。宽高为负时置 0。
3. 并集 = 面积 A + 面积 B − 交集。相除。

| 检查 | 期望 | 类型 |
|---|---|---|
| A=(0,0,2,2)、B=(1,1,2,2) | 0.1429（1/7）| 精确 |
| 自己与自己 | 1.0 | 精确 |
| 完全不重叠 | 0.0 | 精确 |
| A=(0,0,4,4)、B=(2,0,4,4) | 8/24 = 0.3333 | 精确 |

## T2 生成锚框

```python
def make_anchors(h: int, w: int, sizes: list[float],
                 ratios: list[float]) -> torch.Tensor:
    """在 (h, w) 的图上每个像素生成锚框。

    sizes 与 ratios 是相对比例。返回形状 (h*w*(len(sizes)+len(ratios)-1), 4)，
    每行是 (cx, cy, w_a, h_a)。
    """
```

宽高公式（讲义 4.2）：

$$w_a = w \cdot s \sqrt{r}, \qquad h_a = h \cdot s / \sqrt{r}$$

其中 $w$、$h$ 是**图像**的宽高，不是锚框的。

**组合的选取**：$(s_1, r_1)$、$(s_1, r_2)$、…、$(s_1, r_m)$，
再加上 $(s_2, r_1)$、…、$(s_n, r_1)$。共 $n + m - 1$ 个。

| 检查 | 期望 | 类型 |
|---|---|---|
| `make_anchors(4, 4, [0.3, 0.6], [0.5, 1, 2])` | 形状 `(64, 4)` | 精确 |
| 正方形图上 $r=0.5$ 的锚框 | 宽高比 0.5 | 精确 |
| 锚框面积 | 图像面积的 $s^2$ 倍 | 精确 |
| 2:1 的图上 $r=1$ 的锚框 | 宽高比 2 | 精确 |

**最后一条要留意**：宽高比是 $(w/h)\cdot r$，只有在正方形图上才等于 $r$。

## T3 标注锚框

```python
def assign_anchors(anchors: torch.Tensor, gt_boxes: torch.Tensor,
                   iou_threshold: float = 0.5) -> torch.Tensor:
    """给每个锚框分配真实框的下标，-1 表示背景。

    anchors 形状 (n_a, 4)，gt_boxes 形状 (n_g, 4)，都是中心格式。
    返回形状 (n_a,) 的整型张量。
    """
```

**两步**（讲义 4.3）：

1. **强制分配**：每个真实框挑与它 IoU 最大的那个锚框，强制分配给它。
2. **按阈值分配**：对剩下的锚框，取它与真实框的最大 IoU。超过阈值就分配，否则保持 −1。

**第一步不能省**。只按阈值时，很小的真实框可能匹配不上任何锚框，永远学不到。

| 检查 | 期望 | 类型 |
|---|---|---|
| 阈值 0.9 时每个真实框 | 至少分到 1 个锚框 | 精确 |
| 被分配的锚框 | 与真实框的 IoU 满足阈值（强制分配的那个除外）| 精确 |

## T4 非极大值抑制

```python
def nms(boxes: torch.Tensor, scores: torch.Tensor,
        iou_threshold: float = 0.5) -> torch.Tensor:
    """返回保留下来的下标，按分数从高到低。

    boxes 形状 (n, 4) 中心格式，scores 形状 (n,)。
    """
```

**算法**（讲义 5.2）：

1. 按分数从高到低排序。
2. 取分数最高的框，加入结果。
3. 删除与它 IoU 超过阈值的其他框。
4. 重复 2 到 3，直到没有框剩下。

| 检查 | 期望 | 类型 |
|---|---|---|
| 两个高度重叠的框加一个远离的 | 保留第一个与远离的那个 | 精确 |
| 分数不同的三个框 | 结果第一个是分数最高的 | 精确 |
| 阈值 0.9 与 0.1 | 前者保留的框不少于后者 | 趋势 |

## T5 转置卷积的输出尺寸

```python
def deconv_output_size(h_in: int, k: int, stride: int = 1,
                       padding: int = 0) -> int:
    """转置卷积的输出边长。"""
```

公式（讲义 7.2）：

$$H_{out} = (H_{in} - 1)s - 2p + k$$

| 检查 | 期望 | 类型 |
|---|---|---|
| `(16, 3, 2, 1)` | 31 | 精确 |
| `(8, 4, 2, 0)` | 18 | 精确 |
| `(16, 32, 32, 0)` | 至少 512 | 范围 |

## T6 改错

下面这段想实现 IoU 与 NMS。**三处**问题。

```python
def box_iou(box_a, box_b):
    ax1 = box_a[0] - box_a[2] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ay2 = box_a[1] + box_a[3] / 2
    bx1 = box_b[0] - box_b[2] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    by2 = box_b[1] + box_b[3] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = (ix2 - ix1) * (iy2 - iy1)

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b
    return inter / union


def nms(boxes, scores, iou_threshold=0.5):
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                rest.append(j)
        order = rest
    return keep
```

三处分别指：**交集的宽度**、**并集的计算**、**NMS 的保留条件**。

三段提示：第一处让不重叠的框算出负面积；第二处会高估并集；
第三处会保留重叠的框、删掉不重叠的。
