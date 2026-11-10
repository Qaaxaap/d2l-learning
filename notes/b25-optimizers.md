# B25 优化算法

前面所有章节都用 `torch.optim.SGD` 更新参数。这一章打开它，看里面还能换成什么。

如果训练不收敛、调学习率调到崩溃，问题往往在这里：**SGD 对每个参数用同一个学习率**，
而不同方向的梯度大小可能差几个数量级。动量、AdaGrad、RMSProp、Adam 都是在改这一件事。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 为什么要换优化器 | 7.1 优化与深度学习 | [11.1](https://zh.d2l.ai/chapter_optimization/optimization-intro.html) |
| 2 小批量随机梯度下降回顾 | 7.3 小批量随机梯度下降 | 11.5 |
| 3 动量法 | 7.4 动量法 | [11.6](https://zh.d2l.ai/chapter_optimization/momentum.html) |
| 4 AdaGrad | 7.5 AdaGrad | [11.7](https://zh.d2l.ai/chapter_optimization/adagrad.html) |
| 5 RMSProp | 7.6 RMSProp | [11.8](https://zh.d2l.ai/chapter_optimization/rmsprop.html) |
| 6 AdaDelta | 7.7 AdaDelta | 11.9 |
| 7 Adam | 7.8 Adam | [11.10](https://zh.d2l.ai/chapter_optimization/adam.html) |
| 8 与 torch 的对应 | 无，本节补充 | 无，本节补充 |
| 9 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 7.4–7.8 里 `sgd_momentum`、`adagrad`、`rmsprop`、`adadelta`、`adam`
都标了"已保存在 `d2lzh` 中"。这一章要自己写。

## 1 为什么要换优化器

### 1.1 同一个学习率不能适配所有方向

考虑 $f(x_1, x_2) = 0.1x_1^2 + 2x_2^2$。等高线是椭圆的，横向很扁、纵向很瘦。

梯度是 $(0.2x_1, 4x_2)$。**纵向的梯度是横向的 20 倍。**

用固定的学习率 $\eta$：

- $\eta$ 大到能让 $x_1$ 快速下降，$x_2$ 就会震荡甚至发散
- $\eta$ 小到让 $x_2$ 稳定，$x_1$ 就几乎不动

**SGD 的困境**：一个标量学习率要同时应付差 20 倍的两条方向。
真实网络的参数动辄上百万，各方向的梯度尺度差异比 20 倍大得多。

书上把这个叫**病态条件**（ill-conditioning）。它有两个后果：

1. 需要很多步才能到最优点（横向慢慢挪）
2. 学习率稍大就在陡的方向上震荡

### 1.2 换优化器改的是什么

所有下面的算法都在回答同一个问题：**每个参数的学习率该多大？**

| 算法 | 想法 |
|---|---|
| 动量法 | 累积方向一致的梯度，抵消来回震荡的 |
| AdaGrad | 按每个参数的历史梯度大小**调小**它的学习率 |
| RMSProp | 同上，但只看最近的梯度（不会一直降） |
| Adam | 动量 + RMSProp + 偏差修正 |

## 2 回顾：小批量随机梯度下降

这一章的基线。每步：

$$\boldsymbol{g}_t = \frac{1}{|\mathcal{B}_t|}\sum_{i \in \mathcal{B}_t} \nabla f(\boldsymbol{x}_{t-1}; \text{样本}_i)$$

$$\boldsymbol{x}_t = \boldsymbol{x}_{t-1} - \eta \boldsymbol{g}_t$$

**$\eta$ 对所有参数是同一个标量。** 这是后面所有算法要改的地方。

## 3 动量法

### 3.1 公式

$$\boldsymbol{v}_t \leftarrow \gamma \boldsymbol{v}_{t-1} + \eta_t \boldsymbol{g}_t$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{v}_t$$

- $\boldsymbol{v}$ 是**速度**，与参数同形状，初始化为 0
- $\gamma \in [0, 1)$ 是动量系数，常用 0.9
- $\gamma = 0$ 时退化为普通 SGD

### 3.2 为什么有用

把 $\boldsymbol{v}_t$ 展开：

$$\boldsymbol{v}_t = \eta_t \boldsymbol{g}_t + \gamma \eta_{t-1}\boldsymbol{g}_{t-1} + \gamma^2 \eta_{t-2}\boldsymbol{g}_{t-2} + \cdots$$

**它是过去所有梯度的指数加权移动平均**，越久远的权重越小。

两种情形：

- **方向一致**（横向）：各步梯度同号，加起来变大，移动加速
- **来回震荡**（纵向）：各步梯度符号相反，相互抵消，震荡被压住

所以动量法**在同一个学习率下同时解决"横向太慢"与"纵向震荡"**。

### 3.3 与 torch 的对应

`torch.optim.SGD(momentum=0.9)` 的内部是：

$$\boldsymbol{b}_t \leftarrow \gamma \boldsymbol{b}_{t-1} + \boldsymbol{g}_t$$
$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \eta \boldsymbol{b}_t$$

**注意 $\eta$ 的位置不同**：书上的 $\boldsymbol{v}$ 里含 $\eta$，torch 的 $\boldsymbol{b}$ 不含。

两者的关系：令 $\boldsymbol{v}_t = \eta \boldsymbol{b}_t$，代入书上的式子：
$\eta \boldsymbol{b}_t = \gamma \eta \boldsymbol{b}_{t-1} + \eta \boldsymbol{g}_t$，
两边约掉 $\eta$ 就是 torch 的写法。

**等价，但 $\boldsymbol{v}$ 与 $\boldsymbol{b}$ 的数值差一个 $\eta$ 倍**。写代码时不要混。

## 4 AdaGrad

### 4.1 公式

$$\boldsymbol{s}_t \leftarrow \boldsymbol{s}_{t-1} + \boldsymbol{g}_t \odot \boldsymbol{g}_t$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \frac{\eta}{\sqrt{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t$$

- $\boldsymbol{s}$ 是**梯度平方的累加**，与参数同形状，初始化为 0
- $\odot$ 是按元素乘
- $\epsilon$ 是防除零的小数，如 $10^{-6}$

### 4.2 效果

分母 $\sqrt{\boldsymbol{s}_t}$ 是"这个参数历史上梯度有多大"的度量：

- **梯度一直大** → $s$ 大 → 学习率被除小
- **梯度一直小** → $s$ 小 → 学习率保持大

**每个参数有了自己的学习率**。

### 4.3 缺陷

$\boldsymbol{s}_t$ 只增不减，学习率**单调递减**。

训练后期 $\boldsymbol{s}$ 累积得很大，学习率趋近 0，参数几乎不再更新。
如果早期学习率降得太快而解还不好，后面就再也走不动了。

这个缺陷是 RMSProp 出现的原因。

## 5 RMSProp

### 5.1 公式

$$\boldsymbol{s}_t \leftarrow \gamma \boldsymbol{s}_{t-1} + (1 - \gamma) \boldsymbol{g}_t \odot \boldsymbol{g}_t$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \frac{\eta}{\sqrt{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t$$

**唯一的改动**：$\boldsymbol{s}$ 从"累加"变成"指数加权移动平均"。

$\gamma$ 常用 0.9，$1/(1-\gamma) = 10$ 意味着只看最近约 10 步的梯度平方。

**解决了 AdaGrad 的缺陷**：$\boldsymbol{s}$ 会遗忘很久以前的梯度，学习率不会一直降。

## 6 AdaDelta

书上 7.7 节。想法是**连学习率都不用调**：用参数更新的平方的移动平均再去缩放学习率。

$$\Delta\boldsymbol{x}_t = -\sqrt{\frac{\boldsymbol{s}'_t + \epsilon}{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t$$

其中 $\boldsymbol{s}'$ 是更新量平方的移动平均。

**实践里用得比 Adam 少**。这一章不要求实现，知道它解决了"不用调学习率"这一件事即可。

## 7 Adam

### 7.1 公式

Adam = 动量 + RMSProp + **偏差修正**。

$$\boldsymbol{v}_t \leftarrow \beta_1 \boldsymbol{v}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t$$

$$\boldsymbol{s}_t \leftarrow \beta_2 \boldsymbol{s}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t \odot \boldsymbol{g}_t$$

$$\hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1 - \beta_1^t}, \qquad
\hat{\boldsymbol{s}}_t = \frac{\boldsymbol{s}_t}{1 - \beta_2^t}$$

$$\boldsymbol{g}'_t = \frac{\eta \hat{\boldsymbol{v}}_t}{\sqrt{\hat{\boldsymbol{s}}_t} + \epsilon}$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}'_t$$

超参数：$\beta_1 = 0.9$、$\beta_2 = 0.999$、$\epsilon = 10^{-8}$。

### 7.2 为什么需要偏差修正

$\boldsymbol{v}_0 = \boldsymbol{s}_0 = 0$，所以第一步：

$$\boldsymbol{v}_1 = (1 - \beta_1)\boldsymbol{g}_1 = 0.1 \boldsymbol{g}_1$$

**只有真实梯度的 1/10**。前几步 $\boldsymbol{v}_t$ 都偏小（因为初始为 0 拉低了平均）。

把过去各步的权重加起来：$(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i} = 1 - \beta_1^t$。
$t$ 小时这个和远小于 1；$t$ 大时趋近 1。

**除以 $1 - \beta_1^t$ 把权重和归一化**，就是偏差修正。$\boldsymbol{s}$ 同理用 $1 - \beta_2^t$。

### 7.3 为什么 Adam 是默认选择

- 自带动量，不用手调
- 自带自适应学习率
- 偏差修正让前几步也不至于太小

**代价**：内存占用是 SGD 的三倍（要存 $\boldsymbol{v}$ 与 $\boldsymbol{s}$ 两份状态），
而且有些任务上泛化不如调好的 SGD + 动量（尤其是 CV 任务）。

## 8 与 torch 的对应

### 8.1 现成的优化器

```python
torch.optim.SGD(params, lr=0.1, momentum=0.9)
torch.optim.Adagrad(params, lr=0.01)
torch.optim.RMSprop(params, lr=0.01, alpha=0.99)
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

| 书上的符号 | torch 的参数 |
|---|---|
| $\gamma$（动量系数）| `SGD(momentum=γ)` |
| $\eta$ | `lr` |
| $\gamma$（RMSProp 的衰减）| `RMSprop(alpha=γ)` |
| $\beta_1$、$\beta_2$ | `Adam(betas=(β1, β2))` |
| $\epsilon$ | `eps` |

**注意 RMSProp 的 $\gamma$ 在 torch 里叫 `alpha`**，默认 0.99（书上例子常用 0.9）。

### 8.2 状态存在哪

每个优化器都有 `.state_dict()`，里面是**每个参数对应的状态**：

| 优化器 | 状态 |
|---|---|
| SGD（无动量）| 无 |
| SGD + 动量 | `momentum_buffer` |
| Adagrad | `sum` |
| RMSprop | `square_avg` |
| Adam | `step`、`exp_avg`、`exp_avg_sq` |

**保存优化器状态与保存模型参数同样重要**：从 checkpoint 继续训练时，
动量与二阶矩不恢复，轨迹就接不上。

### 8.3 手写与现成对比的方法

写优化算法时最好的验证方式是**同起点跑同一个 bug**：

1. 用 `torch.optim.Adam` 跑 10 步，记下参数
2. 用自己写的 `adam_step` 跑 10 步，记下参数
3. 两者应当逐个元素相同

这一步比"看着损失下降"可靠得多——损失下降只能说明方向对，不能说明公式对。

## 9 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| `lr` 放进状态里再乘一次 | 更新量差 $\eta$ 倍 | 书上的 $\boldsymbol{v}$ 含 $\eta$，torch 的 $\boldsymbol{b}$ 不含（3.3）|
| 忘了偏差修正 | 前几步更新量只有 1/10 | Adam 的状态初始为 0（7.2）|
| 偏差修正用了 $t$ 而不是当前的 $t$ | 前几步正常、后面偏大 | 每步要更新 $t$，$\beta^t$ 里的 $t$ 是步数 |
| 更新参数时破坏计算图 | 反向传播报错 | 优化器的更新要 `torch.no_grad()` 或 `p.data` |
| 状态初始化成横向量 | 广播后形状不对 | 状态要与参数**同形状** |
| 拿 `.item()` 做更新 | 参数不再需要梯度 | `.item()` 返回 Python 数，脱离计算图 |

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.optim.SGD(params, lr, momentum)` | 动量 SGD | 3.3 |
| `torch.optim.Adagrad(params, lr)` | AdaGrad | 4.1 |
| `torch.optim.RMSprop(params, lr, alpha)` | RMSProp | 5.1 |
| `torch.optim.Adam(params, lr, betas, eps)` | Adam | 7.1 |
| `optimizer.state_dict()` | 取优化器状态 | 8.2 |
| `optimizer.load_state_dict()` | 恢复优化器状态 | 8.2 |
| `p.grad` | 当前梯度 | 2 |
| `torch.no_grad()` | 更新参数时切断计算图 | 9 |

## 自测题

1. SGD 的核心问题是什么？举一个两条方向尺度差很多倍的例子。
2. 动量法的两个公式？为什么它能同时加速横向与压制纵向震荡？
3. 书上的 `v` 与 torch 的 `momentum_buffer` 差什么？
4. AdaGrad 的自适应学习率怎么来的？它的缺陷是什么？
5. RMSProp 相对 AdaGrad 改了哪一处？
6. Adam 的偏差修正是干什么的？不修会怎样？
7. 为什么保存优化器状态与保存模型参数同样重要？

## 答案（做完再看）

**1.** 用同一个标量学习率应付所有参数，而各方向梯度尺度差异大。
$f = 0.1x_1^2 + 2x_2^2$ 的纵向梯度是横向的 20 倍：学习率大到能推 $x_1$，$x_2$ 就震荡。

**2.** $v_t = \gamma v_{t-1} + \eta g_t$、$x_t = x_{t-1} - v_t$。
$v_t$ 是过去梯度的指数加权移动平均：方向一致时同号相加变大（加速），
来回震荡时异号相互抵消（压制）。

**3.** 书上的 $v$ 含学习率，torch 的 `momentum_buffer` 不含。令 $v = \eta b$ 两者等价。
写代码时若把 $v$ 当 $b$ 用，更新量会差 $\eta$ 倍。

**4.** 分母 $\sqrt{s_t}$ 是历史梯度大小的度量，梯度大的参数学习率被除小。
缺陷：$s$ 只增不减，学习率单调递减，训练后期几乎不更新。

**5.** $s$ 从"累加"改成"指数加权移动平均"（$\gamma s + (1-\gamma)g^2$）。
这样会遗忘久远的梯度，学习率不会一直降。

**6.** 状态初始为 0 让前几步的平均值偏小（$v_1 = 0.1g_1$）。除以 $1-\beta^t$ 归一化权重和。
不修的话起步几步更新量只有应有的 1/10，训练前半段变慢。

**7.** 动量与二阶矩是"历史"的一部分。只恢复参数不恢复它们，等价于让优化器失忆重启，
轨迹与断点前接不上，损失可能先跳一下。
