# B25 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：选优化器与排查训练问题时直接用
- **了解**（Q5、Q6、Q7）：知道设计动机即可

## 必会

**Q1** SGD 的核心问题是什么？动量法怎么缓解它？

<details><summary>参考答点</summary>

**核心问题**：所有参数用同一个标量学习率，而各方向的梯度尺度可能差几个数量级。
$f = 0.1x_1^2 + 2x_2^2$ 的纵向梯度是横向的 20 倍，学习率调不到两头都合适。

**动量法**：$v = \gamma v + g$、$x -= lr \cdot v$。
$v$ 是过去梯度的指数加权移动平均——方向一致时同号相加变大（加速），
来回震荡时异号抵消（压制）。

</details>

**Q2** AdaGrad 的自适应学习率怎么来的？它的缺陷是什么？RMSProp 怎么改的？

<details><summary>参考答点</summary>

**AdaGrad**：$s$ 累加梯度平方，学习率变成 $lr/\sqrt{s+\epsilon}$。
梯度一直大的参数学习率被除小，梯度一直小的保持大。

**缺陷**：$s$ 只增不减，学习率单调递减，训练后期几乎不更新。

**RMSProp**：$s$ 从累加改成指数加权移动平均
（$s = \gamma s + (1-\gamma)g^2$），会遗忘久远的梯度，学习率不再一路降到 0。

</details>

**Q3** Adam 的偏差修正是干什么的？不做会怎样？

<details><summary>参考答点</summary>

状态 $v$、$s$ 初始为 0，所以第一步 $v_1 = (1-\beta_1)g_1 = 0.1g_1$，
只有真实梯度的 1/10。

**偏差修正**：除以 $1 - \beta_1^t$ 与 $1 - \beta_2^t$，把权重和归一化。

不做的话前几步更新量只有应有的 1/10，训练前半段变慢。
有个直接的现象：修好的 Adam 第一步更新量约等于 `lr`；不修的话只有 `0.1 * lr`。

</details>

**Q4** 从 checkpoint 继续训练时，为什么要连优化器状态一起存？

<details><summary>参考答点</summary>

动量 $v$ 与二阶矩 $s$ 是"历史"的一部分。只恢复参数不恢复它们，
等于让优化器失忆重启，轨迹与断点前接不上，损失可能先跳一下。

`optimizer.state_dict()` 里存着每个参数对应的状态
（SGD 是 `momentum_buffer`、Adam 是 `exp_avg`/`exp_avg_sq`/`step`）。

</details>

## 了解

**Q5** 书上的动量 $v$ 与 torch 的 `momentum_buffer` 差什么？

<details><summary>参考答点</summary>

书上的 $v$ 里含学习率：$v_t = \gamma v_{t-1} + \eta g_t$、$x_t = x_{t-1} - v_t$。

torch 的 buffer 不含：$b_t = \gamma b_{t-1} + g_t$、$x_t = x_{t-1} - \eta b_t$。

令 $v = \eta b$ 两者等价。写代码时若把 $v$ 当 $b$ 用，更新量会差 $\eta$ 倍。

</details>

**Q6** 为什么有些 CV 任务上 Adam 不如调好的 SGD + 动量？

<details><summary>参考答点</summary>

自适应方法在简单问题上收敛快，但在需要精细泛化的任务上（图像分类的经典网络）
一般化差距明显。可能的原因：自适应学习率在后期过大、对噪声梯度过于敏感。

实践里的常见做法是 CV 用 SGD + 动量 + 学习率衰减，NLP 用 Adam。

</details>

**Q7** Adam 的内存占用比 SGD 大多少？为什么？

<details><summary>参考答点</summary>

**三倍**（相对参数本身）：要存 `exp_avg` 与 `exp_avg_sq` 两份，与参数同形状。

SGD 无动量时不额外占内存；有动量时是两倍。

这也是大模型训练里省显存的一个方向（AdamW 的 8-bit 版本之类）。

</details>

## 评分

- **必会**四题全对才算通过
- Q3 是这一章最容易漏的一处
