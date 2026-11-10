# B25 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b25-optimizers.md` | 40 分钟 |
| 纸质书 | 7.1 优化与深度学习、7.4 动量法、7.5 AdaGrad、7.6 RMSProp、7.8 Adam | 50 分钟 |
| 电子版 | [11.1](https://zh.d2l.ai/chapter_optimization/optimization-intro.html)、[11.6](https://zh.d2l.ai/chapter_optimization/momentum.html)–[11.10](https://zh.d2l.ai/chapter_optimization/adam.html) | 50 分钟 |

## 读的时候要能回答

1. SGD 的核心问题是什么？举一个两条方向尺度差很多倍的例子。
2. 动量法的两个公式？为什么它能同时加速横向与压制纵向震荡？
3. 书上的 `v` 与 torch 的 `momentum_buffer` 差什么？
4. AdaGrad 的自适应学习率怎么来的？缺陷是什么？
5. RMSProp 相对 AdaGrad 改了哪一处？
6. Adam 的偏差修正是干什么的？
7. 为什么保存优化器状态与保存模型参数同样重要？

## 顺序

先读 7.1 与讲义第 1 节，把"SGD 为什么不够用"想清楚。这是后面所有算法的动机。

再依次读动量法、AdaGrad、RMSProp、Adam。四个算法是递进的：
每个都在补前一个的短板。

最后看讲义第 8 节，把书上的符号与 torch 的参数对应起来。

## 三处要留意

**动量法的 `v` 在书上含 `lr`，torch 的 `momentum_buffer` 不含**。写代码时混用会差 `lr` 倍。

**Adam 的偏差修正不能省**。状态初始为 0 会让前几步的移动平均偏小，
除以 `1 - beta^t` 才能归一化。

**RMSProp 的 `gamma` 在 torch 里叫 `alpha`**。
