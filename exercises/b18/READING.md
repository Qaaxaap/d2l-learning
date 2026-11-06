# B18 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b18-bn-resnet-densenet.md` | 40 分钟 |
| 纸质书 | 5.10 批量归一化、5.11 残差网络、5.12 稠密连接网络 | 45 分钟 |
| 电子版 | [7.5](https://zh.d2l.ai/chapter_convolutional-modern/batch-norm.html)、[7.6](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)、[7.7](https://zh.d2l.ai/chapter_convolutional-modern/densenet.html) | 45 分钟 |

## 读的时候要能回答

1. BN 训练时做哪四步？`gamma` 与 `beta` 是干什么的？
2. 全连接层与卷积层的 BN 在哪个维度上算均值方差？
3. 预测时 BN 用什么统计量？`momentum` 指什么？
4. 残差块解决什么问题？改通道数时跳跃路径要做什么？
5. DenseNet 的稠密块怎么连接？通道数怎么变？
6. `net.eval()` 对 BN 有什么影响？

## 顺序

先读 BN（5.10），它是这一章唯一需要**手写**的东西。重点看"从零开始实现"那一节，
把训练与预测两种模式的区别弄清楚。

再读 ResNet（5.11）。残差块只有十来行，但它是后面所有深网络的基础结构。

最后读 DenseNet（5.12）。它把 ResNet 的"相加"换成"拼接"，通道数每块增长。

## 三处要留意

**BN 的 `keepdim`**：卷积输入是 `(N,C,H,W)`，算出来的均值如果是 `(C,)` 就没法相减。
必须 `keepdim=True` 让它是 `(1,C,1,1)`。

**BN 在训练与预测两种模式下算法不同**。这是它和 Dropout 一样需要 `net.eval()` 的原因。

**残差是"相加"、稠密是"拼接"**。相加要求形状完全一致，拼接只要求高宽一致。
