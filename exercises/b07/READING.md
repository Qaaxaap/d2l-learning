# B07 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 纸质书 | 3.4 softmax 回归、3.5 图像分类数据集、3.6 从零开始实现、3.7 简洁实现 | 70 分钟 |
| 电子版 | [3.4](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)、[3.5](https://zh.d2l.ai/chapter_linear-networks/image-classification-dataset.html)、[3.6](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html)、[3.7](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html) | 70 分钟 |
| 讲义 | `notes/b07-softmax.md` | 40 分钟 |

四节里 3.4 是原理（三件事：输出怎么变概率、损失怎么定义、结果怎么评估），
3.5 是数据，3.6 与 3.7 是两份实现。

## 读的时候要能回答

1. 为什么不能把类别编号当回归目标？
2. softmax 做了什么？它的两个性质各解决什么问题？
3. 为什么取 `argmax` 时不必先算 softmax？
4. 分类为什么不用平方损失？举一组预测说明它的判断与任务目标相反。
5. 交叉熵公式里为什么只剩一项？
6. 交叉熵对 logits 的导数是什么？这个结果解释了什么？
7. `torch.exp(o) / torch.exp(o).sum()` 什么时候出问题？正确写法做了什么变形？
8. `nn.CrossEntropyLoss` 接收 logits 还是概率？传错了会怎样？
9. `softmax(dim=1)` 写成 `dim=0` 会看到什么？
10. 图像为什么先要拉平成向量？

## 顺序

先读纸质书 3.4（只有公式与文字，没有代码），再读 3.5 与 3.6、3.7，最后读第二版对应四节与讲义。

## 三处要留意

**解锁点**：纸质书这一章有四个函数标着"本函数已保存在 d2lzh 包中方便以后使用"——
`get_fashion_mnist_labels`、`show_fashion_mnist`（3.5）、`evaluate_accuracy`、`train_ch3`（3.6）。
按仓库规则，在书上出现那句话之前要自己写。

**`CrossEntropyLoss` 内含 softmax**：纸质书用的 Gluon 版损失与 torch 的
`nn.CrossEntropyLoss` 都是接收 logits、内部做 softmax 的。讲义 6.2 节有实测数据。
这一条最容易出错，出错不报错，只是训练变慢。

**数值稳定性**：纸质书没展开讲，讲义 5.2 节有实测（朴素写法在 logits=1000 时给出 `nan`）。
从零实现的那一题会考这个。
