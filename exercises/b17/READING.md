# B17 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b17-modern-cnn.md` | 40 分钟 |
| 纸质书 | 5.6 AlexNet、5.7 VGG、5.8 NiN、5.9 GoogLeNet | 50 分钟 |
| 电子版 | [7.1](https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html)、[7.2](https://zh.d2l.ai/chapter_convolutional-modern/vgg.html)、[7.3](https://zh.d2l.ai/chapter_convolutional-modern/nin.html)、[7.4](https://zh.d2l.ai/chapter_convolutional-modern/googlenet.html) | 50 分钟 |

## 读的时候要能回答

1. AlexNet 相对 LeNet 改了哪四处？哪一处最关键？
2. 224×224 输入、11×11 核、步幅 4、无填充，输出多大？
3. VGG 的 `conv_arch` 里两个数字各表示什么？VGG-11 展平后是多少维？
4. NiN 用什么替代全连接层？为什么这样能减少参数？
5. Inception 块里线路 2、3 前面为什么加 1×1 卷积？
6. 全局平均池化怎么做？为什么能取代最后的全连接层？

## 顺序

四个网络按时间顺序读，每个都先看**结构**再看**为什么这么设计**。

**读的时候反复算形状**。这一章的核心技能是"给定结构算出每层输出形状"，
四个网络的结构都能用一条链算下来。

AlexNet 里有两处向下取整（`⌊(224-11)/4⌋+1` 和 `⌊(54-3)/2⌋+1`），最容易算错。

## 三处要留意

**讲义 4.4 节的参数量是本机实测**，与书上论文里的数字不同：
论文的 AlexNet 是 3 通道输入，这里用单通道，少了第一层的 3 倍参数。

**四个网络这一章都不训练。** 它们都在 ImageNet 尺度设计，本机跑 Fashion-MNIST
只是验证结构能前向。代码题也只验形状与参数量。

**NiN 与 GoogLeNet 没有全连接层**（除了 GoogLeNet 最后那个输出层）。
这是这一章最重要的对比结论：参数量能降一个数量级。
