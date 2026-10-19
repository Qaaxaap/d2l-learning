# B11 口试题

关书作答。答不上来不要猜，说"不知道"，我讲完你复述一遍。

**答题要求**：用中文和代码描述即可，**公式不必写 LaTeX**。

题目分两档：

- **必会**（Q1、Q2、Q3、Q5）：后面每一章都在用
- **了解**（Q4、Q6、Q7、Q8）：知道有这回事即可

## 必会

**Q1** 正向传播与反向传播分别按什么顺序？反向为什么要反过来？给出两个理由。

<details><summary>参考答点</summary>

正向从输入到输出，反向从输出到输入。

两个理由：

1. **依赖关系决定顺序**：算 $\partial J/\partial\mathbf{W}^{(1)}$ 要用到 $\partial J/\partial\mathbf{z}$，
   而后者要用到 $\partial J/\partial\mathbf{h}$，每个梯度都依赖它后面的量
2. **中间结果可以复用**：同一个 $\partial J/\partial\mathbf{h}$ 被两层权重共享，
   从后往前算一遍就够

</details>

**Q2** 反向传播经过一个线性层时，两件事各是什么？
（对输入求导、对权重求导）

<details><summary>参考答点</summary>

设正向是 $\mathbf{z} = \mathbf{W}\mathbf{x}$，上游传来 $\partial J/\partial\mathbf{z}$。

- **对输入求导**：$\partial J/\partial\mathbf{x} = \mathbf{W}^\top\frac{\partial J}{\partial\mathbf{z}}$，
  乘权重的**转置**
- **对权重求导**：$\partial J/\partial\mathbf{W} = \frac{\partial J}{\partial\mathbf{z}}\mathbf{x}^\top$，
  上游梯度乘**输入的转置**

两处都出现转置，这是矩阵求导形状约定决定的。

</details>

**Q3** 反向经过激活函数时用哪种乘法？为什么？

<details><summary>参考答点</summary>

**逐元素乘**：$\partial J/\partial\mathbf{z} = \frac{\partial J}{\partial\mathbf{h}} \odot \phi'(\mathbf{z})$。

因为激活函数是逐元素运算，每个元素只影响对应的那个输出元素，
所以局部导数也是逐元素作用，不需要矩阵乘。

</details>

**Q5** 30 层、权重都是 0.2 的网络输出会怎样？对应的训练现象是什么？

<details><summary>参考答点</summary>

输出是输入的 $0.2^{30}\approx 1\times10^{-21}$ 倍，**衰减到消失**。

反向传播时梯度同样在乘这些权重，所以梯度也接近 0。训练现象是参数几乎不动、
损失下降极慢——容易被误认为"学习率太小"。梯度爆炸时反过来，损失会突然变成 `nan`。

两者的症状都不是"报错说数值不好"，而是训练不收敛。

</details>

## 了解

**Q4** 链式法则里的 `prod` 为什么不是普通乘法？

<details><summary>参考答点</summary>

矩阵求导的结果形状取决于约定。`prod` 会**按两个输入的形状**决定要不要转置、
要不要调换位置，然后再做乘法。

它的作用是把这套约定包装起来，让公式写起来不依赖具体形状。

</details>

**Q6** sigmoid 为什么让梯度衰减更快？

<details><summary>参考答点</summary>

sigmoid 的导数最大只有 0.25。反向每过一层就乘上不超过 0.25 的数，
层数一多连乘的结果迅速接近 0。

ReLU 在正半轴导数为 1，不衰减也不放大，缓解了这个问题——这是 B08 说
"隐藏层默认用 ReLU"的深层原因。

</details>

**Q7** 权重必须随机初始化吗？偏置呢？

<details><summary>参考答点</summary>

权重必须随机。如果所有隐藏单元初始化成相同的值，正向算出相同的值、
反向得到相同的梯度、更新后依然相同——无论有多少个单元，隐藏层等价于 1 个。

偏置可以全零，因为随机权重已经让各单元不同了。

</details>

**Q8** Xavier 的区间是什么？它想保证哪两件事？

<details><summary>参考答点</summary>

$$U\left(-\sqrt{\frac{6}{a+b}},\ \sqrt{\frac{6}{a+b}}\right)$$

$a$ 是输入个数，$b$ 是输出个数。

它想保证：初始化后每层**输出**的方差不取决于输入个数 $a$，每层**梯度**的方差不取决于
输出个数 $b$。所以取 $1/(a+b)$ 兼顾两边。

</details>

## 评分

- **必会**四题（Q1、Q2、Q3、Q5）全对才算通过
- **了解**四题答不上来不扣分
- Q2 是手写反向传播的核心，答错进错题本
