# B20 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：训练 RNN 时直接用
- **了解**（Q5、Q6、Q7）：知道原理即可

## 必会

**Q1** 梯度裁剪的公式是什么？为什么必须整体按比例缩放，不能逐元素裁剪？

<details><summary>参考答点</summary>

$$\boldsymbol{g} \leftarrow \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

**整体缩放**：所有元素乘同一个系数，只改长度、不改方向。
**逐元素裁剪**：不同元素除以各自的绝对值，等于把每个分量压到 $\pm\theta$，
方向完全变了——优化方向被破坏。

</details>

**Q2** 困惑度的定义是什么？词表大小 1000 时，困惑度 1000 意味着什么？

<details><summary>参考答点</summary>

$\text{perplexity} = \exp(\text{交叉熵})$。

困惑度 1000 = 词表大小，意味着模型在 1000 个候选里**均匀瞎猜**（交叉熵 $\ln 1000$）。
有效模型的困惑度必须小于词表大小。

</details>

**Q3** 相邻采样训练时为什么要对隐藏状态 `detach`？不做会怎样？

<details><summary>参考答点</summary>

相邻采样的隐藏状态跨批量传递。不 `detach` 的话，每个批量的计算图都接着上一批，
梯度会沿着所有历史批量累积，**计算图无限增长**：
显存越用越多，训练越来越慢，最后 OOM。

`detach()` 保留数值、切断梯度，让每个批量的梯度只看当前批量的序列。

</details>

**Q4** RNN 的输出 `Y` 要过 softmax 吗？为什么？

<details><summary>参考答点</summary>

**不要**。`Y` 是 logits（未归一化分数）。

`nn.CrossEntropyLoss` 内部做 log-softmax + NLL，传 logits 进去正好。
外面再做一次 softmax 会让梯度被严重衰减，训练几乎不动。

</details>

## 了解

**Q5** BPTT 的梯度为什么要沿时间反着算？隐藏状态的梯度是怎么递归的？

<details><summary>参考答点</summary>

因为隐藏状态之间有依赖：$\boldsymbol{h}_t$ 影响 $\boldsymbol{h}_{t+1}$，
而 $\boldsymbol{h}_{t+1}$ 又影响输出。所以算 $\partial L/\partial\boldsymbol{h}_t$
需要先知道 $\partial L/\partial\boldsymbol{h}_{t+1}$——从最后一步往前。

$$\frac{\partial L}{\partial \boldsymbol{h}_t} = \boldsymbol{W}_{hh}^\top
\frac{\partial L}{\partial \boldsymbol{h}_{t+1}} + \boldsymbol{W}_{qh}^\top
\frac{\partial L}{\partial \boldsymbol{o}_t}$$

第一项来自"下一步依赖这一步"，第二项来自"这一步的输出依赖这一步"。

</details>

**Q6** 为什么 RNN 的梯度容易爆炸或消失？

<details><summary>参考答点</summary>

展开后隐藏状态的梯度里有 $(\boldsymbol{W}_{hh}^\top)^{T-t}$ 这样的**连乘**。

- $\boldsymbol{W}_{hh}$ 的特征值都小于 1：连乘趋近 0 → **梯度消失**
- 都大于 1：指数增长 → **梯度爆炸**

序列越长（$T-t$ 越大）越严重。这和 B11 前馈网络的梯度消失是同一件事，
只是连乘次数等于时间步数。

</details>

**Q7** one-hot 有什么缺点？后来的词向量怎么解决？

<details><summary>参考答点</summary>

**缺点**：向量长度等于词表大小。词表 5 万时，每个字符一个 5 万维向量，
绝大部分是 0，存储和计算都浪费。而且任意两个不同字符的 one-hot 向量**距离都相同**，
表达不了"相近"。

**词向量**（B28 的 word2vec）把字符映射到低维稠密向量（几十到几百维），
并通过训练让语义相近的字符离得近。

</details>

## 评分

- **必会**四题全对才算通过
- Q1 与 Q4 是最容易写错实现的两处
