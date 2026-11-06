# B21 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：理解与实现门控单元的核心
- **了解**（Q5、Q6、Q7）：知道设计动机即可

## 必会

**Q1** 门控解决什么问题？门为什么用 sigmoid 算？

<details><summary>参考答点</summary>

解决**长序列上的梯度消失**。原始 RNN 每步整体重写隐藏状态，信息传不远。

门用 sigmoid 算，值域 [0,1]，与信息做按元素乘起到"通过多少"的开关作用。
接近 1 让信息通过，接近 0 挡住。

用 tanh 就不行——值域 [-1,1] 会出现负的"通过比例"，语义乱掉。

</details>

**Q2** GRU 的重置门与更新门各控制什么？

<details><summary>参考答点</summary>

**重置门** $\boldsymbol{R}_t$：控制上一时刻隐藏状态有多少进入**候选状态**。
它与 $\boldsymbol{H}_{t-1}$ 按元素乘后再过矩阵乘，接近 0 时历史被丢掉——管**短期依赖**。

**更新门** $\boldsymbol{Z}_t$：控制旧状态保留多少。
$\boldsymbol{Z}_t = 1$ 时 $\boldsymbol{H}_t = \boldsymbol{H}_{t-1}$，什么都不更新——管**长期依赖**。

</details>

**Q3** GRU 的 `nn.GRU` 与书上公式有哪两处差异？

<details><summary>参考答点</summary>

（a）**更新门语义相反**：书上的 $\boldsymbol{Z}$ 是保留旧状态的比例
（$\boldsymbol{H}_t = \boldsymbol{Z}\boldsymbol{H}_{t-1} + (1-\boldsymbol{Z})\tilde{\boldsymbol{H}}_t$），
torch 的 $z$ 是接受新状态的比例（$h_t = (1-z)n_t + zh_{t-1}$）。

（b）**候选状态里门控的位置不同**：书上 $(\boldsymbol{R}\odot\boldsymbol{H})\boldsymbol{W}_{hh}$
（先门控再矩阵乘），torch $r\odot(\boldsymbol{W}_{hn}h_{t-1}+b_{hn})$（先矩阵乘再门控）。

两条叠加，手写与 `nn.GRU` 直接对比一定对不上。

</details>

**Q4** LSTM 的记忆细胞为什么能缓解梯度消失？

<details><summary>参考答点</summary>

记忆细胞之间是**加法**关系：

$$\boldsymbol{C}_t = \boldsymbol{F}_t\odot\boldsymbol{C}_{t-1} + \boldsymbol{I}_t\odot\tilde{\boldsymbol{C}}_t$$

梯度沿 $\boldsymbol{C}$ 回传时乘的是遗忘门（0 到 1 的向量），
而不是像原始 RNN 那样每步乘一个矩阵 $\boldsymbol{W}_{hh}^\top$。
遗忘门接近 1 时梯度几乎无损地传很远——这条路径叫"恒定误差传送带"。

</details>

## 了解

**Q5** LSTM 与 GRU 的状态数量差在哪？各自返回什么？

<details><summary>参考答点</summary>

GRU 只有隐藏状态 $\boldsymbol{H}$，`nn.GRU` 返回 `(out, h_n)`。

LSTM 有记忆细胞 $\boldsymbol{C}$ 与隐藏状态 $\boldsymbol{H}$ 两份，
`nn.LSTM` 返回 `(out, (h_n, c_n))`。

</details>

**Q6** 深层 RNN 的隐藏状态是什么形状？双向时呢？

<details><summary>参考答点</summary>

`(num_layers, batch, hidden_size)`。

双向时层数翻倍：`(num_layers * 2, batch, hidden_size)`。

输出的最后一维：单向 `h`，双向 `2h`（两个方向的隐藏状态拼接）。

</details>

**Q7** 双向 RNN 为什么不能用于生成？

<details><summary>参考答点</summary>

双向 RNN 的后向部分要从序列**末尾**往前跑，需要看到整个序列。

生成时未来的字符还不存在，无法提供后向输入。所以自回归生成只能用单向 RNN。

双向 RNN 适合序列标注、分类这类"整段输入都能拿到"的任务。

</details>

## 评分

- **必会**四题全对才算通过
- Q3 是这一章最容易踩的坑
