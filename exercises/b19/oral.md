# B19 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：写 RNN 与数据管道时直接用
- **了解**（Q5、Q6、Q7）：知道设计动机即可

## 必会

**Q1** 字符级语言模型的输入与标签是什么关系？如果写成一样会怎样？

<details><summary>参考答点</summary>

**标签是输入右移一位**：输入 `想 要 有 直 升`，标签 `要 有 直 升 机`。

写成一样的话，模型学到的是**恒等映射**——输出等于输入。不报错，损失还能降
（甚至降得很快），但模型什么也学不到，因为它的目标就是复制。

这是这一章最隐蔽的错误。

</details>

**Q2** 写出 RNN 隐藏状态与输出的公式，标出每个张量的形状。

<details><summary>参考答点</summary>

$$\boldsymbol{H}_t = \tanh(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh} + \boldsymbol{b}_h)$$

$$\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q$$

| 张量 | 形状 |
|---|---|
| $\boldsymbol{X}_t$ | $(n, d)$ |
| $\boldsymbol{H}_{t-1}$、$\boldsymbol{H}_t$ | $(n, h)$ |
| $\boldsymbol{W}_{xh}$ | $(d, h)$ |
| $\boldsymbol{W}_{hh}$ | $(h, h)$ |
| $\boldsymbol{b}_h$ | $(h,)$ |
| $\boldsymbol{W}_{hq}$ | $(h, q)$ |
| $\boldsymbol{O}_t$ | $(n, q)$ |

</details>

**Q3** RNN 有多少个参数？序列从 10 步变成 100 步，参数量变吗？为什么？

<details><summary>参考答点</summary>

**五组参数**：$\boldsymbol{W}_{xh}$、$\boldsymbol{W}_{hh}$、$\boldsymbol{b}_h$、
$\boldsymbol{W}_{hq}$、$\boldsymbol{b}_q$。

**不变**。参数在时间维共享——不管序列多长，每一步都用同一组权重。
这是 RNN 与"每步一个全连接层"的根本区别。

</details>

**Q4** 随机采样与相邻采样的区别是什么？各自对隐藏状态有什么要求？

<details><summary>参考答点</summary>

| | 随机采样 | 相邻采样 |
|---|---|---|
| 起点 | 随机挑 | 按顺序切 |
| 相邻批量 | 在序列上不相邻 | 在序列上相接 |
| 隐藏状态 | **每批要重置** | 可以跨批传递（要 `detach`）|

原因：随机采样的两个批量来自序列的不同位置，它们的隐藏状态没有连续性，
沿用上一批的状态等于把不相关的历史带进来。

</details>

## 了解

**Q5** `X @ W_xh + H @ W_hh` 的拼接等价写法是什么？这样写有什么好处？

<details><summary>参考答点</summary>

```python
torch.cat([X, H], dim=1) @ torch.cat([W_xh, W_hh], dim=0)
```

`X` 是 `(n,d)`、`H` 是 `(n,h)`，按列拼成 `(n, d+h)`；
两个权重按行拼成 `(d+h, h)`。乘出来 `(n,h)`，与分开算再相加完全一样。

好处：合并成一次矩阵乘，GPU 上更快（少一次加法，矩阵尺寸更整齐）。

</details>

**Q6** 什么是 n 元语法？它简化了什么、牺牲了什么？

<details><summary>参考答点</summary>

用马尔可夫假设：一个词只与前面 $n-1$ 个词有关。

$$P(w_1,\ldots,w_T) \approx \prod_t P(w_t \mid w_{t-(n-1)},\ldots,w_{t-1})$$

**简化**：条件长度固定，不用存任意长的历史。
**牺牲**：远处依赖被砍掉。一元语法下"你走先"与"你先走"概率相同。

</details>

**Q7** `nn.RNN` 的输入输出各是什么形状？`batch_first=True` 起什么作用？

<details><summary>参考答点</summary>

`batch_first=True` 时：

- 输入 `x`：`(batch, seq_len, input_size)`
- 初始状态 `h_0`：`(num_layers, batch, hidden_size)`
- 输出 `out`：`(batch, seq_len, hidden_size)`，每一步的隐藏状态
- 最终状态 `h_n`：`(num_layers, batch, hidden_size)`

不加 `batch_first=True` 时输入是 `(seq_len, batch, input_size)`——torch 的历史默认。

</details>

## 评分

- **必会**四题全对才算通过
- Q1 与 Q4 是这一章最容易记反的两处
