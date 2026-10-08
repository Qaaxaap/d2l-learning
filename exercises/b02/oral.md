# B02 口试题

关书作答。答不上来不要猜，说"不知道"，我讲完你复述一遍。
三层难度，第一层是保底，第二、三层才是重点。

## 第一层：是什么

**Q1** `x.numel()` 和 `x.size()` 分别返回什么？书上的 `x.size` 对应哪一个？

<details><summary>参考答点</summary>

`numel()` 返回元素总数（一个整数）；`size()` 返回形状（`torch.Size`，可当 tuple 用），
`size(0)` 返回第 0 轴长度。书上 `x.size` 是 MXNet 的属性，表示元素总数，对应 `numel()`。
把 `x.size` 当属性写在 torch 里拿到的是方法对象本身，报错位置离出错位置很远。

</details>

**Q2** `x.reshape(3, 4)` 和 `x.view(3, 4)` 有什么区别？

<details><summary>参考答点</summary>

`view` 只允许在不重新分配内存的前提下重新解释形状，要求张量内存连续；
`reshape` 在不连续时自己复制一份。所以 `view` 更严格，`reshape` 更宽容。
能确定连续时用哪个都行，不确定就用 `reshape`。

</details>

**Q3** 广播的规则是什么？

<details><summary>参考答点</summary>

两个形状从最右边一维开始向左对齐；每一维上，长度相等、或其中一方为 1、或其中一方这一维不存在（视为 1），
才允许广播；结果的每一维取两者在该维上的较大值。

</details>

## 第二层：为什么

**Q4** 为什么 torch 的切片返回视图而不是副本？这样设计带来什么代价？

<details><summary>参考答点</summary>

深度学习里张量很大，每次切片都复制会吃掉大量内存和时间；切片只是换个角度看同一块存储，
代价是别名问题——改切片会改到原张量，而且这种修改常常是无声的。
要副本必须显式 `.clone()`。PyTorch 这样做还让切片上的原地操作能反映到原张量，方便某些实现。

</details>

**Q5** 书上的 `nd.arange(12)` 和 torch 的 `torch.arange(12)` 得到的 dtype 一样吗？这个差异会导致什么后果？

<details><summary>参考答点</summary>

不一样。MXNet 的 `nd.arange` 默认 float32（`mx_real_t`），torch 的 `arange` 默认 int64。
照书翻译时若不显式指定 dtype，后面算梯度和送进线性层都会报 dtype 不匹配；
矩阵乘法两侧 dtype 必须一致，整数张量也不能求 L2 范数。

</details>

**Q6** 为什么 `Y.t().view(-1)` 报错，而 `Y.t().reshape(-1)` 成功？

<details><summary>参考答点</summary>

转置只交换了 stride，没有搬数据，张量不再连续。`view` 要求能用一组新 stride 描述原存储，
而不连续张量展平后下标不再单调，找不到这样的 stride，于是报错。
`reshape` 遇到这种情况会先复制成连续张量再展平，所以能成功。

</details>

## 第三层：应用与陷阱

**Q7** 你训练时写了 `total_loss += loss`（`loss` 是张量），跑了几百轮之后显存缓慢上涨。
原因是什么？怎么改？

<details><summary>参考答点</summary>

`loss` 连着计算图，累加后整条链路都被引用，无法释放，迭代次数越多图越大。
改成 `total_loss += loss.item()`，得到 Python 浮点数，不连图。
注意 `.item()` 只能用于单元素张量。

</details>

**Q8** `a.shape == (3, 1, 4)`、`b.shape == (2, 4)`，`a + b` 的结果形状是什么？说出推理过程。

<details><summary>参考答点</summary>

`(3, 2, 4)`。右对齐：`b` 补成 `(1, 2, 4)`。第 2 维 4 与 4 相同；第 1 维 1 与 2，一方为 1 可广播成 2；
第 0 维 3 与 1，一方为 1 可广播成 3。

</details>

**Q9** 什么时候必须用 `.item()`，什么时候不能用？

<details><summary>参考答点</summary>

需要 Python 标量时用：打印损失、累加统计、当条件判断、传给不认张量的库。
张量元素多于一个时不能用，会抛 `RuntimeError: a Tensor with N elements cannot be converted to Scalar`，
这时先 `.sum()` 或取索引。另外 `.item()` 会切断计算图，如果需要梯度就不能换。

</details>

## 评分

- 第一层全对 → 通过，可以进下一单元
- 第二层答对一半以上 → 通过
- 第二层答不上来 → 回去重读讲义第 2、4 节，隔天再考一次
- 第三层答错 → 通过，但记进错题本，后面章节抽考
