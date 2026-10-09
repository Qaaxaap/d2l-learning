# B04 口试题

关书作答。答不上来不要猜，说"不知道"，我讲完你复述一遍。
三层难度，第一层是保底，第二、三层才是重点。

## 第一层：是什么

**Q1** 导数和微分是什么关系？用微分近似函数值时，误差为什么是二阶的？

<details><summary>参考答点</summary>

导数是函数在某一点的瞬时变化率，是一个数：$f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$。

微分是用导数构造的**线性近似**：$f(x+\Delta x)\approx f(x)+f'(x)\Delta x$。
把右边那部分写成 $\mathrm{d}f=f'(x)\Delta x$，它就是 $\Delta f$ 的主部。

误差是二阶的，因为泰勒展开的下一项是 $\frac{1}{2}f''(x)\Delta x^2$，
所以在 $\Delta x\to 0$ 时误差按 $\Delta x^2$ 缩小。也正因为是线性近似，
梯度下降才能用"沿负梯度方向走一小步"来近似地降低函数值。

</details>

**Q2** 偏导数和梯度是什么关系？梯度指向哪个方向？

<details><summary>参考答点</summary>

偏导数是多元函数对某一个变量求导，其余变量当常数。

梯度把所有偏导数打包成一个向量：$\nabla_{\mathbf{x}}f=\left[\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right]^\top$。

梯度指向函数值**上升最快**的方向，它的反方向下降最快。所以梯度下降沿 $-\nabla f$ 走。
梯度的每个分量是"那个变量单独动一点，函数变化多少"。

</details>

**Q3** `requires_grad=True` 做了什么？`backward()` 做了什么？

<details><summary>参考答点</summary>

`requires_grad=True` 是给这个张量打标记：从它出发的运算都要记录下来，为反传做准备。
标记在叶子张量上设置，中间结果自动继承。

每次对这类张量做运算，torch 会把运算和输入输出记进一张**计算图**。

`backward()` 从调用它的那个张量出发，沿计算图反向走，用链式法则把梯度一层层算回去，
结果累加到各叶子张量的 `.grad` 上。图在反传之后默认被释放。

</details>

## 第二层：为什么

**Q4** $y=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ 的梯度为什么是 $(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$？$\mathbf{A}$ 对称时为什么变成 $2\mathbf{A}\mathbf{x}$？

<details><summary>参考答点</summary>

先把二次型展开成 $\sum_i\sum_j x_iA_{ij}x_j$。对某个 $x_k$ 求偏导时，$x_k$ 可能出现在
第一个因子（$i=k$）或第二个因子（$j=k$）上：

- $i=k$ 的那些项给出第 $k$ 行与 $\mathbf{x}$ 的内积，即 $(\mathbf{A}\mathbf{x})_k$
- $j=k$ 的那些项给出第 $k$ 列与 $\mathbf{x}$ 的内积，即 $(\mathbf{A}^\top\mathbf{x})_k$

两项相加得 $\partial y/\partial x_k=(\mathbf{A}\mathbf{x})_k+(\mathbf{A}^\top\mathbf{x})_k$，
写成向量就是 $(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$。

$\mathbf{A}$ 对称时 $\mathbf{A}^\top=\mathbf{A}$，两项相同，合成 $2\mathbf{A}\mathbf{x}$。
对称时还可以写成 $\nabla(\mathbf{x}^\top\mathbf{A}\mathbf{x})=2\mathbf{A}\mathbf{x}$。

</details>

**Q5** `x.grad` 是 `None` 有哪几种原因？分别怎么排查？

<details><summary>参考答点</summary>

三种：

1. **它不是叶子**。中间结果的 `.grad` 默认不保存，用 `x.is_leaf` 判断。
   确实需要就用 `x.retain_grad()`。
2. **是叶子但还没反传过**。`.grad` 初值就是 `None`，反传之后才有值。
3. **损失不依赖它，或者它没开 `requires_grad`**。后者在反传时会直接报
   `element 0 of tensors does not require grad and does not have a grad_fn`。
   这一种是"参数不更新"的常见原因：比如用 `torch.tensor(...)` 重新包装了参数，
   优化器手里还是旧对象，新张量根本没参与计算图。

排查顺序：先看 `is_leaf` 和 `requires_grad`，再看有没有真的反传过。

</details>

**Q6** 训练循环里为什么每轮都要清零梯度？忘了写会怎样？

<details><summary>参考答点</summary>

因为 `backward()` 是**累加**到 `.grad` 上的，不是覆盖。

忘了清零不会报错。梯度会一轮轮叠加，等效学习率不断变大，训练可能发散、震荡，
也可能表面上还在下降而被忽略——这类错误比崩溃难查。

PyTorch 的写法是 `optimizer.zero_grad()`（或手工 `p.grad = None`）。

顺带一处差异：MXNet 的 `attach_grad()` 默认每次反向**覆盖**梯度缓冲区，
只有显式改成 `'add'` 才累加，所以第一版书里的训练循环没有清零这一步。
把那段代码照搬过来不会报错，只是训练不正常。

</details>

## 第三层：应用与陷阱

**Q7** `retain_graph=True` 解决什么问题？什么时候必须用？代价是什么？

<details><summary>参考答点</summary>

反传结束后图默认被释放（为反向保存的中间结果不再需要）。同一张图再反传一次会报
`Trying to backward through the graph a second time`。`retain_graph=True` 让它保留下来。

必须用的两个场景：一次前向之后既要对输入求导又要对参数求导；计算高阶导数
（配合 `create_graph=True`）。

代价是中间结果不能释放，显存占用上升。能用 `torch.autograd.grad` 指定只求某几个量的梯度时，
优先用它而不是 `retain_graph`。

</details>

**Q8** `with torch.no_grad():` 与 `.detach()` 分别适合什么场合？

<details><summary>参考答点</summary>

`no_grad()` 管**一段代码**：块内所有运算都不建图。适合手工更新参数、验证集评估、推理。

`detach()` 管**一个张量**：把它从图上摘下来当常数，其余运算照常建图。
适合截断沿时间的反向传播（BPTT）、把目标网络的输出当固定目标。

两者可以混用。更新参数的标准写法是：

```python
with torch.no_grad():
    p -= lr * p.grad
```

这样 `p` 不会因为减法变成非叶子，也就不必用 `p.data -= ...` 那种绕过版本计数的写法。

</details>

**Q9** 梯度检验为什么要用 `float64` 和中心差分？步长取多大合适？

<details><summary>参考答点</summary>

**float64**：数值梯度是两个相近的数相减再除以一个很小的步长，舍入误差会被放大。
float32 只有约 7 位有效数字，实测在同样步长下误差能到 1 的量级；float64 降到 1e-9 附近。

**中心差分**：$\frac{f(x+h)-f(x-h)}{2h}$ 的截断误差是 $O(h^2)$，
单侧差分 $\frac{f(x+h)-f(x)}{h}$ 只有 $O(h)$，差一个量级。

**步长**：$10^{-5}$ 到 $10^{-6}$。太大截断误差上升，太小舍入误差被 $h$ 除之后放大，
两头都不好，中间有个最优点。

torch 自带 `torch.autograd.gradcheck`，它同样要求输入是 float64。

</details>

## 评分

- 第一层全对 → 通过，可以进下一单元
- 第二层答对一半以上 → 通过
- 第二层答不上来 → 回去重读讲义第 1、2 节，隔天再考一次
- 第三层答错 → 通过，但记进错题本，后面章节抽考
- Q6 与 Q8 是 B06 之前必须清楚的，答错进错题本
