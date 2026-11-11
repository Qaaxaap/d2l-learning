# B26 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：写训练代码与排查性能问题直接用
- **了解**（Q5、Q6、Q7）：知道有这回事即可

## 必会

**Q1** 训练一个 step 的三种开销分别是什么？Python 开销为什么能占那么大？

<details><summary>参考答点</summary>

三种开销：Python 开销、GPU 计算、显存搬运。

Python 开销大的原因：每次算子调用要走解释器解析、参数检查、构造 C++ 对象、提交任务。
这些是微秒级，但一次小矩阵乘在 GPU 上可能只要 5 微秒。
层数一多，累积开销超过计算本身。

</details>

**Q2** GPU 算子为什么是异步的？哪些操作会强制同步？训练循环里 `print(loss)` 慢在哪？

<details><summary>参考答点</summary>

CPU 只负责把任务放进 GPU 队列，不等结果就返回。这样 CPU 与 GPU 两条流水线能重叠。

强制同步的操作：`.item()`、`print(x)`、`float(x)`、`x.cpu()`、`torch.cuda.synchronize()`。

`print(loss)` 慢有两个原因：它要把张量拿到 CPU（强制同步，清空 GPU 队列），
而且它会触发 `torch.compile` 的图中断。

</details>

**Q3** fp16 训练为什么要 `GradScaler`？bf16 需要吗？

<details><summary>参考答点</summary>

fp16 的最小规格数是 $6\times10^{-8}$。梯度常常比这个小，直接变成 0，参数不更新。

`GradScaler` 先把损失乘一个系数，得到放大后的梯度，更新前再缩回去。
它每步根据有没有出现 `inf` 动态调整系数。

bf16 不需要。它的指数位与 fp32 相同，范围一样，不会下溢。

</details>

**Q4** 怎么正确测量 GPU 上的执行时间？为什么 `time.time()` 直接包住算子测不准？

<details><summary>参考答点</summary>

`time.time()` 只包住了「提交任务」那几微秒，GPU 还没开始算。

正确做法：前后各加一次 `torch.cuda.synchronize()`。
测多次时用 `torch.cuda.Event`，它记录在 GPU 的时间线上，不受 CPU 提交延迟影响。

还要预热。第一次执行包含初始化开销。

</details>

## 了解

**Q5** `torch.compile` 什么时候有用？什么时候反而更慢？

<details><summary>参考答点</summary>

有用：小算子多（算子融合收益大）、输入形状固定、重复执行很多步。
没用的场景：

- 模型已经跑满 GPU，减少 Python 开销没用
- 输入形状每步都变，每次要重新编译
- 代码里控制流多，图中断把收益切碎

</details>

**Q6** 数据并行与模型并行的区别？各自适合什么场景？

<details><summary>参考答点</summary>

数据并行切数据，每张卡一份完整模型，适合模型放得进单卡。
模型并行切模型，每张卡放一部分，适合模型放不进单卡。

数据并行每步要同步梯度，通信量大。模型并行要同步层之间的中间结果，更麻烦。

实践里数据并行用得多。

</details>

**Q7** `DataParallel` 与 `DistributedDataParallel` 的性能差距从哪来？

<details><summary>参考答点</summary>

`DataParallel` 用单进程多线程，有 GIL 瓶颈，每步在 GPU 间汇总梯度开销大，
而且第一个 GPU 要负责汇总，负载不均。

`DistributedDataParallel` 用多进程，每 GPU 一个进程，没有 GIL 问题，
梯度用 ring-allreduce 同步，带宽利用更好。

单机多卡也应该用 DDP。

</details>

## 评分

- **必会**四题全对才算通过
- Q4 是这一章最容易犯的操作错误
