# B13 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 纸质书 | 4.1 模型构造、4.2 模型参数的访问、初始化和共享、4.3 模型参数的延后初始化 | 35 分钟 |
| 电子版 | [5.1 层和块](https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html)、[5.2 参数管理](https://zh.d2l.ai/chapter_deep-learning-computation/parameters.html) | 30 分钟 |
| 讲义 | `notes/b13-model-construction.md` | 35 分钟 |

## 读的时候要能回答

1. 自定义一个模型要写哪两个方法？`super().__init__()` 忘了会怎样？
2. 为什么调用时写 `net(x)` 而不是 `net.forward(x)`？
3. 想让两个层共享参数，torch 里怎么做？
4. 一个张量要跟着训练更新、另一个不更新但要存进模型文件。分别用什么？
5. `net.named_parameters()` 输出什么？怎么用它把权重与偏置分开处理？
6. 为什么改参数要包 `no_grad()`？
7. 书上的"延后初始化"在 torch 里对应什么？

## 顺序

先读纸质书 4.1，一边读一边对照：书上是 `Block`、torch 是 `nn.Module`；
书上的 `_children` 在 torch 里是内部字典，对外用 `children()` 遍历。
再读 4.2（参数）与 4.3（延后初始化），最后读讲义。

**两版差异在这一章最大**：v1 是 MXNet 的 `Block` / `initialize()` / `collect_params()`，
v2 是 torch 的 `nn.Module` / `named_parameters()`。讲义把 v1 的说法当对照，
以 torch 为主。

## 三处要留意

**这一章不加新模型**。它解决的是"怎么组织模型"——后面 B15 的卷积网络、B17 的
AlexNet 到 GoogLeNet，都要靠自己定义模块才能拼出来。`nn.Sequential` 只能表达简单串联。

**4.3 的延后初始化在 torch 里没有**。书上的 `nn.Dense(256)` 只给输出维度，
torch 的 `nn.Linear` 必须两个维度都给。这一段读的时候知道"那是 Gluon 的特性"就够了。

**三个容易搞混的东西**：`nn.Parameter`（会训练、会存）、普通张量属性（都不）、
`register_buffer`（不训练、会存）。讲义 2.3 节有一张表。
