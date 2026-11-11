# B26 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b26-performance.md` | 35 分钟 |
| 纸质书 | 8.1 混合编程、8.2 异步计算、8.4 多 GPU 计算 | 40 分钟 |
| 电子版 | [12.1](https://zh.d2l.ai/chapter_computational-performance/hybridize.html)、[12.2](https://zh.d2l.ai/chapter_computational-performance/async-computation.html)、[12.4](https://zh.d2l.ai/chapter_computational-performance/multiple-gpus.html) | 40 分钟 |

## 读的时候要能回答

1. 训练一个 step 的三种开销分别是什么？
2. 命令式与图式执行的差别在哪三个方面？
3. `torch.compile` 什么时候有用、什么时候没用？
4. 什么是图中断？怎么发现它？
5. GPU 算子为什么是异步的？哪些操作会强制同步？
6. 怎么正确测量 GPU 上的执行时间？
7. fp16 与 bf16 各适合什么场景？
8. fp16 训练为什么需要 `GradScaler`？
9. 数据并行与模型并行的区别？
10. `DataParallel` 与 `DistributedDataParallel` 的性能差距从哪来？

## 顺序

先读讲义第 1 到 3 节，把「为什么慢」与「编译能帮多少」理清。

再读第 4 节。这一节最实用：`print(loss)` 与 `.item()` 拖慢训练是新手最常见的性能问题。

最后读第 5、6 节。混合精度是 Kaggle 上省显存的常规手段，多卡训练本地用不上但要看得懂。

## 三处要留意

**书上大半是 MXNet 的机制**。符号式编程、前后端异步、自动并行这三个概念在 torch 里对应别的东西。
讲义第 2、4 节写了对照，读的时候把书上那些当历史背景。

**`torch.compile` 会改变报错信息**。错误指向生成的代码，与你的源码行号对不上。
调试阶段先不用它。

**bf16 不需要 `GradScaler`**。它的范围与 fp32 相同，不会下溢。
留着一层缩放只会多一次乘除。
