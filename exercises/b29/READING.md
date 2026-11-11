# B29 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b29-nlp-applications.md` | 40 分钟 |
| 纸质书 | 10.8 文本情感分类、10.9 用 RNN、10.10 用 CNN、10.12 自然语言推断、10.13 用注意力 | 60 分钟 |
| 电子版 | [15.2](https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html)、[15.3](https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html)、[15.7](https://zh.d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html) | 60 分钟 |

## 读的时候要能回答

1. 序列分类与语言模型的差别？
2. 截断与填充的规则？为什么要固定长度？
3. 双向 RNN 做分类时取哪个隐藏状态？
4. 一维卷积的输出长度公式？它捕获的是什么？
5. textCNN 为什么要用多个窗口大小？
6. 时序最大池化相比平均池化的优势？
7. 自然语言推断的三个类别？
8. 文本对的三种处理方式与含义？
9. BERT 输入的三个特殊符号？
10. 微调 BERT 要留意哪三个设置？

## 顺序

先读第 2 节。变长文本的表示是所有下游任务的前提，也是最容易写错的地方。

再读第 3、4 节。RNN 与 CNN 两条路线，对比着看能理解各自的取舍。

最后读第 5 到 8 节。自然语言推断与 BERT 微调是现在的主流做法，了解结构即可。

## 三处要留意

**填充必须配 attention_mask**。不加 mask 时填充位置参与注意力，效果会变差。

**双向 RNN 取两个隐藏状态**。只取 `h[-1]` 会丢掉一半信息，
而且形状对不上分类头的输入维。

**`CrossEntropyLoss` 要 logits**。外面不要先做 softmax。
