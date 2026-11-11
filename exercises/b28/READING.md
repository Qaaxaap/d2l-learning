# B28 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b28-word-vectors.md` | 45 分钟 |
| 纸质书 | 10.1 词嵌入基础、10.4 近似训练、10.5 GloVe、10.7 词的相似度和类比 | 50 分钟 |
| 电子版 | [14.1](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word-embedding.html)、[14.2](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/approx-training.html)、[14.5](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/glove.html) | 50 分钟 |

## 读的时候要能回答

1. one-hot 的两个问题是什么？词向量解决了哪个？
2. 跳字模型的设定与条件概率公式？
3. 为什么每个词要两组向量？
4. 连续词袋模型与跳字模型的三处差别？
5. softmax 在词表很大时开销大在哪？两种近似方法的复杂度？
6. 负采样为什么要加负样本？
7. 噪声分布为什么要开 3/4 次方？
8. GloVe 的损失与权重函数？为什么用平方损失？
9. word2vec 与 GloVe 在对称性上的差别？
10. 余弦相似度相比欧氏距离的优势？

## 顺序

先读第 1 到 3 节。跳字模型与连续词袋是整个词向量方法的基础。

再读第 4 到 6 节。这两节解决「词表太大算不动」的问题，负采样是实践里最常用的。

最后读第 7 到 10 节。GloVe 的思路值得看，BERT 的预训练思路是现在的主流。

## 三处要留意

**词向量有两个角色**。一个词当中心词与当背景词用两组向量，训练完取哪一组都行。

**负采样的核心是把多分类改成二分类**。不是为了更准，是为了算得动。

**静态与动态词向量**。word2vec 与 GloVe 一个词一个向量，
BERT 的向量取决于整句话。做现代 NLP 任务时用后者。
