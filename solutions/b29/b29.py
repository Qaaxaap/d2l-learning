"""B29 参考答案。做完再看。"""

import torch
import torch.nn.functional as F


# ---------- T1 截断与填充 ----------

def truncate_pad(line: list[int], num_steps: int,
                 padding_token: int) -> list[int]:
    """把一条索引序列变成固定的 num_steps 长度。

    超长时只留前 num_steps 个，不足时末尾补 padding_token。
    """
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


# ---------- T2 一维卷积的输出长度 ----------

def conv1d_output_len(n_in: int, k: int, stride: int = 1,
                      padding: int = 0) -> int:
    """一维卷积的输出长度取整。"""
    return (n_in + 2 * padding - k) // stride + 1


# ---------- T3 时序最大池化 ----------

def time_max_pool(x: torch.Tensor) -> torch.Tensor:
    """在时间维上取最大值。

    x 形状 (batch, channels, length)，返回 (batch, channels)。
    """
    return x.max(dim=2).values


# ---------- T4 文本对的特征 ----------

def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """把两句的表示拼成分类特征。

    a、b 形状都是 (n, d)。返回 (n, 4*d)，
    依次是 a、b、|a-b|、a*b 四段拼接。
    """
    diff = (a - b).abs()
    prod = a * b
    return torch.cat([a, b, diff, prod], dim=1)


# ---------- T5 分类头 ----------

def build_classifier(embed: torch.Tensor, num_classes: int) -> torch.nn.Module:
    """从最后一层隐藏状态接一个线性分类头。

    embed 是 BERT 的 [CLS] 表示，形状 (batch, hidden)。
    返回一个 nn.Linear，输入维 hidden，输出维 num_classes。
    """
    return torch.nn.Linear(embed.shape[1], num_classes)


"""T6 改错

第 1 处：`count_embeddings` 忘了指定 `padding_idx`
    填充符号的索引也会被训练成一个普通词向量，模型会把它当成有意义的词。
    改法：`nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)`，
    它的向量初始化为 0 且不参与训练。

第 2 处：双向 RNN 只取了 `h[-1]`
    `h` 形状是 `(num_layers*2, batch, h)`。`h[-1]` 只是反向那一层的最终状态。
    前向那一层是 `h[-2]`。要两个都取再拼接。

第 3 处：`logits` 之后又做了一次 softmax 再交给 `CrossEntropyLoss`
    `CrossEntropyLoss` 内部会做 log-softmax。外面再做一次是重复的，
    而且会让梯度变小。
    改法：直接传 logits。
"""


if __name__ == "__main__":
    print("T1 截断:", truncate_pad([1, 2, 3, 4, 5], 3, 0))
    print("T1 填充:", truncate_pad([1, 2], 5, 0))
    print("T2 5 个元素核 2:", conv1d_output_len(5, 2))
    print("T2 16 个词核 3 步幅 2 填充 1:", conv1d_output_len(16, 3, 2, 1))

    torch.manual_seed(0)
    x = torch.randn(2, 4, 6)
    print("T3 最大池化:", time_max_pool(x).shape)

    a = torch.randn(3, 8)
    b = torch.randn(3, 8)
    print("T4 文本对特征:", pair_features(a, b).shape)

    emb = torch.randn(3, 16)
    print("T5 分类头:", build_classifier(emb, 3))
