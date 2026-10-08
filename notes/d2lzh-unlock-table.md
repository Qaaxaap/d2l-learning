# d2l 封装解锁表

第一版书里凡是出现 `# 本函数已保存在d2lzh包中方便以后使用` 的地方，就是从那一刻起"可以不用自己写"。
**在那句话出现之前，对应功能必须自己实现。**

第一版 `d2lzh` 是 MXNet 包，PyTorch 下用不了。替代关系：

| 书上写的（MXNet） | 你可以用的（PyTorch） |
|---|---|
| `d2lzh.xxx` | 你自己写的版本（建议收进 `work/d2llocal.py`），或第二版官方 `d2l` 包的同名 torch 版 |

官方 torch 版 `d2l` 包需要单独安装（`uv add d2l`），本仓库默认不装，等你实现过一遍之后再装。

## 解锁清单

下表来自第一版附录「`d2lzh`包索引」。**解锁章节**指的是：读到那一节的结尾，这个函数就可以直接用了。

| 函数 / 类 | 解锁于（第一版章节） | 你需要在哪一节之前自己写 |
|---|---|---|
| `data_iter` | 3.2 线性回归的从零开始实现 | 3.2 |
| `linreg` | 3.2 线性回归的从零开始实现 | 3.2 |
| `squared_loss` | 3.2 线性回归的从零开始实现 | 3.2 |
| `sgd` | 3.2 线性回归的从零开始实现 | 3.2 |
| `plt` | 3.2 线性回归的从零开始实现 | 3.2 |
| `set_figsize` | 3.2 线性回归的从零开始实现 | 3.2 |
| `use_svg_display` | 3.2 线性回归的从零开始实现 | 3.2 |
| `get_fashion_mnist_labels` | 3.5 图像分类数据集（Fashion-MNIST） | 3.5 |
| `show_fashion_mnist` | 3.5 图像分类数据集（Fashion-MNIST） | 3.5 |
| `train_ch3` | 3.6 softmax 回归的从零开始实现 | 3.6 |
| `semilogy` | 3.11 模型选择、欠拟合和过拟合 | 3.11 |
| `try_gpu` | 5.5 卷积神经网络（LeNet） | 5.5 |
| `train_ch5` | 5.5 卷积神经网络（LeNet） | 5.5 |
| `load_data_fashion_mnist` | 3.5 图像分类数据集（书里说"供后面章节调用"，完整实现放在 5.6） | 3.5 |
| `corr2d` | 5.1 二维卷积层 | 5.1 |
| `Residual` | 5.11 残差网络（ResNet） | 5.11 |
| `resnet18` | 8.4 多 GPU 计算的简洁实现 | 8.4 |
| `to_onehot` | 6.4 循环神经网络的从零开始实现 | 6.4 |
| `grad_clipping` | 6.4 循环神经网络的从零开始实现 | 6.4 |
| `predict_rnn` | 6.4 循环神经网络的从零开始实现 | 6.4 |
| `train_and_predict_rnn` | 6.4 循环神经网络的从零开始实现 | 6.4 |
| `RNNModel` | 6.5 循环神经网络的简洁实现 | 6.5 |
| `predict_rnn_gluon` | 6.5 循环神经网络的简洁实现 | 6.5 |
| `train_and_predict_rnn_gluon` | 6.5 循环神经网络的简洁实现 | 6.5 |
| `load_data_jay_lyrics` | 6.3 语言模型数据集（周杰伦专辑歌词） | 6.3 |
| `data_iter_consecutive` | 6.3 语言模型数据集（周杰伦专辑歌词） | 6.3 |
| `data_iter_random` | 6.3 语言模型数据集（周杰伦专辑歌词） | 6.3 |
| `get_data_ch7` | 7.3 小批量随机梯度下降 | 7.3 |
| `train_ch7` | 7.3 小批量随机梯度下降 | 7.3 |
| `train_gluon_ch7` | 7.3 小批量随机梯度下降 | 7.3 |
| `show_trace_2d` | 7.2 梯度下降和随机梯度下降 | 7.2 |
| `train_2d` | 7.2 梯度下降和随机梯度下降 | 7.2 |
| `Benchmark` | 8.2 异步计算 | 8.2 |
| `show_images` | 9.1 图像增广 | 9.1 |
| `evaluate_accuracy` | 9.1 图像增广 | 9.1 |
| `train` | 9.1 图像增广 | 9.1 |
| `try_all_gpus` | 9.1 图像增广 | 9.1 |
| `bbox_to_rect` | 9.3 目标检测和边界框 | 9.3 |
| `show_bboxes` | 9.4 锚框 | 9.4 |
| `load_data_pikachu` | 9.6 目标检测数据集（皮卡丘） | 9.6 |
| `read_voc_images` | 9.9 语义分割和数据集 | 9.9 |
| `VOC_CLASSES` | 9.9 语义分割和数据集 | 9.9 |
| `VOC_COLORMAP` | 9.9 语义分割和数据集 | 9.9 |
| `voc_label_indices` | 9.9 语义分割和数据集 | 9.9 |
| `voc_rand_crop` | 9.9 语义分割和数据集 | 9.9 |
| `VOCSegDataset` | 9.9 语义分割和数据集 | 9.9 |
| `download_voc_pascal` | 9.9 语义分割和数据集 | 9.9 |
| `mkdir_if_not_exist` | 9.13 实战 Kaggle 比赛：图像分类（CIFAR-10） | 9.13 |
| `download_imdb` | 10.8 情感分析：使用递归神经网络 | 10.8 |
| `read_imdb` | 10.8 情感分析：使用递归神经网络 | 10.8 |
| `get_tokenized_imdb` | 10.8 情感分析：使用递归神经网络 | 10.8 |
| `get_vocab_imdb` | 10.8 情感分析：使用递归神经网络 | 10.8 |
| `preprocess_imdb` | 10.8 情感分析：使用递归神经网络 | 10.8 |
| `predict_sentiment` | 10.8 情感分析：使用递归神经网络 | 10.8 |
| `count_tokens` | 10.8 情感分析：使用递归神经网络 | 10.8 |

## 检查代码时怎么用

看到 `work/` 里的代码调用了上表某个函数时，先看那一段是不是已经读到过解锁它的章节。
没读到就调用，判为越界，要求自己实现。
