// ---- 版面 ----
#set document(title: "B22 编码器-解码器与束搜索", author: "d2l 私教")
#set page(paper: "a4", margin: (left: 46mm, right: 46mm, top: 20mm, bottom: 22mm))
#set page(footer: context {
  set text(size: 8pt, fill: rgb("#555"))
  grid(columns: (1fr, auto),
    text[B22 · 编码器-解码器与束搜索],
    align(right, text[#counter(page).display("1")]),
  )
})
#set text(font: ("Noto Serif CJK SC",), lang: "zh", size: 10pt)
#set par(leading: 0.85em, justify: true, first-line-indent: 0em, spacing: 1.1em)

// 标题
#show heading.where(level: 1): it => {
  set text(size: 17pt, weight: "bold", fill: rgb("#111111"))
  block(above: 1.6em, below: 0.9em, width: 100%,
    [#it
     #v(2pt)
     #line(length: 100%, stroke: 1.2pt + rgb("#111111"))])
}
#show heading.where(level: 2): it => {
  set text(size: 12.5pt, weight: "bold", fill: rgb("#222222"))
  block(above: 1.3em, below: 0.6em, it)
}
#show heading.where(level: 3): it => {
  set text(size: 10.5pt, weight: "bold", fill: rgb("#444"))
  block(above: 0.9em, below: 0.4em, it)
}

// 行内代码
#show raw.where(block: false): it => box(
  fill: rgb("#F0F0F0"), inset: (x: 2.5pt, y: 0.5pt), radius: 1.5pt,
  text(font: ("DejaVu Sans Mono", "Noto Sans CJK SC"), size: 8.8pt, fill: rgb("#333333"), it.text),
)
// 代码块：突破正文宽度
#show raw.where(block: true): it => block(
  width: 100%, outset: (x: 18mm), fill: rgb("#F7F7F7"),
  stroke: (left: 2pt + rgb("#222222")), inset: (x: 8pt, y: 6pt), radius: 0pt,
  text(font: ("DejaVu Sans Mono", "Noto Sans CJK SC"), size: 8.4pt, it),
)

// ---- 组件 ----
#let note(body, title: "要点", color: rgb("#222222"), bg: rgb("#F4F4F4")) = block(
  width: 100%, fill: bg, stroke: (left: 2.5pt + color), radius: (right: 2pt),
  inset: (x: 9pt, y: 7pt), above: 0.9em, below: 0.9em,
  [#text(size: 9pt, weight: "bold", fill: color)[#title] #h(0.4em)
   #text(size: 9.5pt)[#body]],
)
#let warn(body) = note(body, title: "坑", color: rgb("#111111"), bg: rgb("#F4F4F4"))
#let box-node(body, fill: rgb("#F4F4F4"), stroke: rgb("#222222"), inset: 8pt) = box(
  inset: (x: inset, y: 5pt), radius: 3pt, stroke: 0.7pt + stroke, fill: fill,
  align(center, text(size: 9.5pt)[#body]),
)
#let arrow(body) = text(size: 13pt, fill: rgb("#666"))[#body]

// 图：横向数据流，元素用节点函数给
#let flow(..items) = block(
  breakable: false, width: 100%, outset: (x: 18mm), above: 1em, below: 1em,
  grid(columns: (1fr,) * items.pos().len(), align: center + horizon,
    ..items.pos().map(x => x),
  ),
)
#let caption(body) = block(
  width: 100%, above: 0.3em, below: 1.2em,
  align(center, text(size: 8.8pt, fill: rgb("#666"))[#body]),
)

// ---- 封面 ----
#page(footer: none)[
#align(center)[
  #v(38mm)
  #text(size: 11pt, fill: rgb("#888"))[动手学深度学习 · 第一版 · 第 10 章]
  #v(8mm)
  #text(size: 26pt, weight: "bold", fill: rgb("#111111"))[编码器-解码器]
  #text(size: 26pt, weight: "bold", fill: rgb("#111111"))[与束搜索]
  #v(6mm)
  #text(size: 12pt, fill: rgb("#555"))[从 one-hot 到嵌入，从定长到变长]
  #v(20mm)
  #block(width: 60%, stroke: (top: 0.8pt + rgb("#888")))
  #v(6mm)
  #text(size: 9.5pt, fill: rgb("#666"))[
    讲义 § B22 · 读完能自己写出 seq2seq 与束搜索
  ]
]
#v(1fr)
#align(center, text(size: 8.5pt, fill: rgb("#888"))[
  2026-11 · 第二版 · 源文件 notes/pdf/b22.typ
])
]

#pagebreak()

// ---- 目录 ----
#outline()
#v(8pt)
#text(size: 8.5pt, fill: rgb("#888"))[
  页码在页脚。框标「要点」的是该记住的,标「坑」的是常错的。
]

#pagebreak()

// ---- 这章学什么 ----
#heading(level: 1, outlined: false)[这章要解决的问题]

前面所有章节做两类任务。分类是定长输入到一个标签。语言模型是单序列预测下一个词。

机器翻译要*输入一个序列、输出另一个序列*，两者长度不同。这个要求在 RNN 上落不下来。

本章造两个东西。一个把变长输入压成定长向量，另一个从定长向量生成变长输出。两者拼起来叫 seq2seq。
生成段的搜索算法单独讲，从贪婪讲到束搜索。

#note[
  读完这一章，你应该能回答三个问题：
  为什么要两个 RNN、背景变量为什么是瓶颈、束搜索比贪婪好在哪。
  最后一个问题要能算出书上的 0.048 与 0.054。
]

#v(6pt)
#text(size: 9pt, weight: "bold", fill: rgb("#111111"))[全章路线]
#v(4pt)

#block(width: 100%, [
  #let step(n, title, detail, color: rgb("#222222")) = block(
    width: 100%, fill: rgb("#FAFAFA"), stroke: (left: 2.5pt + color),
    inset: (x: 9pt, y: 6pt), radius: (right: 2pt), below: 4pt,
    grid(columns: (auto, auto, 1fr), column-gutter: 8pt, align: horizon,
      text(size: 9pt, weight: "bold", fill: color)[§#n],
      text(size: 9.5pt, weight: "bold")[#title],
      text(size: 8.5pt, fill: rgb("#666"))[#detail],
    )
  )
  #step([2], [嵌入], [词索引怎么变成向量。这一节是后面所有模型的入口], color: rgb("#333333"))
  #step([3-4], [编码器与解码器], [两个 RNN 拼成 seq2seq，把变长压成定长再展开])
  #step([5], [训练], [强制教学与它的代价])
  #step([6-8], [生成], [贪婪、穷举、束搜索，一条从差到好的线])
])

#v(10pt)
#grid(columns: (1fr, 1fr), column-gutter: 18pt,
  block([
    #text(size: 9pt, weight: "bold", fill: rgb("#111111"))[读之前需要的东西]
    #v(3pt)
    #set text(size: 8.5pt)
    #set par(leading: 0.75em)
    - B19 的 one-hot 与两个采样函数
    - B20 的 `rnn_forward` 与 `grad_clipping`
    - B21 的 GRU 与 LSTM 门控
  ]),
  block([
    #text(size: 9pt, weight: "bold", fill: rgb("#111111"))[本章关键词]
    #v(3pt)
    #text(size: 8.5pt)[
      背景变量、嵌入、自回归、强制教学、
      暴露偏差、束宽、长度惩罚
    ]
  ]),
)

#v(12pt)

// ============ 第 1 章 ============
#heading(level: 1)[一、机器翻译难在哪]

#heading(level: 2)[1.1 一个具体例子]

源句是中文「我爱学习」，目标句是英文 “I love learning”。两句话有三处不同。

#grid(columns: (auto, 1fr), column-gutter: 12pt, row-gutter: 7pt,
  text(weight: "bold")[长度], [源句 4 个词，目标句 3 个词],
  text(weight: "bold")[词表], [中英各一套，两边大小不同，词之间没有对应关系],
  text(weight: "bold")[顺序], [逐字换成英文是 “I love study”，英语习惯说 “I love learning”],
)

第一处就把单 RNN 挡住了。

#heading(level: 2)[1.2 单个 RNN 卡在哪]

B19 与 B20 的字符级语言模型，输入 $T$ 个字符，输出 $T$ 个预测。输入与输出的步数相同。

原因是 RNN 的结构固定。它每读一个输入就更新一次隐藏状态，每个隐藏状态对应一次输出。输入几步，输出就几步。

#flow(
  box-node[#text(size: 8.5pt, fill: rgb("#888"))[输入] \ 4 个词],
  arrow[→], box-node[RNN], arrow[→],
  box-node[#text(size: 8.5pt, fill: rgb("#888"))[输出] \ 4 个词],
)
#caption[单 RNN：读几个词，就只能吐几个词。步数绑死。]

机器翻译要求输出步数与输入步数无关。源句 4 个词，目标句可能是 3 个词，也可能是 5 个词。单个 RNN 表达不了。

#heading(level: 2)[1.3 两个 RNN 分工]

把任务拆成两段，每段用一个 RNN。

#flow(
  box-node[#text(size: 8.5pt)[编码器] \ GRU],
  arrow[→],
  box-node(fill: rgb("#E4E4E4"), stroke: rgb("#111111"))[背景变量 $bold(c)$ \ #text(size: 8pt)[定长]],
  arrow[→],
  box-node[#text(size: 8.5pt)[解码器] \ GRU],
)
#caption[编码器读完全部输入，压成一个向量。解码器从这个向量出发，逐词生成。]

长度问题解决了。输入的步数与输出解耦。解码器想生成几个词就生成几个词，由它自己决定什么时候停。

背景变量（context variable）是编码器交出的那个向量。书上叫背景变量，有的资料叫上下文向量。

#warn[
  这一步有个副作用。背景变量是定长的，长短句都压成同样大小。
  这就是后文 3.4 讲的瓶颈，B23 的注意力机制专门解决它。
]



// ============ 第 2 章 ============
#heading(level: 1)[二、词怎么变成向量]

这一节补一个前几章绕过去的问题。词是离散符号，神经网络只吃数字。两者之间要做一次转换。

#heading(level: 2)[2.1 one-hot 是什么]

B19 与 B20 用的是 one-hot。建一个长度等于词表大小的向量，第 $i$ 个词对应第 $i$ 位是 1，其余全 0。

#let bit(v, size: 15pt) = box(
  width: size, height: size,
  fill: if v == 1 { rgb("#222222") } else { rgb("#EDEDED") },
  stroke: 0.6pt + white,
  align(center + horizon,
    text(size: 7.5pt, fill: if v == 1 { white } else { rgb("#999") })[#v]),
)

#align(center, block(breakable: false, above: 0.8em, below: 0.4em, [
  #text(size: 9pt, fill: rgb("#666"))[词表 `['a','b','c','d']` 的 one-hot 矩阵]
  #v(4pt)
  #table(columns: 5, align: center, stroke: none,
    [], table.cell(align: center, text(size: 8pt, fill: rgb("#666"))[a]),
        table.cell(align: center, text(size: 8pt, fill: rgb("#666"))[b]),
        table.cell(align: center, text(size: 8pt, fill: rgb("#666"))[c]),
        table.cell(align: center, text(size: 8pt, fill: rgb("#666"))[d]),
    [a], bit(1), bit(0), bit(0), bit(0),
    [b], bit(0), bit(1), bit(0), bit(0),
    [c], bit(0), bit(0), bit(1), bit(0),
    [d], bit(0), bit(0), bit(0), bit(1),
  )
]))
#caption[深色格是 1。词表一多，这张表几乎全空。]

torch 里用 `F.one_hot(索引, num_classes=V)` 生成。B20 的 `to_onehot` 做的就是这件事。

#heading(level: 2)[2.2 两个问题]

*问题一，太占地方。* 词表 10000 时每个词是 10000 维，按 float32 算占 39 KB。一个 batch 32 条、每条 10 步，输入张量 12 MB，其中 99.99% 是 0。

#block(breakable: false, above: 1em, below: 0.3em, [
  #text(size: 8.5pt, fill: rgb("#666"))[词表 10000 时，一个词的占用]
  #v(5pt)
  #grid(columns: (68pt, 1fr), column-gutter: 8pt, row-gutter: 6pt, align: horizon,
    text(size: 8.5pt)[one-hot],
    block(width: 100%, height: 14pt, fill: rgb("#111111"), radius: 2pt,
      inset: (x: 5pt),
      [#align(left + horizon, text(size: 7.5pt, fill: white, weight: "bold")[39 KB])]),
    text(size: 8.5pt)[128 维嵌入],
    block(width: 4%, height: 14pt, fill: rgb("#222222"), radius: 2pt),
  )
])
#caption[实心块长度按实际占用画。差 78 倍。]

*问题二，词之间没有远近。* $i eq j$ 时两个 one-hot 向量没有一位同时是 1，内积是 0。

$ bold(e)_i dot bold(e)_j = 0 quad (i eq j) $

意思是「猫」与「狗」的距离，和「猫」与「飞机」的距离一样，都是 0。模型学不到词义的相似性。

#heading(level: 2)[2.3 嵌入是什么]

嵌入维护一个矩阵，第 $i$ 行就是第 $i$ 个词的向量。

$ bold(W) in RR^(V times d), quad V = "词表大小", quad d = "每个词的维数" $

查第 $i$ 个词，就是取第 $i$ 行。#align(center)[#text(weight: "bold", size: 11pt)[嵌入的本质是查表。]]

#align(center, block(breakable: false, above: 1em, below: 0.4em, [
  #grid(columns: (86pt, auto, auto), column-gutter: 14pt, align: horizon,
    block(stroke: 0.6pt + rgb("#888"), inset: 5pt, radius: 2pt, width: 100%, [
      #text(size: 7.5pt, fill: rgb("#999"))[权重矩阵 $(V, d)$]
      #v(2pt)
      #grid(columns: 1, row-gutter: 2.5pt,
        [#text(size: 7.5pt, fill: rgb("#999"))[第 0 行]],
        [#text(size: 7.5pt, fill: rgb("#999"))[第 1 行]],
        [#block(width: 100%, fill: rgb("#222222"), inset: 2.5pt, radius: 1.5pt,
          text(size: 7.5pt, fill: white)[第 3 行])],
        [#text(size: 7.5pt, fill: rgb("#999"))[第 4 行]],
        [#text(size: 7.5pt, fill: rgb("#999"))[...]],
      )
    ]),
    text(size: 12pt)[→],
    [#text(size: 8.5pt)[取出的向量]
     #v(2pt)
     #text(size: 7.5pt, fill: rgb("#666"))[形状 $(1, d)$]],
  )
]))
#caption[`emb(索引)` 取出对应的行。同一个索引永远取出同一行。]

与 one-hot 的根本差别在这里。one-hot 给每个词的向量固定、互相无关。嵌入的每一行是可训练参数，训练时会调整，让意思相近的词靠得近。

#note[
  B28 的 word2vec 专门讲怎么把这张表练出语义。那里会看到「国王 − 男人 + 女人 ≈ 女王」这样的结果。
  本章不训练，只要知道这张表存在、怎么查。
]

#heading(level: 2)[2.4 怎么调用]

#raw(block: true, lang: "python", "emb = nn.Embedding(num_embeddings, embedding_dim)")

第一个参数是 $V$，第二个是 $d$。创建时权重随机初始化，形状 `(V, d)`。

传入索引张量，返回对应的行。索引必须是 `0` 到 `V-1` 之间的整数。在 GPU 上要 `.to(device)`。

#note[
  `nn.Embedding` 的名字容易让人以为它做了什么变换。它没有。前向就是按行取值。
  反向时只更新被查到的那几行，不会更新全部 $V$ 行。
]

#heading(level: 2)[2.5 形状规则]

以下是在本机测的，$V = 10000$、$d = 128$。

#table(columns: (auto, auto, 1fr), stroke: (x, y) => (top: 0.5pt + rgb("#999")),
  align: (left, left, left),
  [输入], [输出], [说明],
  [`(3,)`], [`(3, 128)`], [前面几个索引，输出就多几行],
  [`(2, 2)`], [`(2, 2, 128)`], [索引的每一维都保留],
  [`(T, n)`], [`(T, n, 128)`], [本章的输入就是这个形状],
)

三条规律。索引在前面有多少维，输出就保留多少维。最后多加一维是 $d$。相同索引取出相同的行。

#heading(level: 2)[2.6 与 one-hot 等价]

设线性层权重是 $bold(W) in RR^(V times d)$。one-hot 向量 $bold(e)_i$ 只有第 $i$ 位是 1，乘上矩阵等于取第 $i$ 行。

$ bold(e)_i bold(W) = bold(W)[i] $

矩阵乘法的每个输出位都是一行乘一列。$bold(e)_i$ 只有第 $i$ 位非零，所以只有 $bold(W)$ 的第 $i$ 行参与。实测验证过，两者结果完全相同。

#warn[
  既然等价，那为什么还要嵌入。因为 one-hot 那条路要先造出 `(batch, V)` 的稠密矩阵，
  再和 `(V, d)` 相乘。嵌入直接查行，不产生中间那个大矩阵。词表越大，省得越多。
]

#heading(level: 2)[2.7 这一章用多少维]

本章题面统一用 `embed_size` 表示 $d$，书上取 32。序列长度 $T$ 加批量 $n$ 之后，编码器的输入形状是：

#table(columns: (auto, auto, 1fr), stroke: (x, y) => (top: 0.5pt + rgb("#999")),
  align: (left, left, left),
  [步骤], [形状], [说明],
  [原始索引], [`(T, n)`], [$T$ 个时间步，每步 $n$ 个整数],
  [嵌入之后], [`(T, n, e)`], [每个整数换成 $e$ 维向量],
  [送进 GRU], [同上], [GRU 吃 `(seq_len, batch, input_size)`],
)

torch 的 RNN 默认把序列长度放第 0 维。不加 `batch_first=True` 时，输入就是 `(seq_len, batch, d)`。



// ============ 第 3 章 编码器 ============
#heading(level: 1)[三、编码器]

#heading(level: 2)[3.1 做什么]

编码器是一个普通 RNN。它把整个输入序列读完，取最后一步的隐藏状态作为背景变量。

$ bold(h)_t = f(bold(x)_t, bold(h)_(t-1)), quad bold(c) = q(bold(h)_1, dots, bold(h)_T) $

书上的默认选择是 $q = bold(h)_T$，只取最后一个。RNN 的隐藏状态每步都在累积信息，第 $T$ 步看过前面所有输入，信息最全。

#heading(level: 2)[3.2 前向返回两个东西]

`nn.GRU` 的前向返回元组 `out, h_n = rnn(x, h_0)`。两者的含义不同。

#align(center, block(breakable: false, above: 0.9em, below: 0.4em, [
  #grid(columns: (auto, auto, 1fr), column-gutter: 12pt, row-gutter: 8pt, align: horizon,
    box-node[`out`], [`(T, n, h)`], [每个时间步的输出，共 $T$ 个],
    box-node[`h_n`], [`(num_layers, n, h)`], [每层最后一步的隐藏状态],
  )
]))
#caption[背景变量取 `h_n`，不取 `out`。]

#note[
  单层 GRU 时 `out[-1]` 与 `h_n[0]` 相等，两种写法结果一样。多层时不一样。
  `h_n` 是每层喂给下一层的最终状态，`out` 只反映最后一层。书上取 `h_n`，题面按 `h_n` 写。
]

#heading(level: 2)[3.3 形状走一遍]

设序列长 $T$、批量 $n$、词表 $V$、嵌入维 $e$、隐藏维 $h$、层数 1。

#flow(
  box-node(fill: rgb("#F0F0F0"))[索引 `(T, n)`],
  arrow[→],
  box-node[嵌入 `(T, n, e)`],
  arrow[→],
  box-node(fill: rgb("#E4E4E4"), stroke: rgb("#111111"))[`h_n` `(1, n, h)`],
)
#caption[从词索引走到背景变量。解码器需要的正是最后那个。]

#heading(level: 2)[3.4 背景变量是瓶颈]

背景变量是定长的。输入 5 个词与输入 50 个词，解码器拿到的向量形状相同、容量相同。

#align(center, block(breakable: false, above: 0.9em, below: 0.3em, [
  #grid(columns: (auto, auto, auto), column-gutter: 10pt, align: horizon,
    [#grid(columns: 1, row-gutter: 2pt,
      [#text(size: 7.5pt)[5 个词]],
      [#block(width: 90pt, height: 7pt, fill: rgb("#222222"), radius: 1pt)])],
    arrow[→],
    box-node(fill: rgb("#F4F4F4"), stroke: rgb("#111111"), inset: 5pt)[`(1, n, h)`],
  )
]))
#caption[不管输入多长,都压成同一个形状。50 个词的细节要挤进这个向量。]

这是 seq2seq 的固有瓶颈。B23 的注意力机制为它设计，那时不再压成一个向量，让解码器每一步重新看一遍整个输入。

// ============ 第 4 章 解码器 ============
#heading(level: 1)[四、解码器]

#heading(level: 2)[4.1 做什么]

解码器是另一个 RNN。在输出序列的第 $t'$ 步：

$ bold(s)_(t') = g(y_(t'-1), bold(c), bold(s)_(t'-1)), quad bold(o)_(t') = bold(s)_(t') bold(W)_(h q) + bold(b)_q $

三个输入。上一步的输出 $y_(t'-1)$、背景变量 $bold(c)$、上一步的隐藏状态 $bold(s)_(t'-1)$。然后过输出层得到 logits。

实现上直接输出 logits，不在这里做 softmax。softmax 交给损失函数，`CrossEntropyLoss` 内部先做 log_softmax，自己再算一遍会重复。

#heading(level: 2)[4.2 自回归]

$y_(t'-1)$ 是上一步的输出。解码器用自己的输出作为下一步的输入。

#align(center, block(breakable: false, above: 0.9em, below: 0.3em, [
  #grid(columns: (auto, auto, auto, auto, auto), column-gutter: 6pt, align: horizon,
    box-node(fill: rgb("#F0F0F0"))[$y_0$],
    arrow[→], box-node[解码器], arrow[→],
    box-node(fill: rgb("#F4F4F4"))[$y_1$ #text(size: 7.5pt, fill: rgb("#888"))[回填]],
  )
]))
#caption[上一步的输出转一圈回到输入。这叫自回归。]

生成时只能自回归。没有标签可用,手上只有刚生成的词。

#heading(level: 2)[4.3 初始状态与 detach]

解码器的初始隐藏状态来自编码器的最终状态。传之前要 `detach()`。

#align(center, block(breakable: false, above: 0.9em, below: 0.3em, [
  #grid(columns: (auto, auto, auto), column-gutter: 10pt, align: horizon,
    box-node[编码器的计算图],
    text(size: 10pt, fill: rgb("#111111"))[✂ detach],
    box-node[解码器的计算图],
  )
]))
#caption[detach 剪断两张图之间的梯度链。]

#warn[
  不 detach 的后果有两条。编码器的计算图会一直挂到解码器,显存翻倍。
  梯度会沿这条链回传到编码器,多算一遍。本章不训练,这条仍然要遵守,断言查 `requires_grad`。
]

// ============ 第 5 章 训练 ============
#heading(level: 1)[五、训练]

#heading(level: 2)[5.1 目标函数]

最大化输出序列的条件概率,取负对数得到损失。

$ -log P = -sum_(t'=1)^(T') log P(y_(t') | y_1, dots, y_(t'-1), bold(c)) $

每个时间步一个交叉熵。整个序列求和或求平均。

#heading(level: 2)[5.2 强制教学]

训练时解码器的输入有两种选择。

#block(above: 1em, below: 0.6em, grid(columns: (1fr, 1fr), column-gutter: 14pt,
  block(fill: rgb("#F4F4F4"), inset: 9pt, radius: 2pt, [
    #text(weight: "bold", fill: rgb("#111111"))[自回归]
    #v(3pt)
    输入是上一步的*预测* $hat(y)_(t'-1)$。
    #v(3pt)
    早期预测全错,输入就是噪声,误差累积,学得很慢。
  ]),
  block(fill: white, stroke: 1.2pt + rgb("#111111"), inset: 9pt, radius: 2pt, [
    #text(weight: "bold", fill: rgb("#111111"))[强制教学 teacher forcing]
    #v(3pt)
    输入是上一步的*标签* $y_(t'-1)$。
    #v(3pt)
    每一步都在正确的上下文里学习,收敛快。
  ]),
))

#note[为什么要快。模型每一步都在正确的上下文里学习「下一个词是什么」。用预测值当输入时,早期模型预测几乎全错,学不到东西。]

#heading(level: 2)[5.3 暴露偏差]

训练用强制教学,预测只能自回归。两者不一致,带来一个问题。

#align(center, block(breakable: false, above: 0.8em, below: 0.3em, [
  #table(columns: (auto, auto, 1fr), stroke: none,
    align: (left, left, left), inset: (y: 3pt), column-gutter: 12pt,
    [*训练时*], [`我 爱 学 习` → 预测 `学`], [#text(size: 8pt, fill: rgb("#888"))[看到正确的过去]],
    [*预测时*], [`我 爱 跑 习` → 预测 ?], [#text(size: 8pt, fill: rgb("#888"))[看到自己生成的过去]],
  )
]))
#caption[两者的分布不同。这个差异叫暴露偏差。]

后果是模型一旦某步出错,后面会越错越远。它从没在错误的上下文里训练过。缓解办法有 scheduled sampling,本章不展开。

#heading(level: 2)[5.4 输入与标签错一位]

#align(center, block(breakable: false, above: 0.8em, below: 0.4em, [
  #table(columns: (auto, auto), stroke: (x, y) => (top: 0.5pt + rgb("#999")),
    align: (left, left),
    [解码器输入], [`<bos>` 我 爱 学 习],
    [解码器标签], [我 爱 学 习 `<eos>`],
  )
]))

`<bos>` 是序列开始符,`<eos>` 是结束符。实现上输入是 `labels[:, :-1]`,标签是 `labels[:, 1:]`。



// ============ 第 6 章 贪婪 ============
#heading(level: 1)[六、贪婪搜索]

#heading(level: 2)[6.1 做法]

每一步取分数最大的词。生成时每步 `torch.argmax(logits, dim=-1)`,遇到 `<eos>` 或达到最大长度就停。

#heading(level: 2)[6.2 为什么不是最优]

词表有 A、B、C、`<eos>` 四个词,每步概率如下。

#align(center, block(breakable: false, above: 0.9em, below: 0.3em, [
  #table(columns: (auto, auto, auto), stroke: (x, y) => (top: 0.5pt + rgb("#999")),
    align: (left, left, center),
    [路径], [每步概率], [总概率],
    [贪婪选出 `A B C <eos>`], [$0.5 times 0.4 times 0.4 times 0.6$], [#text(fill: rgb("#111111"), weight: "bold")[0.048]],
    [另一条 `A C B <eos>`], [$0.5 times 0.3 times 0.6 times 0.6$], [#text(fill: rgb("#111111"), weight: "bold")[0.054]],
  )
]))
#caption[第二条第 2 步是 0.3,不是最大。但之后每步更高,总概率反而更大。]

贪婪在每步最优,总和不最优。原因在第 2 步选了不同的词之后,后面每一步的概率分布都变了。

// ============ 第 7 章 穷举 ============
#heading(level: 1)[七、穷举搜索]

把所有可能的序列都算一遍,取总概率最大的。计算量 $O(|cal(Y)|^(T'))$。

词表 10000、长度 10 时是 $10^40$ 个序列。不可能完成。

// ============ 第 8 章 束搜索 ============
#heading(level: 1)[八、束搜索]

#heading(level: 2)[8.1 三种策略的位置]

#align(center, block(breakable: false, above: 0.9em, below: 0.4em, [
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 10pt,
    block(fill: rgb("#F0F0F0"), inset: 8pt, radius: 2pt, [
      #align(center, [#text(weight: "bold")[贪婪]])
      #align(center, [#text(size: 8.5pt)[每步留 1 个]])
      #v(3pt)
      #align(center, [#text(fill: rgb("#666"), size: 8pt)[$O(|cal(Y)| T')$]])
    ]),
    block(fill: white, stroke: 1.2pt + rgb("#111111"), inset: 8pt, radius: 2pt, [
      #align(center, [#text(weight: "bold", fill: rgb("#222222"))[束搜索]])
      #align(center, [#text(size: 8.5pt)[每步留 $k$ 个]])
      #v(3pt)
      #align(center, [#text(fill: rgb("#666"), size: 8pt)[$O(k |cal(Y)| T')$]])
    ]),
    block(fill: rgb("#F0F0F0"), inset: 8pt, radius: 2pt, [
      #align(center, [#text(weight: "bold")[穷举]])
      #align(center, [#text(size: 8.5pt)[每步留全部]])
      #v(3pt)
      #align(center, [#text(fill: rgb("#666"), size: 8pt)[$O(|cal(Y)|^(T'))$]])
    ]),
  )
]))
#caption[束搜索在贪婪与穷举之间折中。贪婪是束宽为 1 的束搜索。]

#heading(level: 2)[8.2 算法三步]

#grid(columns: (auto, 1fr), column-gutter: 10pt, row-gutter: 7pt,
  text(weight: "bold")[第 1 步], [从 $|cal(Y)|$ 个词里取概率最大的 $k$ 个,作为 $k$ 个候选],
  text(weight: "bold")[之后每步], [对 $k$ 个候选各扩展 $|cal(Y)|$ 种,从 $k|cal(Y)|$ 个新候选里取前 $k$ 个],
  text(weight: "bold")[结束], [收集所有出现过的候选,筛出含 `<eos>` 的,砍掉之后的部分],
)

书上用束宽 2 举例。第 1 步选出 A 与 C 两条,第 2 步对 A 与 C 各扩展 5 种,从 10 个里取前 2 个,最后得到 6 条候选序列。

#heading(level: 2)[8.3 长度惩罚]

对数概率都是负数,项越多总和越负。长序列天然吃亏。

$ "score" = 1 / L^alpha sum_(t'=1)^L log P(y_(t') | dots), quad alpha = 0.75 $

两个极端。$alpha = 0$ 不惩罚,倾向选长序列。$alpha = 1$ 按长度完全归一化,倾向选短序列。

#warn[
  是 $log P$ 除以 $L^alpha$,不是 $P$ 开 $L$ 次方再加权。两者数学等价,
  但实现时用对数形式避免下溢。
]

#heading(level: 2)[8.4 实现要点]

#grid(columns: (auto, 1fr), column-gutter: 12pt, row-gutter: 8pt,
  text(weight: "bold")[候选表示], [「token 列表, 累积对数概率, 隐藏状态」三元组。每个候选有自己的状态,扩展时要用],
  text(weight: "bold")[扩展], [用最后 token 与状态做一次解码,得到 $|cal(Y)|$ 个对数概率,加到累积分数上],
  text(weight: "bold")[为什么用对数], [概率连乘到十几步就下溢成 0。对数把连乘变成相加],
  text(weight: "bold")[什么时候停], [候选生成 `<eos>` 就移入「完成集合」,停止扩展。完成集合空时从没完成的里选],
)

// ============ 第 9 章 坑 ============
#heading(level: 1)[九、实现会踩的坑]

#table(columns: (auto, auto), stroke: (x, y) => (top: 0.4pt + rgb("#999")),
  align: (left, left), inset: (y: 4pt),
  [#text(weight: "bold")[坑]], [#text(weight: "bold")[症状与原因]],
  [背景变量取了 `out`], [形状对但语义错。多层时 `out` 只反映最后一层,该取 `h_n`],
  [忘了建嵌入层], [把整数索引直接喂给 GRU。GRU 吃浮点向量,不吃整数],
  [解码器输入忘了 `<bos>`], [第一个词总是错。输入要是 `labels[:, :-1]`],
  [训练时用预测值], [收敛极慢。该用强制教学],
  [初始状态没 detach], [显存翻倍。编码器计算图连进了解码器],
  [束搜索分数用乘积], [下溢成 0,所有候选无法比较。要用对数概率相加],
  [没做长度惩罚], [总是选最短或最长。除以 $L^alpha$],
  [完成了的候选还在扩展], [输出多出无意义字符。遇到 `<eos>` 就停止],
  [忘了截断 `<eos>` 之后], [输出带一堆填充。砍掉 `<eos>` 之后的部分],
  [候选不存隐藏状态], [形状错误或结果错。每个候选要带自己的状态],
)



// ============ 速查 ============
#heading(level: 1)[命令速查]

#table(columns: (auto, 1fr, auto), stroke: (x, y) => (top: 0.4pt + rgb("#999")),
  align: (left, left, center), inset: (y: 4pt),
  [#text(weight: "bold")[命令]], [#text(weight: "bold")[作用]], [#text(weight: "bold")[节]],
  [`nn.Embedding(V, d)`], [建词嵌入表,权重形状 `(V, d)`], [2.4],
  [`emb(索引张量)`], [查表,索引几维输出就多一维 $d$], [2.5],
  [`F.one_hot(t, num_classes=V)`], [生成 one-hot,B20 用过], [2.1],
  [`nn.GRU(input_size, hidden_size)`], [编码器与解码器的主干], [3.1],
  [`out, h_n = rnn(x, h_0)`], [前向,`h_n` 是最后状态], [3.2],
  [`state.detach()`], [切断编码器的计算图], [4.3],
  [`torch.softmax(logits, dim=-1)`], [概率分布], [4.1],
  [`torch.argmax(logits, dim=-1)`], [贪婪搜索取词], [6.1],
  [`torch.log_softmax(logits, dim=-1)`], [对数概率], [8.4],
  [`logits.topk(k)`], [取前 k 个], [8.4],
)

// ============ 自测 ============
#heading(level: 1)[自测题]

#set enum(numbering: "1.")
+ seq2seq 为什么需要两个 RNN,给三条理由。
+ one-hot 的两个问题各是什么,各给一个具体数字。
+ 嵌入层在做什么,权重是什么形状。
+ 解释为什么 one-hot 后接线性层与嵌入层等价。
+ 一个 `nn.Embedding(5000, 64)` 接受 `(5, 8)` 的索引,输出什么形状。
+ 编码器前向的两个返回值、各自形状、各自含义。
+ 背景变量的瓶颈在哪,B23 用什么技术解决。
+ 解码器在训练时与预测时的输入来源分别是什么。
+ 强制教学的定义与代价。
+ 贪婪为什么不能保证最优,算出书上的 0.048 与 0.054。
+ 穷举与束搜索的计算量,束宽为 1 时退化成什么。
+ 束搜索为什么需要长度惩罚,写出打分公式与 $alpha$ 的常用值。
+ 束搜索实现时为什么用对数概率相加。

#heading(level: 1)[答案]

#block(fill: rgb("#F4F4F4"), stroke: (left: 2.5pt + rgb("#111111")), inset: (x: 9pt, y: 6pt), radius: (right: 2pt), above: 0.6em, below: 1em, [
  #text(size: 9pt)[先自己做一遍。看一眼答案能想起来,合上纸还是写不出。]
])

#text(size: 9pt)[
*1.* 长度不同、词表不同、顺序习惯不同。单个 RNN 的输出长度与输入绑定。

*2.* 占地方。词表 10000 时每词 39 KB。词之间没有远近。任意两个 one-hot 内积为 0。

*3.* 维护 `(V, d)` 矩阵,第 $i$ 行是第 $i$ 个词的向量,查词就是取行。权重形状 `(V, d)`。

*4.* one-hot 只有第 $i$ 位是 1,乘 `(V, d)` 矩阵只有第 $i$ 行参与,结果就是第 $i$ 行。嵌入直接查这一行,不产生中间大矩阵。

*5.* `(5, 8, 64)`。

*6.* `out` 是 `(T, n, h)`,每个时间步的输出。`h_n` 是 `(num_layers, n, h)`,每层最后一步的状态。背景变量取 `h_n`。

*7.* 定长,长短句压成同样大小,长句丢信息。B23 的注意力让解码器每步重看整个输入。

*8.* 训练用上一步的标签(强制教学),预测用上一步的预测(自回归)。

*9.* 训练时用真实标签而非预测当输入。代价是暴露偏差,训练看到正确的过去,预测看到自己生成的过去。

*10.* 贪心每步取局部最大,第 2 步选次优可能让后面每步更高。$0.5 times 0.4 times 0.4 times 0.6 = 0.048$, $0.5 times 0.3 times 0.6 times 0.6 = 0.054$。

*11.* 穷举 $O(|cal(Y)|^(T'))$,束搜索 $O(k |cal(Y)| T')$。束宽 1 退化成贪婪。

*12.* 对数概率是负数,项越多越负,长序列吃亏。$"score" = 1/L^alpha sum log P$,$alpha = 0.75$。

*13.* 概率连乘十几步就下溢成 0,无法比较。对数把连乘变成相加。
]
