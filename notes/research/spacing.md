# 间隔重复与间隔练习：实证查证与可执行复习排期

面向编程与数学学习的复习排期。每条结论标注来源与证据强度。

## 结论速览

1. **间隔效应成立**，分散练习优于集中练习。最有力的元分析：Cepeda 等 (2006)、Donovan & Radosevich (1999, 平均效应量 0.46)、Mawson & Kang (2025, 课堂研究 d = 0.54)、Murray 等 (2025, 专做数学, g = 0.28)。
2. **有定量结论，形式是比例**。Cepeda 等 (2008)：最优复习间隔随目标保持期增加，但比例下降，从 1 周保持期的 20%–40% 降到 1 年保持期的 5%–10%。Cepeda 等 (2009) 实测：保持期 6 个月时最优间隔 28 天。间隔过短的损失远大于间隔过长。
3. **排期表在第 9 节与附录 A**：一个单元学完后在第 1、3、7、16、35、85 天回看（前 6 次必做），第 180 天可选。形式依次为闭卷自测、变式题、交错混合、限时闭卷、综合应用、跨单元综合。
4. **事实与技能不一样**。事实类记忆证据强、效应量大（g = 0.28–0.93）；数学程序性技能效应明显更小（g = 0.28），且数学上"测试 vs 重学"的结论不稳健（g = 0.18，置信区间跨零）。技能中间隔起作用主要靠提取与"问题—策略配对"，不靠重新输入。
5. **检索练习与间隔重复是两个可叠加的机制**，分别管"做什么"和"什么时候做"。重读被评为低效用，因为它提高的是当下的熟悉感，不是延迟后的提取能力。
6. **交错练习**（混合题型练习）元分析 g = 0.42，数学上 d = 0.83（RCT）。适合需要辨别方法的情境，对单词表反而有害（g = −0.39），对软件工具操作类技能未见收益。

本文第 8 节列出检索缺口：编程学习中纯粹操纵间隔长度的随机对照试验、Anki/SM-2 用于编程的对照研究、编程题上纯粹的交错练习实验，均未找到。

证据强度分级：

- **强**：元分析（多研究汇总），或多篇独立对照实验一致
- **中**：单项对照实验（随机分配，或有明确控制条件）
- **弱**：相关性研究、无对照的工具/案例报告、单一小样本、理论推测或流行说法

本文的数字均来自论文摘要或全文原文。二手转述会明确标注。查不到的项目单列在第 8 节。

---

## 1. 间隔效应（spacing effect）的核心结论

**结论**：同一份材料分多次学习，比集中一次学完，长期保持更好。这就是间隔效应，也叫分散练习效应（distributed practice effect）。这个结论在 1885 年 Ebbinghaus 的自实验中被首次记录，此后 130 余年有数百项研究，是现代学习科学中证据最扎实的几条之一。**证据强度：强。**

最有力的几篇元分析（按重要性排序）：

| 文献 | 规模 | 关键数字 |
|---|---|---|
| Cepeda, Pashler, Vul, Wixted, Rohrer (2006), *Psychological Bulletin* 132(3), 354–380 | 184 篇文章中的 317 个实验、839 个评估 | 学段间隔（ISI）与保持间隔共同决定最终保持；最优 ISI 随保持间隔增加。**该文没有给出间隔效应的单一平均效应量**（Latimier 等 2021 明确指出这一点，常被误引） |
| Donovan & Radosevich (1999), *Journal of Applied Psychology* 84(5), 795–805 | 63 项研究、112 个效应量 | 加权平均效应量 **0.46**，分散优于集中；任务性质、试次间隔及二者交互显著调节该效应 |
| Mawson & Kang (2025), *Behavioral Sciences* 15(6), 771 | 22 篇报告、31 个效应量、N > 3000（只收课堂研究） | 分散优于集中 **d = 0.54, 95% CI [0.31, 0.77]**；保持间隔更长、学习者教育水平更高、材料重复暴露次数更少时效应量更大 |
| Murray, Horner, Göbel (2025), *Educational Psychology Review*（DOI 10.1007/s10648-025-10035-1） | 34 项研究、85 个效应量（只收数学材料） | 数学中分散优于集中 **g = 0.282, se = 0.045, 95% CI [0.188, 0.376]**；拆开看，孤立技能学习 g = 0.427 [0.179, 0.675]，课程内嵌 g = 0.24 |

效应量的横向比较（除标注为原文者外均为二手转述，**证据强度：中**）：一般任务 d = 0.46（Donovan & Radosevich 1999 原文）；二语词汇 g = 0.80（Kim & Webb 2022，转述）；有论文把语言材料的间隔效应记为 d = 0.85 并归到 Cepeda 等 (2006)，但该文并未给出这一数字，其出处未能核实；数学 g = 0.28（Murray 等 2025 原文）。**数学与程序性任务的间隔效应小于言语材料，排期设计需要考虑这一点。**

---

## 2. 最佳复习间隔的定量结论

有定量结论。结论以比例形式给出，没有跨情境通用的固定天数。

### 2.1 比例规则（核心数字）

Cepeda, Vul, Rohrer, Wixted, Pashler (2008), *Psychological Science* 19(11), 1095–1102。**证据强度：强**（1,354 名以上被试，学段间隔最长 3.5 个月，最终测试延迟最长 1 年）。

- 最优间隔随测试延迟增加而增加。
- 但作为测试延迟的**比例**，最优间隔随延迟增加而**下降**：
  - 1 周测试延迟 → 最优间隔约为延迟的 **20% 到 40%**（即 1.4–2.8 天）
  - 1 年测试延迟 → 最优间隔约为延迟的 **5% 到 10%**（即 18–36 天）

Cepeda, Coburn, Rohrer, Wixted, Mozer, Pashler (2009), *Experimental Psychology* 56(4), 236–246。**证据强度：中**（两项实验室实验，三学段设计）。

- 实验 1（斯瓦希里语—英语词对，学段间隔 5 分钟至 14 天，测试延迟 10 天）：最优间隔为 **1 天**（拟合函数给出 3.7 天）。
- 实验 2（冷门事实与陌生物体图片名称，学段间隔 20 分钟至 6 个月，最终测试在第二学段后 6 个月）：最优间隔为 **28 天**（拟合值：事实 25.6 天，物体 37.1 天）。
  - 28 天间隔的最终回忆比 0 天间隔高 **151%**；1 天间隔只高 **18%**。
  - 间隔从 28 天增至 168 天，保持量只下降 **23%**。
- 该文图 5 汇总了 Cepeda 等 2006 元分析中所有含最优间隔的研究加本文数据（共 48 个数据点）：**分钟量级的测试延迟，最优间隔/延迟比接近 1.0；多日量级的延迟，该比例接近 0.1。**
- 两条实践含义（原文）：间隔应当按"月"量级安排，天或周量级偏短；**间隔过短的惩罚远大于间隔过长的惩罚**。

### 2.2 怎么用这两个数字

把"目标保持期"（你希望多久之后还能用出这个知识）乘以比例：

| 目标保持期 | 最优首次复习间隔（按 5%–40% 区间取） | 来源 |
|---|---|---|
| 1 周（下周考试） | 1–2 天 | Cepeda 2008 |
| 1 个月 | 3–7 天 | Cepeda 2009 实验 2 外推 |
| 6 个月 | 约 4 周 | Cepeda 2009 实验 2 实测 28 天 |
| 1 年 | 3–5 周 | Cepeda 2008 |
| 数年（职业技能） | 1–3 个月 | Cepeda 2009 的"月量级"建议 |

**必须说明的限制**：这些比例全部来自言语材料（词对、冷门事实、图片名称）。没有找到把它们直接外推到编程技能或数学推导的研究。用于技能时这只属于合理外推，缺少实测支持（**证据强度：弱**）。

### 2.3 扩展间隔还是均匀间隔

常见说法是"复习间隔应逐次拉长"（SuperMemo/Anki 类算法的核心假设）。**元分析不支持这个说法。**

Latimier, Peyre, Ramus (2021), *Educational Psychology Review*（预印本 PsyArXiv kzy7u；发表版 DOI 10.1007/s10648-020-09572-8，本文数字来自预印本全文，发表版数字未核对）。**证据强度：强（元分析），但结果"无差异"需谨慎解读**。

- 子集 1（分散检索 vs 集中检索）：11 项研究、39 个比较，g = 1.02, 95% CI [0.68, 1.36]；发表偏倚校正（trim-and-fill）后 **g = 0.74, 95% CI [0.55, 0.92]**，I² = 48%。
  - 注：Murray 等 2025 引用该文为 g = 1.01，对应的是校正前的数值。
- 子集 2（扩展间隔 vs 均匀间隔）：16 项研究、54 个比较，**g = 0.032, 95% CI [−0.10, 0.17], p = 0.62, I² = 0%**。55% 的效应量为正，43% 为负。
  - 唯一接近显著的调节变量：每个知识单元的暴露次数超过 4 次时，扩展间隔略优（g = 0.2, 95% CI [−0.07, 0.46], p = 0.09，**证据强度：弱**）。
- 原文结论：结果支持把重复提取分散到不同时间，但**不支持"复习间隔应逐步延长直到测试"这一广泛流传的看法**。

**对排期表的直接影响**：不必执着于严格的扩展间隔。均匀间隔（例如固定每 7 天一次）与扩展间隔在证据上没有差别。真正重要的是"分散到不同的日子"和"每次都做提取"。

---

## 3. 检索练习与间隔重复的关系，以及重读为什么效果差

（先回答第 5 问，因为排期表的"复习形式"一栏完全依赖这部分结论。）

### 3.1 两者的关系

它们是两个独立、可叠加的机制：

- **间隔重复**管的是"什么时候复习"（时间安排）
- **检索练习（retrieval practice / testing effect）**管的是"复习时做什么"，要求从记忆中提取，与重新输入相对

Latimier 等 2021 的汇总认为两者效应大致可加：单独的间隔效应 g ≈ 0.71，单独的检索练习效应 g = 0.50–0.61；把两者结合起来（分散安排的检索练习）与集中安排的检索练习相比，g = 0.74（发表偏倚校正后）到 1.02（**证据强度：中**，该文明确说"现有数据不足以直接检验两者的交互"）。两者结合称为**间隔检索练习（spaced retrieval practice）**，是证据最强的组合。

检索练习的元分析证据：

| 文献 | 规模 | 关键数字 |
|---|---|---|
| Rowland (2014), *Psychological Bulletin* 140(6), 1432–1463 | 元分析 | 测试 vs 重学 **g = 0.50, 95% CI [0.42, 0.58]**；保持间隔 <1 天 g = 0.41，>1 天 g = 0.69；有反馈 g = 0.73，无反馈 g = 0.39 |
| Adesope, Trevisan, Sundararajan (2017), *Review of Educational Research* 87(3), 659–701 | 元分析 | 测试 vs 无活动 **g = 0.93**；测试 vs 重读 **g = 0.51**；总体 g = 0.61 |
| Yang, Luo, Vadillo, Yu, Shanks (2021), *Psychological Bulletin* 147(4), 399–435 | 222 项独立研究、48,478 名学生（真实课堂） | 总体 **g = 0.499**；有纠正性反馈 g = 0.537，无反馈 g = 0.374 |
| Murray 等 (2025)（数学专用） | 7 项研究、32 个效应量 | 测试 vs 重学 **g = 0.18，95% CI 跨零，不稳健** |

**数学是例外**。数学材料上"测试 vs 重学"的元分析结论不成立（置信区间跨零）。数学上证据更强的是**交错练习**（见第 5 节），单纯的"把重读换成自测"在数学上获益有限。

即使把对照组从重读换成更主动的精细学习，结果也不变。Karpicke & Blunt (2011), *Science* 331(6018), 772–775：在科学文本上，检索练习带来的学习收益大于用概念图做精细学习；这一优势在理解题与推理题上成立，甚至当终测本身就是"画概念图"时也成立。**证据强度：中（单项实验，但设计严谨、结论被后续研究重复）。**

### 3.2 重读为什么效果差

**结论：重读被评为低效用技术，与自测的直接比较中一致落后。证据强度：强（元分析级评审）。**

Dunlosky, Rawson, Marsh, Nathan, Willingham (2013), *Psychological Science in the Public Interest* 14(1), 4–58。这篇评审对 10 种学习技术做效用评级：

- **高效用**：练习测试（practice testing）、分散练习（distributed practice）
- **中等效用**：精细提问、自我解释、交错练习
- **低效用**：总结、划线/高亮、关键词记忆法、文本学习中的想象、**重读**

该文关于重读的具体判断（原文第 7 节）：

- 多数显示重读效应的研究，终测在最后一次学习后几分钟内进行；在 1–2 天延迟后，多项研究未发现显著效应（Callender & McDaniel 2009；Cranney 等 2009；Hinze & Wiley 2011；Rawson & Kintsch 2005）。
- 与精细提问、自我解释、练习测试的直接比较中，重读一致更差。
- 没有任何实验研究在教育情境中评估过重读的效果。

机制层面的解释（**证据强度：中**，属于理论模型而非直接测量）：

- Roediger & Karpicke (2006), *Psychological Science* 17(3), 249–255：重复学习提高了学生对自己记忆的信心，但延迟测试成绩更低。这种"感觉学会了"与实际保持能力脱钩的现象，通常称为**流畅性错觉**。
- Karpicke (2009), *Journal of Experimental Psychology: General* 138(4), 469–486：学生一旦能回忆某项目，就倾向于停止练习，而不是继续做提取练习，导致保持变差。
- Soderstrom & Bjork (2015), *Perspectives on Psychological Science* 10(2), 176–199：训练过程中可观察到的**表现**（performance）不是长期**学习**（learning）的可靠指标，某些操作对二者有相反效果。重读提高的是当下的表现。

Roediger & Karpicke (2006) 的具体数字（**证据强度：中**，单项实验，n ≈ 40/组）：

| 终测时间 | 重复学习组 | 测试组 | 差异 |
|---|---|---|---|
| 5 分钟后 | 81% | 75% | 重读占优，d = 0.52 |
| 2 天后 | 54% | 68% | 测试占优，d = 0.95 |
| 1 周后 | 42% | 56% | 测试占优，d = 0.83 |

实验 2（1 周后回忆）：多次测试组 61% > 部分测试组 56% > 全部重读组 40%。而全部重读组对 1 周后记忆的信心反而最高。

**一个容易被忽略的补充**：把重读本身分散开，效果也不好。Greving & Richter (2019), *Frontiers in Psychology* 9:2517（191 名七年级学生，预注册）：分散重读被感知为更难、学生预测的成功率更低；短保持间隔时集中重读更好；分散重读组没有出现遗忘，但在长间隔测试上与集中重读持平。**原文结论是分散重读未显示出有益效果。证据强度：中。**

这解释了排期表的设计原则：**间隔必须配合提取才有价值。把复习安排成"再看一遍"是浪费排期。**

---

## 4. 间隔重复对"记住事实"与"掌握技能"的效果是否相同

**直接回答：不相同。事实类材料证据强、效应量大；技能类证据中等、效应量小，且高度依赖任务复杂度和练习场景。**

### 4.1 事实/陈述性知识

证据强。词对、冷门事实、图片名称、医学术语、外语词汇，见第 1、2 节的元分析与实验。这类材料的效应量约在 g = 0.46–0.93 之间，间隔比例规则有实测数字。

间隔不只促进记忆，也促进泛化。Gluckman, Vlach, Sandhofer (2014), *Applied Cognitive Psychology* 28(2), 266–273：36 名小学低年级儿童接受三种排期的科学课（集中、成簇、分散），1 周后测试，分散组在**记忆与泛化**两项上都显著优于其他组，且两项表现之间没有相关。**证据强度：中（小样本单项实验）。**

### 4.2 数学（程序性技能）

证据中等，效应量明显更小：

- 数学间隔效应 g = 0.282（Murray 等 2025 元分析，27 项研究、53 个效应量）。孤立技能学习 g = 0.427，课程内嵌 g = 0.24。
- 数学中"测试 vs 重学"g = 0.18，95% CI 跨零，**不稳健**。
- 数学中证据最强的是交错练习：Rohrer, Dedrick, Hartwig, Cheung (2020), *Journal of Educational Psychology* 112(1), 40–52，预注册集群随机对照试验，54 个七年级班级、700 余名学生、4 个月。1 个月后突击测试：交错组 **61%** vs 分块组 **38%，d = 0.83**。教师无需培训即可实施。**证据强度：强。**
- Rohrer, Dedrick, Stershic (2015), *Journal of Educational Psychology* 107(3), 900–908（126 名七年级学生，3 个月）：交错 vs 分块，即时测试 d = 0.42，30 天后 d = 0.79。**证据强度：中。**

### 4.3 复杂运动/操作技能

证据中等，且"实验室有效、真实场景常常无效"：

- Czyż, Wójcik, Solarská, Kiper (2024), *Scientific Reports*（54 项研究元分析）：高背景干扰（随机练习）对保持有中等有益效应。实验室情境有效，**应用情境中效应几乎可忽略**；老年人效应大，成人中等，**年轻参与者可忽略**。**证据强度：强（元分析），但结论是"有条件成立"。**
- Czyż, Wójcik, Solarská (2024), *Frontiers in Psychology*（迁移，34 项研究元分析）：整体迁移 SMD = 0.55；实验室 0.75（显著），应用场景 0.34（不显著）；成人 0.54，老年人 1.28，年轻人 0.12。
- Moulton 等 (2006), *Annals of Surgery* 244(3), 400–409：38 名初级外科住院医师随机分配。集中组 1 天内 4 个训练时段，分散组 4 周内每周 1 次，总练习时间相同。训练后即时测试两组无差异；**1 个月后的保持与迁移（活体大鼠）分散组显著更好**。**证据强度：中。注：数字与结论来自一篇评论文章的转述，未读到原始全文。**

这条外科技能实验与 Soderstrom & Bjork (2015) 的"学习 vs 表现"区分完全吻合：分散练习在训练期间看起来没优势，差异只在延迟后出现。

### 4.4 编程

直接证据比数学和运动技能少，但近年来出现了若干针对 CS 教育的对照研究：

- **Li, Ning, Zhang, Yang, Zhang (2021), *Journal of Pacific Rim Psychology* 15**：编程教育中目前唯一直接操纵练习节奏的对照实验。200 名大一新生（C 语言课）通过同一移动平台收到**相同数量**的多选题，唯一差别是练习节奏的引导：对照组被鼓励每 7 天练一次，实验组每 3 天练一次。结果实验组的期末成绩显著更高，首次作答正确率也更高。**证据强度：中。**设计上的混淆需要指出：改变的是练习节奏的**引导**（催促频率），两组实际完成的时间分布与投入未必相同，因此差异不能完全归因于间隔长度本身。（有二手转述给出期末 85.67 vs 77.40, t = 4.76, p < .001，我未能独立核对原文；摘要级结论已核实。）
- **YeckehZaare & Resnick (2025), *npj Science of Learning***：两项随机对照试验。课程内随机分配 143 名学生；另一项随机分配 71 名教师。对照条件按"答题数"给分，处理条件按"练习天数"给分（Counting Days）。课程内实验中，按天数给分的组**考试成绩更高，中介变量是在更多天里练习**；**对低 GPA 学生特别有益**——该组课程成绩与既往 GPA 的相关显著更低（即降低了对先验能力的依赖）。教师间实验中，练习天数与题数都显著更高。**证据强度：中偏强（两个 RCT，但教师间实验无法比较学习结果）。**
- **YeckehZaare, Resnick, Ericson (2019), ICER '19**（大型入门 Python 课程）：按"每天答够最少题数"给分以强制间隔；间隔重复算法调度**主题**而非具体题目；提供排期可视化支持元认知。回归模型（控制多个混淆）：**每使用工具 1 小时，期末成绩提高 1.04%**；193 名学生中 62 人（32%）自愿使用超过要求的 45 天。**证据强度：中（回归分析，非随机）。**
- **Smith, Emeka, Fowler, West, Zilles (2023), SIGCSE TS '23**：跨学期准实验，入门 CS 课程。频繁测试学期（4 次小测 + 4 次考试）的学生在**代码写作题**上比不频繁测试学期（1 次期中 + 1 次期末）高 **9.1 到 13.5 个百分点**。**证据强度：中（准实验，跨学期比较存在混淆）。**
- **Moraes, Lionelle, Ghosh, Folkestad (2023)**, ACM 会议论文（CS1 课程）：把测验改造为低风险形成性检索练习活动，学生可在学期内以间隔、交错方式反复自测；配合学习行为可视化与反思。结果显示间隔+交错练习的学生在期末笔试、期末编码考试、课程总评上均显著提高。**证据强度：中（准实验，且是多成分组合干预，无法分离单项）。**
- **Tate & Naidu (2026), SIGCSE TS**：每周 workbook（概念概览 + 3 个递进作业），"早期结果"提示保持改善与成绩提高。**证据强度：弱（无对照，作者自述为 early results）。**

方向相反的一条（未能独立核实）：Herman, Patel, Emeka, Zilles, West (2025), ICER '25 报告，在一门计算机体系结构课中比较三种测试制度（11 次考试无补考、3 次考试有补考、4 次考试有补考），**共同期末考试成绩无统计显著差异**。该文摘要未能获取（ACM 与 Cloudflare 拦截），结论来自二手转述，不纳入本文的判断依据。

### 4.5 技能学习里间隔起作用的是哪个机制

对技能而言，起作用的机制与"巩固记忆痕迹"不同。现有证据指向三条：

1. **提取/重建过程本身**。Karpicke & Roediger (2008), *Science* 319(5865), 966–968：学生学会词对后，继续重复学习对 1 周后回忆无影响；继续测试则产生大幅提升。学生对自己表现的预测与实际成绩不相关。**证据强度：中（单项实验，但结论被多项元分析支持）。**
2. **问题与策略的配对能力（discriminative contrast）**。Taylor & Rohrer (2010), *Applied Cognitive Psychology* 24(6), 837–848：儿童练习四类数学题，交错或分块，**间隔程度被固定**（这一设计排除了"交错只是变相间隔"的解释）。交错降低了练习阶段的表现，但一天后测试分数**翻倍**。错误分析表明交错提高成绩的途径是改善"把问题与正确程序配对"的能力。**证据强度：中。**
3. **遗忘后的再巩固（reconsolidation）**。Smith & Scarf (2017), *Frontiers in Psychology* 8:962 综述了 24 小时以上间隔的研究，涵盖技能类、语言类任务与泛化，区分学习（训练末表现）与保持（延迟后表现），提出间隔通过影响后续的巩固与再巩固过程起作用。**证据强度：弱到中（理论综述，非直接测量）。**

对编程的直接含义：**"看懂了"和"能写出来"是两种状态，只有后者依赖提取。** 复习编程内容时，重读代码或笔记练到的是识别能力；从空白文件开始写练到的才是生成能力。

---

## 5. 交错练习（interleaving）

### 5.1 定义

把不同类型的问题混合编排，避免同一类型连续刷（`abcabc` 优于 `aaabbbccc`）。它与间隔练习相关但不同：分块练习时，同一技能的两次练习是连续的；交错练习天然使同一技能的练习分散开。Taylor & Rohrer (2010) 通过固定间隔程度证明，交错本身有独立于间隔的效应。

### 5.2 证据

Brunmair & Richter (2019), *Psychological Bulletin* 145(11), 1029–1052。**证据强度：强（元分析）**。59 项研究、238 个效应量、158 个样本。

| 材料类型 | 交错效应 |
|---|---|
| 总体 | **g = 0.42** |
| 绘画 | g = 0.67 |
| 其他视觉材料 | 优于绘画 |
| 数学任务 | **g = 0.34** |
| 说明性文本、味觉 | 不显著 |
| 单词 | **g = −0.39（分块反而更好）** |

调节变量：类别之间相似度高、类别内部相似度低、材料更复杂时，交错效应更大。

教育场景证据：Rohrer 等 (2020) 的集群 RCT（见 4.2 节），交错组 61% vs 分块组 38%，d = 0.83。

### 5.3 适合与不适合

适合：

- 需要辨别"该用哪个方法"的领域（数学题型、算法选择、诊断题）
- 类别之间容易混淆的材料（相似度高的概念）
- 复杂材料

不适合：

- 单词表、词汇配对（分块更好，g = −0.39）
- 说明性文本（无显著效应）
- 尚未掌握单项技能的阶段（交错的前提是每个技能已经会了，交错练的是选择）
- **软件工具的操作类学习**：van der Meij & Nuketayeva (2023), *Computers & Education* 199, 104786（49 名大学生学 Word 进阶排版），分块、交错、混合三种练习安排对概念知识与程序性知识的发展均无显著影响；van der Meij & Maseland (2021), *Social Sciences & Humanities Open* 3(1), 100133（小学新手学 Word），分块与交错无显著差异，趋势偏向分块，作者据此建议初次技能学习仍用分块。**证据强度：中（两项独立实验，样本较小）。**

这一条对编程有直接含义：编程学习中有一部分内容接近软件操作技能（记住工具、API、快捷键、IDE 操作），交错未必带来收益。交错真正发挥作用的地方是需要**判断用哪个方法**的场合（算法选择、调试策略、题型识别）。

Hartwig & Rohrer (2025), *Behavioral Sciences* 15(8), 1047（174 + 233 名七年级学生调查）：**绝大多数学生认为交错练习既不喜欢也无效**，而它实际有效。这是实施中的主要障碍。**证据强度：中（调查，非实验）。**

---

## 6. 这些结论在什么条件下不适用

### 6.1 材料太难、首次提取失败时

检索练习要求学习者能"努力地想起来"。想不起来时它退化为无效甚至有害。

- Rowland (2014) 的一个子分析（**二手转述**，来自一篇 2024 年论文对该元分析的引用）：在无反馈的研究中，初始测试表现低于或等于 50% 时，测试效应不可靠（g = 0.03）。
- Yang 等 (2021) 原文数据：课堂测验中提供纠正性反馈 g = 0.537，不提供 g = 0.374。**证据强度：强。**
- 同一篇 Yang 等 2021 转述：Adesope 等 (2017) 未发现反馈的调节作用（有反馈 g = 0.63，无反馈 g = 0.60）。两篇元分析结论不一致。

**操作含义**：复习时正确率低于大约一半，说明材料还没学会，此时应当回到学习（看讲解、看示例、请教他人），不应硬撑着继续自测。每次自测都必须能核对答案。

### 6.2 学习者水平

证据方向不一致，需要如实并列：

- Mawson & Kang (2025)：**教育水平更高**的学习者，效应量更大。
- YeckehZaare & Resnick (2025)：**低 GPA 学生**从"按天给分"的间隔激励中获益特别大，该组课程成绩与既往 GPA 的相关显著降低。
- Czyż 等 (2024)（运动技能）：年轻参与者的背景干扰效应可忽略（SMD = 0.12），成人与老年人效应明显。

这两条并不必然矛盾（一条讲教育阶段，一条讲同一阶段内的成绩分布），但现有证据不足以给出"什么水平的人该用多长的间隔"这样的定量建议。**这一项证据强度：中，且方向不一致。**

### 6.3 任务复杂度

三个来源给出不同方向，原因是"复杂度"的操作定义不同：

- Donovan & Radosevich (1999)：任务越复杂，间隔效应越小。多篇后续论文据此认为，高复杂度、高心理与体力要求、且练习间隔超过一天的任务，间隔未必有益（**二手转述**）。
- Brunmair & Richter (2019)：材料越复杂，**交错**效应越大（元分析原始数据，**强**）。
- Murray (2025) 博士论文《Spacing and Task Complexity in Mathematics Learning》：用四个实验操纵程序复杂度（步骤数）与元素交互性（工作记忆同时保持的元素数），**未发现复杂度与间隔效应的交互**；作者的解释是间隔效应有多种底层机制相互竞争，因而跨任务稳健。**证据强度：中（博士论文，尚未全部同行评审）。**

**操作含义**：不要依据"任务复杂所以间隔没用"来放弃间隔。但也不要指望间隔对复杂技能产生像对词汇那样大的效果。

### 6.4 材料类型

交错练习对单词表有害（g = −0.39），对说明性文本无显著效应（Brunmair & Richter 2019）。间隔+自测对事实类材料效果最大，对数学程序性技能效果明显更小（Murray 等 2025）。

### 6.5 学习目标包含"近期就要用"时

Roediger & Karpicke (2006)：5 分钟后测试，重复学习组 81% 优于测试组 75%。Soderstrom & Bjork (2015) 系统论述了训练期表现与长期学习可能反向。**如果目标是明天或三天后的考试，集中练习、重读、连续刷同类题在短期表现上不差。**

### 6.6 自愿执行的间隔安排会崩溃

- Maligaya (2026)，Queen's University 硕士论文：1,706 名普通化学学生中，318 人使用了自愿间隔练习工具，61 人完成全部四个有机化学会话，**只有 5 人按预期的间隔安排使用**。原文结论是"自愿间隔在真实条件下崩溃"。**证据强度：弱到中（学位论文；其中第二项研究只有 6 名完成者）。**
- YeckehZaare 等 (2019)：必须靠"按天给分"这类制度设计才能让学生真的分散练习。193 名学生中 32% 自愿超额完成。

**操作含义**：排期表如果不落到日历、提醒或某种外部约束上，实际执行率会很低。这一条的证据比多数学习技巧的证据更贴近真实使用。

### 6.7 外推限制

第 2 节的间隔比例数字全部来自言语材料。把它们用于编程技能或数学推导属于合理外推，没有直接实测支持（**弱**）。第 4.4 节的编程证据全部来自 CS 入门课程，没有针对有经验开发者的研究。

---

## 7. 工具与算法：能信到什么程度

- **SM-2 / Anki 类算法的具体参数没有直接的随机对照证据。** SuperMemo 的 SM-2 是 1987 年提出的启发式规则，其"间隔逐次乘以难度系数"的设计假设被 Latimier 等 (2021) 的元分析削弱（扩展间隔与均匀间隔无显著差异，g = 0.032）。**证据强度：弱（算法本身），强（对扩展间隔假设的否定）。**
- **自适应排期有理论与自然实验支持。** Tabibian 等 (2019), *PNAS* 116(10), 3988–3993：把排期设计成最优控制问题，得出最优复习时刻由"回忆概率本身"决定（MEMORIZE 算法）；用 Duolingo 数据做的大规模自然实验显示，遵循该算法排期的学习者记忆效果优于多种启发式排期。**证据强度：中（自然实验，非随机对照）。**
- **半衰期回归能预测回忆率。** Settles & Meeder (2016), ACL：用 Duolingo 数据拟合，预测回忆率误差比多个基线降低 45% 以上；运营用户研究中日活提升 12%。注意这测的是预测精度与参与度，不是学习效果的因果证据。**证据强度：中。**
- **按天给分有效。** YeckehZaare & Resnick (2025) 的两个 RCT（见 4.4）。**证据强度：中偏强。**

**实践结论**：算法的具体公式不重要，被证据支持的是它的三个行为要求——把复习分散到不同的日子、每次都做提取、每次都能核对答案。任何满足这三条的安排（纸质日历、Anki、自建脚本）在证据上没有优劣之分。

---

## 8. 检索缺口（查不到的部分）

以下项目经过检索未找到可用证据。列出来是为了避免把它们当作已知结论。

1. **编程学习中纯粹操纵"间隔长度"的随机对照试验：没有找到。** 最接近的两项是 Li 等 (2021)（3 天 vs 7 天的练习节奏引导，200 名大一新生，两组实际练习分布未被完全控制）与 YeckehZaare & Resnick (2025)（随机化的是激励方式，不是间隔长度）。检索过的关键词包括 spaced practice / distributed practice / interleaved practice + programming education / CS1 / novice programmers / computer science education / code，检索过 OpenAlex、Semantic Scholar、Crossref、ERIC 与网页搜索。
2. **把 Anki 或 SM-2 类算法用于编程学习、并与对照组比较学习结果的随机试验：没有找到。** 找到的是工具设计与回归分析（YeckehZaare 等 2019）。另有两条线索未能核实，不列入结论：竞赛编程平台的 FSRS 调度 pilot（Wang, Lim, Yu & Tan 2026，摘要未给样本量与效应量）；以及一份报告称 SM-2 驱动的自适应间隔系统在某代数课程中与无练习组无显著差异（Cao & Carvalho，OSF 预印本，未能检索到该文）。
3. **编程/算法题上纯粹的"交错 vs 分块"对照实验：没有找到。** 最接近的 Jacinto, Medeiros & de Sousa (2024), FIE 把交错学习与间隔复习捆绑成一个处理条件，与"集中学习 + 集中复习"对照，无法分离交错效应；Ghosh 等 (2025), SIGCSE TS 比较的是测验形式（多样化 vs 简单填空），两组都使用交错。交错的纯粹对照实验集中在数学、绘画、运动技能，以及两项得出无差异结论的软件操作技能实验（van der Meij 等）。
4. **数学推导类内容（多步证明、实分析、抽象代数）的间隔重复实验：没有找到。** 现有数学研究集中在算术、代数入门、组合计数、统计入门、微积分课程。
5. **"最佳间隔比例"在技能材料上的专门研究：没有找到。** 第 2 节的 5%–40% 比例全部来自言语材料。
6. **Cepeda 等 (2006) 的单一效应量：不存在。** 该文没有估计间隔效应本身的平均效应量（Latimier 等 2021 明确指出）。文献与网络上流传的"间隔效应 d = 0.42 / d = 0.85"等具体数字，我未能核实其出自该文。
7. **Moulton 等 (2006) 的原始全文：未获取。** 第 4.3 节的细节来自一篇评论文章的转述。
8. **Latimier 等 (2021) 发表版与预印本的差异：未核实。** 本文数字取自预印本全文（PsyArXiv kzy7u）。Murray 等 (2025) 引用该文为 g = 1.01，与预印本校正前的 1.02 一致，推测发表版未改动主要结果，但这一点未逐项核对。
9. **Rawson & Dunlosky (2012) 的"3–4 次后续会话"排期：只见到摘要级描述。** 该文出处已核实为 *Educational Psychology Review* 24(3), 419–435，但全文未获取，其具体排期的实验细节未核对。
10. **间隔重复对"有经验的开发者"的效果：没有找到任何研究。** 所有编程证据都来自入门课程的学生。

---

## 9. 复习排期表

### 9.1 使用前提

- **单元（unit）**定义为一个可以独立练习的最小技能或知识点。编程的例子：二分查找的边界处理、装饰器、SQL 的 JOIN 类型、某类动态规划的状态定义。数学的例子：分部积分、特征值计算、某类极限的处理方法。
- 目标保持期假定为**数月至数年**（希望长期可用）。短期应试变体见 9.5。
- 三条硬规则：**分散到不同的日子**、**每次都要提取**（重看不计入）、**每次都能核对答案**。

### 9.2 主表：一个单元学完后的复习排期

| 序号 | 第几天 | 做什么（形式） | 时长 | 依据 |
|---|---|---|---|---|
| 0 | Day 0 | 首次学习。读完/看完讲解后，立刻合上材料写一遍或做一道题 | — | 学习阶段 |
| 1 | **Day 1** | **闭卷自测**：不看材料写出要点、流程或定义；然后重做 1 道核心题；核对答案 | 10–15 分钟 | Cepeda 等 2008（1 周保持期的最优间隔约 1–2 天）；Roediger & Karpicke 2006（测试优势在 2 天后出现） |
| 2 | **Day 3** | **自测 + 变式**：闭卷自测，再做 1 道变式题（换参数、换情境、反向提问） | 15 分钟 | Cepeda 等 2009 实验 1（10 天保持期下的最优间隔）；Brunmair & Richter 2019（辨别与迁移） |
| 3 | **Day 7** | **交错混合练习**：把本单元与 2–3 个已学单元混排成一张练习，题目不标注属于哪个单元 | 20–30 分钟 | Rohrer 等 2020（数学交错 d = 0.83）；Taylor & Rohrer 2010（问题—策略配对） |
| 4 | **Day 16** | **限时闭卷**：从空白开始，在规定时间内完成一个完整任务 | 20–40 分钟 | Karpicke & Roediger 2008（提取才是关键）；Soderstrom & Bjork 2015（表现≠学习） |
| 5 | **Day 35** | **综合应用**：把本单元放进一个更大的任务、项目或跨章节题目里 | 30–60 分钟 | Cepeda 等 2009 实验 2（6 个月保持期的最优间隔为 28 天；28 天比 0 天提升 151%） |
| 6 | **Day 85** | **跨单元综合 + 交错** | 30–60 分钟 | Cepeda 等 2009（间隔应为"月量级"）；Kerfoot 等 2006（6–11 个月后效应量最大） |
| 7 | **Day 180**（可选） | 快速自测，只记录还不熟的项，不重学已掌握的 | 10–15 分钟 | Kerfoot 等 2006 |

回看日序列：**1、3、7、16、35、85、180 天**。相邻间隔比值约 2–2.5 倍。由于扩展间隔与均匀间隔在元分析中没有差别（Latimier 等 2021, g = 0.032），这个序列不必严格遵守，改成 1、3、7、14、30、90、180 天同样有依据。

与另一条独立建议的交叉印证：Rawson & Dunlosky (2012) 给出的高效排期是"初次学习时练习提取直到目标信息被正确回忆一次，随后在 3–4 个会话中各自重新学到一次正确回忆"。主表的前五次复习（Day 1/3/7/16/35）在结构上与之一致（**证据强度：中，仅有摘要级描述**）。

### 9.3 各次复习的具体形式（分编程与数学）

**编程**

| 序号 | 形式 |
|---|---|
| Day 1 | 合上编辑器和笔记，用文字写出算法步骤或数据流；再从零写一遍核心函数并跑通 |
| Day 3 | 不看示例，写出同一功能的变体（换数据结构、换边界条件），用测试用例验证 |
| Day 7 | 混合练习：本单元题目与另外 2–3 个单元混排，不提示用哪个方法 |
| Day 16 | 限时闭卷：给定需求，在限定时段内从空白文件写出可运行代码 |
| Day 35 | 在真实项目或大作业里用到该技能，或做一次相关重构 |
| Day 85 | 参与跨主题任务，或给别人讲解并现场写出来 |

依据：Karpicke & Roediger 2008（提取）；Taylor & Rohrer 2010（策略配对）；Rohrer 等 2020（交错）；YeckehZaare 等 2019 与 2025（分散到多天、按天给分）。

注意：读代码、看视频、重看笔记练到的是**识别**能力，**生成**能力要靠自己写。复习位不要用它们填充（Dunlosky 等 2013 把重读评为低效用；Greving & Richter 2019 显示分散重读也没有收益）。

**数学**

| 序号 | 形式 |
|---|---|
| Day 1 | 合上笔记默写定义、定理的条件与结论；做 1 道基础题 |
| Day 3 | 做 1–2 道变式题（改变条件、反向提问、举反例） |
| Day 7 | 混合题型练习：不同章节的题混排，不提示该用哪个方法 |
| Day 16 | 限时闭卷完成一整套混合题 |
| Day 35 | 综合题或证明题，需要串联多个单元 |
| Day 85 | 跨章节综合，或给别人讲一遍推导 |

依据：Rohrer 等 2020（数学交错，61% vs 38%，d = 0.83）；Murray 等 2025（数学的间隔效应 g = 0.28，明显小于言语材料；数学上"测试 vs 重学"g = 0.18 且置信区间跨零，所以**数学复习必须落到实际做题，默述概念的形式在数学上证据最弱**）。

### 9.4 复习失败时怎么调整

| 自测正确率 | 处理 |
|---|---|
| 低于 50% | 停止自测，回到学习材料（看讲解、看示例）。依据：Rowland (2014) 的转述结论，无反馈且初始正确率 ≤50% 时测试效应不可靠 |
| 50%–80% | 把下一次复习提前（例如原定 Day 7 改为 Day 5），并增加变式题 |
| 高于 80% | 按表执行，或把下一次间隔适度延长 |

每次复习必须有可核对的答案或可运行的测试。依据：Yang 等 2021 原文，课堂测验中有纠正性反馈 g = 0.537，无反馈 g = 0.374（**强**）。

### 9.5 短期应试变体（目标是 1–2 周后的考试）

回看日：**Day 1、Day 2、Day 4、Day 7，考前 1 天**。依据：Cepeda 等 2008，1 周保持期的最优间隔是 1–2 天。形式以限时混合题为主。

### 9.6 执行载体：每周混合池

除单元自身的排期外，每周固定一天做一张混合练习：从最近 4 周学过的所有单元中随机取 5–8 个，混合排题，不标注来源单元。这一步同时满足交错（Rohrer 等 2020）与滚动间隔（Cepeda 等 2009 的累积复习建议）。

### 9.7 决定成败的执行细节

1. **把排期落到日历、提醒或软件上。** 依据：Maligaya (2026)，自愿的间隔安排在真实条件下崩溃（1,706 名学生中只有 5 人按预期间隔使用）。
2. **给自己一个按"练习天数"计量、而不是按"答题数"计量的指标。** 依据：YeckehZaare & Resnick (2025) 的两个 RCT，按天给分提高成绩，对低 GPA 学生尤其有效。
3. **每次复习必须有反馈。** 依据：Yang 等 2021。
4. **不要用重读填充复习位。** 依据：Dunlosky 等 2013；Greving & Richter 2019。
5. **一次复习涉及的单元数宁少勿滥。** 依据：Latimier 等 2021，暴露次数过多时扩展与均匀的差异才略微显现，说明单次复习的负载会影响效果。

---

## 10. 参考文献

方括号内为本文给出的证据强度评级。DOI 与卷期页均经 Crossref API 核对（2026-10 查询）。

- Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). Rethinking the use of tests: A meta-analysis of practice testing. *Review of Educational Research*, 87(3), 659–701. https://doi.org/10.3102/0034654316689306 【强】
- Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin*, 145(11), 1029–1052. https://doi.org/10.1037/bul0000209 【强】
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354 【强】
- Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: A temporal ridgeline of optimal retention. *Psychological Science*, 19(11), 1095–1102. https://doi.org/10.1111/j.1467-9280.2008.02209.x 【强】
- Cepeda, N. J., Coburn, N., Rohrer, D., Wixted, J. T., Mozer, M. C., & Pashler, H. (2009). Optimizing distributed practice: Theoretical analysis and practical implications. *Experimental Psychology*, 56(4), 236–246. https://doi.org/10.1027/1618-3169.56.4.236 【中】
- Czyż, S. H., Wójcik, A. M., & Solarská, P. (2024). The effect of contextual interference on transfer in motor learning — a systematic review and meta-analysis. *Frontiers in Psychology*, 15, 1377122. https://doi.org/10.3389/fpsyg.2024.1377122 【强】
- Czyż, S. H., Wójcik, A. M., Solarská, P., & Kiper, P. (2024). High contextual interference improves retention in motor learning: Systematic review and meta-analysis. *Scientific Reports*, 14, 14235. https://doi.org/10.1038/s41598-024-65753-3 【强】
- Donovan, J. J., & Radosevich, D. J. (1999). A meta-analytic review of the distribution of practice effect: Now you see it, now you don't. *Journal of Applied Psychology*, 84(5), 795–805. https://doi.org/10.1037/0021-9010.84.5.795 【强】
- Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest*, 14(1), 4–58. https://doi.org/10.1177/1529100612453266 【强】
- Ebbinghaus, H. (1885). *Über das Gedächtnis*. （历史溯源，未直接核实原文）【仅作历史引用】
- Gluckman, M., Vlach, H. A., & Sandhofer, C. M. (2014). Spacing simultaneously promotes multiple forms of learning in children's science curriculum. *Applied Cognitive Psychology*, 28(2), 266–273. https://doi.org/10.1002/acp.2997 【中】
- Greving, C. E., & Richter, T. (2019). Distributed learning in the classroom: Effects of rereading schedules depend on time of test. *Frontiers in Psychology*, 9, 2517. https://doi.org/10.3389/fpsyg.2018.02517 【中】
- Hartwig, M. K., & Rohrer, D. (2025). Students' perceptions of effective math learning strategies. *Behavioral Sciences*, 15(8), 1047. https://doi.org/10.3390/bs15081047 【中】
- Herman, G. L., Patel, K., Emeka, C., Zilles, C., & West, M. (2025). Frequent testing vs. second-chance testing: An exploration. *Proceedings of the 2025 ACM Conference on International Computing Education Research*, 354–366. https://doi.org/10.1145/3702652.3744210 【未独立核实，摘要未获取】
- Jacinto, A. S., de Medeiros, F. P. A., & de Sousa, M. P. (2024). An approach to spaced repetition methodology for teaching markup and scripting programming languages. *2024 IEEE Frontiers in Education Conference (FIE)*, 1–9. https://doi.org/10.1109/FIE61694.2024.10893444 【中，交错与间隔复习捆绑，无法分离】
- Karpicke, J. D. (2009). Metacognitive control and strategy selection: Deciding to practice retrieval during learning. *Journal of Experimental Psychology: General*, 138(4), 469–486. https://doi.org/10.1037/a0017341 【中】
- Karpicke, J. D., & Blunt, J. R. (2011). Retrieval practice produces more learning than elaborative studying with concept mapping. *Science*, 331(6018), 772–775. https://doi.org/10.1126/science.1199327 【中】
- Karpicke, J. D., & Roediger, H. L. (2008). The critical importance of retrieval for learning. *Science*, 319(5865), 966–968. https://doi.org/10.1126/science.1152408 【中】
- Kerfoot, B. P., DeWolf, W. C., Masser, B. A., Church, P. A., & Federman, D. D. (2006). Spaced education improves the retention of clinical knowledge by medical students: A randomised controlled trial. *Medical Education*, 41(1), 23–31. https://doi.org/10.1111/j.1365-2929.2006.02644.x 【中】
- Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review*, 33(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8（预印本 https://doi.org/10.31234/osf.io/kzy7u ，本文数字取自预印本全文）【强】
- Li, B., Ning, F., Zhang, L., Yang, B., & Zhang, L. (2021). Evaluation of a practice system supporting distributed practice for novice programming students. *Journal of Pacific Rim Psychology*, 15. https://doi.org/10.1177/18344909211008264 【中】
- Maligaya, R. J. (2026). *Spaced retrieval practice for learning IUPAC nomenclature*（硕士论文）. Queen's University Library, QSpace. 【弱到中】
- Mawson, R. D., & Kang, S. H. K. (2025). The distributed practice effect on classroom learning: A meta-analytic review of applied research. *Behavioral Sciences*, 15(6), 771. https://doi.org/10.3390/bs15060771 【强】
- Moraes, M. C., Lionelle, A., Ghosh, S., & Folkestad, J. E. (2023). Teach students to study using quizzes, study behavior visualization, and reflection: A case study in an introduction to programming course. *The 15th International Conference on Education Technology and Computers*, 409–415. https://doi.org/10.1145/3629296.3629362 【中】
- Moulton, C. A., Dubrowski, A., MacRae, H., Graham, B., Grober, E., & Reznick, R. (2006). Teaching surgical skills: What kind of practice makes perfect? A randomized, controlled trial. *Annals of Surgery*, 244(3), 400–409.（全文未获取，细节来自二手转述）【中】
- Murray, E., Horner, A. J., & Göbel, S. M. (2025). A meta-analytic review of the effectiveness of spacing and retrieval practice for mathematics learning. *Educational Psychology Review*, 37(3). https://doi.org/10.1007/s10648-025-10035-1 【强】
- Murray, E. (2025). *Spacing and task complexity in mathematics learning*（博士论文）. University of York. https://etheses.whiterose.ac.uk/id/eprint/38200/ 【中】
- Rawson, K. A., & Dunlosky, J. (2012). When is practice testing most effective for improving the durability and efficiency of student learning? *Educational Psychology Review*, 24(3), 419–435. https://doi.org/10.1007/s10648-012-9203-1 【中】
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x 【中】
- Rohrer, D., Dedrick, R. F., & Stershic, S. (2015). Interleaved practice improves mathematics learning. *Journal of Educational Psychology*, 107(3), 900–908. https://doi.org/10.1037/edu0000001 【中】
- Rohrer, D., Dedrick, R. F., Hartwig, M. K., & Cheung, C.-N. (2020). A randomized controlled trial of interleaved mathematics practice. *Journal of Educational Psychology*, 112(1), 40–52. https://doi.org/10.1037/edu0000367 【强】
- Rowland, C. A. (2014). The effect of testing versus restudy on retention: A meta-analytic review of the testing effect. *Psychological Bulletin*, 140(6), 1432–1463. https://doi.org/10.1037/a0037559 【强】
- Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1848–1858. https://doi.org/10.18653/v1/P16-1174 【中】
- Smith, C. D., & Scarf, D. (2017). Spacing repetitions over long timescales: A review and a reconsolidation explanation. *Frontiers in Psychology*, 8, 962. https://doi.org/10.3389/fpsyg.2017.00962 【弱到中】
- Smith, D. H., Emeka, C., Fowler, M., West, M., & Zilles, C. (2023). Investigating the effects of testing frequency on programming performance and students' behavior. *Proceedings of the 54th ACM Technical Symposium on Computer Science Education*, 757–763. https://doi.org/10.1145/3545945.3569821 【中，准实验】
- Soderstrom, N. C., & Bjork, R. A. (2015). Learning versus performance: An integrative review. *Perspectives on Psychological Science*, 10(2), 176–199. https://doi.org/10.1177/1745691615569000 【强】
- Tabibian, B., Upadhyay, U., De, A., Zarezade, A., Schölkopf, B., & Gomez-Rodriguez, M. (2019). Enhancing human learning via spaced repetition optimization. *PNAS*, 116(10), 3988–3993. https://doi.org/10.1073/pnas.1815156116 【中】
- Tate, E., & Naidu, S. (2026). Encouraging learning through repetition: Effects of multiple practice opportunities in a large intro programming course. *Proceedings of the 57th ACM Technical Symposium on Computer Science Education*, 1047–1053. https://doi.org/10.1145/3770762.3772638 【弱】
- Taylor, K., & Rohrer, D. (2010). The effects of interleaved practice. *Applied Cognitive Psychology*, 24(6), 837–848. https://doi.org/10.1002/acp.1598 【中】
- van der Meij, H., & Maseland, J. (2021). Practice schedules in a video-based software training arrangement. *Social Sciences & Humanities Open*, 3(1), 100133. https://doi.org/10.1016/j.ssaho.2021.100133 【中】
- van der Meij, H., & Nuketayeva, K. (2023). Effects of practice schedules in video tutorials for software training. *Computers & Education*, 199, 104786. https://doi.org/10.1016/j.compedu.2023.104786 【中】
- Yang, C., Luo, L., Vadillo, M. A., Yu, R., & Shanks, D. R. (2021). Testing (quizzing) boosts classroom learning: A systematic and meta-analytic review. *Psychological Bulletin*, 147(4), 399–435. https://doi.org/10.1037/bul0000309 【强】
- YeckehZaare, I., Resnick, P., & Ericson, B. (2019). A spaced, interleaved retrieval practice tool that is motivating and effective. *Proceedings of the 2019 ACM Conference on International Computing Education Research*, 71–79. https://doi.org/10.1145/3291279.3339411 【中】
- YeckehZaare, I., Aronoff, C., & Grot, G. (2022). Retrieval-based teaching incentivizes spacing and improves grades in computer science education. *Proceedings of the 53rd ACM Technical Symposium on Computer Science Education*, 892–898. https://doi.org/10.1145/3478431.3499408 【中，摘要未获取，未引用其具体数字】
- YeckehZaare, I., & Resnick, P. (2025). Counting days is a spacing incentive that unlocks the potential of low GPA students. *npj Science of Learning*, 10(1). https://doi.org/10.1038/s41539-025-00322-5 【中偏强】

---

## 附录 A：复习排期表（可直接执行）

**单元定义**：一个可独立练习的最小技能或知识点。
**目标保持期**：数月至数年。
**三条硬规则**：分散到不同的日子、每次都要提取、每次都能核对答案。

| 第几天 | 回看形式 | 编程的具体做法 | 数学的具体做法 | 依据（对应本文结论） |
|---|---|---|---|---|
| **Day 0** | 首次学习 | 看讲解/示例后立刻合上材料写一遍 | 看完推导后立刻独立做 1 题 | 学习阶段（§9.1） |
| **Day 1** | 闭卷自测 + 重做核心题 | 用文字写出算法步骤，再从零写一遍核心函数并跑通 | 默写定义与定理条件，做 1 道基础题 | §2.1 比例规则（Cepeda 2008：1 周保持期最优间隔 1–2 天）；§3.2 重读 vs 自测（Roediger & Karpicke 2006） |
| **Day 3** | 自测 + 变式题 | 不看示例写同功能的变体（换数据结构、换边界） | 做变式题（改条件、反向提问、举反例） | §2.1（Cepeda 2009 实验 1）；§5.2 交错与辨别（Brunmair & Richter 2019） |
| **Day 7** | 交错混合练习 | 本单元与 2–3 个旧单元混排，不提示用哪个方法 | 不同章节题目混排，不提示方法 | §4.2 / §5.2（Rohrer 等 2020，d = 0.83）；§4.5 问题—策略配对（Taylor & Rohrer 2010） |
| **Day 16** | 限时闭卷，从空白开始 | 给定需求，限定时间内写出可运行代码 | 限时完成一套混合题 | §3.1 提取优于重复学习（Karpicke & Roediger 2008）；§6.5 表现≠学习（Soderstrom & Bjork 2015） |
| **Day 35** | 综合应用 | 在真实项目里用到该技能 | 做需要串联多单元的综合题或证明 | §2.1（Cepeda 2009 实验 2：28 天为最优间隔，比 0 天高 151%） |
| **Day 85** | 跨单元综合 + 交错 | 跨主题任务，或讲解并现场写出来 | 跨章节综合，或给别人讲推导 | §2.1（Cepeda 2009 的"月量级"建议）；§4.2（Kerfoot 等 2006：6–11 个月后效应量最大） |
| **Day 180**（可选） | 快速自测，只记录薄弱项 | 同左 | 同左 | §4.2（Kerfoot 等 2006） |

补充机制：**每周一次混合池**，从最近 4 周学过的单元里随机取 5–8 个混排练习。

调整规则：正确率 < 50% 时回到学习材料；50%–80% 时把下次复习提前；> 80% 时按表执行或适度延长。

变化形式：把 1/3/7/16/35/85/180 改成 1/3/7/14/30/90/180 天同样有依据（Latimier 等 2021 显示扩展间隔与均匀间隔无显著差异）。

短期应试（目标 1–2 周后的考试）：回看日改为 **Day 1、2、4、7，考前 1 天**，形式以限时混合题为主（Cepeda 等 2008）。
