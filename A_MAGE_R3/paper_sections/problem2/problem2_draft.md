# 5 问题2：同题论文可量化文本特征关联分析与小样本质量预测模型

## 5.1 问题分析与建模目标

第二问的目标不是重新建立一套完整评分体系，而是在第一问封版智能评价系统的基础上，研究附件2同一赛题论文中哪些可量化文本特征能够解释论文质量差异。附件2共包含10篇同一赛题论文，在赛题背景相同的条件下进行比较，可以相对削弱题目差异带来的影响，更集中地考察论文结构、模型方法、结果解释和写作规范等因素。

由于样本量仅为 N=10，直接建立强预测模型存在明显小样本风险。因此，本问采用稳健标准化、Spearman相关分析、灰色关联度、PLS-VIP、Bootstrap稳定性、删除单样本敏感性分析以及成对排序辅助检验等方法，形成以“特征解释和稳定性审计”为主的分析框架。需要强调的是，本文使用的质量标签 `Q_i` 来自第一问封版智能评价系统，是弱监督质量标签，不是官方真实分数、专家分数或人工真值。

因此，第二问识别的是“在第一问封版评价系统下，同题样本弱监督质量标签对可量化文本特征的敏感关系”，而不是基于外部真实质量标签的因果发现或独立人工评审结论。该定位可以避免将第一问评价指标与第二问解释特征的同源关系误写为外部真值验证。

## 5.2 弱监督质量标签 Q_i 与特征体系

附件2每篇论文的质量标签记为 `Q_i`，其来源为第一问封版智能评价系统输出，`Q_source = Problem1 sealed evaluation system`，`label_note = weak-supervised label, not official ground truth`。由结果表可得，`Q_i` 的取值范围为 `35.892975 ~ 70.222043`。

由于 `Q_i` 由第一问封版的文本评价系统计算得到，第二问后续相关性、灰色关联和PLS-VIP分析均用于解释该封版系统下的质量差异来源，而不将 `Q_i` 解释为官方质量标签。换言之，本问回答的是弱监督评价系统内部的可审计敏感性问题。

本文提取的表层文本特征 `X` 包括五类：篇幅结构类、数学表达类、结果展示类、逻辑表达类和学术规范类。深层质量校正特征 `H` 包括 `task_coverage`、`data_credibility`、`method_fit`、`formula_explanation`、`result_closure` 和 `stacking_penalty`。其中前五个指标为正向特征，`stacking_penalty` 为负向风险特征。

在章节识别过程中，若某些章节只能作为候选段落识别，则按 0.5 的低置信权重参与特征计算，不将其提升为确定章节。Step12B 审计显示，`reference_norm_rate` 为常数列，后续从主相关排序和PLS建模中剔除；`appendix_code_presence` 为低方差特征，在主PLS模型中剔除或谨慎处理。

## 5.3 稳健标准化与相关性分析

为降低小样本和极端值对特征尺度的影响，本文采用基于中位数和MAD的稳健标准化：

$$z_{ij}=\frac{x_{ij}-\operatorname{median}(x_j)}{\operatorname{MAD}(x_j)+\varepsilon},$$

其中，$\operatorname{MAD}(x_j)=\operatorname{median}(|x_{ij}-\operatorname{median}(x_j)|)$，$\varepsilon$ 为防止分母为零的极小正数。由于样本量仅为10，本文主要采用Spearman秩相关衡量特征与弱监督质量标签之间的单变量关系，不强调显著性检验。

Spearman绝对值排名前8的特征为：method_fit=0.612121、citation_norm_rate=0.589077、section_coverage=0.553912、task_coverage=0.454545、figure_table_explanation_rate=0.443813、objective_constraint_completeness=0.443046、total_chars=0.442424、paragraph_balance=-0.345455。其中 `method_fit`、`section_coverage`、`task_coverage`、`figure_table_explanation_rate` 和 `objective_constraint_completeness` 与质量标签关系较为明显。`total_chars` 只能解释为信息承载量和结构完整性相关，不能解释为篇幅越长越好。`stacking_penalty` 与 `Q_i` 的Spearman相关为 `-0.018182`，方向符合负向风险特征设定，但强度很弱，应谨慎表述。

## 5.4 灰色关联度与初步关键性指数

为从序列贴近程度角度补充相关性分析，本文以 `Q_i` 序列为参考序列，各文本特征为比较序列，计算灰色关联度。设第 $j$ 个特征在第 $i$ 篇论文上的标准化序列为 $x_j(i)$，参考序列为 $x_0(i)$，则灰色关联系数可写为：

$$\xi_j(i)=\frac{\Delta_{\min}+\rho\Delta_{\max}}{|x_0(i)-x_j(i)|+\rho\Delta_{\max}},$$

其中 $\rho=0.5$，灰色关联度 $G_j$ 为各样本关联系数的均值。进一步构造初步关键性指数：

$$K_{pre,j}=0.6|r^S_j|+0.4G_j,$$

灰色关联度Top 8为：total_chars、page_count、objective_constraint_completeness、figure_table_numbering_rate、figure_table_explanation_rate、variable_definition_coverage、cross_reference_rate、logic_connective_density。K_pre Top 8为：total_chars、page_count、objective_constraint_completeness、figure_table_explanation_rate、method_fit、figure_table_numbering_rate、section_coverage、conclusion_echo_rate。其中 `K_pre` 仅用于初步筛选，不作为最终关键特征结论。

## 5.5 PLS质量预测模型与VIP关键特征识别

为在多特征共线性条件下进一步识别关键特征，本文采用偏最小二乘回归（PLS）进行小样本建模，并用VIP指标衡量特征在潜变量中的贡献。候选潜变量数取 $A\in\{1,2\}$，通过留一交叉验证（LOOCV）选择模型复杂度。

需要说明的是，PLS使用的是Step14形成的稳健标准化特征矩阵，预处理并未在每一折内部重新估计。因此该LOOCV结果应视为近似留一审计，用于诊断小样本预测稳定性和筛选VIP解释特征，不能写作严格的外部泛化能力验证。

结果显示，最终选择 `A=1`。PLS-LOOCV结果为：MAE=9.196813，RMSE=11.176952，R2_LOO=-0.344322，Spearman=0.393939。其中R2_LOO为负，说明PLS不适合作为强预测模型。本文主要使用PLS的VIP结果识别关键解释特征，而不是夸大其预测性能。

VIP Top 8为：page_count=1.616451、method_fit=1.550686、total_chars=1.475844、task_coverage=1.454779、section_coverage=1.397529、objective_constraint_completeness=1.243901、figure_table_explanation_rate=1.161569、abstract_ratio=1.045178。VIP大于1的特征包括：page_count, method_fit, total_chars, task_coverage, section_coverage, objective_constraint_completeness, figure_table_explanation_rate, abstract_ratio, citation_norm_rate。其中 `page_count` 和 `total_chars` 只表示信息承载量和完整性相关。预测误差最大的样本为 `2-1`，其绝对误差为 `23.530734`。

## 5.6 质量调整因子 QAF

考虑到PLS可能受表层统计特征影响，本文构造质量调整因子（QAF）进行保守校正。首先定义深层质量校正得分：

$$u_i=\operatorname{mean}(task\_coverage_i,data\_credibility_i,method\_fit_i,formula\_explanation_i,result\_closure_i)-stacking\_penalty_i.$$

将其中心化为：

$$u_i^c=u_i-\overline{u}.$$

质量调整因子为：

$$\phi_i=\operatorname{clip}(1+\eta u_i^c,0.90,1.10),$$

最终预测为：

$$\hat Q^{QAF}_i=\operatorname{clip}(\phi_i\tilde Q_i,Q_{\min},Q_{\max}).$$

结果表明，最优 `eta=0.05`，`phi_i` 范围为 `0.973304 ~ 1.018579`。PLS基线MAE=9.196813，QAF后MAE=9.122450；PLS基线RMSE=11.176952，QAF后RMSE=11.049784；PLS基线R2=-0.344322，QAF后R2=-0.313906；PLS基线Spearman=0.393939，QAF后Spearman=0.406061。

QAF只带来小幅改善，应写作保守校正，不能写作显著提升。样本 `2-1` 的误差未被改善；样本 `2-10` 下调后误差改善。`2-1`、`2-2`、`2-10` 因 `stacking_penalty` 较高受到约束，但局部样本效果并不完全一致。

## 5.7 小样本稳定性分析

为检验关键特征结论在小样本下的稳定性，本文进行Bootstrap VIP稳定性分析。Bootstrap次数为 `B=1000`，有效次数为 `1000/1000`。稳定特征包括：

- `method_fit`：mean_VIP=1.334101，P(VIP>1)=0.883，符号一致率=0.991
- `task_coverage`：mean_VIP=1.254036，P(VIP>1)=0.797，符号一致率=0.992
- `section_coverage`：mean_VIP=1.208467，P(VIP>1)=0.779，符号一致率=0.967
- `page_count`：mean_VIP=1.189910，P(VIP>1)=0.703，符号一致率=0.845
- `total_chars`：mean_VIP=1.150335，P(VIP>1)=0.683，符号一致率=0.838

综合Spearman、灰色关联度、Bootstrap平均VIP和符号一致率，构造最终关键性指数：

$$K_j=0.35|r^S_j|+0.25G_j+0.25VIP^{norm}_j+0.15SC_j,$$

其中 $SC_j$ 表示Bootstrap系数符号一致率。K_final Top 8为：

- `total_chars`：K_final=0.727882
- `method_fit`：K_final=0.698960
- `page_count`：K_final=0.677210
- `section_coverage`：K_final=0.627357
- `objective_constraint_completeness`：K_final=0.626433
- `task_coverage`：K_final=0.619769
- `figure_table_explanation_rate`：K_final=0.613688
- `conclusion_echo_rate`：K_final=0.491281

最终关键特征为：total_chars, method_fit, page_count, section_coverage, objective_constraint_completeness, task_coverage, figure_table_explanation_rate, conclusion_echo_rate。删除单样本敏感性分析识别出的高影响论文包括：2-1, 2-3, 2-4, 2-6, 2-8, 2-10。特别地，删除 `2-1` 后，RMSE变化为 `-3.194589`，Spearman变化为 `-0.677273`，Top5特征Jaccard为 `0.429`，说明 `2-1` 是高影响样本。总体而言，小样本稳定性可审计但有限。

## 5.8 成对排序辅助检验

附件2共有10篇论文，可构造 $C(10,2)=45$ 个论文对。需要强调的是，这45对并非45个独立监督样本，不能作为独立样本验证模型性能。本文仅使用 `final_key_feature + K_final` 固定加权差分进行排序辅助检验，不进行复杂模型训练。

该检验中的相对优劣方向由 `Q_i` 的差值确定，而非外部人工排序或官方排序，因此其作用是检查最终关键特征能否较好复现弱监督标签诱导的相对排序，而不是提供独立外部验证。

此外，Step14稳健标准化日志显示 `figure_table_explanation_rate` 等特征存在 `MAD=0` 情况，采用 $\varepsilon$ 兜底缩放后，在成对差分加权中可能放大局部贡献量级。因此，Step19结果只作为排序一致性辅助审计和复现性风险提示，不作为支撑关键特征结论的主要证据。

结果显示，成对样本数为 `45`，near_tie数量为 `3`，总体pairwise accuracy为 `0.622222`，留一论文组平均accuracy为 `0.622222`，accuracy标准差为 `0.140546`，去除near_tie后accuracy为 `0.595238`，`2-1` 留一组accuracy为 `0.888889`。

该结果仅说明最终关键特征与弱监督标签诱导排序之间存在一定一致性参考，但受小样本、弱监督标签、成对样本非独立性以及MAD=0特征尺度放大的共同限制，不能写作独立验证或中等强度支撑证据。

## 5.9 第二问结果总结

综合相关性分析、灰色关联度、PLS-VIP、Bootstrap稳定性和删除单样本敏感性分析，第二问最终识别出的关键特征为：total_chars, method_fit, page_count, section_coverage, objective_constraint_completeness, task_coverage, figure_table_explanation_rate, conclusion_echo_rate。成对排序仅作为上述结论的辅助一致性审计，不作为主证据。

其中，`total_chars` 和 `page_count` 仅表示信息承载量和结构完整性相关，并不意味着篇幅越长越好；`method_fit`、`task_coverage`、`section_coverage`、`objective_constraint_completeness`、`figure_table_explanation_rate` 和 `conclusion_echo_rate` 更能体现论文的实质质量。由此可见，影响数学建模论文质量的关键因素不是单纯篇幅、公式数量或图表数量，而是方法是否贴题、任务是否覆盖、结构是否完整、目标函数与约束是否清晰、图表是否被解释以及结论是否形成闭环。

本问的局限性也需要明确说明：`Q_i` 是弱监督标签而非官方真实分数；附件2样本量仅为10；PLS/QAF预测能力有限且R2为负；高影响样本较多；成对排序只作为辅助检验。若后续获得官方标签或专家评分，可进一步验证关键特征的稳健性。

同时，由于 `Q_i` 和部分解释特征均来源于文本量化过程，结论存在同源解释风险。本文通过候选章节低置信处理、低方差特征剔除、Bootstrap稳定性、删除单样本敏感性和成对排序辅助审计降低该风险，但仍应将第二问表述为弱监督系统下的关键特征解释，而非独立真实质量预测。
