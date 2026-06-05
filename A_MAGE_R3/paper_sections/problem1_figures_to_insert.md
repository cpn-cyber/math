# 问题1建议插入图表清单

以下图均已由前序步骤生成，建议在论文正文中按“图X”占位编号，最终排版时统一调整编号。

| 图号占位 | 建议图题 | 来源文件 | 建议位置 | 建议图注 |
|---|---|---|---|---|
| 图X | AHP-熵权组合权重分布图 | `output/charts/appendix1_weights_ahp_entropy.png` | 4.3 | 该图展示21个二级指标的组合权重大小，可用于说明结果分析、灵敏度分析、附录代码和变量定义等指标在评价体系中的相对重要性。 |
| 图X | TOPSIS基础得分分布图 | `output/charts/topsis_score_distribution.png` | 4.4 | 该图展示30篇论文的TOPSIS基础分分布，反映样本内部质量差异。 |
| 图X | TOPSIS基础评分排名柱状图 | `output/charts/topsis_ranking_bar.png` | 4.4 | 该图按 `S_base` 降序展示论文基础排序，用于说明TOPSIS只是相对贴近度评分，不是传统百分制考试分。 |
| 图X | BT得分与TOPSIS基础分散点图 | `output/charts/bt_vs_topsis_scatter.png` | 4.5 | 该图展示BT校准分与TOPSIS基础分的关系，用于说明规则化rubric成对比较提供了不同于单纯TOPSIS贴近度的内部复核信息。 |
| 图X | 推荐融合方案排名变化图 | `output/charts/rank_change_v2.png` | 4.5 | 该图展示采用 `S_BT_scaled + lambda=0.85` 后各论文排名变化，最大排名变化为7，说明BT只进行温和校准。 |
| 图X | 最终融合分排名柱状图 | `output/charts/final_ranking_bar.png` | 4.7 | 该图展示30篇论文最终 `S_rank_v2` 排名及五级等级颜色，是问题1最终结果的核心图。 |
| 图X | 五级质量等级分布图 | `output/charts/grade_distribution.png` | 4.6 | 该图展示优秀2篇、良好6篇、中等10篇、及格9篇、不及格3篇的等级分布。 |
| 图X | 优秀-中等-不及格代表论文一级指标雷达图 | `output/charts/high_mid_low_radar.png` | 4.7 | 该图比较不同等级代表论文在A1结构、A2逻辑、A3建模、A4结果和A5写作五个一级指标上的差异。 |
| 图X | 最终融合分直方图 | `output/charts/final_score_histogram.png` | 4.6或4.7 | 该图展示最终融合分的总体分布，可与五级分级结果配合说明分数区间。 |
| 图X | KMeans五级分级散点图 | `output/charts/kmeans_grade_scatter.png` | 4.6 | 该图按最终排名和 `S_rank_v2` 展示KMeans五级分级结果，说明分级边界的样本分布。 |
| 图X | KMeans与Jenks等级对比图 | `output/charts/kmeans_jenks_comparison.png` | 4.6 | 该图展示KMeans与Jenks分级结果完全一致，对应一致率为1.000000，用于说明一维最终分数的自然断点与聚类边界一致。 |
| 图X | 多智能体可审计论文评估框架图 | `output/charts/multi_agent_framework.png` | 4.8 | 该图展示结构规范、逻辑一致、数学建模、结果验证、写作应用五类Agent如何向综合仲裁Agent提供可追溯证据。 |
| 图X | 权重扰动下的排名稳定性图 | `output/charts/weight_perturbation_rank_stability.png` | 4.8 | 该图展示300次权重扰动中各论文排名标准差，用于说明最终排序对权重小幅变化的敏感程度。 |
| 图X | Bootstrap理想解扰动排名稳定性图 | `output/charts/bootstrap_rank_stability.png` | 4.8 | 该图展示Bootstrap改变TOPSIS理想解参考集合后各论文排名标准差，用于说明理想解选取的稳健性。 |
| 图X | 最终等级置信度与边界风险图 | `output/charts/boundary_grade_confidence.png` | 4.8 | 该图展示每篇论文的等级置信度，并标记边界等级样本，重点用于解释02.txt和07.txt。 |

## 正文图优先级建议

若正文篇幅有限，建议优先保留以下5张图：

1. AHP-熵权组合权重分布图；
2. TOPSIS基础评分排名柱状图；
3. 推荐融合方案排名变化图；
4. 最终融合分排名柱状图；
5. 优秀-中等-不及格代表论文一级指标雷达图；
6. 多智能体可审计论文评估框架图；
7. 最终等级置信度与边界风险图。

其余图可放入附录或作为论文支撑材料。
