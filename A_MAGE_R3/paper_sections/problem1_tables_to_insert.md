# 问题1建议插入表格清单

以下表格均来自已生成结果文件，建议在论文正文中按“表X”占位编号，最终排版时统一调整编号。

| 表号占位 | 建议表题 | 来源文件 | 来源sheet | 建议位置 | 说明 |
|---|---|---|---|---|---|
| 表X | 附件1论文文本特征提取概况 | `output/tables/appendix1_features_raw.xlsx`、`output/tables/appendix1_features_normalized.xlsx` | 默认sheet | 4.1 | 展示30篇论文、21个指标、文本长度范围、主要零值/缺失指标。 |
| 表X | 问题1综合评价指标体系 | `output/tables/appendix1_features_normalized.xlsx` | 默认sheet | 4.2 | 列出5个一级指标与21个二级指标，可结合指标量化方法整理为正文表。 |
| 表X | AHP一致性检验结果 | `output/tables/appendix1_weights_ahp_entropy.xlsx` | `ahp_consistency` | 4.3 | 展示准则层和各一级指标内部判断矩阵的 `lambda_max`、`CI`、`CR` 和一致性结论。 |
| 表X | AHP-熵权组合权重Top 5 | `output/tables/appendix1_weights_ahp_entropy.xlsx` | `combined_weights` | 4.3 | 展示组合权重排名前5的二级指标：I16、I04、I14、I11、I05。 |
| 表X | 一级指标组合权重汇总 | `output/tables/appendix1_weights_ahp_entropy.xlsx` | `group_summary` | 4.3 | 展示A1至A5的AHP组权重、熵权组权重和组合组权重。 |
| 表X | TOPSIS基础评分前5与后5 | `output/tables/appendix1_topsis_scores.xlsx` | `topsis_scores` | 4.4 | 展示 `rank_base`、`paper_id`、`filename`、`S_base`。 |
| 表X | BT融合敏感性分析结果 | `output/tables/bt_lambda_sensitivity.xlsx` | `summary` | 4.5 | 比较原始 `S_BT` 与 `S_BT_scaled` 在不同 `lambda` 下的Spearman、最大排名变化、平均绝对排名变化。 |
| 表X | BT异常排名变化审计表 | `output/tables/bt_rank_change_audit.xlsx` | `large_rank_changes` | 4.5 | 展示Step 7B原始融合中 `|rank_change| >= 8` 的论文，用于解释为何采用缩放BT。 |
| 表X | 最终五级质量分布 | `output/tables/grade_distribution.xlsx` | `grade_distribution` | 4.6 | 展示优秀、良好、中等、及格、不及格的数量和分数区间。 |
| 表X | KMeans分级中心与分数区间 | `output/tables/kmeans_grade_details.xlsx` | `kmeans_grade_summary` | 4.6 | 展示五个等级的聚类中心、分数范围和论文数量。 |
| 表X | Jenks自然断点与稳健性验证 | `output/tables/jenks_grade_details.xlsx` | `jenks_breaks`、`kmeans_jenks_comparison` | 4.6 | 展示Jenks断点，并说明KMeans与Jenks一致率为1.000000。 |
| 表X | 附件1最终排名前5与后5 | `output/tables/final_problem1_ranking.xlsx` | `final_ranking` | 4.7 | 展示最终 `S_rank_v2`、等级和TOPSIS/BT融合分。 |
| 表X | 问题1最终结果一致性审计 | `output/tables/problem1_final_audit.xlsx` | `summary`、`checks` | 4.7或附录 | 展示所有一致性检查均通过，结论为“问题1结果文件一致，可进入论文写作阶段”。 |

## 正文表格优先级建议

若正文篇幅有限，建议优先保留以下6张表：

1. 问题1综合评价指标体系；
2. AHP-熵权组合权重Top 5；
3. TOPSIS基础评分前5与后5；
4. BT融合敏感性分析结果；
5. 最终五级质量分布；
6. 附件1最终排名前5与后5。

其余表格可放入附录，作为结果可追溯性支撑。

