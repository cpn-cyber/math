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
| 表X | Jenks自然断点一致性校验 | `output/tables/jenks_grade_details.xlsx` | `jenks_breaks`、`kmeans_jenks_comparison` | 4.6 | 展示Jenks断点，并说明KMeans与Jenks一致率为1.000000；该表用于内部一致性校验，不作为外部分级正确性的证明。 |
| 表X | 附件1最终排名前5与后5 | `output/tables/final_problem1_ranking.xlsx` | `final_ranking` | 4.7 | 展示最终 `S_rank_v2`、等级和TOPSIS/BT融合分。 |
| 表X | 问题1最终结果一致性审计 | `output/tables/problem1_final_audit.xlsx` | `summary`、`checks` | 4.7或附录 | 展示所有一致性检查均通过，结论为“问题1结果文件一致，可进入论文写作阶段”。 |
| 表X | 问题1稳健性补充审计 | `output/tables/problem1_robustness_audit.xlsx` | `weight_perturb_summary`、`bootstrap_summary`、`leave_one_indicator` | 4.8 | 展示权重扰动、指标删除和Bootstrap理想解扰动下的排名稳定性。 |
| 表X | 边界论文等级置信度 | `output/tables/problem1_robustness_audit.xlsx` | `boundary_confidence` | 4.8 | 展示每篇论文的等级置信度、最近等级边界和风险类型，重点解释02.txt与07.txt。 |
| 表X | 证据型解释摘要 | `output/tables/problem1_robustness_audit.xlsx` | `evidence_summary`、`representative_evidence` | 4.7或4.8 | 展示每篇论文的主要正向证据、扣分证据、一级指标强弱项和风险类型。 |
| 表X | 反循环论证审计表 | `output/tables/problem1_robustness_audit.xlsx` | `anti_circular_audit` | 4.8 | 说明BT来源、AHP一致性、KMeans/Jenks关系和OCR风险提示的修正表述。 |
| 表X | Claude评委意见修订审计汇总 | `output/tables/problem1_judge_revision_audit.xlsx` | `summary` | 4.8或附录 | 汇总OCR细节、BT独立性、AHP敏感性、二值指标剔除、绝对护栏和外部锚状态。 |
| 表X | BT独立性指数与边际贡献审计 | `output/tables/problem1_judge_revision_audit.xlsx` | `bt_independence` | 4.5或4.8 | 展示BT排序与TOPSIS排序的Spearman、Kendall tau、成对方向独立性指数和最终排名影响。 |
| 表X | AHP-熵权alpha敏感性分析 | `output/tables/problem1_judge_revision_audit.xlsx` | `alpha_sensitivity` | 4.3或4.8 | 展示 \(\alpha=0.4,0.5,0.6,0.7\) 下最终排名的稳定性。 |
| 表X | 稀疏二值指标剔除审计 | `output/tables/problem1_judge_revision_audit.xlsx` | `binary_indicator_ablation` | 4.3或4.8 | 专门回应I16、I04等稀疏二值指标权重偏高问题。 |
| 表X | 绝对护栏审计表 | `output/tables/problem1_judge_revision_audit.xlsx` | `absolute_guardrail_audit` | 4.6或附录 | 说明最终等级为批内相对等级，并标记18.txt等需复核风险。 |
| 表X | 外部锚复核一致性检查 | `output/tables/problem1_judge_revision_audit.xlsx`、`output/tables/external_anchor_blind_review_filled.xlsx` | `external_anchor_check`、`external_anchor_review_detail`、`blind_pairwise_review` | 4.5或附录 | 展示12对边界论文复核结果、与规则化winner的一致率75.00%，以及A02、A06、A10等不一致样本对。 |

## 正文表格优先级建议

若正文篇幅有限，建议优先保留以下6张表：

1. 问题1综合评价指标体系；
2. AHP-熵权组合权重Top 5；
3. TOPSIS基础评分前5与后5；
4. BT融合敏感性分析结果；
5. 最终五级质量分布；
6. 附件1最终排名前5与后5；
7. 问题1稳健性补充审计；
8. BT独立性指数与边际贡献审计；
9. 稀疏二值指标剔除审计。

其余表格可放入附录，作为结果可追溯性支撑。
