# 第二问关键结果汇总

本文件只汇总 Step 12-Step 19 已生成的真实结果，不重新计算模型。

## Q_label
- **range**：35.892975 ~ 70.222043

## Step14
- **Spearman Top 8**：method_fit:0.612121; citation_norm_rate:0.589077; section_coverage:0.553912; task_coverage:0.454545; figure_table_explanation_rate:0.443813; objective_constraint_completeness:0.443046; total_chars:0.442424; paragraph_balance:-0.345455

## Step15
- **K_pre Top 8**：total_chars:0.665455; page_count:0.560188; objective_constraint_completeness:0.554763; figure_table_explanation_rate:0.534650; method_fit:0.504981; figure_table_numbering_rate:0.464433; section_coverage:0.451457; conclusion_echo_rate:0.412927

## Step16
- **PLS metrics**：A=1, MAE=9.196813, RMSE=11.176952, R2_LOO=-0.344322, Spearman=0.393939
- **VIP Top 8**：page_count:1.616451; method_fit:1.550686; total_chars:1.475844; task_coverage:1.454779; section_coverage:1.397529; objective_constraint_completeness:1.243901; figure_table_explanation_rate:1.161569; abstract_ratio:1.045178

## Step17
- **QAF metrics**：eta=0.05, MAE=9.122450, RMSE=11.049784, R2=-0.313906, Spearman=0.406061

## Step18
- **K_final Top 8**：total_chars:0.727882; method_fit:0.698960; page_count:0.677210; section_coverage:0.627357; objective_constraint_completeness:0.626433; task_coverage:0.619769; figure_table_explanation_rate:0.613688; conclusion_echo_rate:0.491281
- **final_key_feature**：total_chars, method_fit, page_count, section_coverage, objective_constraint_completeness, task_coverage, figure_table_explanation_rate, conclusion_echo_rate

## Step19
- **pairwise accuracy**：total_pairs=45, near_tie=3, overall=0.622222, group_mean=0.622222, no_near_tie=0.595238

## Conclusion
- **writeable conclusion**：Key features have moderate auxiliary ranking support; PLS/QAF are limited and auditable, not strong prediction models.

## 最终可写结论
- 第二问以第一问封版智能评估系统输出的 `Q_label` 作为弱监督质量标签，而非官方真实质量分。
- `method_fit`、`section_coverage`、`task_coverage`、`objective_constraint_completeness`、`figure_table_explanation_rate` 等指标对质量差异具有较稳定解释力。
- `total_chars` 与 `page_count` 只能解释为信息承载量和完整性相关，不代表篇幅越长越好。
- PLS 和 QAF 的预测能力有限，应作为关键特征解释、稳健性审计和保守校正工具，而非强预测模型。
- 成对排序辅助检验为中等支持，不能写成强验证。
