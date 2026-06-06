# 第二问结果摘要

- Q_label范围：35.892975 ~ 70.222043
- Spearman Top 8：method_fit=0.612121、citation_norm_rate=0.589077、section_coverage=0.553912、task_coverage=0.454545、figure_table_explanation_rate=0.443813、objective_constraint_completeness=0.443046、total_chars=0.442424、paragraph_balance=-0.345455
- K_pre Top 8：total_chars=0.665455、page_count=0.560188、objective_constraint_completeness=0.554763、figure_table_explanation_rate=0.534650、method_fit=0.504981、figure_table_numbering_rate=0.464433、section_coverage=0.451457、conclusion_echo_rate=0.412927
- PLS：A=1，MAE=9.196813，RMSE=11.176952，R2_LOO=-0.344322，Spearman=0.393939
- VIP Top 8：page_count=1.616451、method_fit=1.550686、total_chars=1.475844、task_coverage=1.454779、section_coverage=1.397529、objective_constraint_completeness=1.243901、figure_table_explanation_rate=1.161569、abstract_ratio=1.045178
- QAF：eta=0.05，MAE=9.122450，RMSE=11.049784，R2=-0.313906，Spearman=0.406061
- K_final Top 8：total_chars:0.727882; method_fit:0.698960; page_count:0.677210; section_coverage:0.627357; objective_constraint_completeness:0.626433; task_coverage:0.619769; figure_table_explanation_rate:0.613688; conclusion_echo_rate:0.491281
- final_key_feature：total_chars, method_fit, page_count, section_coverage, objective_constraint_completeness, task_coverage, figure_table_explanation_rate, conclusion_echo_rate
- 成对排序：total_pairs=45，near_tie=3，overall_accuracy=0.622222，group_mean=0.622222，accuracy_without_near_tie=0.595238

最终可写结论：第二问识别出的关键因素集中在方法匹配、任务覆盖、结构完整性、目标函数与约束完整性、图表解释和结论闭环等方面。篇幅类特征只表示信息承载量和完整性相关，不能解释为篇幅越长越好。由于 `Q_label` 是第一问封版评价系统产生的弱监督标签，第二问结论应写成该封版系统下的特征敏感性和可审计解释，不能写成外部真实质量因果发现。PLS/QAF预测能力有限，PLS的LOOCV应称为近似留一审计，成对排序也只是由 `Q_label` 诱导排序的一致性辅助检查。

