# 问题1最终结果一致性审计报告

审计结论：**问题1结果文件一致，可进入论文写作阶段**

## 审计范围

- final_problem1_ranking.xlsx
- grade_distribution.xlsx
- kmeans_grade_details.xlsx
- jenks_grade_details.xlsx
- appendix1_rank_fusion_v2.xlsx
- appendix1_topsis_scores.xlsx
- appendix1_weights_ahp_entropy.xlsx

## 关键结果

- 论文数量：30
- S_rank_v2 范围：23.616646 - 71.670038
- KMeans 与 Jenks 一致率：1.000000

## 五级分布

| 等级 | 数量 | 分数区间 |
|---|---:|---|
| 优秀 | 2 | 67.393314 - 71.670038 |
| 良好 | 6 | 52.534990 - 59.108593 |
| 中等 | 10 | 44.605967 - 49.764098 |
| 及格 | 9 | 35.870799 - 43.190744 |
| 不及格 | 3 | 23.616646 - 31.817805 |

## 最终排名前5

| 排名 | 论文 | S_rank_v2 | 等级 |
|---:|---|---:|---|
| 1 | 28.txt | 71.670038 | 优秀 |
| 2 | 25.txt | 67.393314 | 优秀 |
| 3 | 26.txt | 59.108593 | 良好 |
| 4 | 01.txt | 55.063840 | 良好 |
| 5 | 29.txt | 54.907091 | 良好 |

## 最终排名后5

| 排名 | 论文 | S_rank_v2 | 等级 |
|---:|---|---:|---|
| 26 | 22.txt | 36.254877 | 及格 |
| 27 | 04.txt | 35.870799 | 及格 |
| 28 | 19.txt | 31.817805 | 不及格 |
| 29 | 13.txt | 25.783640 | 不及格 |
| 30 | 12.txt | 23.616646 | 不及格 |

## 审计项

| 检查项 | 状态 | 说明 |
|---|---|---|
| final_problem1_ranking.xlsx contains 30 papers | PASS | actual_rows=30 |
| paper_id has no duplicates | PASS | duplicate_ids=[] |
| paper_id has no missing IDs against 01-30/fusion/TOPSIS references | PASS | missing=[]; extra=[] |
| S_rank_v2 matches appendix1_rank_fusion_v2.xlsx | PASS | inconsistent_paper_ids=[] |
| rank_final follows S_rank_v2 descending order | PASS | rank_mismatch_ids=[]; rank_set_ok=True |
| grade_distribution final counts sum to 30 | PASS | count_sum=30 |
| grade_distribution has five expected grade labels | PASS | grades=['优秀', '良好', '中等', '及格', '不及格'] |
| grade_final, grade_kmeans, and grade_jenks are identical in final ranking | PASS | inconsistent_ids=[] |
| KMeans/Jenks detail workbooks match final ranking | PASS | kmeans_detail_ok=True; jenks_detail_ok=True |
| KMeans and Jenks consistency rate equals 1.0 | PASS | computed_consistency=1.000000; detail_sheet_consistency=1.000000 |
| Final top 5 and bottom 5 match grade_visualization.log | PASS | top_ok=True; bottom_ok=True |
| All final chart files exist and are non-empty | PASS | missing_or_empty=[] |
| Weight table has 21 indicators and normalized combined weights | PASS | indicator_count=21; combined_weight_sum=1.000000000000; missing_columns=[] |
