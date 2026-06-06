# 第二问论文写作风险提醒

以下提醒必须在论文写作时遵守，避免把弱监督小样本模型写得过强。

1. Q_i is a weak-supervised label from the Problem 1 sealed intelligent evaluation system, not official ground truth.
2. Because Q_i and several explanatory features are derived from text quantification, describe Problem 2 as sensitivity explanation under the sealed evaluation system, not external causal discovery.
3. PLS/QAF prediction ability is limited; do not describe them as strong prediction models.
4. R2_LOO is negative and must be reported honestly.
5. PLS LOOCV is an approximate leave-one audit based on the Step14 robust-scaled matrix, not strict external generalization validation.
6. QAF is a conservative calibration with small improvement, not a significant performance boost.
7. Pairwise ranking is auxiliary consistency checking only; 45 pairs are not 45 independent samples and their direction is induced by Q_i.
8. Step19 pairwise scores may be affected by MAD=0 robust-scaled features such as figure_table_explanation_rate, so do not call it moderate or strong support.
9. total_chars and page_count represent information-carrying amount/completeness, not longer-is-better.
10. There are multiple high-influence samples, so conclusions should be conservative.
11. Small-sample stability is auditable but limited.
