# 第二问论文写作风险提醒

以下提醒必须在论文写作时遵守，避免把弱监督小样本模型写得过强。

1. Q_i is a weak-supervised label from the Problem 1 sealed intelligent evaluation system, not official ground truth.
2. PLS/QAF prediction ability is limited; do not describe them as strong prediction models.
3. R2_LOO is negative and must be reported honestly.
4. QAF is a conservative calibration with small improvement, not a significant performance boost.
5. Pairwise ranking is auxiliary validation only; 45 pairs are not 45 independent samples.
6. total_chars and page_count represent information-carrying amount/completeness, not longer-is-better.
7. There are multiple high-influence samples, so conclusions should be conservative.
8. Small-sample stability is auditable but limited.
