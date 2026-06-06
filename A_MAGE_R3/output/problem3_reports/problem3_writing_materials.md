# 第三问论文写作素材

## 第三问整体方法流程

第三问承接第一问封版评分系统和第二问关键特征解释结果，对附件3三篇中等质量论文建立“当前复评—逻辑诊断—AI辅助写作风险提示—多智能体分歧分析—修改动作优化—优化后预测—稳健性检验”的闭环流程。该流程强调可解释、可审计和保守预测，不将风险提示指标等同于质量扣分。

## Step 26 AI 风险识别结果摘要

| paper_id | R_AI | risk_level | main_risk_source |
| --- | --- | --- | --- |
| 3-1 | 0.2381516335582713 | 低风险 | 数据不可追溯风险(0.688)、方法到结果跳跃风险(0.412) |
| 3-2 | 0.2294708656161272 | 低风险 | 数据不可追溯风险(0.666)、方法到结果跳跃风险(0.459) |
| 3-3 | 0.2725711516103803 | 低风险 | 数据不可追溯风险(0.623)、方法到结果跳跃风险(0.452) |

三篇论文 AI 辅助写作风险均处于低风险区间，主要风险来源集中在数据不可追溯风险和方法到结果跳跃风险。

## Step 27 多智能体分歧分析结果摘要

| paper_id | agent_score_std | disagreement_level | min_score_agent | max_score_agent |
| --- | --- | --- | --- | --- |
| 3-1 | 16.85800620104913 | 高分歧 | result_validation_reviewer | application_value_reviewer |
| 3-2 | 16.77380556065495 | 高分歧 | logic_reviewer | structure_reviewer |
| 3-3 | 11.50905384561031 | 高分歧 | result_validation_reviewer | structure_reviewer |

三篇论文均表现为高分歧，说明不同评审维度对同一论文的评价差异明显，后续修改应优先修复最低评分维度。

## Step 28 修改动作优化结果摘要

| paper_id | selected_actions_knapsack | expected_total_gain_knapsack | total_cost_knapsack | revision_priority_level |
| --- | --- | --- | --- | --- |
| 3-1 | A7,A10,A11 | 62.0182427517348 | 10 | 高优先级 |
| 3-2 | A11,A12,A14 | 65.28515662803656 | 10 | 高优先级 |
| 3-3 | A10,A11,A12 | 61.80249006637455 | 10 | 高优先级 |

动作优化在预算10内给出推荐组合，其中 A11、A10、A12 是最稳定的修改主干。

## Step 29 优化后质量预测结果摘要

| paper_id | current_score | score_gain | predicted_score_after_revision | current_level | predicted_level_after_revision | R_AI_before | R_AI_after_pred | agent_score_std_before | agent_score_std_after_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-1 | 28.56295053412095 | 7.442189130208176 | 36.00513966432913 | 待改进 | 待改进 | 0.2381516335582713 | 0.1781516335582713 | 16.85800620104913 | 10.11480372062948 |
| 3-2 | 63.34107920361653 | 7.834218795364387 | 71.17529799898092 | 合格 | 合格 | 0.2294708656161272 | 0.1544708656161272 | 16.77380556065495 | 10.06428333639297 |
| 3-3 | 53.59409217967413 | 7.416298807964946 | 61.01039098763907 | 待改进 | 合格 | 0.2725711516103803 | 0.1975711516103803 | 11.50905384561031 | 6.905432307366186 |

预测结果显示三篇论文均有正向提升，3-3 可由待改进提升至合格，3-1 提升后仍处于待改进，说明其需要更大幅度结构性重构。

## Step 30 稳健性分析结果摘要

| paper_id | pass_or_above_ratio | mean_score_gain | min_score_gain | max_score_gain |
| --- | --- | --- | --- | --- |
| 3-1 | 0.0 | 7.266201831090148 | 3.066925894577087 | 12 |
| 3-2 | 1.0 | 7.389927363682976 | 2.963822224073481 | 12 |
| 3-3 | 0.56 | 7.138148357108827 | 2.983590239882115 | 12 |

| action_id | selected_count | selection_rate |
| --- | --- | --- |
| A11 | 675 | 1.0 |
| A10 | 405 | 0.6 |
| A12 | 405 | 0.6 |
| A14 | 180 | 0.2666666666666667 |
| A4 | 180 | 0.2666666666666667 |
| A7 | 180 | 0.2666666666666667 |
| A2 | 45 | 0.06666666666666667 |

225组参数扰动下，预测提升稳定为正；A11在所有参数组合中均被选中，A10和A12为次稳定主干动作。

## 可直接放入论文的关键表格建议

- 表X：附件3 AI辅助写作风险识别结果，来源 `ai_risk_ds_fusion.xlsx`。
- 表X：多智能体评分分歧分析结果，来源 `multi_agent_subjectivity_analysis.xlsx`。
- 表X：修改动作优化推荐方案，来源 `revision_action_optimization.xlsx`。
- 表X：优化后质量预测结果，来源 `quality_prediction_after_revision.xlsx`。
- 表X：稳健性分析关键结果，来源 `robustness_summary.xlsx`。

## 可直接放入论文的图表建议

| figure_name | suggested_caption | suggested_section | paper_usage_priority |
| --- | --- | --- | --- |
| paper_disagreement_radar.png | Step27多智能体评分雷达图：展示各维度评分差异来源 | 正文或附录 | 中 |
| revision_gain_cost_scatter.png | Step28动作收益-成本散点图：展示动作收益和成本结构 | 附录 | 中 |
| risk_before_after.png | Step29风险提示指标前后对比图：展示AI辅助写作风险提示指标下降 | 正文或附录 | 中 |
| robustness_score_gain_sensitivity.png | Step30收益系数敏感性图：展示预测提升对收益系数扰动的响应 | 附录 | 中 |
| robustness_budget_sensitivity.png | Step30预算敏感性图：展示预算变化下平均预测得分和动作数量 | 正文或附录 | 中 |
| robustness_disagreement_sensitivity.png | Step30分歧收敛敏感性图：展示分歧标准差下降趋势 | 附录 | 中 |
| ai_risk_radar.png | Step26 AI风险证据雷达图：展示三篇论文四类AI辅助写作风险证据 | 正文 | 高 |
| ai_risk_bar.png | Step26 AI风险柱状图：展示三篇论文R_AI均为低风险 | 正文 | 高 |
| agent_disagreement_bar.png | Step27多智能体分歧柱状图：展示三篇论文均存在高分歧 | 正文 | 高 |
| revision_priority_bar.png | Step28修改优先级柱状图：展示预计收益和修改成本 | 正文 | 高 |
| score_before_after.png | Step29修改前后质量得分对比图：展示优化后预测质量提升 | 正文 | 高 |
| disagreement_before_after.png | Step29分歧前后对比图：展示多智能体分歧收敛 | 正文 | 高 |
| robustness_action_frequency.png | Step30动作稳定性频次图：展示A11、A10、A12等稳定推荐动作 | 正文 | 高 |

## 第三问主要结论

| conclusion | note | write_to_paper_recommendation |
| --- | --- | --- |
| 三篇论文 AI 辅助写作风险均为低风险 | risk_levels=['低风险', '低风险', '低风险'] | 建议写入正文 |
| 三篇论文主要风险来源集中在数据不可追溯和方法到结果跳跃 | 数据不可追溯风险(0.688)、方法到结果跳跃风险(0.412) 数据不可追溯风险(0.666)、方法到结果跳跃风险(0.459) 数据不可追溯风险(0.623)、方法到结果跳跃风险(0.452) | 建议写入正文 |
| 三篇论文多智能体评分均为高分歧 | levels=['高分歧', '高分歧', '高分歧'] | 建议写入正文 |
| A11 是最稳定修改动作 | top_actions=['A11', 'A10', 'A12'] | 建议写入正文 |
| A10、A12 是次稳定主干动作 | top_actions=['A11', 'A10', 'A12'] | 建议写入正文 |
| 3-1 即使优化后仍难达到合格 | 3-1 pass ratio=0.0 | 建议写入正文 |
| 3-2 优化后稳定保持合格 | 3-2 pass ratio=1.0 | 建议写入正文 |
| 3-3 有较大概率提升到合格 | 3-3 pass ratio=0.560 | 建议写入正文 |
| AI 风险提示指标在稳健性分析中均呈下降趋势 | all paper risk decrease proportions are 1.0 | 建议写入正文 |
| 多智能体分歧整体呈下降趋势 | mean std reduction > 0 for all papers | 建议写入正文 |

## 模型边界与免责声明

该指标仅表示文本存在 AI辅助写作风险，不构成学术不端判断，也不等同于AI生成判定。

第三问中的优化后质量预测和稳健性分析均为模型参数扰动下的预测稳定性检验，不代表真实人工修改后的必然结果，也不替代专家复评。
