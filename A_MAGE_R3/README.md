# A_MAGE_R3

中青杯数学建模竞赛 A 题问题 1 项目。

当前版本已经完成 PDF解析、扫描件OCR兜底、章节切分、21项二级指标提取、AHP-熵权组合赋权、TOPSIS基础评分、规则化Bradley-Terry成对比较复核、五级分级、最终一致性审计、稳健性与可解释性补强，以及 Claude 评委意见专项修订审计。

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据放置

将附件 1 的 30 篇论文 PDF 放入：

```text
data/appendix1_papers/
```

普通 PDF 后续由 PyMuPDF 提取文本；扫描型 PDF 后续通过 OCR 模块处理。

将附件 2 的 10 篇同赛题论文 PDF 放入：

```text
data/appendix2_papers/
```

第二问将承接第一问封版评分体系，把第一问智能评估系统输出的综合质量分 \(Q_i\) 作为弱监督标签，用于分析可量化文本特征与论文质量之间的关系。

将附件 3 的 3 篇“中等质量论文” PDF 放入：

```text
data/appendix3_papers/
```

第三问将承接前两问封版系统，完成当前质量复评、逻辑断层诊断、AI辅助写作风险评估、多智能体评审主观性差异分析、修改动作优化、优化后质量预测和稳健性分析。

## 运行步骤

请在 `A_MAGE_R3` 目录下运行。

```bash
python scripts/run_step1_setup.py
python scripts/run_step2_parse_pdf.py
python scripts/run_step2b_ocr_parse_failed.py
python scripts/run_step3_split_sections.py
python scripts/run_step4_extract_features.py
python scripts/run_step5_weighting.py
python scripts/run_step6_topsis.py
python scripts/run_step7a_generate_pairwise_template.py
python scripts/run_step7a_fill_pairwise_surrogate.py
python scripts/run_step7b_pairwise_quality_check.py
python scripts/run_step7b_bradley_terry.py
python scripts/run_step7c_bt_audit_sensitivity.py
python scripts/run_step8_grade_visualize.py
python scripts/run_step8b_final_audit.py
python scripts/run_step10_problem1_enhance.py
python scripts/run_step11_problem1_judge_revision.py
```

第二问工程结构检查与后续预留步骤：

```bash
python scripts/run_step10_problem2_setup.py
python scripts/run_step11_appendix2_parse_sections.py
python scripts/run_step12_appendix2_features_labels.py
python scripts/run_step13_deep_quality_features.py
python scripts/run_step14_robust_correlation.py
python scripts/run_step15_grey_key_index.py
python scripts/run_step16_pls_vip_prediction.py
python scripts/run_step17_quality_adjustment.py
python scripts/run_step18_small_sample_validation.py
python scripts/run_step19_pairwise_ranking_check.py
python scripts/run_step20_problem2_final_audit.py
python scripts/run_step21_problem2_draft.py
```

第三问工程结构检查与后续预留步骤：

```bash
python scripts/run_step22_problem3_setup.py
python scripts/run_step23_appendix3_parse_sections.py
python scripts/run_step24_appendix3_current_eval.py
python scripts/run_step25_argument_chain_diagnosis.py
python scripts/run_step26_ai_risk_ds_fusion.py
python scripts/run_step27_reviewer_agents.py
python scripts/run_step28_revision_action_optimization.py
python scripts/run_step29_post_revision_prediction.py
python scripts/run_step30_problem3_robustness.py
python scripts/run_step31_problem3_final_audit.py
python scripts/run_step32_problem3_draft.py
```

也可以使用一键流程：

```bash
python scripts/run_all_problem1.py --dry-run
python scripts/run_all_problem1.py
```

## Step 说明

- Step 1：检查项目目录和基础文件是否完整。
- Step 2：解析 PDF，生成论文文本文件。
- Step 2B：对解析失败的扫描型 PDF 进行 OCR 兜底，重点处理 25.pdf。
- Step 3：按摘要、问题分析、模型建立、结果分析等章节切分文本。
- Step 4：提取结构、逻辑、方法、公式、图表、参考文献等质量特征。
- Step 5：计算主观权重、客观权重和融合权重。
- Step 6：基于 TOPSIS 计算综合质量得分。
- Step 7A：生成并规则化填写成对比较模板。
- Step 7B：运行 Bradley-Terry 模型。
- Step 7C：做 BT 结果审计和融合系数敏感性分析。
- Step 8：输出等级分类结果和可视化图表。
- Step 8B：问题1最终结果一致性审计。
- Step 10：补充稳健性、等级置信度、OCR质量代理和证据解释。
- Step 11：按 Claude 评委意见补 BT 独立性、AHP alpha敏感性、稀疏二值指标剔除、绝对护栏审计，并生成真人盲评/专家AHP外部锚模板。
- Step 10（问题2）：建立并检查第二问工程结构与配置。
- Step 11（问题2）：预留附件2 PDF解析、文本抽取和章节切分入口。
- Step 12（问题2）：预留附件2可量化特征与弱监督质量标签 \(Q_i\) 构造入口。
- Step 13（问题2）：预留任务覆盖率、数据可信度、方法匹配度等深层质量特征入口。
- Step 14（问题2）：预留稳健标准化与相关性分析入口。
- Step 15（问题2）：预留灰色关联和关键特征指数入口。
- Step 16（问题2）：预留PLS-VIP小样本预测模型入口。
- Step 17（问题2）：预留质量调整因子 \(\phi_i\) 入口。
- Step 18（问题2）：预留LOOCV、Bootstrap和删除单样本敏感性检验入口。
- Step 19（问题2）：预留45对成对排序稳定性检查入口。
- Step 20（问题2）：预留第二问最终一致性审计入口。
- Step 21（问题2）：预留第二问论文写作素材生成入口。
- Step 22（问题3）：建立并检查第三问工程结构与配置。
- Step 23（问题3）：预留附件3 PDF解析、OCR兜底和章节识别入口。
- Step 24（问题3）：预留基于前两问封版系统的当前质量复评入口。
- Step 25（问题3）：预留五元论证链和逻辑断层诊断入口。
- Step 26（问题3）：预留AI辅助写作风险D-S证据融合入口。
- Step 27（问题3）：预留多智能体评审主观性差异分析入口。
- Step 28（问题3）：预留修改动作库和多目标修复优化入口。
- Step 29（问题3）：预留优化后质量预测入口。
- Step 30（问题3）：预留Bootstrap和参数扰动稳健性分析入口。
- Step 31（问题3）：预留第三问最终一致性审计入口。
- Step 32（问题3）：预留第三问论文写作素材生成入口。

## 外部锚说明

`output/tables/external_anchor_blind_review_template.xlsx` 是给队员真实填写的盲评与专家AHP模板。当前已检测到 `output/tables/external_anchor_blind_review_filled.xlsx`，Step 11 输出显示12对边界论文复核全部有效，与规则化winner一致率为75.00%。专家AHP模板尚未作为真实专家矩阵进入权重模型，因此AHP部分仍按“预设重要性向量构造的一致偏好矩阵”表述。

## 输出目录

- `output/tables/`：表格结果。
- `output/charts/`：图表结果。
- `output/logs/`：运行日志。
- `output/problem2_tables/`：第二问表格结果。
- `output/problem2_charts/`：第二问图表结果。
- `output/problem2_logs/`：第二问运行日志。
- `paper_sections/problem2/`：第二问论文写作素材。
- `output/problem3_tables/`：第三问表格结果。
- `output/problem3_charts/`：第三问图表结果。
- `output/problem3_logs/`：第三问运行日志。
- `output/problem3_reports/`：第三问诊断和优化报告。
- `paper_sections/problem3/`：第三问论文写作素材。
