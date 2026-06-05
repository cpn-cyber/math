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

## 外部锚说明

`output/tables/external_anchor_blind_review_template.xlsx` 是给队员真实填写的盲评与专家AHP模板。当前已检测到 `output/tables/external_anchor_blind_review_filled.xlsx`，Step 11 输出显示12对边界论文复核全部有效，与规则化winner一致率为75.00%。专家AHP模板尚未作为真实专家矩阵进入权重模型，因此AHP部分仍按“预设重要性向量构造的一致偏好矩阵”表述。

## 输出目录

- `output/tables/`：表格结果。
- `output/charts/`：图表结果。
- `output/logs/`：运行日志。
