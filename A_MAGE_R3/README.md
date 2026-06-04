# A_MAGE_R3

中青杯数学建模竞赛 A 题问题 1 项目骨架。

当前版本只完成 Step 1：项目结构、基础配置文件、模块骨架和脚本骨架。
暂未实现 PDF 解析、特征提取、权重计算、TOPSIS、Bradley-Terry 或分级模型。

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
python scripts/run_step3_split_sections.py
python scripts/run_step4_extract_features.py
python scripts/run_step5_weighting.py
python scripts/run_step6_topsis.py
python scripts/run_step7_bradley_terry.py
python scripts/run_step8_grade_visualize.py
```

## Step 说明

- Step 1：检查项目目录和基础文件是否完整。
- Step 2：解析 PDF，生成论文文本文件。
- Step 3：按摘要、问题分析、模型建立、结果分析等章节切分文本。
- Step 4：提取结构、逻辑、方法、公式、图表、参考文献等质量特征。
- Step 5：计算主观权重、客观权重和融合权重。
- Step 6：基于 TOPSIS 计算综合质量得分。
- Step 7：基于 Bradley-Terry 模型做论文两两偏好校准。
- Step 8：输出等级分类结果和可视化图表。

## 输出目录

- `output/tables/`：表格结果。
- `output/charts/`：图表结果。
- `output/logs/`：运行日志。
