# A_MAGE_R3 工作记录

本文档记录中青杯数学建模竞赛 A 题问题 1 工程部分目前已完成的工作，覆盖赛题材料阅读、项目结构搭建、PDF 文本解析、章节识别、二级指标特征提取，以及过程中遇到的问题和处理方式。

当前进度：已完成 Step 1 到 Step 4。尚未实现 AHP、熵权、CRITIC、TOPSIS、Bradley-Terry、最终分级或可视化评分。

## 一、项目背景

竞赛题目为 A 题：数学建模论文智能评估系统与多智能体优化方法。

问题 1 的核心任务是：对附件 1 中 30 篇 2025 年中青杯参赛论文进行质量特征分析，建立数学建模论文质量综合评价指标体系，并构建自动评分模型与质量分级结果。

当前工程项目 `A_MAGE_R3` 先按流水线分步实现：

1. Step 1：项目结构与基础配置。
2. Step 2：PDF 解析与文本提取。
3. Step 3：章节识别与结构切分。
4. Step 4：二级指标特征提取。
5. Step 5 之后：尚未实现，后续会进行权重计算、综合评价、分级与可视化。

## 二、原始材料说明

项目上层目录包含以下原始材料：

| 文件或目录 | 说明 | 当前处理情况 |
|---|---|---|
| `A题：数学建模论文智能评估系统与多智能体优化方法.pdf` | 赛题正文，定义问题 1、问题 2、问题 3 的任务要求 | 已阅读，问题 1 被拆解为“结构、逻辑、方法、结果、规范”等特征体系 |
| `2026年第八届中青杯全国大学生数学建模竞赛参赛细则.pdf` | 参赛格式、AI 工具使用声明、论文命名、参考文献格式等规则 | 已阅读，用于后续“写作规范”“参考文献规范”类指标依据 |
| `中青杯全国大学生数学建模竞赛诚信参赛告知书.pdf` | 诚信规范、AI 工具使用约束、违规行为说明 | 已阅读，提醒后续论文正文需要披露 AI 使用 |
| `2026年第八届中青杯全国大学生数学建模竞赛论文模板.doc` | 官方论文模板 | 已粗读，确认摘要页、正文格式、标题格式等要求 |
| `附件1/` | 问题 1 的 30 篇参赛论文 | 已复制到 `A_MAGE_R3/data/appendix1_papers/` 并完成 Step 2-4 |
| `附件2/` | 问题 2 的 10 篇同题论文 | 初步检查过，其中 `2-8.pdf` 也是扫描型 PDF；当前项目暂未处理附件 2 |
| `附件3/` | 问题 3 的 3 篇“中等”质量论文 | 初步检查过，当前项目暂未处理附件 3 |

特别注意：

- 附件 1 的 `25.pdf` 是扫描型 PDF，普通文本解析无法提取正文。
- 附件 2 的 `2-8.pdf` 也是扫描型 PDF。
- 在项目创建之前，曾用 PyMuPDF 渲染页面并用 OCR 检查过扫描件可读性；但当前 Step 2 按用户要求只实现 PyMuPDF 优先解析，没有把 OCR 正文并入项目流水线。

## 三、项目目录说明

项目目录为：

```text
A_MAGE_R3/
├── main_problem1.py
├── config.yaml
├── requirements.txt
├── README.md
├── WORK_LOG.md
├── data/
│   ├── appendix1_papers/
│   ├── extracted_text/
│   └── intermediate/
│       └── sections/
├── modules/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── section_splitter.py
│   ├── feature_extractor.py
│   ├── weighting.py
│   ├── topsis.py
│   ├── bradley_terry.py
│   ├── grade_classifier.py
│   └── visualization.py
├── output/
│   ├── tables/
│   ├── charts/
│   └── logs/
└── scripts/
    ├── run_step1_setup.py
    ├── run_step2_parse_pdf.py
    ├── run_step3_split_sections.py
    ├── run_step4_extract_features.py
    ├── run_step5_weighting.py
    ├── run_step6_topsis.py
    ├── run_step7_bradley_terry.py
    └── run_step8_grade_visualize.py
```

## 四、根目录文件说明

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `main_problem1.py` | 项目主入口，打印当前流水线步骤 | 已实现基础入口 |
| `config.yaml` | 全局配置文件，保存路径、PDF 解析参数、章节关键词、逻辑连接词、公式识别规则、特征提取关键词等 | 已持续补充到 Step 4 |
| `requirements.txt` | 依赖库清单 | 已写入 PyMuPDF、pandas、openpyxl、scikit-learn 等依赖 |
| `README.md` | 简要说明每一步如何运行 | 已创建基础说明 |
| `WORK_LOG.md` | 当前工作记录文档 | 本次新增 |

## 五、模块文件说明

| 模块 | 作用 | 当前实现情况 |
|---|---|---|
| `modules/pdf_parser.py` | Step 2 PDF 解析模块 | 已实现 PyMuPDF 文本提取、页码标记、空白页日志、解析报告 |
| `modules/section_splitter.py` | Step 3 章节识别模块 | 已实现规则型标题识别、章节切分、JSON 输出、章节报告 |
| `modules/feature_extractor.py` | Step 4 二级指标特征提取模块 | 已实现 21 个二级指标的 raw 和 normalized 表输出 |
| `modules/weighting.py` | Step 5 权重计算模块 | 目前仅保留 TODO 骨架，尚未实现 |
| `modules/topsis.py` | Step 6 TOPSIS 综合评价模块 | 目前仅保留 TODO 骨架，尚未实现 |
| `modules/bradley_terry.py` | Step 7 Bradley-Terry 成对偏好校准模块 | 目前仅保留 TODO 骨架，尚未实现 |
| `modules/grade_classifier.py` | Step 8 等级分类模块 | 目前仅保留 TODO 骨架，尚未实现 |
| `modules/visualization.py` | Step 8 可视化模块 | 目前仅保留 TODO 骨架，尚未实现 |

## 六、脚本文件说明

| 脚本 | 作用 | 当前状态 |
|---|---|---|
| `scripts/run_step1_setup.py` | 检查项目目录与基础文件是否完整 | 已实现，可独立运行 |
| `scripts/run_step2_parse_pdf.py` | 批量解析 `data/appendix1_papers/*.pdf` | 已实现，可独立运行 |
| `scripts/run_step3_split_sections.py` | 批量切分 `data/extracted_text/*.txt` | 已实现，可独立运行 |
| `scripts/run_step4_extract_features.py` | 批量提取 21 个二级指标 | 已实现，可独立运行 |
| `scripts/run_step5_weighting.py` | 权重计算入口 | 占位，尚未实现 |
| `scripts/run_step6_topsis.py` | TOPSIS 评分入口 | 占位，尚未实现 |
| `scripts/run_step7_bradley_terry.py` | Bradley-Terry 校准入口 | 占位，尚未实现 |
| `scripts/run_step8_grade_visualize.py` | 分级与可视化入口 | 占位，尚未实现 |

## 七、Step 1：项目结构与基础配置

### 已实现内容

Step 1 创建了完整项目结构：

- `data/appendix1_papers/`
- `data/extracted_text/`
- `data/intermediate/`
- `modules/`
- `scripts/`
- `output/tables/`
- `output/charts/`
- `output/logs/`

同时创建了：

- `main_problem1.py`
- `config.yaml`
- `requirements.txt`
- `README.md`
- 所有模块 `.py` 文件
- 所有 Step 脚本 `.py` 文件

### 自检方式

```powershell
cd C:\Users\13178\Desktop\A题\A_MAGE_R3
python scripts/run_step1_setup.py
```

自检结果：

```text
Step 1 setup check passed. Project structure is complete.
```

### 设计原则

- 只建立项目骨架，不实现模型。
- 每个模块先给出可扩展函数接口。
- 后续 Step 按脚本逐步推进，避免把解析、特征、评分混在一起。

## 八、Step 2：PDF 解析与文本提取

### 输入

```text
data/appendix1_papers/*.pdf
```

共 30 篇 PDF：

```text
01.pdf 到 30.pdf
```

### 输出

```text
data/extracted_text/01.txt 到 30.txt
output/tables/pdf_parse_report.xlsx
output/logs/pdf_parse.log
```

### 已实现功能

`modules/pdf_parser.py` 实现：

- `parse_pdf_to_text(pdf_path, output_txt_path)`
- `parse_all_pdfs(input_dir, output_dir, report_path, log_path)`
- PyMuPDF 优先提取文本。
- 每页保留页码标记：

```text
[PAGE 1]
[PAGE 2]
```

- 如果某页提取文本为空，写入日志。
- 单个 PDF 失败不会中断全部流程。
- 生成 Excel 报告。

### 报告字段

`pdf_parse_report.xlsx` 字段：

```text
文件名、页数、字数、是否解析成功、空白页数量、空白页页码、错误信息、输出文本路径
```

### 运行命令

```powershell
cd C:\Users\13178\Desktop\A题\A_MAGE_R3
python scripts/run_step2_parse_pdf.py
```

### 运行结果

- 解析 PDF 数量：30
- 成功打开并解析：30
- 解析失败：0
- 输出 TXT 数量：30

### 重要问题

`25.pdf` 是扫描型 PDF。

PyMuPDF 可以打开文件并识别页数，但提取不到正文文本：

- 页数：36
- 字数：0
- 空白页数量：36
- 空白页页码：1 到 36

处理方式：

- 不编造文本。
- `25.txt` 仅保留 `[PAGE n]` 页码标记。
- 在 `pdf_parse.log` 中逐页记录空白页。
- 后续如果进入 OCR Step，需要对 `25.pdf` 单独做图像渲染和 OCR。

### 依赖与环境问题

过程中发现：

- 默认 `python` 环境不一定安装 PyMuPDF。
- Codex 内置 Python 运行时有 PyMuPDF、pandas、openpyxl，但缺 PyYAML。

处理方式：

- 代码仍按 `requirements.txt` 正式依赖编写。
- `run_step2_parse_pdf.py` 对 PyYAML 缺失提供默认配置兜底，保证脚本可运行。
- 正式复现实验建议先执行：

```powershell
python -m pip install -r requirements.txt
```

## 九、Step 3：章节识别与结构切分

### 输入

```text
data/extracted_text/*.txt
output/tables/pdf_parse_report.xlsx
```

其中实际使用文本文件作为主输入，解析报告用于了解空白页和扫描件情况。

### 输出

```text
data/intermediate/sections/01.json 到 30.json
output/tables/section_split_report.xlsx
output/logs/section_split.log
```

### JSON 格式

每篇论文输出一个 JSON：

```json
{
  "paper_id": "01",
  "filename": "01.txt",
  "sections": {
    "abstract": "...",
    "problem_statement": "...",
    "assumptions": "...",
    "symbols": "...",
    "model_building": "...",
    "model_solving": "...",
    "results": "...",
    "sensitivity_analysis": "...",
    "error_analysis": "...",
    "model_evaluation": "...",
    "references": "...",
    "appendix": "..."
  },
  "missing_sections": [...]
}
```

### 核心章节

当前识别的核心章节包括：

- 摘要：`abstract`
- 问题重述：`problem_statement`
- 模型假设：`assumptions`
- 符号说明：`symbols`
- 模型建立：`model_building`
- 模型求解：`model_solving`
- 结果分析：`results`
- 灵敏度分析：`sensitivity_analysis`
- 误差分析：`error_analysis`
- 模型评价：`model_evaluation`
- 参考文献：`references`
- 附录：`appendix`

另有辅助章节：

- 关键词：`keywords`
- 问题分析：`problem_analysis`

辅助章节不进入报告核心字段，但在 Step 4 的特征提取中可用于补充语义判断。

### 已实现功能

`modules/section_splitter.py` 实现：

- `split_sections(text)`
- `split_all_texts(input_dir, output_dir, report_path, log_path)`
- 支持中文标题关键词。
- 支持常见编号格式：

```text
一、问题重述
1 问题重述
1.1 问题分析
（一）模型假设
模型的建立与求解
```

- 自动忽略目录行中的点线页码。
- 对识别失败或缺失章节写入日志。

### 报告字段

`section_split_report.xlsx` 字段：

```text
paper_id、filename、是否有摘要、是否有问题重述、是否有模型假设、是否有符号说明、是否有模型建立、是否有结果分析、是否有参考文献、缺失章节数量
```

### 运行命令

```powershell
cd C:\Users\13178\Desktop\A题\A_MAGE_R3
python scripts/run_step3_split_sections.py
```

### 运行结果

- 输入 TXT 数量：30
- 输出 JSON 数量：30
- 至少识别到一个核心章节的论文：29
- 章节识别成功率：96.67%

唯一完全无法识别章节的是：

```text
25.txt
```

原因是 Step 2 对扫描型 `25.pdf` 没有提取到正文。

### 最常缺失章节

从 `missing_sections` 统计：

```text
sensitivity_analysis：25 篇缺失
error_analysis：25 篇缺失
model_evaluation：17 篇缺失
results：12 篇缺失
appendix：10 篇缺失
symbols：8 篇缺失
assumptions：7 篇缺失
model_solving：6 篇缺失
references：5 篇缺失
problem_statement：4 篇缺失
abstract：1 篇缺失
model_building：1 篇缺失
```

### 遇到的问题与修正

问题 1：泛关键词误判。

早期规则把“假设”“结果”“求解”等短词作为标题关键词，导致正文句子如“假设综合成本>...”被误识别为章节标题。

处理方式：

- 收紧关键词，使用“模型假设”“基本假设”“假设条件”等更明确的标题。
- 将“结果”改为“结果分析”“求解结果”“模型结果”“结果与分析”。
- 删除过泛的“符号”“优缺点”“附表”“附图”等标题关键词。

问题 2：扫描件无正文。

处理方式：

- 对 `25.txt` 输出空章节 JSON。
- `missing_sections` 标记全部核心章节缺失。
- 日志中记录无标题识别。

## 十、Step 4：二级指标特征提取

### 输入

```text
data/intermediate/sections/*.json
data/extracted_text/*.txt
output/tables/section_split_report.xlsx
```

实际计算主要使用：

- 章节 JSON
- 原始页码标记 TXT

`section_split_report.xlsx` 用于人工核查章节缺失情况，当前代码不依赖它来生成指标。

### 输出

```text
output/tables/appendix1_features_raw.xlsx
output/tables/appendix1_features_normalized.xlsx
output/logs/feature_extraction.log
```

### 已实现函数

`modules/feature_extractor.py` 已实现：

- `extract_structure_features(...)`
- `extract_logic_features(...)`
- `extract_modeling_features(...)`
- `extract_result_features(...)`
- `extract_writing_features(...)`
- `normalize_features(...)`
- `extract_all_features(...)`

此外保留了：

- `extract_features(...)`
- `extract_feature_table(...)`

便于单篇论文或批量表格生成。

### 21 个二级指标

当前输出列名为：

```text
I01_核心章节完整率
I02_摘要完整性
I03_图表编号规范率
I04_附录代码存在性
I05_问题重述覆盖率
I06_模型假设与问题匹配度
I07_逻辑连接词密度
I08_结果结论一致性
I09_模型数量与任务匹配度
I10_公式密度
I11_变量定义覆盖率
I12_目标函数约束完整性
I13_方法合理性语义评分
I14_结果完整率
I15_图表解释率
I16_灵敏度分析存在性
I17_误差分析存在性
I18_参考文献规范率
I19_语言可读性
I20_创新性表达
I21_推广应用价值
```

### 指标类型划分

硬指标：

```text
I01_核心章节完整率
I03_图表编号规范率
I04_附录代码存在性
I05_问题重述覆盖率
I07_逻辑连接词密度
I10_公式密度
I12_目标函数约束完整性
I14_结果完整率
I16_灵敏度分析存在性
I17_误差分析存在性
I18_参考文献规范率
```

半硬指标：

```text
I02_摘要完整性
I09_模型数量与任务匹配度
I11_变量定义覆盖率
I15_图表解释率
I19_语言可读性
I20_创新性表达
I21_推广应用价值
```

软指标：

```text
I06_模型假设与问题匹配度
I08_结果结论一致性
I13_方法合理性语义评分
```

软指标没有调用大语言模型，全部使用可复现近似规则：

- TF-IDF 字符 n-gram 余弦相似度。
- 关键词覆盖率。
- 结构完整性加权。

### 规则来源

Step 4 使用的规则写在 `config.yaml`：

- `logic_connectives`：逻辑连接词列表。
- `feature_extraction.formula_line_patterns`：公式行识别规则。
- `feature_extraction.model_keywords`：模型方法关键词。
- `feature_extraction.validation_keywords`：验证类关键词。
- `feature_extraction.innovation_keywords`：创新表达关键词。
- `feature_extraction.application_keywords`：推广应用关键词。
- `feature_extraction.chart_explanation_keywords`：图表解释关键词。

### 运行命令

```powershell
cd C:\Users\13178\Desktop\A题\A_MAGE_R3
python scripts/run_step4_extract_features.py
```

### 运行结果

- 处理论文数：30
- 生成指标列数：21
- raw 表：30 行，24 列，包括 `paper_id`、`filename`、`text_chars` 和 21 个指标。
- normalized 表：30 行，24 列。
- 所有非缺失归一化指标均位于 `[0,1]`。
- 缺失值数量：40。

### 缺失值说明

缺失值主要来自两类情况：

1. `25.txt` 无正文。

由于 `25.pdf` 是扫描型 PDF，Step 2 未做 OCR，`25.txt` 只有页码标记。Step 4 修正了字符统计逻辑，忽略 `[PAGE n]` 标记后：

```text
25.txt text_chars = 0
```

因此 `25.txt` 的 21 个指标全部为 `NaN`，并写入日志。

2. 某些论文缺少必要章节。

例如：

- 没有模型假设时，`I06_模型假设与问题匹配度` 无法计算。
- 没有结果或结论相关文本时，`I08_结果结论一致性` 无法计算。

这些情况均未填 0，而是按“无法提取”填 `NaN`，符合“不允许编造指标值”的要求。

### 遇到的问题与修正

问题 1：扫描件页码标记被当作正文。

最初 `_char_count` 直接统计非空字符，导致 `25.txt` 的 `[PAGE 1]` 等页码标记被计入字符数，部分指标变为 0 而不是 `NaN`。

处理方式：

- 在 `_clean_text` 中先移除 `[PAGE n]` 标记。
- 重新运行 Step 4。
- 修正后 `25.txt text_chars = 0`，21 个指标均为 `NaN`。

问题 2：没有本地 embedding 模型。

处理方式：

- 不调用大模型。
- 使用 `sklearn` 的 `TfidfVectorizer(analyzer="char", ngram_range=(2,4))`。
- 若 sklearn 不可用，则回退到字符 bigram Jaccard 相似度。

问题 3：不同论文标题格式差异大。

处理方式：

- 指标计算不只依赖章节名。
- 还结合全文正则、关键词覆盖、模型关键词、图表编号等规则。

## 十一、输出文件总览

### 数据输入

| 路径 | 数量 | 说明 |
|---|---:|---|
| `data/appendix1_papers/*.pdf` | 30 | 附件 1 原始论文 PDF |

### Step 2 输出

| 路径 | 数量或规模 | 说明 |
|---|---:|---|
| `data/extracted_text/*.txt` | 30 | 每篇 PDF 对应一个文本文件 |
| `output/tables/pdf_parse_report.xlsx` | 30 行 | PDF 解析报告 |
| `output/logs/pdf_parse.log` | 1 个 | PDF 解析日志 |

### Step 3 输出

| 路径 | 数量或规模 | 说明 |
|---|---:|---|
| `data/intermediate/sections/*.json` | 30 | 每篇论文一个章节 JSON |
| `output/tables/section_split_report.xlsx` | 30 行 | 章节识别报告 |
| `output/logs/section_split.log` | 1 个 | 章节切分日志 |

### Step 4 输出

| 路径 | 数量或规模 | 说明 |
|---|---:|---|
| `output/tables/appendix1_features_raw.xlsx` | 30 行，24 列 | 原始特征表 |
| `output/tables/appendix1_features_normalized.xlsx` | 30 行，24 列 | 归一化特征表 |
| `output/logs/feature_extraction.log` | 1 个 | 特征提取日志 |

## 十二、当前实现边界

当前已经完成：

- 项目结构。
- PDF 解析。
- 章节识别。
- 21 个二级指标特征提取。
- raw 与 normalized 特征表。
- 日志与报告。

当前尚未完成：

- OCR 模块正式接入。
- AHP 主观权重。
- 熵权法或 CRITIC 客观权重。
- 主客观融合权重。
- TOPSIS 综合评价。
- Bradley-Terry 成对偏好校准。
- 质量等级分类。
- 可视化图表。
- 论文正文中的模型公式与结果表述。

## 十三、后续建议

### 优先建议 1：先接入 OCR 或补齐 `25.pdf`

由于 `25.pdf` 是附件 1 中唯一完全无法提取正文的样本，如果不做 OCR，后续权重和评分会出现一篇大量缺失的样本。

建议后续新增一个 OCR Step 或在 Step 2 中增加可配置 OCR fallback：

1. PyMuPDF 渲染页面为图片。
2. Tesseract 或其他中文 OCR 引擎识别。
3. 输出 `25.txt` 的 OCR 文本。
4. 重新运行 Step 3 和 Step 4。

### 优先建议 2：进入 Step 5 权重计算

当前已经有 normalized 特征表，可以进行：

- AHP 主观权重。
- CRITIC 或熵权客观权重。
- 融合权重。
- 权重敏感性分析。

### 优先建议 3：记录指标解释表

后续建模论文中需要说明每个二级指标的含义、计算公式、取值范围和评价方向。当前代码已经可追溯，建议另行输出一份 `feature_definitions.xlsx` 或 Markdown 表格，方便直接写入论文。

## 十四、复现实验命令

从项目根目录依次运行：

```powershell
cd C:\Users\13178\Desktop\A题\A_MAGE_R3
python scripts/run_step1_setup.py
python scripts/run_step2_parse_pdf.py
python scripts/run_step3_split_sections.py
python scripts/run_step4_extract_features.py
```

如果依赖不完整：

```powershell
python -m pip install -r requirements.txt
```

## 十五、重要结论

目前工程已经形成了问题 1 自动评估系统的前处理基础：

- PDF 到文本。
- 文本到章节。
- 章节到 21 个可解释指标。

所有指标均来自文本、章节或显式规则，没有编造指标值，没有调用大模型直接打分。对于无法提取的内容，保留 `NaN` 并写日志，为后续 OCR、缺失值处理和权重模型提供透明依据。
