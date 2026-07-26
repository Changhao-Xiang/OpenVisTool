# OpenVisTool 方法图真实调用案例候选

本目录用于人工选择 `figures/synthesis_pipeline_refined.drawio` 中 “Representative Tool-Use Patterns” 三条主 lane 的真实案例，并额外提供 `InRangeColorTool` 的 Chart 专题候选。

所有候选都直接来自主数据集 `dataset/OpenVisTool/*.jsonl`（正确性 ∩ tool-gain 版本），没有重写工具调用或伪造工具输出。每条记录均提供原始 JSONL 行的 SHA-256、底层 rollout session（若存在）和全部媒体，便于回查。

面向论文与人工选样的主视图均已压缩为 **Query → 关键 Tool Call / Observation → Answer**；完整轨迹只放在候选目录中作来源审计，不建议直接展示在论文图里。

## 建议先看

| 图中 lane | 首选 | 另一种风格 |
|---|---|---|
| Chart / Table | `A1_table_survey_row`：顺序链最干净、答案无歧义 | `A2_chart_sunburst_regions`：彩色、缩放后更醒目 |
| GUI / Visual Search | `B1_gui_calendar_button`：完整 crop→bbox→click | `B2_search_dog_scarf` / `B3_search_teacup_count`：自然场景定位 |
| Web-to-HTML | `C1_web_tech_company`：首尾布局差异最明显 | `C3_web_bookstore`：最终结构最接近参考图 |
| Chart / InRangeColorTool | `D2_chart_calorie_segments`：三色分解最具代表性 | `D1_chart_2021_funding`：单次调用最干净；`D3_chart_book_growth`：多序列增长比较 |

浏览方式：先打开各 lane 的 `contact_sheet.jpg`，再进入候选目录查看 `paper_card.jpg` 和 `paper_case.md`。`index.html` 也提供了本地画廊。

## Chart / Table — fine-grained reading

[总览 contact sheet](01_chart_table/contact_sheet.jpg)

| ID | 推荐度 | 工具序列 | 适合点 | 注意点 |
|---|---:|---|---|---|
| [A1_table_survey_row](01_chart_table/A1_table_survey_row/paper_case.md) | ★★★★★ | `enhance_contrast → crop` | 调用链最干净：原表 → 对比度增强 → 目标行裁剪；问题和答案都短，适合在小版面讲清证据获取。 | 黑白表格的视觉冲击力弱于彩色图表。 |
| [A2_chart_sunburst_regions](01_chart_table/A2_chart_sunburst_regions/paper_case.md) | ★★★★★ | `read_file → crop → crop → crop` | 色彩丰富且“整图 → 局部证据”的视觉关系非常直观；三次真实 crop 均保留在 media/。 | 两个展示 crop 是并列读取；教师思考还指出图中一个子项小计不完全一致，因此若强调数值严谨性优先选 A1。 |
| [A3_chart_enrollment_subplots](01_chart_table/A3_chart_enrollment_subplots/paper_case.md) | ★★★★☆ | `write_file → crop → crop → exec` | 能体现工具在多子图中分别获取证据，再进行数值计算的模式。 | “average History enrollment”被教师解释为 monthly History，语义有歧义；不建议作为首选。 |
| [A4_table_social_metrics](01_chart_table/A4_table_social_metrics/paper_case.md) | ★★★★☆ | `crop → crop` | 问题简单，两个 crop 分别提供比较所需的数值，读者无需理解复杂背景。 | 两次 crop 为并列证据，不含额外视觉变换。 |

## GUI / Visual Search — precise localization

[总览 contact sheet](02_gui_visual_search/contact_sheet.jpg)

| ID | 推荐度 | 工具序列 | 适合点 | 注意点 |
|---|---:|---|---|---|
| [B1_gui_calendar_button](02_gui_visual_search/B1_gui_calendar_button/paper_case.md) | ★★★★★ | `crop → draw_bbox → computer_use` | 标准且完整的 crop → draw_bbox → computer_use 轨迹；目标按钮语义明确，截图干净。 | 最终 bbox 在全图中较小，排版时建议使用 crop 与 bbox 的局部视图。 |
| [B2_search_dog_scarf](02_gui_visual_search/B2_search_dog_scarf/paper_case.md) | ★★★★★ | `crop → crop` | 从大场景到狗再到项圈文字的两级嵌套 crop 非常直观，缩略图下仍能看出证据逐步显现。 | 最终标签文字带有幽默色彩，论文风格是否合适可人工决定。 |
| [B3_search_teacup_count](02_gui_visual_search/B3_search_teacup_count/paper_case.md) | ★★★★★ | `crop → crop → draw_bbox` | 最终证据图用四色 bbox 标出四个杯子，工具结果的用途一眼可见。 | 完整轨迹有两次 crop，三联画跳过了第一层较宽的 crop；该图仍保留在 media/。 |
| [B4_gui_center_alignment](02_gui_visual_search/B4_gui_center_alignment/paper_case.md) | ★★★★☆ | `crop → draw_bbox → computer_use` | 真实桌面应用、目标图标清楚，同样覆盖 crop → bbox → click 的典型模式。 | 全图中的最终 bbox 很小，需在正式图中放大局部。 |
| [B5_search_water_bottles](02_gui_visual_search/B5_search_water_bottles/paper_case.md) | ★★★★★ | `crop → draw_bbox` | crop 后的目标组清楚，最终六个红框很醒目；比纯文字读取更能体现定位工具。 | 场景色调偏浅，正式排版时需要保证打印对比度。 |

## Web-to-HTML — render-and-revise

[总览 contact sheet](03_web_to_html/contact_sheet.jpg)

| ID | 推荐度 | 工具序列 | 适合点 | 注意点 |
|---|---:|---|---|---|
| [C1_web_tech_company](03_web_to_html/C1_web_tech_company/paper_case.md) | ★★★★★ | `read_file → write_file → render_html → edit_file → edit_file → edit_file → edit_file → render_html → edit_file → render_html → read_file` | 初稿错误非常明显（页面被压成窄列），修订后恢复为接近参考图的横向布局；小尺寸也看得出变化。 | 完整轨迹有三次 render；三联画只展示第一次与最后一次。 |
| [C2_web_car_company](03_web_to_html/C2_web_car_company/paper_case.md) | ★★★★★ | `write_file → render_html → edit_file → edit_file → render_html → read_file` | 红蓝绿和评论卡片让布局对应关系清楚；仅两次 render，调用链短。 | 配色较强，可能与论文主图的低饱和配色冲突。 |
| [C3_web_bookstore](03_web_to_html/C3_web_bookstore/paper_case.md) | ★★★★★ | `read_file → in_range_color → write_file → render_html → edit_file → edit_file → render_html → edit_file → edit_file → render_html → read_file` | 初稿中的巨大占位图与修订后的布局差异明显，最终结果和参考截图结构非常接近。 | 三联画省略了中间 render；全部渲染结果均在 media/。 |
| [C4_web_travel_agency](03_web_to_html/C4_web_travel_agency/paper_case.md) | ★★★★☆ | `read_file → in_range_color → in_range_color → in_range_color → write_file → render_html → edit_file → edit_file → edit_file → render_html → edit_file → edit_file → edit_file → render_html → read_file` | 蓝黄大色块在论文缩放后依旧清楚，初稿大占位图到最终布局的变化明显。 | 颜色较鲜艳，且完整轨迹包含额外颜色分析和三次 render。 |

## Chart — InRangeColorTool candidates

[总览 contact sheet](04_chart_in_range_color/contact_sheet.jpg)

| ID | 推荐度 | 工具序列 | 适合点 | 注意点 |
|---|---:|---|---|---|
| [D1_chart_2021_funding](04_chart_in_range_color/D1_chart_2021_funding/paper_case.md) | ★★★★★ | `read_file → in_range_color → exec → exec` | 单次颜色调用、输出最干净，黄色年度序列与问题直接对应；即使缩到论文尺寸也能看出十根柱子。 | 只有一个关键视觉 observation；后续两次 exec 只是把 bbox 高度换算并求比例。 |
| [D2_chart_calorie_segments](04_chart_in_range_color/D2_chart_calorie_segments/paper_case.md) | ★★★★★ | `in_range_color → in_range_color → in_range_color → write_file → exec` | 最典型的堆叠柱颜色分解：三次真实 HSV 调用分别取得 Low、Moderate、High，颜色掩膜与数值计算关系直观。 | 三联主卡只放 Low 与 High；Moderate 的真实输出保留在 media/ 和完整记录中。 |
| [D3_chart_book_growth](04_chart_in_range_color/D3_chart_book_growth/paper_case.md) | ★★★★★ | `in_range_color → in_range_color → in_range_color → write_file → exec` | 三种颜色序列都得到规整的十根柱，能清楚体现“分离序列 → 比较首尾增长 → 计算答案”。 | 主卡省略 Poetry mask，但全部三次调用及输出都保留在 media/。 |
| [D4_chart_retail_sunburst](04_chart_in_range_color/D4_chart_retail_sunburst/paper_case.md) | ★★★★★ | `in_range_color → crop` | 非柱状图案例且调用链只有两步，能展示颜色工具也可用于层级图中的语义区域隔离。 | 问题表述略抽象；最终数值还需要在 crop 中读取叶节点标签。 |
| [D5_chart_chrome_bug_growth](04_chart_in_range_color/D5_chart_chrome_bug_growth/paper_case.md) | ★★★★☆ | `enhance_contrast → crop → in_range_color → in_range_color → write_file → exec` | 两个颜色 observation 都很干净，且直接支撑“局部增长占总增长比例”的像素测量。 | 完整调用链含 enhance/crop 和脚本计算；结果是基于像素高度的近似值。 |

## 文件说明

- `manifest.json` / `manifest.csv`：候选来源、行号、hash、工具序列和 panel 映射。
- `<lane>/contact_sheet.jpg`：按 Query → Tool/Observation → Answer 组织的候选总览。
- `<candidate>/paper_card.jpg`：论文友好的三段式摘要卡片。
- `<candidate>/paper_case.md`：三段式文字摘要与 figure-ready panel 清单。
- `<candidate>/triptych.jpg`：仅含三格视觉结果的辅助预览。
- `<candidate>/panel_*.{png,jpg}`：按三格顺序命名的原始媒体副本。
- `<candidate>/media/`：该 OpenVisTool 记录引用的全部去重媒体。
- `<candidate>/record.jsonl`：从最终 OpenVisTool 文件逐字节复制的原始行。
- `<candidate>/record.pretty.json`：便于阅读的完整 JSON。
- `<candidate>/source_session.jsonl`：底层教师 rollout session 的原始记录。
- `<candidate>/trajectory.md`：仅供审计的调用链；完整内容仍在 JSON 中。

重新生成：先按仓库约定 `conda activate nanobot`，再运行 `python OpenVisTool_paper/figure_case_candidates/_build_candidates.py`。
