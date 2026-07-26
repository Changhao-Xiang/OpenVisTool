#!/usr/bin/env python3
"""Materialize paper-figure candidates from real OpenVisTool trajectories.

The candidate specification below is intentionally line-addressed against the
final correctness-and-tool-gain OpenVisTool JSONL files.  Each output folder
contains the exact source record, its underlying rollout session when present,
all referenced media, and a three-panel preview for manual paper selection.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "dataset" / "OpenVisTool"
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

LANES = {
    "01_chart_table": {
        "title": "Chart / Table — fine-grained reading",
        "figure_slot": "流程图第 2 阶段第一行",
    },
    "02_gui_visual_search": {
        "title": "GUI / Visual Search — precise localization",
        "figure_slot": "流程图第 2 阶段第二行",
    },
    "03_web_to_html": {
        "title": "Web-to-HTML — render-and-revise",
        "figure_slot": "流程图第 2 阶段第三行",
    },
    "04_chart_in_range_color": {
        "title": "Chart — InRangeColorTool candidates",
        "figure_slot": "右侧 Representative Tool-Use Patterns 的颜色工具备选",
    },
}

# panel_indices index into the de-duplicated `images` field of the final
# OpenVisTool record.  Every referenced image is still copied into media/.
CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "A1_table_survey_row",
        "lane": "01_chart_table",
        "source_jsonl": "Table_5k.jsonl",
        "source_line": 732,
        "domain": "Table",
        "rating": 5,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Input table", "Contrast enhancement", "Cropped evidence"],
        "short_title": "Survey table: enhance, then isolate one row",
        "paper_query": "Which organization conducted the survey in 2020?",
        "paper_steps": [
            {
                "call": "enhance_contrast(image)",
                "observation": "The low-contrast table text becomes easier to read.",
            },
            {
                "call": "crop(enhanced_image, row=2020)",
                "observation": "The row reads: 2020 · 72% · Public Policy Polling.",
            },
        ],
        "paper_answer": "Public Policy Polling.",
        "why": "调用链最干净：原表 → 对比度增强 → 目标行裁剪；问题和答案都短，适合在小版面讲清证据获取。",
        "caveat": "黑白表格的视觉冲击力弱于彩色图表。",
        "recommended": True,
    },
    {
        "id": "A2_chart_sunburst_regions",
        "lane": "01_chart_table",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 8420,
        "domain": "Chart",
        "rating": 5,
        "panel_indices": [0, 1, 3],
        "panel_labels": ["Sunburst input", "Crop: Produce", "Crop: Meats"],
        "short_title": "Sunburst chart: inspect multiple dense regions",
        "paper_query": "What is the combined share of the top product in each of the three largest categories?",
        "paper_steps": [
            {
                "call": "crop(chart, region=Produce)",
                "observation": "The largest Produce item is Leafy Greens at 9%.",
            },
            {
                "call": "crop(chart, region=Meats)",
                "observation": "Free-range Chicken is 4%; a separate Dairy crop finds Greek Yogurt at 6%.",
            },
        ],
        "paper_answer": "9% + 6% + 4% = 19%.",
        "why": "色彩丰富且“整图 → 局部证据”的视觉关系非常直观；三次真实 crop 均保留在 media/。",
        "caveat": "两个展示 crop 是并列读取；教师思考还指出图中一个子项小计不完全一致，因此若强调数值严谨性优先选 A1。",
        "recommended": True,
    },
    {
        "id": "A3_chart_enrollment_subplots",
        "lane": "01_chart_table",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 1889,
        "domain": "Chart",
        "rating": 4,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Multi-chart input", "Monthly subplot", "Yearly subplot"],
        "short_title": "Chart dashboard: acquire evidence from two subplots",
        "paper_query": "How large is the monthly-to-yearly CS increase relative to average History enrollment?",
        "paper_steps": [
            {
                "call": "crop(chart, subplot=monthly)",
                "observation": "Monthly averages: CS 76.67; History 58.33.",
            },
            {
                "call": "crop(chart, subplot=yearly) + exec(calc)",
                "observation": "Yearly CS average is 79.5; the normalized increase is 0.0486.",
            },
        ],
        "paper_answer": "Approximately 0.049 (4.9%).",
        "why": "能体现工具在多子图中分别获取证据，再进行数值计算的模式。",
        "caveat": "“average History enrollment”被教师解释为 monthly History，语义有歧义；不建议作为首选。",
        "recommended": False,
    },
    {
        "id": "A4_table_social_metrics",
        "lane": "01_chart_table",
        "source_jsonl": "Table_5k.jsonl",
        "source_line": 878,
        "domain": "Table",
        "rating": 4,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Input table", "Twitter row", "Facebook row"],
        "short_title": "Social metrics table: compare two distant rows",
        "paper_query": "Were Twitter retweets more numerous than Facebook posts?",
        "paper_steps": [
            {
                "call": "crop(table, row=Twitter Retweets)",
                "observation": "Twitter Retweets = 800.",
            },
            {
                "call": "crop(table, row=Facebook Posts)",
                "observation": "Facebook Posts = 400.",
            },
        ],
        "paper_answer": "True — 800 retweets vs. 400 Facebook posts.",
        "why": "问题简单，两个 crop 分别提供比较所需的数值，读者无需理解复杂背景。",
        "caveat": "两次 crop 为并列证据，不含额外视觉变换。",
        "recommended": False,
    },
    {
        "id": "B1_gui_calendar_button",
        "lane": "02_gui_visual_search",
        "source_jsonl": "GUI-Grounding_11k.jsonl",
        "source_line": 816,
        "domain": "GUI Grounding",
        "rating": 5,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["GUI input", "Target crop", "BBox verification"],
        "short_title": "Calendar GUI: localize and click “Edit Details…”",
        "paper_query": "Click “Edit Details…” in the calendar event dialog.",
        "paper_steps": [
            {
                "call": "crop(screenshot, event_dialog)",
                "observation": "The target button becomes legible in the dialog.",
            },
            {
                "call": "draw_bbox(target) + computer_use(click)",
                "observation": "The verified target center is [353, 453].",
            },
        ],
        "paper_answer": "Click issued at [353, 453].",
        "why": "标准且完整的 crop → draw_bbox → computer_use 轨迹；目标按钮语义明确，截图干净。",
        "caveat": "最终 bbox 在全图中较小，排版时建议使用 crop 与 bbox 的局部视图。",
        "recommended": True,
    },
    {
        "id": "B2_search_dog_scarf",
        "lane": "02_gui_visual_search",
        "source_jsonl": "VisualSearch_2k.jsonl",
        "source_line": 1823,
        "domain": "Visual Search",
        "rating": 5,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Cluttered scene", "Dog crop", "Text evidence"],
        "short_title": "Visual search: two-stage zoom to read a scarf",
        "paper_query": "What text appears on the dog’s scarf?",
        "paper_steps": [
            {
                "call": "crop(scene, dog)",
                "observation": "The dog and its blue checkered scarf become visible.",
            },
            {
                "call": "crop(dog_crop, neck_label)",
                "observation": "The red label resolves into readable white text.",
            },
        ],
        "paper_answer": "“FILTHY.”",
        "why": "从大场景到狗再到项圈文字的两级嵌套 crop 非常直观，缩略图下仍能看出证据逐步显现。",
        "caveat": "最终标签文字带有幽默色彩，论文风格是否合适可人工决定。",
        "recommended": True,
    },
    {
        "id": "B3_search_teacup_count",
        "lane": "02_gui_visual_search",
        "source_jsonl": "VisualSearch_2k.jsonl",
        "source_line": 1035,
        "domain": "Visual Search",
        "rating": 5,
        "panel_indices": [0, 2, 3],
        "panel_labels": ["Cluttered scene", "Close crop", "Counted objects"],
        "short_title": "Visual search: zoom and mark four tea cups",
        "paper_query": "How many white tea cups are on the coffee table?",
        "paper_steps": [
            {
                "call": "crop(scene, coffee_table) × 2",
                "observation": "The tea tray is isolated at a readable scale.",
            },
            {
                "call": "draw_bbox(cups)",
                "observation": "Four separate cups are marked on the tray.",
            },
        ],
        "paper_answer": "4 white tea cups.",
        "why": "最终证据图用四色 bbox 标出四个杯子，工具结果的用途一眼可见。",
        "caveat": "完整轨迹有两次 crop，三联画跳过了第一层较宽的 crop；该图仍保留在 media/。",
        "recommended": False,
    },
    {
        "id": "B4_gui_center_alignment",
        "lane": "02_gui_visual_search",
        "source_jsonl": "GUI-Grounding_11k.jsonl",
        "source_line": 240,
        "domain": "GUI Grounding",
        "rating": 4,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["GUI input", "Toolbar crop", "BBox verification"],
        "short_title": "Writer GUI: localize the center-alignment button",
        "paper_query": "Click the center-alignment button in the formatting toolbar.",
        "paper_steps": [
            {
                "call": "crop(screenshot, alignment_toolbar)",
                "observation": "The four alignment icons become distinguishable.",
            },
            {
                "call": "draw_bbox(target) + computer_use(click)",
                "observation": "The verified center-align icon is at [768, 183].",
            },
        ],
        "paper_answer": "Click issued at [768, 183].",
        "why": "真实桌面应用、目标图标清楚，同样覆盖 crop → bbox → click 的典型模式。",
        "caveat": "全图中的最终 bbox 很小，需在正式图中放大局部。",
        "recommended": False,
    },
    {
        "id": "B5_search_water_bottles",
        "lane": "02_gui_visual_search",
        "source_jsonl": "VisualSearch_2k.jsonl",
        "source_line": 1915,
        "domain": "Visual Search",
        "rating": 5,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Cluttered scene", "Counter crop", "Counted bottles"],
        "short_title": "Visual search: localize and count six bottles",
        "paper_query": "How many mineral-water bottles are on the mini-bar counter?",
        "paper_steps": [
            {
                "call": "crop(scene, mini_bar_counter)",
                "observation": "A compact group of red-capped bottles is isolated.",
            },
            {
                "call": "draw_bbox(bottles)",
                "observation": "Six individual bottles are marked.",
            },
        ],
        "paper_answer": "6 mineral-water bottles.",
        "why": "crop 后的目标组清楚，最终六个红框很醒目；比纯文字读取更能体现定位工具。",
        "caveat": "场景色调偏浅，正式排版时需要保证打印对比度。",
        "recommended": False,
    },
    {
        "id": "C1_web_tech_company",
        "lane": "03_web_to_html",
        "source_jsonl": "VinciCoder_11k.jsonl",
        "source_line": 908,
        "domain": "Web-to-HTML",
        "rating": 5,
        "panel_indices": [0, 1, 3],
        "panel_labels": ["Reference", "First render", "Revised render"],
        "short_title": "Tech-company page: repair a narrow first render",
        "paper_query": "Recreate the reference tech-company webpage in HTML and CSS.",
        "paper_steps": [
            {
                "call": "render_html(index.html)",
                "observation": "Oversized product/footer placeholders make the first render narrow and overlong.",
            },
            {
                "call": "edit_file × 5 + iterative render_html",
                "observation": "Revisions remove product images and reduce the icons and logo.",
            },
        ],
        "paper_answer": "Final self-contained tech-company HTML/CSS returned.",
        "why": "初稿错误非常明显（页面被压成窄列），修订后恢复为接近参考图的横向布局；小尺寸也看得出变化。",
        "caveat": "完整轨迹有三次 render；三联画只展示第一次与最后一次。",
        "recommended": True,
    },
    {
        "id": "C2_web_car_company",
        "lane": "03_web_to_html",
        "source_jsonl": "VinciCoder_11k.jsonl",
        "source_line": 1216,
        "domain": "Web-to-HTML",
        "rating": 5,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Reference", "First render", "Revised render"],
        "short_title": "Car-company page: fix layout and oversized media",
        "paper_query": "Recreate the reference car-company webpage in HTML and CSS.",
        "paper_steps": [
            {
                "call": "render_html(index.html)",
                "observation": "The first render exposes an oversized footer logo.",
            },
            {
                "call": "edit_file × 2 + render_html(index.html)",
                "observation": "The footer logo is reduced to a 20-pixel inline icon.",
            },
        ],
        "paper_answer": "Final self-contained car-company HTML/CSS returned.",
        "why": "红蓝绿和评论卡片让布局对应关系清楚；仅两次 render，调用链短。",
        "caveat": "配色较强，可能与论文主图的低饱和配色冲突。",
        "recommended": False,
    },
    {
        "id": "C3_web_bookstore",
        "lane": "03_web_to_html",
        "source_jsonl": "VinciCoder_11k.jsonl",
        "source_line": 1751,
        "domain": "Web-to-HTML",
        "rating": 5,
        "panel_indices": [0, 2, 4],
        "panel_labels": ["Reference", "First render", "Revised render"],
        "short_title": "Bookstore page: shrink placeholder and recover layout",
        "paper_query": "Recreate the reference bookstore webpage in HTML and CSS.",
        "paper_steps": [
            {
                "call": "render_html(index.html)",
                "observation": "A giant placeholder dominates the first render.",
            },
            {
                "call": "edit_file × 4 + iterative render_html",
                "observation": "A small inline logo and stronger headings restore the page proportions.",
            },
        ],
        "paper_answer": "Final self-contained bookstore HTML/CSS returned.",
        "why": "初稿中的巨大占位图与修订后的布局差异明显，最终结果和参考截图结构非常接近。",
        "caveat": "三联画省略了中间 render；全部渲染结果均在 media/。",
        "recommended": True,
    },
    {
        "id": "C4_web_travel_agency",
        "lane": "03_web_to_html",
        "source_jsonl": "VinciCoder_11k.jsonl",
        "source_line": 1480,
        "domain": "Web-to-HTML",
        "rating": 4,
        "panel_indices": [0, 3, 5],
        "panel_labels": ["Reference", "First render", "Revised render"],
        "short_title": "Travel page: revise logo scale and section layout",
        "paper_query": "Recreate the reference travel-agency webpage in HTML and CSS.",
        "paper_steps": [
            {
                "call": "render_html(index.html)",
                "observation": "The first render contains a giant logo placeholder.",
            },
            {
                "call": "edit_file × 6 + iterative render_html",
                "observation": "Revisions shrink the logo, widen nav spacing, and match title wrapping.",
            },
        ],
        "paper_answer": "Final self-contained travel-agency HTML/CSS returned.",
        "why": "蓝黄大色块在论文缩放后依旧清楚，初稿大占位图到最终布局的变化明显。",
        "caveat": "颜色较鲜艳，且完整轨迹包含额外颜色分析和三次 render。",
        "recommended": False,
    },
    {
        "id": "D1_chart_2021_funding",
        "lane": "04_chart_in_range_color",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 10155,
        "domain": "Chart",
        "rating": 5,
        "panel_indices": [0, 1],
        "panel_labels": ["Grouped-bar input", "2021 series isolated"],
        "short_title": "Grouped bars: isolate one year across all countries",
        "paper_query": "In 2021, what fraction of total funding equals the amount by which the U.S. exceeded China?",
        "paper_steps": [
            {
                "call": "InRangeColorTool(chart, yellow/2021)",
                "observation": "Ten clean yellow bars are isolated; their bounding boxes provide the 2021 heights.",
            },
        ],
        "paper_answer": "(820 − 310) / 2,510 ≈ 20.3%.",
        "why": "单次颜色调用、输出最干净，黄色年度序列与问题直接对应；即使缩到论文尺寸也能看出十根柱子。",
        "caveat": "只有一个关键视觉 observation；后续两次 exec 只是把 bbox 高度换算并求比例。",
        "recommended": True,
    },
    {
        "id": "D2_chart_calorie_segments",
        "lane": "04_chart_in_range_color",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 10682,
        "domain": "Chart",
        "rating": 5,
        "panel_indices": [0, 1, 3],
        "panel_labels": ["Stacked-bar input", "Low-calorie segments", "High-calorie segments"],
        "short_title": "Stacked bars: split caloric levels by color",
        "paper_query": "What is the average ratio of high-calorie consumption to combined low- and moderate-calorie consumption?",
        "paper_steps": [
            {
                "call": "InRangeColorTool(chart, pink HSV)",
                "observation": "The low-calorie segment of every food-category bar is isolated.",
            },
            {
                "call": "InRangeColorTool(chart, green HSV)",
                "observation": "The high-calorie segments are isolated; a third real call extracts the moderate segments.",
            },
        ],
        "paper_answer": "Average High / (Low + Moderate) ≈ 1.062.",
        "why": "最典型的堆叠柱颜色分解：三次真实 HSV 调用分别取得 Low、Moderate、High，颜色掩膜与数值计算关系直观。",
        "caveat": "三联主卡只放 Low 与 High；Moderate 的真实输出保留在 media/ 和完整记录中。",
        "recommended": True,
    },
    {
        "id": "D3_chart_book_growth",
        "lane": "04_chart_in_range_color",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 3712,
        "domain": "Chart",
        "rating": 5,
        "panel_indices": [0, 1, 3],
        "panel_labels": ["Grouped-bar input", "Children's Books", "Academic Texts"],
        "short_title": "Grouped bars: separate three genres before comparison",
        "paper_query": "What is the ratio of Children's Books growth to the combined growth of Poetry and Academic Texts?",
        "paper_steps": [
            {
                "call": "InRangeColorTool(chart, orange HSV)",
                "observation": "Ten Children's Books bars are isolated, exposing the 2013 and 2022 endpoints.",
            },
            {
                "call": "InRangeColorTool(chart, pink HSV)",
                "observation": "Academic Texts are isolated; a third real call similarly extracts Poetry.",
            },
        ],
        "paper_answer": "Approximately 0.55 (about 5:9).",
        "why": "三种颜色序列都得到规整的十根柱，能清楚体现“分离序列 → 比较首尾增长 → 计算答案”。",
        "caveat": "主卡省略 Poetry mask，但全部三次调用及输出都保留在 media/。",
        "recommended": True,
    },
    {
        "id": "D4_chart_retail_sunburst",
        "lane": "04_chart_in_range_color",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 12016,
        "domain": "Chart",
        "rating": 5,
        "panel_indices": [0, 1, 2],
        "panel_labels": ["Sunburst input", "Retail hierarchy isolated", "Retail crop"],
        "short_title": "Sunburst: isolate the dominant hierarchy by color",
        "paper_query": "Within the most prevalent main category, what share does one specific business type typically occupy?",
        "paper_steps": [
            {
                "call": "InRangeColorTool(sunburst, blue HSV)",
                "observation": "The complete blue Retail hierarchy is separated from the other main categories.",
            },
            {
                "call": "crop(chart, Retail sector)",
                "observation": "Six leaf shares are readable: 9%, 6%, 7%, 5%, 6%, and 4%.",
            },
        ],
        "paper_answer": "37% / 6 ≈ 6.17% (about 6%).",
        "why": "非柱状图案例且调用链只有两步，能展示颜色工具也可用于层级图中的语义区域隔离。",
        "caveat": "问题表述略抽象；最终数值还需要在 crop 中读取叶节点标签。",
        "recommended": False,
    },
    {
        "id": "D5_chart_chrome_bug_growth",
        "lane": "04_chart_in_range_color",
        "source_jsonl": "Chart_14k.jsonl",
        "source_line": 12163,
        "domain": "Chart",
        "rating": 4,
        "panel_indices": [0, 3, 4],
        "panel_labels": ["Stacked-bar input", "Chrome segments", "Total-height anchors"],
        "short_title": "Stacked trend: measure one component of total growth",
        "paper_query": "What percentage of the total bug increase from Oct. 2022 to Mar. 2023 was caused by Chrome?",
        "paper_steps": [
            {
                "call": "InRangeColorTool(chart, orange HSV)",
                "observation": "Chrome segments are isolated across all dates; endpoint heights correspond to about 95 and 235 bugs.",
            },
            {
                "call": "InRangeColorTool(chart, purple HSV)",
                "observation": "Top segments provide full-bar anchors for the labeled totals 190 and 497.",
            },
        ],
        "paper_answer": "(235 − 95) / (497 − 190) ≈ 45.5%.",
        "why": "两个颜色 observation 都很干净，且直接支撑“局部增长占总增长比例”的像素测量。",
        "caveat": "完整调用链含 enhance/crop 和脚本计算；结果是基于像素高度的近似值。",
        "recommended": False,
    },
]


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "Ubuntu-B.ttf" if bold else "Ubuntu-R.ttf",
    ]
    roots = [
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/ubuntu"),
    ]
    for root in roots:
        for name in names:
            path = root / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def unique_paths(values: list[str]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(Path(value))
    return result


def slug(value: str) -> str:
    value = value.lower().replace("→", "to")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "panel"


def text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox(xy, value, font=font)
    width, height = draw.textsize(value, font=font)
    return (xy[0], xy[1], xy[0] + width, xy[1] + height)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap English display text to a measured pixel width."""
    lines: list[str] = []
    for paragraph in value.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if text_bbox(draw, (0, 0), trial, font)[2] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_spacing: int = 5,
    max_lines: int | None = None,
) -> int:
    """Draw wrapped text and return the bottom y coordinate."""
    lines = wrap_text(draw, value, font, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = f"{lines[-1].rstrip('.')}…"
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        box = text_bbox(draw, (x, y), line or "Ag", font)
        y += box[3] - box[1] + line_spacing
    return y


def extract_question(record: dict[str, Any]) -> str:
    content = next(
        message["content"]
        for message in record["messages"]
        if message.get("role") == "user"
    )
    if "\n\n" in content:
        return content.split("\n\n", 1)[1].strip()
    return content.strip()


def parsed_tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for message in record["messages"]:
        if message.get("role") == "tool_call":
            result.append(json.loads(message["content"]))
    return result


def infer_session_dir(media: list[Path]) -> Path | None:
    for path in media:
        if "workspaces" in path.parts and path.parent.name.startswith("session_"):
            return path.parent
    return None


def copy_media(media: list[Path], candidate_dir: Path) -> list[str]:
    media_dir = candidate_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for index, source in enumerate(media):
        if not source.exists():
            raise FileNotFoundError(source)
        target = media_dir / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(candidate_dir)))
        cropmeta = Path(f"{source}.cropmeta.json")
        if cropmeta.exists():
            shutil.copy2(cropmeta, media_dir / f"{index:02d}_{cropmeta.name}")
    return copied


def materialize_panels(
    candidate: dict[str, Any],
    media: list[Path],
    candidate_dir: Path,
) -> list[str]:
    copied: list[str] = []
    for number, (index, label) in enumerate(
        zip(candidate["panel_indices"], candidate["panel_labels"]),
        start=1,
    ):
        source = media[index]
        target = candidate_dir / f"panel_{number}_{slug(label)}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied.append(target.name)
    return copied


def render_triptych(
    candidate: dict[str, Any],
    question: str,
    panel_paths: list[Path],
    target: Path,
) -> None:
    width, height = 1500, 590
    canvas = Image.new("RGB", (width, height), "#f8faf8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    body_font = load_font(18)
    label_font = load_font(19, bold=True)
    note_font = load_font(15)

    draw.text(
        (24, 16),
        f"{candidate['id']}  ·  {candidate['short_title']}",
        font=title_font,
        fill="#1f5f31",
    )
    short_question = re.sub(r"\s+", " ", question)
    if len(short_question) > 180:
        short_question = f"{short_question[:177]}..."
    draw.text((24, 55), short_question, font=body_font, fill="#242424")

    panel_width = 464
    panel_height = 405
    panel_top = 128
    gap = 28
    panel_count = len(panel_paths)
    panel_left = (width - (panel_count * panel_width + (panel_count - 1) * gap)) // 2
    for index, (path, label) in enumerate(
        zip(panel_paths, candidate["panel_labels"])
    ):
        left = panel_left + index * (panel_width + gap)
        draw.rounded_rectangle(
            (left, panel_top, left + panel_width, panel_top + panel_height),
            radius=10,
            fill="#ffffff",
            outline="#8eaa94",
            width=2,
        )
        with Image.open(path) as source:
            image = ImageOps.contain(
                source.convert("RGB"),
                (panel_width - 20, panel_height - 64),
                method=LANCZOS,
            )
        image_left = left + (panel_width - image.width) // 2
        image_top = panel_top + 45 + (panel_height - 58 - image.height) // 2
        canvas.paste(image, (image_left, image_top))
        draw.text((left + 12, panel_top + 12), label, font=label_font, fill="#245d32")
        if index < panel_count - 1:
            arrow_x = left + panel_width + 6
            arrow_y = panel_top + panel_height // 2
            draw.line((arrow_x, arrow_y, arrow_x + 15, arrow_y), fill="#39804a", width=4)
            draw.polygon(
                [
                    (arrow_x + 15, arrow_y - 7),
                    (arrow_x + 25, arrow_y),
                    (arrow_x + 15, arrow_y + 7),
                ],
                fill="#39804a",
            )

    tools = " → ".join(candidate["tool_sequence"])
    if len(tools) > 150:
        tools = f"{tools[:147]}..."
    draw.text((24, 552), f"Real tool sequence: {tools}", font=note_font, fill="#4b5e50")
    canvas.save(target, quality=94)


def render_paper_card(
    candidate: dict[str, Any],
    panel_paths: list[Path],
    target: Path,
) -> None:
    """Render the paper-facing Query → key evidence → Answer summary."""
    width, height = 1500, 820
    canvas = Image.new("RGB", (width, height), "#f7faf7")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    section_font = load_font(18, bold=True)
    query_font = load_font(23, bold=True)
    call_font = load_font(17, bold=True)
    body_font = load_font(16)
    answer_font = load_font(25, bold=True)
    footer_font = load_font(14)

    draw.text(
        (24, 15),
        f"{candidate['id']}  ·  {candidate['short_title']}",
        font=title_font,
        fill="#225f33",
    )

    draw.rounded_rectangle(
        (24, 56, 1476, 146),
        radius=12,
        fill="#edf4ff",
        outline="#6d91bf",
        width=2,
    )
    draw.text((42, 70), "QUERY", font=section_font, fill="#2f62a1")
    draw_wrapped(
        draw,
        (145, 67),
        candidate["paper_query"],
        query_font,
        "#172b46",
        1300,
        line_spacing=4,
        max_lines=2,
    )

    draw.text(
        (24, 165),
        "KEY TOOL CALLS / OBSERVATIONS",
        font=section_font,
        fill="#2a6c3c",
    )
    panel_width = 464
    panel_height = 430
    panel_top = 196
    gap = 28
    columns = [
        {
            "call": "INPUT",
            "observation": "Original visual context.",
        },
        *candidate["paper_steps"],
    ]
    panel_count = len(panel_paths)
    panel_left = (width - (panel_count * panel_width + (panel_count - 1) * gap)) // 2
    for index, (path, column) in enumerate(zip(panel_paths, columns)):
        left = panel_left + index * (panel_width + gap)
        draw.rounded_rectangle(
            (left, panel_top, left + panel_width, panel_top + panel_height),
            radius=10,
            fill="#ffffff",
            outline="#8eaa94",
            width=2,
        )
        call_color = "#315a8c" if index == 0 else "#246b35"
        draw_wrapped(
            draw,
            (left + 14, panel_top + 12),
            column["call"],
            call_font,
            call_color,
            panel_width - 28,
            line_spacing=2,
            max_lines=2,
        )
        with Image.open(path) as source:
            image = ImageOps.contain(
                source.convert("RGB"),
                (panel_width - 24, 260),
                method=LANCZOS,
            )
        image_left = left + (panel_width - image.width) // 2
        image_top = panel_top + 82 + (260 - image.height) // 2
        canvas.paste(image, (image_left, image_top))
        draw.line(
            (left + 14, panel_top + 355, left + panel_width - 14, panel_top + 355),
            fill="#d6e1d8",
            width=1,
        )
        draw_wrapped(
            draw,
            (left + 14, panel_top + 365),
            f"Observation: {column['observation']}",
            body_font,
            "#26342a",
            panel_width - 28,
            line_spacing=3,
            max_lines=3,
        )
        if index < panel_count - 1:
            arrow_x = left + panel_width + 6
            arrow_y = panel_top + panel_height // 2
            draw.line((arrow_x, arrow_y, arrow_x + 15, arrow_y), fill="#39804a", width=4)
            draw.polygon(
                [
                    (arrow_x + 15, arrow_y - 7),
                    (arrow_x + 25, arrow_y),
                    (arrow_x + 15, arrow_y + 7),
                ],
                fill="#39804a",
            )

    draw.rounded_rectangle(
        (24, 650, 1476, 777),
        radius=12,
        fill="#e9f6eb",
        outline="#559a63",
        width=2,
    )
    draw.text((42, 670), "ANSWER", font=section_font, fill="#28733a")
    draw_wrapped(
        draw,
        (155, 664),
        candidate["paper_answer"],
        answer_font,
        "#174d27",
        1280,
        line_spacing=4,
        max_lines=2,
    )
    draw.text(
        (42, 785),
        "Condensed from a real OpenVisTool record; full trajectory retained only for provenance.",
        font=footer_font,
        fill="#617066",
    )
    canvas.save(target, quality=94)


def compact_text(value: str, limit: int = 1200) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n\n… [truncated; see record.pretty.json for the exact full message]"


def readable_trajectory(
    candidate: dict[str, Any],
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    lines = [
        f"# {candidate['id']}",
        "",
        f"- Lane: {LANES[candidate['lane']]['title']}",
        f"- Domain: {candidate['domain']}",
        f"- Source: `dataset/OpenVisTool/{candidate['source_jsonl']}` line {candidate['source_line']} (1-based)",
        f"- SHA-256: `{metadata['source_record_sha256']}`",
        f"- Paper fit: {'★' * candidate['rating']}{'☆' * (5 - candidate['rating'])}",
        f"- Why: {candidate['why']}",
        f"- Caveat: {candidate['caveat']}",
        "",
        "## Recommended panel mapping",
        "",
    ]
    for index, (label, path) in enumerate(
        zip(candidate["panel_labels"], metadata["panel_files"]),
        start=1,
    ):
        lines.append(f"{index}. **{label}** — `{path}`")
    lines.extend(
        [
            "",
            "## Tool sequence",
            "",
            "`" + " → ".join(metadata["tool_sequence"]) + "`",
            "",
            "## Readable transcript",
            "",
            "This view truncates very long HTML/code payloads. `record.jsonl` and "
            "`record.pretty.json` preserve the complete OpenVisTool record.",
            "",
        ]
    )
    for message in record["messages"]:
        role = message.get("role", "unknown")
        if role == "system":
            continue
        content = message.get("content", "")
        if role == "tool_call":
            parsed = json.loads(content)
            content = json.dumps(parsed, ensure_ascii=False, indent=2)
        lines.extend(
            [
                f"### {role}",
                "",
                "```text",
                compact_text(content),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def concise_paper_case(
    candidate: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    lines = [
        f"# {candidate['id']}",
        "",
        "![Query → Tool Call / Observation → Answer](paper_card.jpg)",
        "",
        "## Query",
        "",
        candidate["paper_query"],
        "",
        "## Key tool calls / observations",
        "",
    ]
    for index, step in enumerate(candidate["paper_steps"], start=1):
        lines.extend(
            [
                f"{index}. `{step['call']}`",
                f"   - Observation: {step['observation']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Answer",
            "",
            f"**{candidate['paper_answer']}**",
            "",
            "## Figure-ready assets",
            "",
        ]
    )
    for index, (label, path) in enumerate(
        zip(candidate["panel_labels"], metadata["panel_files"]),
        start=1,
    ):
        lines.append(f"- Panel {index}, {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Provenance (for audit only)",
            "",
            f"- Final OpenVisTool source: `dataset/OpenVisTool/{candidate['source_jsonl']}` "
            f"line {candidate['source_line']} (1-based)",
            f"- Record SHA-256: `{metadata['source_record_sha256']}`",
            "- Full record: `record.jsonl` / `record.pretty.json`",
            "- Original rollout: `source_session.jsonl`",
            "- Full readable trajectory: `trajectory.md`",
            "",
        ]
    )
    return "\n".join(lines)


def write_lane_contact_sheet(
    lane: str,
    candidates: list[dict[str, Any]],
    target: Path,
) -> None:
    row_width, row_height = 1500, 820
    header_height = 80
    canvas = Image.new(
        "RGB",
        (row_width, header_height + row_height * len(candidates)),
        "#eef4ef",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 20),
        LANES[lane]["title"],
        font=load_font(30, bold=True),
        fill="#1f5f31",
    )
    for index, candidate in enumerate(candidates):
        paper_card = Image.open(OUTPUT_ROOT / lane / candidate["id"] / "paper_card.jpg")
        canvas.paste(paper_card, (0, header_height + index * row_height))
    canvas.save(target, quality=92)


def write_readme(materialized: list[dict[str, Any]]) -> None:
    by_lane: dict[str, list[dict[str, Any]]] = {key: [] for key in LANES}
    for item in materialized:
        by_lane[item["lane"]].append(item)

    lines = [
        "# OpenVisTool 方法图真实调用案例候选",
        "",
        "本目录用于人工选择 `figures/synthesis_pipeline_refined.drawio` 中 "
        "“Representative Tool-Use Patterns” 三条主 lane 的真实案例，并额外提供 "
        "`InRangeColorTool` 的 Chart 专题候选。",
        "",
        "所有候选都直接来自主数据集 `dataset/OpenVisTool/*.jsonl`（正确性 ∩ "
        "tool-gain 版本），没有重写工具调用或伪造工具输出。每条记录均提供原始 "
        "JSONL 行的 SHA-256、底层 rollout session（若存在）和全部媒体，便于回查。",
        "",
        "面向论文与人工选样的主视图均已压缩为 "
        "**Query → 关键 Tool Call / Observation → Answer**；完整轨迹只放在候选目录中作来源审计，"
        "不建议直接展示在论文图里。",
        "",
        "## 建议先看",
        "",
        "| 图中 lane | 首选 | 另一种风格 |",
        "|---|---|---|",
        "| Chart / Table | `A1_table_survey_row`：顺序链最干净、答案无歧义 | `A2_chart_sunburst_regions`：彩色、缩放后更醒目 |",
        "| GUI / Visual Search | `B1_gui_calendar_button`：完整 crop→bbox→click | `B2_search_dog_scarf` / `B3_search_teacup_count`：自然场景定位 |",
        "| Web-to-HTML | `C1_web_tech_company`：首尾布局差异最明显 | `C3_web_bookstore`：最终结构最接近参考图 |",
        "| Chart / InRangeColorTool | `D2_chart_calorie_segments`：三色分解最具代表性 | `D1_chart_2021_funding`：单次调用最干净；`D3_chart_book_growth`：多序列增长比较 |",
        "",
        "浏览方式：先打开各 lane 的 `contact_sheet.jpg`，再进入候选目录查看 "
        "`paper_card.jpg` 和 `paper_case.md`。`index.html` 也提供了本地画廊。",
        "",
    ]
    for lane, lane_info in LANES.items():
        lines.extend(
            [
                f"## {lane_info['title']}",
                "",
                f"[总览 contact sheet]({lane}/contact_sheet.jpg)",
                "",
                "| ID | 推荐度 | 工具序列 | 适合点 | 注意点 |",
                "|---|---:|---|---|---|",
            ]
        )
        for item in by_lane[lane]:
            sequence = " → ".join(item["tool_sequence"])
            lines.append(
                f"| [{item['id']}]({lane}/{item['id']}/paper_case.md) "
                f"| {'★' * item['rating']}{'☆' * (5 - item['rating'])} "
                f"| `{sequence}` | {item['why']} | {item['caveat']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 文件说明",
            "",
            "- `manifest.json` / `manifest.csv`：候选来源、行号、hash、工具序列和 panel 映射。",
            "- `<lane>/contact_sheet.jpg`：按 Query → Tool/Observation → Answer 组织的候选总览。",
            "- `<candidate>/paper_card.jpg`：论文友好的三段式摘要卡片。",
            "- `<candidate>/paper_case.md`：三段式文字摘要与 figure-ready panel 清单。",
            "- `<candidate>/triptych.jpg`：仅含三格视觉结果的辅助预览。",
            "- `<candidate>/panel_*.{png,jpg}`：按三格顺序命名的原始媒体副本。",
            "- `<candidate>/media/`：该 OpenVisTool 记录引用的全部去重媒体。",
            "- `<candidate>/record.jsonl`：从最终 OpenVisTool 文件逐字节复制的原始行。",
            "- `<candidate>/record.pretty.json`：便于阅读的完整 JSON。",
            "- `<candidate>/source_session.jsonl`：底层教师 rollout session 的原始记录。",
            "- `<candidate>/trajectory.md`：仅供审计的调用链；完整内容仍在 JSON 中。",
            "",
            "重新生成：先按仓库约定 `conda activate nanobot`，再运行 "
            "`python OpenVisTool_paper/figure_case_candidates/_build_candidates.py`。",
            "",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_gallery(materialized: list[dict[str, Any]]) -> None:
    cards = []
    for item in materialized:
        lane = item["lane"]
        candidate_id = item["id"]
        cards.append(
            f"""
            <article class="card">
              <div class="meta">{html.escape(LANES[lane]['title'])} · {'★' * item['rating']}</div>
              <h2>{html.escape(candidate_id)}</h2>
              <p>{html.escape(item['short_title'])}</p>
              <a href="{lane}/{candidate_id}/paper_card.jpg">
                <img src="{lane}/{candidate_id}/paper_card.jpg" alt="{html.escape(candidate_id)}">
              </a>
              <p><strong>Query:</strong> {html.escape(item['paper_query'])}</p>
              <p><strong>Answer:</strong> {html.escape(item['paper_answer'])}</p>
              <p>{html.escape(item['why'])}</p>
              <p class="links">
                <a href="{lane}/{candidate_id}/paper_case.md">concise case</a>
                <a href="{lane}/{candidate_id}/record.pretty.json">full record (audit)</a>
              </p>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenVisTool figure case candidates</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, "Noto Sans CJK SC", sans-serif; }}
    body {{ margin: 0; background: #edf4ee; color: #1e2b21; }}
    header {{ padding: 28px 4vw 16px; max-width: 1500px; margin: auto; }}
    h1 {{ margin: 0 0 8px; color: #1f6534; }}
    main {{ max-width: 1500px; margin: auto; padding: 12px 4vw 48px; display: grid; gap: 22px; }}
    .card {{ background: white; border: 1px solid #aac6af; border-radius: 14px; padding: 18px; box-shadow: 0 4px 16px #234b2a15; }}
    .card img {{ display: block; width: 100%; border: 1px solid #d4e0d6; border-radius: 8px; }}
    .meta {{ color: #40734b; font-weight: 700; }}
    h2 {{ margin: 6px 0; }}
    p {{ line-height: 1.55; }}
    a {{ color: #17642c; }}
    .links {{ display: flex; gap: 18px; }}
  </style>
</head>
<body>
  <header>
    <h1>OpenVisTool 方法图真实调用案例候选</h1>
    <p>按 Query → 关键 Tool Call / Observation → Answer 组织；完整记录仅用于来源审计。</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (OUTPUT_ROOT / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    requested_by_file: dict[str, set[int]] = {}
    for candidate in CANDIDATES:
        requested_by_file.setdefault(candidate["source_jsonl"], set()).add(
            candidate["source_line"]
        )

    raw_records: dict[tuple[str, int], tuple[bytes, dict[str, Any]]] = {}
    for filename, requested_lines in requested_by_file.items():
        path = DATA_ROOT / filename
        with path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if line_number in requested_lines:
                    raw_records[(filename, line_number)] = (
                        raw_line,
                        json.loads(raw_line),
                    )
                if len(
                    [
                        key
                        for key in raw_records
                        if key[0] == filename
                    ]
                ) == len(requested_lines):
                    break

    missing = [
        (candidate["source_jsonl"], candidate["source_line"])
        for candidate in CANDIDATES
        if (candidate["source_jsonl"], candidate["source_line"]) not in raw_records
    ]
    if missing:
        raise RuntimeError(f"Missing source records: {missing}")

    materialized: list[dict[str, Any]] = []
    for candidate_spec in CANDIDATES:
        candidate = dict(candidate_spec)
        key = (candidate["source_jsonl"], candidate["source_line"])
        raw_line, record = raw_records[key]
        media = unique_paths(record.get("images", []))
        if max(candidate["panel_indices"]) >= len(media):
            raise IndexError(
                f"{candidate['id']} panel index exceeds {len(media)} unique images"
            )
        calls = parsed_tool_calls(record)
        candidate["tool_sequence"] = [call["name"] for call in calls]

        candidate_dir = OUTPUT_ROOT / candidate["lane"] / candidate["id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        copied_media = copy_media(media, candidate_dir)
        panel_files = materialize_panels(candidate, media, candidate_dir)
        panel_paths = [candidate_dir / name for name in panel_files]

        (candidate_dir / "record.jsonl").write_bytes(raw_line)
        (candidate_dir / "record.pretty.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "tool_calls.json").write_text(
            json.dumps(calls, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        session_dir = infer_session_dir(media)
        source_session = None
        if session_dir is not None:
            session_jsonls = sorted(session_dir.glob("*.jsonl"))
            if session_jsonls:
                source_session = session_jsonls[0]
                shutil.copy2(source_session, candidate_dir / "source_session.jsonl")

        question = extract_question(record)
        metadata = {
            **candidate,
            "source_path": str(DATA_ROOT / candidate["source_jsonl"]),
            "source_line_1_based": candidate["source_line"],
            "source_record_sha256": hashlib.sha256(raw_line).hexdigest(),
            "question": question,
            "source_images": [str(path) for path in media],
            "copied_media": copied_media,
            "panel_files": panel_files,
            "session_dir": str(session_dir) if session_dir else None,
            "source_session": str(source_session) if source_session else None,
        }
        (candidate_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "trajectory.md").write_text(
            readable_trajectory(candidate, record, metadata),
            encoding="utf-8",
        )
        (candidate_dir / "paper_case.md").write_text(
            concise_paper_case(candidate, metadata),
            encoding="utf-8",
        )
        render_triptych(
            candidate,
            question,
            panel_paths,
            candidate_dir / "triptych.jpg",
        )
        render_paper_card(
            candidate,
            panel_paths,
            candidate_dir / "paper_card.jpg",
        )
        materialized.append(metadata)

    for lane in LANES:
        lane_dir = OUTPUT_ROOT / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        lane_candidates = [item for item in materialized if item["lane"] == lane]
        write_lane_contact_sheet(
            lane,
            lane_candidates,
            lane_dir / "contact_sheet.jpg",
        )

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_ROOT / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "id",
                "lane",
                "domain",
                "rating",
                "recommended",
                "source_jsonl",
                "source_line_1_based",
                "source_record_sha256",
                "tool_sequence",
                "question",
                "session_dir",
            ],
        )
        writer.writeheader()
        for item in materialized:
            writer.writerow(
                {
                    **{
                        key: item.get(key)
                        for key in writer.fieldnames
                    },
                    "tool_sequence": " -> ".join(item["tool_sequence"]),
                }
            )

    write_readme(materialized)
    write_gallery(materialized)
    print(
        f"Materialized {len(materialized)} candidates in "
        f"{OUTPUT_ROOT.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
