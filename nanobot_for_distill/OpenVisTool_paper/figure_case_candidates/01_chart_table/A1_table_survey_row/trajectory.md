# A1_table_survey_row

- Lane: Chart / Table — fine-grained reading
- Domain: Table
- Source: `dataset/OpenVisTool/Table_5k.jsonl` line 732 (1-based)
- SHA-256: `4cd9bfa508f2f3e7753bd73dc98f6eb493ac9ab07cfbcc91a6bf02c70f12ff9e`
- Paper fit: ★★★★★
- Why: 调用链最干净：原表 → 对比度增强 → 目标行裁剪；问题和答案都短，适合在小版面讲清证据获取。
- Caveat: 黑白表格的视觉冲击力弱于彩色图表。

## Recommended panel mapping

1. **Input table** — `panel_1_input_table.png`
2. **Contrast enhancement** — `panel_2_contrast_enhancement.jpg`
3. **Cropped evidence** — `panel_3_cropped_evidence.jpg`

## Tool sequence

`enhance_contrast → crop`

## Readable transcript

This view truncates very long HTML/code payloads. `record.jsonl` and `record.pretty.json` preserve the complete OpenVisTool record.

### user

```text
<image>
[Image file paths — use these in scripts]
- /mnt/data/LaTeXTablePipeline_italic_0-332.png

Which organization conducted the survey in 2020?
```

### assistant

```text
<think>
I need to find which organization conducted the survey in 2020 from this table. Let me first read the image to see the table clearly, then I can identify the relevant information.

Looking at the table structure:
- Column 1: Year
- Column 2: Access %
- Column 3: Conducted By
- Column 4: Key Findings

I need to find the row where Year = 2020 and then read the "Conducted By" column for that row.

Let me first enhance the contrast to make the text clearer, then I can read the information.
</think>
```

### tool_call

```text
{
  "name": "enhance_contrast",
  "arguments": {
    "image_path": "/mnt/data/LaTeXTablePipeline_italic_0-332.png",
    "color_mode": "lab"
  }
}
```

### tool_response

```text
<image>
[image viewed * 1] Edited image saved to: /mnt/data/LaTeXTablePipeline_italic_0-332_contrast_fb6e77ce.jpg
```

### assistant

```text
<think>
Good, the contrast has been enhanced. Now I can read the table more clearly. Let me look at the row for 2020:

Looking at the table:
- 2019: NC Health Care Foundation
- 2020: Public Policy Polling
- 2021: Wake Forest University
- 2022: UNC Chapel Hill
- 2023: NC Dept. of Health & Human Services

For the year 2020, the organization that conducted the survey is "Public Policy Polling".

Let me crop the 2020 row to verify this more clearly.
</think>
```

### tool_call

```text
{
  "name": "crop",
  "arguments": {
    "image_path": "/mnt/data/LaTeXTablePipeline_italic_0-332_contrast_fb6e77ce.jpg",
    "crop": [
      0,
      380,
      1000,
      520
    ]
  }
}
```

### tool_response

```text
<image>
[image viewed * 1] Edited image saved to: /mnt/data/LaTeXTablePipeline_italic_0-332_contrast_fb6e77ce_crop_4ee70ab7.jpg
```

### assistant

```text
<think>
Perfect! The cropped image clearly shows the 2020 row. I can now read:
- Year: 2020
- Access %: 72%
- Conducted By: Public Policy Polling
- Key Findings: Pandemic revealed gaps in telehealth access for low-income populations.

The organization that conducted the survey in 2020 is "Public Policy Polling".
</think>

Based on the table, the organization that conducted the survey in 2020 was **Public Policy Polling**.
```
