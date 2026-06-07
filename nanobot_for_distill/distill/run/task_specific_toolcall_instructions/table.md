## Table VQA Tool-Use Instructions

Below are the vision tools that frequently help on table VQA. For each tool, a concrete trigger is listed — when the situation matches, call the corresponding tool instead of guessing from the raw image.

### Geometric transforms
- `crop`: when only a specific cell, row block, column block, header region, or small text is relevant, or when the image also contains surrounding captions, footnotes, or other tables. Zooming in reduces distraction and lets you read fine digits/units reliably.

### Annotation / alignment aids
- `draw_bbox`: when you need to highlight and track a specific target cell or a set of candidate cells while cross-checking row label × column header × value.
- `draw_line`: when you must align a row with a column across a wide table; drawing a horizontal line across the target row or a vertical line down the target column avoids off-by-one row/column mismatches.

### Contrast / readability enhancement
- `enhance_contrast`: when the table has low contrast (faded scans, light-gray zebra stripes, watermark bleed-through), small digits are hard to read, or cell backgrounds differ in subtle shades that interfere with text.

### Color-based lookup
- `in_range_color`: when the question depends on cells of a specific color (highlighted rows, conditional formatting, colored status cells); the HSV range mask isolates them and returns per-component bboxes.

### Compute
- If the question involves calculation (sum, mean, ratio, ranking, percentage change, etc.), use `exec` or `write_file` to create and run a Python script instead of relying on mental math.
