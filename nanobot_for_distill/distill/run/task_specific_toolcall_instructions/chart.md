## Chart VQA Tool-Use Instructions

Below are the vision tools that frequently help on chart VQA. For each tool, a concrete trigger is listed — when the situation matches, call the corresponding tool instead of guessing from the raw image.

### Geometric transforms
- `crop`: when the chart contains multiple subplots, inset views, dense legends, or small tick labels, or when only a specific region (a single subplot, a legend box, an axis area) is relevant. Zooming in reduces distraction and makes labels/values legible.

### Annotation / alignment aids
- `draw_line`: when reading a value off an axis — place a vertical guide at the queried x-value or a horizontal guide at the queried y-value to avoid mis-aligning bar tops / line points with the axis ticks.
- `draw_bbox`: when a specific region (a bar group, a legend entry, a highlighted area) must be tracked while cross-referencing it with the axis or legend.
- `draw_circle`: when pointing to a single data point (a scatter marker, a line peak, a pie slice) to confirm it is the one the question asks about.

### Contrast / readability enhancement
- `enhance_contrast`: when grid lines, low-contrast bars/lines, small tick labels, or compressed chart details are hard to read.

### Color-based lookup
- `in_range_color`: when the question depends on identifying a category, legend color, series color, or colored bar/line/area, or when estimating how much of a chart region belongs to a specific color. Prefer HSV ranges for robust selection under anti-aliasing/compression. Pass `region` to restrict matching to the plot area so legends, titles, and surrounding decorations are excluded.

### Compute
- If the question involves calculation (sum, mean, ratio, ranking, percentage change, slope, etc.), use `exec` or `write_file` to create and run a Python script instead of relying on mental math.
