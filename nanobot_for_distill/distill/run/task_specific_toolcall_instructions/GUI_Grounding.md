## GUI Grounding Tool-Use Instructions

The input is a screenshot of a GUI, and the query asks you to locate a specific UI element (e.g. "click the Submit button", "find the search bar", "where is the settings icon?"). Your job is to locate that element precisely and return its click position as the final answer.

### Required workflow
**First, use visual tools to find and verify the target.** Do not guess the coordinate from the raw screenshot. Always confirm the element's position with the tools below before committing to a final click. Each tool call should have a clear hypothesis to confirm or reject — only call when it actually reduces ambiguity.

**The key step — predict the click on a zoomed crop, then map it back.** Returning a click directly in the full screenshot's coordinate space is hard and imprecise: the target is small, so a small error in your estimate is a large error on screen. Instead:

1. **`crop`** a tight rectangle around the candidate target (keep a little margin so the element is fully inside). This zooms in so the target is large and easy to localize.
2. Decide the click position **on the cropped image** — its center, read in the crop's own 0-1000 space.
3. **`locate_in_crop`** with that crop image's path and your `coordinate` on the crop. It returns the exactly-equivalent point in the **original** image's 0-1000 space. Because the crop covers only a fraction of the original, this amplifies your precision — a coarse click on a tight crop becomes a fine click on the full image. Do **not** re-estimate the coordinate on the full screenshot yourself; use exactly the value `locate_in_crop` returns.

**Finally, produce the answer as a single `computer_use` tool_call with `action: "*_click"`.** Set `coordinate: [x, y]` to the value returned by `locate_in_crop` (original-image 0-1000 space — never raw pixels). It must point at the **center** of the target element, not its edge or corner. Emit exactly one `computer_use` call; it terminates the trajectory and is treated as your final output. Do not follow it with any other tool call or free-form text.

### Tools
- `crop`: zoom into the candidate region. Required before the final click so you can localize precisely. Also useful to read small labels, verify icons, or disambiguate between nearby elements. Prefer **one decisive, tight crop** around the target right before answering (chained crops still map back correctly, but keep it simple).
- `locate_in_crop`: convert your click on the cropped image into the original-image coordinate. This is the bridge between "where the target is in the zoomed view" and "what coordinate to emit". Always run it on your final crop before emitting `computer_use`.
- `draw_bbox`: when multiple candidate elements exist ("the third item in the list", "the button next to X"), or to visually confirm in advance that the element you plan to click is the intended one.
- `in_range_color`: when the target is primarily identified by color ("the red alert", "the green confirm button") and shape alone is ambiguous; the HSV mask returns per-component bboxes.
- `enhance_contrast` / `adjust_brightness`: when the screenshot is dim, washed out, or has heavy dark-mode shadows that hide the target.
- `detect_edges` / `find_contours`: when the target is defined by a thin outline (icon silhouette, table border, dividing line) that is hard to separate visually.
