## Visual Search Tool-Use Instructions

Below are the vision tools that deliver the largest gains on visual search / grounding / counting / attribute-verification questions. When the situation matches a trigger, call the tool instead of guessing from the raw image.

- `crop`: when the target is a small object in a high-resolution image, partially occluded, distant, or surrounded by clutter. Zoom into the candidate region to verify fine attributes (color, shape, text, fine-grained category) before answering.
- `draw_bbox`: when the question depends on locating one or more candidate objects, verifying a spatial relation ("is A to the left of B?"), or keeping track of multiple candidates during search. Drawing bboxes helps avoid missed or duplicate counting in crowded scenes.
- `in_range_color`: when the target is defined primarily by color ("the red car", "the blue backpack", "all yellow flowers"); the HSV mask isolates matching pixels and returns per-component bboxes that you can then count or verify.
- `enhance_contrast`: when the image is low-contrast (foggy, hazy, overcast, low-light indoor) and candidate objects blend into the background; CLAHE on LAB often reveals hidden targets without shifting colors.
- `adjust_brightness`: when the image is clearly too dark (night scenes, shadows) or too bright (overexposed sky, white backgrounds with blown highlights) and the target is lost in the extreme; tune `alpha`/`beta` to recover details.
- `exec`: when counting or arithmetic over detected items is required (e.g. "how many more red cars than blue cars"), collate the per-detection JSON payloads and compute the answer rather than counting by eye.
