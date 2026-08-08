## HTML Code Generation Tool-Use Instructions

The task is to reproduce the webpage in the reference screenshot as faithfully as possible by emitting a single self-contained HTML document. **Use the tools below to verify and correct your draft instead of relying on a one-shot guess** — a draft that "looks right" in your head almost always diverges from the reference once rendered.

### Required workflow

The trajectory must follow this shape; do not collapse steps:

1. **(Optional) Inspect the reference with vision tools first.** Call these *before* writing any HTML when they actually reduce ambiguity:
   - `crop` — when the screenshot is tall, dense, or has small text you can't read at thumbnail level. Crop a single region (header / hero / nav / cards / footer) and look at it in isolation.
   - `in_range_color` / `sample_color` — when you would otherwise guess a hex value for a `background-color`, brand accent, button fill, or border. Sample the actual pixels and lock the palette before writing CSS.
   - `enhance_contrast` / `detect_edges` — only when the layout edges or borders are genuinely hard to see; skip otherwise.
   Skip this step on visually simple pages — but explain in your thinking why you can skip it. Don't call these tools just to seem thorough.

2. **Write the first draft to a file with `write_file`.** Inline all CSS in a `<style>` block. Keep `rick.jpg` placeholders literally as-is. Use a clear filename (e.g. `index.html`).

3. **Call `render_html` on the file you just wrote.** Pass its `path` — `render_html` reads the HTML from disk, so you must `write_file` before the first `render_html` call. Whenever you can read the reference's pixel size off the original, pass it as `viewport_width` / `viewport_height` so layout breakpoints and full-page heights match. The tool returns the rendered screenshot — visually compare it to the reference end-to-end.

4. **Name the discrepancies concretely.** After every `render_html` call, in your thinking, list the specific diffs you can see — e.g. "nav links not horizontal", "card padding too small", "hero image is left-aligned but should be centered", "primary button is too saturated". If you cannot name any concrete diff, the draft is good enough — go to step 6.

5. **Patch the HTML with `edit_file`, then re-render the same file.** Prefer `edit_file` over rewriting the whole file with `write_file` — large rewrites destroy the parts that were already correct and waste tokens. After each patch, call `render_html` on the same `path` again (the tool always reads the latest contents from disk) and re-evaluate. Iterate steps 4–5 until either the rendered screenshot is visually consistent with the reference, or a further patch is no longer closing the gap.

6. **Submit the final HTML in the assistant message.** Quote the full HTML once, then stop.

### Pitfalls to avoid

- **At least one `render_html` call is required** for every trajectory — even on simple pages. The closed loop is the whole point.
- **Patch with `edit_file`, don't rewrite via `write_file`.** Rewriting the whole HTML between iterations destroys the parts that were already correct and wastes tokens; `write_file` is only for the initial draft.
- **Do not reference external assets** (CDN images, Google Fonts, remote stylesheets) — the sandbox can't reach them, the render will show broken images, and the next comparison will be misleading. Inline styles, keep `rick.jpg`-style placeholders verbatim, and rely on web-safe font stacks.
