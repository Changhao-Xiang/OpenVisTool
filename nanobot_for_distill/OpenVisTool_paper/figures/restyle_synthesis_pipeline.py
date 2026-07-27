from __future__ import annotations

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PALETTE = {
    "text": "#24313F",
    "muted": "#66717E",
    "line": "#CDD5DD",
    "stage1": "#3E638A",
    "stage1_fill": "#F2F6FA",
    "stage1_canvas": "#F8FAFC",
    "stage1_border": "#9EB3C7",
    "stage2": "#2F766D",
    "stage2_fill": "#F1F7F5",
    "stage2_canvas": "#F9FBFA",
    "stage2_border": "#9ABCB5",
    "stage3": "#8F5573",
    "stage3_fill": "#FAF4F7",
    "stage3_canvas": "#FCF9FB",
    "stage3_border": "#C7A6B7",
    "stage3_dark": "#653A50",
    "bar_neutral": "#B8C1CA",
    "white": "#FFFFFF",
}


COLOR_REPLACEMENTS = {
    # Stage 1: blue -> muted blue-gray.
    "#F8FBFF": PALETTE["stage1_canvas"],
    "#AFC8E8": PALETTE["stage1_border"],
    "#1F62B5": PALETTE["stage1"],
    "#1759A6": PALETTE["stage1"],
    "#2F6FCC": PALETTE["stage1"],
    "#2f6fcc": PALETTE["stage1"],
    "#315A8C": PALETTE["stage1"],
    "#314A68": PALETTE["text"],
    "#8EB1DF": PALETTE["stage1_border"],
    "#A8BFE1": PALETTE["stage1_border"],
    "#7F9FC6": PALETTE["stage1_border"],
    "#F2F6FC": PALETTE["stage1_fill"],
    "#EAF3FF": PALETTE["stage1_fill"],
    # Stage 2: green -> muted teal.
    "#F8FCF8": PALETTE["stage2_canvas"],
    "#A8CCAE": PALETTE["stage2_border"],
    "#23743A": PALETTE["stage2"],
    "#246B35": PALETTE["stage2"],
    "#315E38": PALETTE["stage2"],
    "#235F31": PALETTE["stage2"],
    "#1F5B2E": PALETTE["stage2"],
    "#3E7D49": PALETTE["stage2"],
    "#3E8A4B": PALETTE["stage2"],
    "#5A9865": PALETTE["stage2"],
    "#589a63": PALETTE["stage2"],
    "#A9C6AF": PALETTE["stage2_border"],
    "#78A883": PALETTE["stage2_border"],
    "#6DAA78": PALETTE["stage2_border"],
    "#7db387": PALETTE["stage2_border"],
    "#EEF7EF": PALETTE["stage2_fill"],
    "#eef7ef": PALETTE["stage2_fill"],
    "#F4F9F5": PALETTE["stage2_fill"],
    "#F2F8F3": PALETTE["stage2_fill"],
    # Stage 3: purple -> muted ochre.
    "#FBFAFE": PALETTE["stage3_canvas"],
    "#B8ADD1": PALETTE["stage3_border"],
    "#B9ADCF": PALETTE["stage3_border"],
    "#A18BC2": PALETTE["stage3_border"],
    "#8B78B8": PALETTE["stage3_border"],
    "#6D5AA7": PALETTE["stage3"],
    "#5A4785": PALETTE["stage3"],
    "#553F79": PALETTE["stage3"],
    "#503D78": PALETTE["stage3"],
    "#554278": PALETTE["stage3"],
    "#4A386E": PALETTE["stage3"],
    "#7562A1": PALETTE["stage3"],
    "#7762A3": PALETTE["stage3"],
    "#8B75B3": PALETTE["stage3"],
    "#433365": PALETTE["stage3_dark"],
    "#F1EDF9": PALETTE["stage3_fill"],
    "#F7F4FC": PALETTE["stage3_fill"],
    "#FCFBFE": PALETTE["stage3_fill"],
    "#EEE9F7": PALETTE["stage3_fill"],
    "#E9E4F5": PALETTE["stage3_fill"],
    # Neutral structure and text.
    "#CFD9D2": PALETTE["line"],
    "#D5E2D7": PALETTE["line"],
    "#22324A": PALETTE["text"],
    "#263B54": PALETTE["text"],
    "#4F6754": PALETTE["muted"],
    "#405748": PALETTE["muted"],
    "#5D6B62": PALETTE["muted"],
    "#58718F": PALETTE["muted"],
    "#6E677C": PALETTE["muted"],
    "#65717C": PALETTE["muted"],
    "#8E8A98": PALETTE["muted"],
    "#B6BCC6": PALETTE["bar_neutral"],
    "#ededed": PALETTE["line"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the publication palette and typography to synthesis_pipeline.drawio."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font",
        choices=("Helvetica", "Times New Roman"),
        default="Helvetica",
    )
    return parser.parse_args()


def replace_colors(value: str) -> str:
    for old, new in COLOR_REPLACEMENTS.items():
        value = value.replace(old, new)
    return value.replace("rgb(49, 74, 104)", "rgb(36, 49, 63)")


def set_style(cell: ET.Element, **properties: str | int | float) -> None:
    parts = [part for part in cell.get("style", "").split(";") if part]
    positions: dict[str, int] = {}
    for index, part in enumerate(parts):
        if "=" in part:
            positions[part.split("=", 1)[0]] = index

    for key, raw_value in properties.items():
        value = str(raw_value)
        item = f"{key}={value}"
        if key in positions:
            parts[positions[key]] = item
        else:
            positions[key] = len(parts)
            parts.append(item)
    cell.set("style", ";".join(parts) + ";")


def set_geometry(
    cell: ET.Element,
    *,
    x: float | None = None,
    y: float | None = None,
    width: float | None = None,
    height: float | None = None,
) -> None:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        raise ValueError(f"Cell {cell.get('id')} has no mxGeometry")
    for key, value in (
        ("x", x),
        ("y", y),
        ("width", width),
        ("height", height),
    ):
        if value is not None:
            geometry.set(key, f"{value:g}")


def set_explicit_edge_route(
    cell: ET.Element,
    *,
    source_point: tuple[float, float],
    target_point: tuple[float, float],
    points: tuple[tuple[float, float], ...],
) -> None:
    # Cross-group connectors can inherit a misleading terminal direction from
    # target perimeter routing. Explicit endpoints make the final segment
    # deterministic and keep the arrowhead pointing horizontally to the right.
    cell.attrib.pop("source", None)
    cell.attrib.pop("target", None)
    geometry = cell.find("mxGeometry")
    if geometry is None:
        raise ValueError(f"Cell {cell.get('id')} has no mxGeometry")
    for child in list(geometry):
        geometry.remove(child)
    ET.SubElement(
        geometry,
        "mxPoint",
        {
            "x": f"{source_point[0]:g}",
            "y": f"{source_point[1]:g}",
            "as": "sourcePoint",
        },
    )
    ET.SubElement(
        geometry,
        "mxPoint",
        {
            "x": f"{target_point[0]:g}",
            "y": f"{target_point[1]:g}",
            "as": "targetPoint",
        },
    )
    route = ET.SubElement(geometry, "Array", {"as": "points"})
    for x, y in points:
        ET.SubElement(route, "mxPoint", {"x": f"{x:g}", "y": f"{y:g}"})


def require_cells(root: ET.Element) -> dict[str, ET.Element]:
    cells = {cell.get("id", ""): cell for cell in root.iter("mxCell")}
    required = {
        "s1-bg",
        "s1-title",
        "source-pool",
        "hard-test",
        "keep-query",
        "s2-bg",
        "s2-title",
        "teacher-agent-title",
        "lane-d3-bg",
        "lane-b1-bg",
        "s3-bg",
        "s3-title",
        "candidate",
        "dataset",
    }
    missing = sorted(required - cells.keys())
    if missing:
        raise ValueError(f"Unexpected diagram schema; missing cells: {', '.join(missing)}")
    return cells


def update_typography_and_text(cells: dict[str, ET.Element], font: str) -> None:
    # Enforce one AAAI-compatible family for every call-out.
    for cell in cells.values():
        style = cell.get("style", "")
        if "fontFamily=" in style:
            set_style(cell, fontFamily=font)

    values = {
        "source-pool": (
            f'<b><font color="{PALETTE["stage1"]}" style="font-size:18px">'
            "Source pool</font></b>"
        ),
        "pcsyo0guieg9A1sTuubV-4": (
            '<i><font style="font-size:16px">Five visual domains</font></i>'
        ),
        "baseline": (
            '<b>No-tool baseline</b><br>'
            '<font style="font-size:15px;font-weight:normal">Image + query</font>'
        ),
        "hard-test": (
            '<b>Hard without tools?</b><br>'
            '<font style="font-size:18px">p<sub>0</sub> ≤ 0.5</font>'
        ),
        "keep-query": (
            '<b>Keep query</b><br>'
            '<font style="font-size:15px;font-weight:normal">'
            'cache baseline p<sub>0</sub></font>'
        ),
        "s2-subtitle": (
            '<i><font style="font-size:16px">'
            "Shared tools · domain-conditioned evidence</font></i>"
        ),
        "query-header": "Input",
        "tool-header": "Tool use",
        "obs-header": "Observation &amp; answer",
        "d3-label": (
            "<b>Chart / Table</b><br>"
            '<font style="font-size:16px;font-weight:normal">'
            "isolate &amp; compute</font>"
        ),
        "d3-query": (
            "<b>Query</b><br>"
            '<font style="font-size:16px">'
            "Growth ratio:<br>Children’s /<br>(Poetry + Academic)</font>"
        ),
        "d3-color": "color_range",
        "d3-compute": "compute",
        "d3-childrens-caption": "Children’s <b>+50</b>",
        "d3-academic-caption": "Poetry <b>+45</b> · Academic <b>+45</b>",
        "LwsBief1fKGYkRnShY69-2": (
            '<font style="font-size:18px">'
            "Answer: 50 / (45 + 45) = 0.56</font>"
        ),
        "b1-label": (
            "<b>GUI / Search</b><br>"
            '<font style="font-size:16px;font-weight:normal">'
            "localize &amp; act</font>"
        ),
        "b1-query": (
            "<b>Query</b><br>"
            '<font style="font-size:17px">Click<br>'
            "Edit Details…</font>"
        ),
        "b1-answer": (
            '<font style="font-size:17px">'
            "computer_use · left_click · [353, 453]</font>"
        ),
        "teacher-agent-title": "<b>Teacher agent loop</b>",
        "teacher-agent-tool": "<b>Tool call</b>",
        "s3-subtitle": (
            '<i><font style="font-size:16px">Both tests required</font></i>'
        ),
        "retain": "<b>Instructive trace</b>",
        "causal-gain-delta": "<b>Δp ≥ δ</b>",
    }
    for cell_id, value in values.items():
        cells[cell_id].set("value", value)

    # Titles and headers: readable after full-width AAAI scaling without dominating.
    set_style(cells["s1-title"], fontSize=23, fontColor=PALETTE["stage1"])
    set_style(cells["s2-title"], fontSize=24, fontColor=PALETTE["stage2"])
    set_style(cells["s3-title"], fontSize=23, fontColor=PALETTE["stage3"])
    for cell_id in ("pcsyo0guieg9A1sTuubV-4", "s2-subtitle", "s3-subtitle"):
        set_style(cells[cell_id], fontSize=16, fontColor=PALETTE["muted"])
    for cell_id in ("query-header", "tool-header", "obs-header"):
        set_style(cells[cell_id], fontSize=18, fontColor=PALETTE["text"])

    # Body text is neutral; stage colors remain reserved for hierarchy and flow.
    neutral_text = (
        "dom-chart",
        "dom-table",
        "dom-gui",
        "dom-search",
        "dom-web",
        "baseline",
        "d3-label",
        "d3-query",
        "d3-childrens-caption",
        "d3-academic-caption",
        "b1-label",
        "b1-query",
        "teacher-agent-reason",
        "teacher-agent-tool",
        "teacher-agent-observe",
        "candidate",
        "both-panel",
        "outcome-test",
        "causal-card",
    )
    for cell_id in neutral_text:
        set_style(cells[cell_id], fontColor=PALETTE["text"])

    for cell_id in ("d3-color", "d3-compute", "b1-crop", "b1-bbox"):
        set_style(cells[cell_id], fontSize=16, fontColor=PALETTE["text"])

    set_style(cells["hard-test"], fontSize=17, fontColor=PALETTE["text"])
    set_style(cells["keep-query"], fontSize=17, fontColor=PALETTE["text"])
    set_style(cells["baseline"], fontSize=17)
    set_style(cells["d3-label"], fontSize=18)
    set_style(cells["b1-label"], fontSize=18)
    set_style(cells["d3-query"], fontSize=17)
    set_style(cells["b1-query"], fontSize=17)
    set_style(cells["teacher-agent-title"], fontSize=16, fontColor=PALETTE["text"])
    for cell_id in ("teacher-agent-reason", "teacher-agent-tool", "teacher-agent-observe"):
        set_style(cells[cell_id], fontSize=15)
    set_style(cells["candidate"], fontSize=18)
    set_style(cells["both-panel"], fontSize=16)
    set_style(cells["outcome-test"], fontSize=17)
    set_style(cells["causal-card"], fontSize=17)
    set_style(cells["retain"], fontSize=17, fontColor=PALETTE["text"])
    set_style(cells["dataset"], fontSize=16)
    set_style(cells["causal-gain-delta"], fontSize=14, fontColor=PALETTE["stage3"])

    # Times New Roman has a smaller x-height than Helvetica. A one-pixel
    # compensation keeps small call-outs equally legible at AAAI full-width scale.
    if font == "Times New Roman":
        for cell in cells.values():
            style = cell.get("style", "")
            if "fontFamily=Times New Roman" not in style:
                continue
            matches = re.findall(r"(?:^|;)fontSize=([0-9]+(?:\.[0-9]+)?)", style)
            if matches:
                set_style(cell, fontSize=f"{float(matches[-1]) + 1:g}")
            value = cell.get("value", "")
            value = re.sub(
                r"font-size:([0-9]+)px",
                lambda match: f"font-size:{int(match.group(1)) + 1}px",
                value,
            )
            cell.set("value", value)


def update_palette(cells: dict[str, ET.Element]) -> None:
    for cell in cells.values():
        cell.set("style", replace_colors(cell.get("style", "")))
        cell.set("value", replace_colors(cell.get("value", "")))

    # Large-area canvases stay almost white; stronger tints are used on local cards.
    set_style(
        cells["s1-bg"],
        fillColor=PALETTE["stage1_canvas"],
        strokeColor=PALETTE["stage1_border"],
    )
    set_style(
        cells["s2-bg"],
        fillColor=PALETTE["stage2_canvas"],
        strokeColor=PALETTE["stage2_border"],
    )
    set_style(
        cells["s3-bg"],
        fillColor=PALETTE["stage3_canvas"],
        strokeColor=PALETTE["stage3_border"],
    )

    for cell_id in ("s1-badge",):
        set_style(cells[cell_id], fillColor=PALETTE["stage1"])
    for cell_id in ("s2-badge",):
        set_style(cells[cell_id], fillColor=PALETTE["stage2"])
    for cell_id in ("s3-badge",):
        set_style(cells[cell_id], fillColor=PALETTE["stage3"])

    for cell_id in (
        "lane-d3-bg",
        "lane-b1-bg",
        "lane-d3-qcard",
        "lane-b1-qcard",
        "lane-d3-ocard",
        "lane-b1-ocard",
    ):
        set_style(cells[cell_id], fillColor=PALETTE["white"], strokeColor=PALETTE["line"])

    for cell_id in ("lane-d3-label-bg", "lane-b1-label-bg"):
        set_style(cells[cell_id], fillColor=PALETTE["stage2_fill"])
    for cell_id in ("lane-d3-tcard", "lane-b1-tcard", "teacher-agent-bg"):
        set_style(
            cells[cell_id],
            fillColor=PALETTE["stage2_fill"],
            strokeColor=PALETTE["stage2_border"],
        )
    for cell_id in (
        "d3-color",
        "d3-compute",
        "b1-crop",
        "b1-bbox",
        "teacher-agent-reason",
        "teacher-agent-tool",
        "teacher-agent-observe",
    ):
        set_style(
            cells[cell_id],
            fillColor=PALETTE["white"],
            strokeColor=PALETTE["stage2_border"],
        )

    set_style(
        cells["LwsBief1fKGYkRnShY69-2"],
        fillColor=PALETTE["stage2_fill"],
        strokeColor=PALETTE["stage2_border"],
        fontColor=PALETTE["stage2"],
    )
    set_style(
        cells["b1-answer"],
        fillColor=PALETTE["stage2_fill"],
        strokeColor=PALETTE["stage2_border"],
        fontColor=PALETTE["stage2"],
    )

    for cell_id in ("candidate", "outcome-test", "causal-card", "retain", "and-badge"):
        set_style(
            cells[cell_id],
            fillColor=PALETTE["stage3_fill"],
            strokeColor=PALETTE["stage3_border"],
        )
    set_style(cells["both-panel"], fillColor=PALETTE["white"], strokeColor=PALETTE["stage3_border"])
    set_style(
        cells["dataset"],
        fillColor=PALETTE["stage3"],
        strokeColor=PALETTE["stage3_dark"],
        fontColor=PALETTE["white"],
    )
    set_style(cells["bar-noobs"], fillColor=PALETTE["bar_neutral"])
    set_style(cells["bar-obs"], fillColor=PALETTE["stage3"])


def update_geometry(cells: dict[str, ET.Element]) -> None:
    # Shorter copy lets the first-stage flow breathe and reduces arrow runs.
    set_geometry(cells["baseline"], y=400, height=70)
    set_geometry(cells["hard-test"], y=500, height=86)
    set_geometry(cells["keep-query"], y=636, height=68)

    # Keep the center title clear of the teacher-loop panel.
    set_geometry(cells["s2-title"], width=485)
    set_geometry(cells["s2-subtitle"], width=485, y=58)
    set_geometry(cells["query-header"], y=147, height=28)
    set_geometry(cells["tool-header"], y=147, height=28)
    set_geometry(cells["obs-header"], y=147, height=28)

    # Compact tool stacks and captions without reducing the evidence panels.
    for cell_id in ("d3-color", "b1-crop"):
        set_geometry(cells[cell_id], y=58, height=48)
    for cell_id in ("d3-compute", "b1-bbox"):
        set_geometry(cells[cell_id], y=154, height=48)
    set_geometry(cells["d3-childrens-caption"], y=139, height=44)
    set_geometry(cells["d3-academic-caption"], y=139, height=44)

    # The third-stage subtitle now fits on one line; use the space for a cleaner flow.
    set_geometry(cells["s3-subtitle"], y=87, height=28)
    set_geometry(cells["candidate"], y=145, height=60)
    set_geometry(cells["both-panel"], y=235, height=315)
    set_style(cells["both-panel"], align="center", spacingLeft=0, spacingRight=0)
    set_geometry(cells["outcome-test"], y=300, height=58)
    set_geometry(cells["and-badge"], y=367)
    set_geometry(cells["causal-card"], y=405, height=130)
    set_geometry(cells["retain"], y=578, height=58)
    set_geometry(cells["causal-gain-delta"], x=158, y=454, width=58, height=24)


def update_arrows(cells: dict[str, ET.Element]) -> None:
    stage1_edges = ("s1-pool-base", "s1-base-hard", "s1-hard-keep")
    lane_edges = (
        "lane-d3-q-to-t",
        "lane-d3-t-to-o",
        "lane-b1-q-to-t",
        "lane-b1-t-to-o",
    )
    tool_edges = ("d3-color-compute", "b1-crop-bbox")
    teacher_edges = (
        "teacher-agent-reason-tool",
        "teacher-agent-tool-observe",
        "teacher-agent-return",
    )
    verification_edges = (
        "candidate-both",
        "both-retain",
        "retain-dataset",
        "causal-gain-arrow",
    )

    for cell_id in stage1_edges:
        set_style(
            cells[cell_id],
            strokeColor=PALETTE["stage1"],
            strokeWidth=2,
            endSize=7,
        )
    set_style(
        cells["s1-hard-keep"],
        fontColor=PALETTE["stage1"],
        fontSize=15,
    )
    for cell_id in lane_edges:
        set_style(
            cells[cell_id],
            strokeColor=PALETTE["stage2"],
            strokeWidth=1.8,
            endSize=7,
        )
    for cell_id in tool_edges:
        set_style(
            cells[cell_id],
            strokeColor=PALETTE["stage2"],
            strokeWidth=1.7,
            endSize=7,
        )
    for cell_id in teacher_edges:
        set_style(
            cells[cell_id],
            strokeColor=PALETTE["stage2"],
            strokeWidth=1.7,
            endSize=6,
        )
    for cell_id in verification_edges:
        set_style(
            cells[cell_id],
            strokeColor=PALETTE["stage3"],
            strokeWidth=1.8,
            endSize=8,
        )
    set_style(cells["causal-gain-guide"], strokeColor=PALETTE["stage3"], strokeWidth=1.5)
    set_style(cells["bar-axis"], strokeColor=PALETTE["muted"], strokeWidth=1.2)
    set_style(
        cells["stage1-to-stage2"],
        strokeColor=PALETTE["stage2"],
        strokeWidth=2,
        endSize=8,
    )
    set_style(
        cells["stage2-to-stage3"],
        strokeColor=PALETTE["stage3"],
        strokeWidth=2,
        endSize=8,
        endArrow="block",
        endFill=1,
    )
    set_explicit_edge_route(
        cells["stage2-to-stage3"],
        source_point=(1339, 420),
        target_point=(1387, 185),
        points=((1350, 420), (1350, 185)),
    )


def main() -> None:
    args = parse_args()
    tree = ET.parse(args.input)
    root = tree.getroot()
    cells = require_cells(root)

    update_palette(cells)
    update_typography_and_text(cells, args.font)
    update_geometry(cells)
    update_arrows(cells)

    mxfile = root
    mxfile.set("agent", f"Codex · {args.font} publication palette")
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=False)
    print(f"Wrote {args.output} ({args.font})")


if __name__ == "__main__":
    main()
