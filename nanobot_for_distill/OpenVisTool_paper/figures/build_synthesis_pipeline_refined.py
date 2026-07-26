from __future__ import annotations

import base64
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFile, ImageOps


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
CANDIDATES = PAPER / "figure_case_candidates"
ASSET_DIR = HERE / "synthesis_pipeline_assets"
OUTPUT = HERE / "synthesis_pipeline_refined.drawio"

FONT = "fontFamily=Helvetica;"
MONO = "fontFamily=Courier New;"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_asset(
    source: Path,
    name: str,
    size: tuple[int, int],
    *,
    crop: tuple[int, int, int, int] | None = None,
    fit: bool = False,
    quality: int = 91,
) -> Path:
    """Create a compact, figure-ready derivative while preserving the real tool output."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSET_DIR / name
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if crop is not None:
                image = image.crop(crop)

            if fit:
                canvas = ImageOps.fit(
                    image,
                    size,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
            else:
                contained = ImageOps.contain(
                    image,
                    (size[0] - 10, size[1] - 10),
                    method=Image.Resampling.LANCZOS,
                )
                canvas = Image.new("RGB", size, "white")
                canvas.paste(
                    contained,
                    ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
                )

            draw = ImageDraw.Draw(canvas)
            draw.rectangle(
                (0, 0, size[0] - 1, size[1] - 1),
                outline=(203, 213, 222),
                width=2,
            )
            canvas.save(out, "JPEG", quality=quality, optimize=True, subsampling=0)
    except OSError:
        # Candidate bundles may contain unavailable/zero-byte source panels.
        # Preserve a previously generated derivative when it is already present.
        if not out.exists() or out.stat().st_size == 0:
            raise
    return out


def data_uri(path: Path) -> str:
    # diagrams.net stores embedded raster payloads as data:image/<type>,BASE64.
    return "data:image/jpeg," + base64.b64encode(path.read_bytes()).decode("ascii")


def geometry(
    parent: ET.Element,
    *,
    x: float | None = None,
    y: float | None = None,
    width: float | None = None,
    height: float | None = None,
    relative: bool = False,
) -> ET.Element:
    attrs: dict[str, str] = {"as": "geometry"}
    if x is not None:
        attrs["x"] = f"{x:g}"
    if y is not None:
        attrs["y"] = f"{y:g}"
    if width is not None:
        attrs["width"] = f"{width:g}"
    if height is not None:
        attrs["height"] = f"{height:g}"
    if relative:
        attrs["relative"] = "1"
    return ET.SubElement(parent, "mxGeometry", attrs)


def group(
    root: ET.Element,
    cell_id: str,
    parent_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": "",
            "style": "group;",
            "vertex": "1",
            "connectable": "0",
            "parent": parent_id,
        },
    )
    geometry(cell, x=x, y=y, width=width, height=height)
    return cell


def vertex(
    root: ET.Element,
    cell_id: str,
    parent_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    style: str,
    value: str = "",
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": parent_id,
        },
    )
    geometry(cell, x=x, y=y, width=width, height=height)
    return cell


def edge(
    root: ET.Element,
    cell_id: str,
    parent_id: str,
    *,
    source: str | None = None,
    target: str | None = None,
    value: str = "",
    style: str,
    points: list[tuple[float, float]] | None = None,
    source_point: tuple[float, float] | None = None,
    target_point: tuple[float, float] | None = None,
) -> ET.Element:
    attrs = {
        "id": cell_id,
        "value": value,
        "style": style,
        "edge": "1",
        "parent": parent_id,
    }
    if source is not None:
        attrs["source"] = source
    if target is not None:
        attrs["target"] = target
    cell = ET.SubElement(root, "mxCell", attrs)
    geo = geometry(cell, relative=True)
    if source_point is not None:
        ET.SubElement(
            geo,
            "mxPoint",
            {"x": f"{source_point[0]:g}", "y": f"{source_point[1]:g}", "as": "sourcePoint"},
        )
    if target_point is not None:
        ET.SubElement(
            geo,
            "mxPoint",
            {"x": f"{target_point[0]:g}", "y": f"{target_point[1]:g}", "as": "targetPoint"},
        )
    if points:
        array = ET.SubElement(geo, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(array, "mxPoint", {"x": f"{px:g}", "y": f"{py:g}"})
    return cell


def image_vertex(
    root: ET.Element,
    cell_id: str,
    parent_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    image_path: Path,
) -> ET.Element:
    return vertex(
        root,
        cell_id,
        parent_id,
        x,
        y,
        width,
        height,
        (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;strokeColor=none;fillColor=none;"
            f"image={data_uri(image_path)};"
        ),
    )


def build_assets() -> dict[str, Path]:
    d3 = CANDIDATES / "04_chart_in_range_color" / "D3_chart_book_growth"
    b1 = CANDIDATES / "02_gui_visual_search" / "B1_gui_calendar_button"
    c3 = CANDIDATES / "03_web_to_html" / "C3_web_bookstore"

    return {
        "chart_input": make_asset(
            d3 / "paper_card.jpg",
            "d3_chart_input.jpg",
            (720, 356),
            crop=(58, 288, 478, 520),
            fit=True,
        ),
        "childrens_books": make_asset(
            d3 / "panel_2_children_s_books.jpg",
            "d3_childrens_books.jpg",
            (560, 277),
            crop=(0, 95, 2781, 1475),
            fit=True,
        ),
        "academic_texts": make_asset(
            d3 / "panel_3_academic_texts.jpg",
            "d3_academic_texts.jpg",
            (560, 277),
            crop=(0, 95, 2781, 1475),
            fit=True,
        ),
        "gui_input": make_asset(
            b1 / "panel_1_gui_input.png",
            "b1_gui_input.jpg",
            (700, 470),
            crop=(380, 150, 1120, 650),
            fit=True,
        ),
        "gui_bbox": make_asset(
            b1 / "panel_3_bbox_verification.jpg",
            "b1_gui_bbox.jpg",
            (850, 350),
            crop=(500, 305, 1050, 540),
            fit=True,
        ),
        "web_reference": make_asset(
            c3 / "panel_1_reference.png",
            "c3_web_reference.jpg",
            (720, 405),
            crop=(0, 0, 1280, 640),
            fit=True,
        ),
        "web_draft": make_asset(
            c3 / "panel_2_first_render.png",
            "c3_web_draft.jpg",
            (300, 430),
            fit=False,
        ),
        "web_final": make_asset(
            c3 / "panel_3_revised_render.png",
            "c3_web_final.jpg",
            (720, 405),
            crop=(0, 0, 1000, 565),
            fit=True,
        ),
    }


def build_drawio(assets: dict[str, Path]) -> None:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-07-26T00:00:00.000Z",
            "agent": "Codex",
            "version": "26.0.9",
            "type": "device",
        },
    )
    diagram = ET.SubElement(
        mxfile,
        "diagram",
        {"id": "openvistool-method-refined", "name": "Method Overview — Real Cases"},
    )
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1660",
            "dy": "900",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1660",
            "pageHeight": "900",
            "math": "0",
            "shadow": "0",
            "background": "#FFFFFF",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    # Shared styles.
    stage_frame = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=3;strokeWidth=2;"
        "shadow=0;"
    )
    title = (
        "text;html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;fontSize=30;fontStyle=1;" + FONT
    )
    subtitle = (
        "text;html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;fontSize=18;fontStyle=2;" + FONT
    )
    text = (
        "text;html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;fontSize=20;" + FONT
    )
    centered = (
        "text;html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontSize=20;" + FONT
    )
    card = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=7;strokeWidth=1.5;"
        "fillColor=#FFFFFF;strokeColor=#CFD9D2;" + FONT
    )
    tool_card = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=7;strokeWidth=1.5;"
        "fillColor=#F4F9F5;strokeColor=#A9C6AF;" + FONT
    )
    tool_chip = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=15;strokeWidth=1.5;"
        "fillColor=#FFFFFF;strokeColor=#78A883;fontColor=#235F31;"
        "fontSize=18;fontStyle=1;" + MONO
    )
    flow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
        "endFill=1;strokeColor=#3E8A4B;strokeWidth=2.5;"
    )
    thin_flow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
        "endFill=1;strokeColor=#5A9865;strokeWidth=2;"
    )

    # Stage groups are real groups so the layout remains editable.
    group(root, "stage1", "1", 12, 12, 250, 876)
    group(root, "stage2", "1", 276, 12, 1094, 876)
    group(root, "stage3", "1", 1384, 12, 264, 876)

    # Stage 1 — Difficulty screening.
    vertex(
        root,
        "s1-bg",
        "stage1",
        0,
        0,
        250,
        876,
        stage_frame + "fillColor=#F8FBFF;strokeColor=#AFC8E8;",
    )
    vertex(
        root,
        "s1-badge",
        "stage1",
        14,
        16,
        40,
        40,
        "ellipse;html=1;aspect=fixed;fillColor=#1F62B5;strokeColor=none;"
        "fontColor=#FFFFFF;fontSize=24;fontStyle=1;" + FONT,
        "1",
    )
    vertex(
        root,
        "s1-title",
        "stage1",
        62,
        8,
        174,
        66,
        title + "fontColor=#1759A6;",
        "Difficulty<br>Screening",
    )
    vertex(
        root,
        "s1-subtitle",
        "stage1",
        20,
        82,
        210,
        32,
        subtitle + "fontColor=#58718F;",
        "Five visual domains",
    )
    vertex(
        root,
        "source-pool",
        "stage1",
        18,
        125,
        214,
        215,
        card + "strokeColor=#8EB1DF;verticalAlign=top;spacingTop=12;fontSize=18;",
        '<b><font color="#1759A6">Source pool</font></b>',
    )
    domain_style = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#F2F6FC;"
        "strokeColor=#A8BFE1;fontColor=#315A8C;fontSize=15;" + FONT
    )
    vertex(root, "dom-chart", "stage1", 30, 165, 90, 48, domain_style, "Chart")
    vertex(root, "dom-table", "stage1", 130, 165, 90, 48, domain_style, "Table")
    vertex(root, "dom-gui", "stage1", 30, 225, 90, 52, domain_style, "GUI<br>grounding")
    vertex(root, "dom-search", "stage1", 130, 225, 90, 52, domain_style, "Visual<br>search")
    vertex(root, "dom-web", "stage1", 58, 290, 134, 38, domain_style, "Web-to-HTML")
    vertex(
        root,
        "baseline",
        "stage1",
        32,
        375,
        186,
        76,
        card
        + "fillColor=#FFFFFF;strokeColor=#7F9FC6;fontColor=#263B54;"
        "fontSize=20;fontStyle=1;",
        'No-tool baseline<br><font style="font-size:16px;font-weight:normal">same query + image</font>',
    )
    vertex(
        root,
        "hard-test",
        "stage1",
        32,
        495,
        186,
        94,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#EAF3FF;"
        "strokeColor=#2F6FCC;strokeWidth=2;fontColor=#1759A6;"
        "fontSize=20;fontStyle=1;" + FONT,
        'Hard without tools?<br><font style="font-size:22px">p₀ ≤ 0.5</font>',
    )
    vertex(
        root,
        "keep-query",
        "stage1",
        25,
        650,
        200,
        86,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#EAF3FF;"
        "strokeColor=#2F6FCC;strokeWidth=2;fontColor=#1759A6;"
        "fontSize=20;fontStyle=1;" + FONT,
        'Keep query<br><font style="font-size:16px;font-weight:normal">cache baseline p₀</font>',
    )
    vertex(
        root,
        "drop-query",
        "stage1",
        55,
        795,
        140,
        46,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#FAFAFA;"
        "strokeColor=#A9A9A9;dashed=1;fontColor=#777777;fontSize=16;" + FONT,
        "Easy → drop",
    )
    edge(root, "s1-pool-base", "stage1", source="source-pool", target="baseline", style=flow)
    edge(root, "s1-base-hard", "stage1", source="baseline", target="hard-test", style=flow)
    edge(
        root,
        "s1-hard-keep",
        "stage1",
        source="hard-test",
        target="keep-query",
        value="Yes",
        style=flow + "fontColor=#1759A6;fontSize=16;" + FONT,
    )
    edge(
        root,
        "s1-hard-drop",
        "stage1",
        source="hard-test",
        target="drop-query",
        value="No",
        style=(
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
            "strokeColor=#999999;strokeWidth=1.2;dashed=1;fontColor=#777777;"
            "fontSize=16;" + FONT
        ),
        points=[(236, 610), (236, 818)],
    )

    # Stage 2 — Real domain-specific trajectories.
    vertex(
        root,
        "s2-bg",
        "stage2",
        0,
        0,
        1094,
        876,
        stage_frame + "fillColor=#F8FCF8;strokeColor=#A8CCAE;",
    )
    vertex(
        root,
        "s2-badge",
        "stage2",
        16,
        16,
        40,
        40,
        "ellipse;html=1;aspect=fixed;fillColor=#23743A;strokeColor=none;"
        "fontColor=#FFFFFF;fontSize=24;fontStyle=1;" + FONT,
        "2",
    )
    vertex(
        root,
        "s2-title",
        "stage2",
        64,
        8,
        880,
        48,
        title + "fontColor=#246B35;",
        "Domain-Specific Tool Use",
    )
    vertex(
        root,
        "s2-subtitle",
        "stage2",
        64,
        55,
        950,
        30,
        subtitle + "fontColor=#4F6754;",
        "Real trajectories · shared tools · domain-conditioned evidence acquisition",
    )
    column_header = (
        "text;html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontColor=#315E38;fontSize=20;fontStyle=1;" + FONT
    )
    vertex(root, "query-header", "stage2", 98, 90, 286, 26, column_header, "Query + Image")
    vertex(root, "tool-header", "stage2", 396, 90, 190, 26, column_header, "Tool pattern")
    vertex(
        root,
        "obs-header",
        "stage2",
        598,
        90,
        454,
        26,
        column_header,
        "Observation + Answer",
    )

    for lane_id, y in (("lane-d3", 120), ("lane-b1", 365), ("lane-c3", 610)):
        group(root, lane_id, "stage2", 14, y, 1066, 230)
        vertex(
            root,
            f"{lane_id}-bg",
            lane_id,
            0,
            0,
            1066,
            230,
            "rounded=1;whiteSpace=wrap;html=1;arcSize=4;fillColor=#FFFFFF;"
            "strokeColor=#D5E2D7;strokeWidth=1;" + FONT,
        )
        vertex(
            root,
            f"{lane_id}-label-bg",
            lane_id,
            0,
            0,
            88,
            230,
            "rounded=1;whiteSpace=wrap;html=1;arcSize=4;fillColor=#EEF7EF;"
            "strokeColor=none;" + FONT,
        )
        vertex(root, f"{lane_id}-qcard", lane_id, 98, 15, 286, 200, card)
        vertex(root, f"{lane_id}-tcard", lane_id, 396, 15, 190, 200, tool_card)
        vertex(root, f"{lane_id}-ocard", lane_id, 598, 15, 454, 200, card)
        edge(
            root,
            f"{lane_id}-q-to-t",
            lane_id,
            source=f"{lane_id}-qcard",
            target=f"{lane_id}-tcard",
            style=flow,
        )
        edge(
            root,
            f"{lane_id}-t-to-o",
            lane_id,
            source=f"{lane_id}-tcard",
            target=f"{lane_id}-ocard",
            style=flow,
        )

    # D3: isolate three color-coded series, then compare endpoint growth.
    vertex(
        root,
        "d3-label",
        "lane-d3",
        5,
        52,
        78,
        126,
        centered + "fontColor=#246B35;fontSize=21;fontStyle=1;",
        'Chart<br><font style="font-size:17px;font-weight:normal">isolate series<br>compute ratio</font>',
    )
    image_vertex(root, "d3-input-img", "lane-d3", 108, 72, 140, 82, assets["chart_input"])
    vertex(
        root,
        "d3-query",
        "lane-d3",
        255,
        35,
        118,
        165,
        text + "fontColor=#22324A;fontSize=20;",
        '<font color="#58718F" style="font-size:17px"><b>Query</b></font><br>'
        '<b>Growth ratio</b><br>(2013–22):<br>Children’s /<br>(Poetry +<br>Academic)',
    )
    vertex(
        root,
        "d3-color",
        "lane-d3",
        413,
        42,
        156,
        60,
        tool_chip + "fontSize=18;",
        'in_range_color<br><font style="font-size:17px">× 3 series</font>',
    )
    vertex(root, "d3-compute", "lane-d3", 420, 142, 142, 46, tool_chip, "compute ratio")
    edge(root, "d3-color-compute", "lane-d3", source="d3-color", target="d3-compute", style=thin_flow)
    image_vertex(
        root,
        "d3-childrens-img",
        "lane-d3",
        610,
        43,
        146,
        90,
        assets["childrens_books"],
    )
    image_vertex(
        root,
        "d3-academic-img",
        "lane-d3",
        764,
        43,
        146,
        90,
        assets["academic_texts"],
    )
    vertex(
        root,
        "d3-childrens-caption",
        "lane-d3",
        606,
        137,
        154,
        52,
        centered + "fontColor=#405748;fontSize=17;",
        'Children’s<br><b>+50</b>',
    )
    vertex(
        root,
        "d3-academic-caption",
        "lane-d3",
        760,
        137,
        154,
        52,
        centered + "fontColor=#405748;fontSize=16;",
        'Poetry <b>+45</b><br>Academic <b>+45</b>',
    )
    vertex(
        root,
        "d3-answer",
        "lane-d3",
        926,
        50,
        118,
        130,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#EEF7EF;"
        "strokeColor=#6DAA78;strokeWidth=1.5;fontColor=#1F5B2E;fontSize=22;" + FONT,
        '<font style="font-size:17px"><b>Answer</b></font><br>'
        '<b>50 /<br>(45 + 45)<br>= 0.55</b>',
    )

    # B1: linear localization and action.
    vertex(
        root,
        "b1-label",
        "lane-b1",
        5,
        52,
        78,
        126,
        centered + "fontColor=#246B35;fontSize=20;fontStyle=1;",
        'GUI / Search<br><font style="font-size:17px;font-weight:normal">localize<br>then act</font>',
    )
    image_vertex(root, "b1-input-img", "lane-b1", 108, 66, 140, 94, assets["gui_input"])
    vertex(
        root,
        "b1-query",
        "lane-b1",
        255,
        45,
        118,
        140,
        text + "fontColor=#22324A;fontSize=21;",
        '<font color="#58718F" style="font-size:17px"><b>Query</b></font><br>'
        'Click<br><b>“Edit Details…”</b>',
    )
    vertex(root, "b1-crop", "lane-b1", 421, 30, 140, 40, tool_chip, "crop")
    vertex(root, "b1-bbox", "lane-b1", 413, 91, 156, 40, tool_chip, "draw_bbox")
    vertex(root, "b1-click", "lane-b1", 407, 152, 168, 40, tool_chip, "computer_use")
    edge(root, "b1-crop-bbox", "lane-b1", source="b1-crop", target="b1-bbox", style=thin_flow)
    edge(root, "b1-bbox-click", "lane-b1", source="b1-bbox", target="b1-click", style=thin_flow)
    image_vertex(root, "b1-output-img", "lane-b1", 610, 52, 305, 126, assets["gui_bbox"])
    vertex(
        root,
        "b1-answer",
        "lane-b1",
        926,
        50,
        118,
        130,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#EEF7EF;"
        "strokeColor=#6DAA78;strokeWidth=1.5;fontColor=#1F5B2E;fontSize=22;" + FONT,
        '<font style="font-size:17px"><b>Answer</b></font><br>'
        '<b>Target<br>clicked</b>',
    )

    # C3: render–edit feedback loop.
    vertex(
        root,
        "c3-label",
        "lane-c3",
        5,
        52,
        78,
        126,
        centered + "fontColor=#246B35;fontSize=20;fontStyle=1;",
        'Web-to-<br>HTML<br><font style="font-size:17px;font-weight:normal">render<br>and revise</font>',
    )
    image_vertex(root, "c3-input-img", "lane-c3", 108, 74, 140, 79, assets["web_reference"])
    vertex(
        root,
        "c3-query",
        "lane-c3",
        255,
        45,
        118,
        140,
        text + "fontColor=#22324A;fontSize=21;",
        '<font color="#58718F" style="font-size:17px"><b>Query</b></font><br>'
        'Recreate page<br>in <b>HTML/CSS</b>',
    )
    vertex(root, "c3-render", "lane-c3", 412, 43, 158, 44, tool_chip, "render_html")
    vertex(root, "c3-edit", "lane-c3", 420, 143, 142, 44, tool_chip, "edit_file")
    edge(root, "c3-render-edit", "lane-c3", source="c3-render", target="c3-edit", style=thin_flow)
    edge(
        root,
        "c3-edit-render",
        "lane-c3",
        source="c3-edit",
        target="c3-render",
        value="iterate",
        style=(
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
            "strokeColor=#5A9865;strokeWidth=2;dashed=1;fontColor=#3E7D49;"
            "fontSize=16;" + FONT
        ),
        points=[(580, 165), (580, 65)],
    )
    image_vertex(root, "c3-draft-img", "lane-c3", 610, 43, 80, 115, assets["web_draft"])
    vertex(
        root,
        "c3-draft-label",
        "lane-c3",
        603,
        163,
        94,
        24,
        centered + "fontColor=#5D6B62;fontSize=16;",
        "initial",
    )
    vertex(
        root,
        "c3-revise-arrow",
        "lane-c3",
        695,
        82,
        28,
        40,
        centered + "fontColor=#3E8A4B;fontSize=28;fontStyle=1;",
        "→",
    )
    image_vertex(root, "c3-final-img", "lane-c3", 724, 48, 190, 107, assets["web_final"])
    vertex(
        root,
        "c3-final-label",
        "lane-c3",
        720,
        163,
        198,
        24,
        centered + "fontColor=#5D6B62;fontSize=16;",
        "revised",
    )
    vertex(
        root,
        "c3-answer",
        "lane-c3",
        926,
        50,
        118,
        130,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#EEF7EF;"
        "strokeColor=#6DAA78;strokeWidth=1.5;fontColor=#1F5B2E;fontSize=21;" + FONT,
        '<font style="font-size:17px"><b>Answer</b></font><br>'
        '<b>HTML/CSS<br>returned</b>',
    )

    # Stage 3 — Verification and retention.
    vertex(
        root,
        "s3-bg",
        "stage3",
        0,
        0,
        264,
        876,
        stage_frame + "fillColor=#FBFAFE;strokeColor=#B8ADD1;",
    )
    vertex(
        root,
        "s3-badge",
        "stage3",
        14,
        16,
        40,
        40,
        "ellipse;html=1;aspect=fixed;fillColor=#6D5AA7;strokeColor=none;"
        "fontColor=#FFFFFF;fontSize=24;fontStyle=1;" + FONT,
        "3",
    )
    vertex(
        root,
        "s3-title",
        "stage3",
        60,
        8,
        190,
        68,
        title + "fontColor=#5A4785;fontSize=28;",
        "Supervision<br>Verification",
    )
    vertex(
        root,
        "s3-subtitle",
        "stage3",
        20,
        84,
        224,
        44,
        subtitle + "fontColor=#6E677C;fontSize=17;",
        "Keep trajectories that<br>pass both tests",
    )
    vertex(
        root,
        "candidate",
        "stage3",
        24,
        140,
        216,
        70,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#F1EDF9;"
        "strokeColor=#8B78B8;strokeWidth=2;fontColor=#503D78;"
        "fontSize=21;fontStyle=1;" + FONT,
        "Candidate τ",
    )
    vertex(
        root,
        "both-panel",
        "stage3",
        16,
        235,
        232,
        350,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=5;fillColor=#FFFFFF;"
        "strokeColor=#B9ADCF;strokeWidth=2;verticalAlign=top;spacingTop=12;"
        "fontColor=#5A4785;fontSize=18;fontStyle=1;" + FONT,
        "Both required",
    )
    vertex(
        root,
        "outcome-test",
        "stage3",
        28,
        282,
        208,
        82,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#F7F4FC;"
        "strokeColor=#A18BC2;strokeWidth=1.5;fontColor=#553F79;fontSize=21;fontStyle=1;" + FONT,
        'Outcome validity <font style="font-size:20px">✓</font>',
    )
    vertex(
        root,
        "causal-card",
        "stage3",
        28,
        390,
        208,
        165,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#FCFBFE;"
        "strokeColor=#A18BC2;strokeWidth=1.5;fontColor=#553F79;fontSize=19;fontStyle=1;"
        "verticalAlign=top;spacingTop=10;" + FONT,
        "Causal utility",
    )
    vertex(
        root,
        "bar-noobs",
        "stage3",
        72,
        458,
        32,
        36,
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#B6BCC6;strokeColor=none;",
    )
    vertex(
        root,
        "bar-obs",
        "stage3",
        166,
        425,
        32,
        69,
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#8B75B3;strokeColor=none;",
    )
    edge(
        root,
        "bar-axis",
        "stage3",
        style="html=1;endArrow=none;strokeColor=#8E8A98;strokeWidth=1.5;",
        source_point=(52, 495),
        target_point=(216, 495),
    )
    vertex(
        root,
        "bar-noobs-label",
        "stage3",
        42,
        498,
        92,
        24,
        centered + "fontColor=#65717C;fontSize=16;",
        "no tool",
    )
    vertex(
        root,
        "bar-obs-label",
        "stage3",
        142,
        498,
        92,
        24,
        centered + "fontColor=#5A4785;fontSize=16;",
        "+ tool trace",
    )
    vertex(
        root,
        "gain-label",
        "stage3",
        64,
        524,
        136,
        26,
        centered + "fontColor=#5A4785;fontSize=18;fontStyle=1;",
        "Δp ≥ δ",
    )
    vertex(
        root,
        "and-badge",
        "stage3",
        109,
        566,
        46,
        30,
        "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#EEE9F7;"
        "strokeColor=#8B75B3;fontColor=#554278;fontSize=15;fontStyle=1;" + FONT,
        "And",
    )
    vertex(
        root,
        "retain",
        "stage3",
        40,
        650,
        184,
        72,
        "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=#E9E4F5;"
        "strokeColor=#7762A3;strokeWidth=2;fontColor=#4A386E;"
        "fontSize=21;fontStyle=1;" + FONT,
        "Instructive<br>keep",
    )
    vertex(
        root,
        "dataset",
        "stage3",
        37,
        780,
        190,
        72,
        "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;"
        "backgroundOutline=1;size=14;fillColor=#5A4785;strokeColor=#433365;"
        "fontColor=#FFFFFF;fontSize=21;fontStyle=1;" + FONT,
        "OpenVisTool-42K",
    )
    verify_flow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
        "endFill=1;strokeColor=#7562A1;strokeWidth=2;"
    )
    edge(root, "candidate-both", "stage3", source="candidate", target="both-panel", style=verify_flow)
    edge(root, "both-retain", "stage3", source="both-panel", target="retain", style=verify_flow)
    edge(root, "retain-dataset", "stage3", source="retain", target="dataset", style=verify_flow)

    # Only two top-level connectors, both left-to-right.
    edge(
        root,
        "stage1-stage2",
        "1",
        style=(
            "rounded=1;html=1;endArrow=block;endFill=1;"
            "strokeColor=#2F6FCC;strokeWidth=2.5;"
        ),
        source_point=(262, 693),
        target_point=(276, 450),
        points=[(269, 693), (269, 450)],
    )
    edge(
        root,
        "stage2-stage3",
        "1",
        style=(
            "rounded=1;html=1;endArrow=block;endFill=1;"
            "strokeColor=#6D5AA7;strokeWidth=2.5;"
        ),
        source_point=(1370, 450),
        target_point=(1408, 175),
        points=[(1377, 450), (1377, 175)],
    )

    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(OUTPUT, encoding="utf-8", xml_declaration=False)


def main() -> None:
    assets = build_assets()
    build_drawio(assets)
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {len(assets)} compact assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
