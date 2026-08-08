"""Vision tools for image analysis and annotation."""

from nanobot.agent.tools.vision.color import (
    ColorClustersTool,
    ColorSegmentsTool,
    SampleColorTool,
)
from nanobot.agent.tools.vision.computer_use import ComputerUseTool
from nanobot.agent.tools.vision.draw import (
    DrawBBoxTool,
    DrawCircleTool,
    DrawLineTool,
)
from nanobot.agent.tools.vision.enhance import (
    AdjustBrightnessTool,
    DetectEdgesTool,
    EnhanceContrastTool,
    GrayscaleTool,
)
from nanobot.agent.tools.vision.feature import (
    ConnectedComponentsTool,
    FindContoursTool,
    HoughCirclesTool,
    HoughLinesTool,
    TemplateMatchTool,
)
from nanobot.agent.tools.vision.in_range_color import InRangeColorTool
from nanobot.agent.tools.vision.render_html import RenderHtmlTool
from nanobot.agent.tools.vision.transform import CropTool, FlipTool, RotateTool

__all__ = [
    "AdjustBrightnessTool",
    "ColorClustersTool",
    "ColorSegmentsTool",
    "ComputerUseTool",
    "ConnectedComponentsTool",
    "CropTool",
    "DetectEdgesTool",
    "DrawBBoxTool",
    "DrawCircleTool",
    "DrawLineTool",
    "EnhanceContrastTool",
    "FindContoursTool",
    "FlipTool",
    "GrayscaleTool",
    "HoughCirclesTool",
    "HoughLinesTool",
    "InRangeColorTool",
    "RenderHtmlTool",
    "RotateTool",
    "SampleColorTool",
    "TemplateMatchTool",
]
