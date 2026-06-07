"""Count tool-call distribution from a training/export JSONL file.

Supported common formats:
1. Top-level {"messages": [...]} with message role "tool_call" and JSON content
2. Top-level {"messages": [...]} with assistant messages containing "tool_calls"
3. Top-level {"conversations": [...]} with the same message variants

Example:
    python distill/utils/stat_tool_call_distribution.py \
        workspaces/qwen35plus/ChartVerse-SFT-9K_agent_sandbox.jsonl

Optional JSON export:
    python distill/utils/stat_tool_call_distribution.py input.jsonl \
        --output tool_distribution.json

Optional pie chart export:
    python distill/utils/stat_tool_call_distribution.py input.jsonl \
        --pie-output tool_distribution.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 JSONL 文件中的工具调用分布。")
    parser.add_argument("input", help="输入 JSONL 文件路径。")
    parser.add_argument(
        "--output",
        default=None,
        help="可选：将统计结果写入 JSON 文件。",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="可选：只显示前 N 个工具；默认显示全部。",
    )
    parser.add_argument(
        "--pie-output",
        default=None,
        help="可选：将工具调用次数分布绘制为饼状图并保存到指定路径。",
    )
    parser.add_argument(
        "--pie-title",
        default=None,
        help="可选：饼状图标题；默认 Tool Call Distribution。",
    )
    return parser.parse_args()


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_name_from_tool_call(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None

    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    name = call.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    return None


def _iter_tool_names_from_payload(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, list):
        names: list[str] = []
        for item in payload:
            names.extend(_iter_tool_names_from_payload(item))
        return names
    if isinstance(payload, dict):
        name = _extract_name_from_tool_call(payload)
        return [name] if name else []
    return []


def extract_tool_names_from_message(message: dict[str, Any]) -> list[str]:
    names: list[str] = []

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            name = _extract_name_from_tool_call(tool_call)
            if name:
                names.append(name)

    role = str(message.get("role", "")).strip()
    if role == "tool_call":
        payload = _maybe_parse_json(message.get("content"))
        names.extend(_iter_tool_names_from_payload(payload))

    return names


def _get_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "conversations"):
        value = sample.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def build_summary(input_path: Path) -> dict[str, Any]:
    tool_counter: Counter[str] = Counter()
    tool_sample_counter: Counter[str] = Counter()
    total_samples = 0
    samples_with_tools = 0
    total_tool_calls = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc

            if not isinstance(sample, dict):
                continue

            total_samples += 1
            sample_tool_names: list[str] = []
            for message in _get_messages(sample):
                sample_tool_names.extend(extract_tool_names_from_message(message))

            if sample_tool_names:
                samples_with_tools += 1
                total_tool_calls += len(sample_tool_names)
                tool_counter.update(sample_tool_names)
                tool_sample_counter.update(set(sample_tool_names))

    sorted_tools = sorted(tool_counter.items(), key=lambda item: (-item[1], item[0]))
    return {
        "input": str(input_path),
        "total_samples": total_samples,
        "samples_with_tools": samples_with_tools,
        "samples_without_tools": total_samples - samples_with_tools,
        "total_tool_calls": total_tool_calls,
        "distinct_tools": len(tool_counter),
        "tools": [
            {
                "name": name,
                "count": count,
                "ratio_of_calls": (count / total_tool_calls) if total_tool_calls else 0.0,
                "sample_count": tool_sample_counter[name],
                "sample_coverage": (tool_sample_counter[name] / total_samples) if total_samples else 0.0,
                "avg_calls_per_sample": (count / total_samples) if total_samples else 0.0,
            }
            for name, count in sorted_tools
        ],
    }


def print_summary(summary: dict[str, Any], top_n: int | None = None) -> None:
    tools = summary["tools"]
    if top_n is not None:
        tools = tools[:top_n]

    print(f"输入文件: {summary['input']}")
    print(f"样本数: {summary['total_samples']}")
    print(f"含工具调用样本数: {summary['samples_with_tools']}")
    print(f"无工具调用样本数: {summary['samples_without_tools']}")
    print(f"总工具调用次数: {summary['total_tool_calls']}")
    print(f"工具种类数: {summary['distinct_tools']}")
    print()

    if not tools:
        print("未发现工具调用。")
        return

    print(
        f"{'排名':>4}  {'工具名':<24} {'调用次数':>10} {'占总调用比例':>14} "
        f"{'覆盖样本比例':>14} {'平均每样本调用':>14}"
    )
    for index, item in enumerate(tools, 1):
        print(
            f"{index:>4}  {item['name']:<24} {item['count']:>10} "
            f"{item['ratio_of_calls'] * 100:>13.2f}% {item['sample_coverage'] * 100:>13.2f}% "
            f"{item['avg_calls_per_sample']:>14.2f}"
        )


def save_pie_chart(summary: dict[str, Any], output_path: Path, title: str | None = None) -> None:
    title_fontsize = 18
    slice_label_fontsize = 14
    legend_fontsize = 14

    tools = summary["tools"]
    if not tools:
        raise ValueError("未发现工具调用，无法绘制饼状图。")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("绘制饼状图需要安装 matplotlib。") from exc

    pie_tools = tools[:8]
    tool_names = [item["name"] for item in pie_tools]
    counts = [item["count"] for item in pie_tools]
    ratios = [item["ratio_of_calls"] * 100 for item in pie_tools]
    legend_labels = tool_names

    colors = [plt.get_cmap("tab20")(index) for index in range(len(counts))]

    slice_index = 0

    def format_large_slice(_pct: float) -> str:
        nonlocal slice_index
        index = slice_index
        slice_index += 1
        ratio = ratios[index]
        if ratio <= 10:
            return ""
        return f"{tool_names[index]}\n{ratio:.1f}%"

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, autotexts = ax.pie(
        counts,
        colors=colors,
        autopct=format_large_slice,
        pctdistance=0.65,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": slice_label_fontsize, "weight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_ha("center")
        autotext.set_va("center")

    ax.axis("equal")
    ax.set_title(title or "Tool Call Distribution", fontsize=title_fontsize)
    fig.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        frameon=False,
        fontsize=legend_fontsize,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.16, 0.98, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    summary = build_summary(input_path)
    print_summary(summary, top_n=args.top)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"JSON 结果已写入: {output_path}")

    if args.pie_output:
        pie_output_path = Path(args.pie_output).expanduser().resolve()
        save_pie_chart(summary, pie_output_path, title=args.pie_title)
        print()
        print(f"饼状图已写入: {pie_output_path}")


if __name__ == "__main__":
    main()
