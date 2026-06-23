# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

工具调用 vision agent 的离线评测框架：跑一个 benchmark，让被测模型循环调用一组视觉工具回答问题，答案先经 LLM extractor 抽取，再经 LLM judge 多次投票打分。

详细的使用说明（配置文件字段、resume、rejudge、看结果、排查表）见 `README.md`。本文件聚焦架构与开发要点。

## 常用命令

```bash
uv sync                                                          # 准备 .venv（uv 只管 venv，不打包项目）
./eval.sh --model-config configs/qwen35_9b_openvistool.json     # 跑评测
./eval.sh --model-config configs/xxx.json -n 5 --no-tools       # 跑前 5 题 / 关工具基线
./eval.sh --model-config configs/xxx.json --bench dataset/CharXiv_val.jsonl
./eval.sh --resume workspace/<model>/run_xxx                     # 中断后续跑（config 从 run dir 冻结快照读）
./rejudge.sh workspace/<model>/run_xxx                           # 只重判分，不重跑 agent
WORKERS=16 MAX_STEPS=30 AVG_K=4 ./eval.sh --model-config ...     # 环境变量覆盖（avg@k 多采样）
```

源码改动立即生效，无需 reinstall。`eval.sh` / `rejudge.sh` 会探测 `configs/*.json` 里的内网 base_url 并自动 unset 代理。

跑单题调试：用 `-n 1` 或先把 bench 截成一行 jsonl 再传 `--bench`。没有测试套件（验证靠 `uv run python -m py_compile` + 直接 `import harness`）。

## 评测数据集来源与收集方法

`multi_eval.sh` 默认跑五个评测集，分别覆盖 Chart、GUI、Table、VisualSearch 和 Web2HTML。评测数据放在 `dataset/*.jsonl` 或子目录下，当前仓库通常不提交 `dataset/` 大文件，只保留脚本中的路径约定。

| 领域 | 默认 bench | 来源与收集方法 |
| --- | --- | --- |
| Chart | `dataset/Chart_tool_bench_union.jsonl` | 基于 CharXiv-reasoning 和 ChartMuseum 筛选。对每个样本分别用 GPT-5.4、Gemini-3.0-Flash、Qwen3.5-Plus 做 no-tool 推理和 with-tools 推理，计算 avg@5；保留“给工具后 avg@5 变高但仍 `< 1`”的样本，三模型筛选结果取并集。 |
| Table | `dataset/Table_tool_bench_union.jsonl` | 基于 MMT-Bench 和 TableVQA-Bench 筛选。筛选流程与 Chart 相同：三个强模型分别比较 no-tool / with-tools 的 avg@5，保留工具带来提升但未完全解决（avg@5 `< 1`）的样本，并取三模型并集。 |
| GUI | `dataset/GUI_tool_bench_union.jsonl` | 直接使用 ScreenSpot-Pro 中目标元素占整张图片比例最小的一部分，共 117 条，评测时走 `gui` 模式的 point-in-bbox 几何评分。 |
| VisualSearch | `dataset/VisualProbe_Hard.jsonl` | 直接使用 VisualProbe-Hard。 |
| Web2HTML | `dataset/Vision2Web/Vision2Web-webpage.jsonl` | 直接使用 Vision2Web-Level1，评测时按 Vision2Web 路径渲染生成 HTML 并做组件级视觉保真评分。 |

Chart / Table 的筛选目标不是只找“难题”，而是找工具确实能提供增益、但强模型在有工具时仍未达到满分的样本；这样评测能区分模型是否真正利用视觉工具，而不是只考察已有的静态看图能力。

## 包结构

库代码集中在 `harness/`：四个职责子包 + 一个 `tools/`（agent 调用的工具实现）。依赖方向**自上而下单向**（`core ← agents/scoring/runtime`，`scoring` 不依赖 `agents`，`runtime` 可依赖 `scoring`；`tools` 被 `agents`/`runtime` 调用、内部自洽不反依赖其它子包），图无环。入口脚本 `eval.py` / `rejudge.py` 按子包导入（`from harness.agents import run` 等）；`harness/__init__.py` 另提供一个扁平 facade（`from harness import run`）。

```
harness/
├── core/        # 无业务依赖的地基
│   ├── config.py        # AppConfig + load_config；含 is_qwen25vl()/is_qwen3vl()/is_code_agent()
│   ├── llm.py           # AsyncOpenAI + retry + 超时 + 连接池
│   ├── benchmark.py     # bench loader + task_key
│   ├── io.py            # 多模态 content 编码 / base64 strip
│   ├── path_mapper.py   # sandbox ↔ /mnt/data 虚拟路径映射
│   └── image.py         # smart_resize + 像素预算常量（Qwen2.5-VL 几何，单一来源）
├── agents/      # rollout 驱动：把「模型+工具」跑成一个 pred
│   ├── prompts.py       # 所有 SYSTEM_PROMPT_* + build_*_prompt（纯文本/模板）
│   ├── function_call.py # run()：原生 OpenAI tool-calling 主循环（默认）
│   ├── code_agent.py    # run_code_agent()：Thyme / DeepEyesV2 代码即工具协议
│   └── grounding.py     # run_qwen25_grounding()：base Qwen2.5-VL 单轮 + 共享 grounding 协议
├── scoring/     # 把 pred 变成分数
│   ├── extractor.py     # LLM 抽取最终答案
│   ├── judge.py         # 投票 YES/NO judge + Vision2Web 组件保真 judge（0–100）
│   ├── gui_grounding.py # 几何 point-in-bbox 评分（score_grounding / score_grounding_pyautogui）
│   └── vision2web.py    # HTML 解析/回退 + 无头渲染 + 组件级评分编排 + 数据集谓词
├── runtime/     # 跑时记账
│   ├── sandbox.py       # 单任务/单样本 sandbox + 工具注册
│   ├── results.py       # 聚合 task_*/result.json → results.jsonl + 统计
│   └── dashboard.py     # rich.Live 实时表格（非 TTY 自动降级 no-op）
└── tools/       # agent 调用的工具实现（内部自洽，不反依赖其它子包）
    ├── base.py / registry.py             # Tool 基类 + 动态注册表（to_schema 导出 function-calling）
    ├── filesystem.py / shell.py          # sandbox 内文件读写 / exec
    ├── code_kernel.py / code_sandbox.py  # code-agent baseline 的子进程内核 + 包装
    └── vision/                           # crop / enhance / draw / transform / feature / render_html / computer_use / ...
```

## 架构

一次 run 的数据流（`eval.py` 是入口，编排下面这些）：

1. **`harness/core/benchmark.py`** 加载 bench jsonl（每行一个 item：image + question + gt）。
2. 对每个 item 并发（`workers` 个）执行：
   - **`harness/runtime/sandbox.py`** `setup_sandbox` 给该 task 建 `task_<id>/` 目录、symlink 输入图；`build_registry` 注册工具。
   - **`harness/agents/function_call.py`** `run()` 跑 agent 主循环：LLM call → 解析 tool_calls → `ToolRegistry.execute` → 把工具结果（含图）回灌 messages → 再 call，直到模型给最终答案或到 `max_steps`。
   - **`harness/scoring/extractor.py`** 从模型自由文本里抽出结构化答案；**`harness/scoring/judge.py`** 用 judge 模型投票（`n_votes`）打分。web 代码生成 / GUI grounding 走各自的特殊打分路径（见下文 eval_mode），分别用 **`harness/scoring/vision2web.py`**（HTML→PNG + 组件评分）和 **`harness/scoring/gui_grounding.py`**（几何判点）。
3. **`harness/runtime/results.py`** 聚合所有 `task_<id>/result.json` → 重写 `results.jsonl` + `meta.json`（原子写）。
4. **`harness/runtime/dashboard.py`** 在 TTY 下用 `rich.Live` 显示实时表格，被管道时自动关闭。

**两种 eval_mode**（`general_config.json` 里设，`core/config.py::EVAL_MODES=("judge","gui")`）：

- `judge`：默认。`eval.py::_run_one_sample` 按 item 的 `source_dataset` 再分两条子路径：
  - **chart / QA**（默认）：extractor 抽答案 → judge LLM 投票（`n_votes`）打 **0/1**。
  - **Vision2Web 网页代码生成**（`source_dataset=="Vision2Web-webpage"`，谓词 `scoring/vision2web.py::is_vision2web_item`）：被测模型按代码生成 prompt（`SYSTEM_PROMPT_VISION2WEB_WEBPAGE`）产出 HTML；`resolve_submitted_html` 取最终 HTML → 按 `workflow.json` 里的 desktop/tablet/mobile 视口（`vision2web_viewports`，缺则用 `DEFAULT_VIEWPORTS`）各渲一张（`render_html_to_viewport_images`，Playwright 无头 Chromium）→ `judge_vision2web_viewports` 逐视口调 `judge.judge_vision2web_static` 做**组件级**打分（把页面切成 header/hero/卡片/footer 等块，每块取 {0,.25,.5,.75,1}，`n_votes` 取中位数 ×100）→ 三视口分均值即该 sample 分（**连续 0–100 分**，不是 0/1）。参考图为 sandbox 内 symlink 的 `desktop.jpg`/`tablet.jpg`/`mobile.jpg`，渲染图为 `generated_<device>.png`，明细写入 `result.json` 的 `vision2web` 字段。配套 `configs/general_config_vision2web.json`。
    - **HTML 来源回退**：`resolve_submitted_html` 先 `extract_html(pred)`，若最后一轮没有 HTML 代码块（agentic openvistool 模型常以 render_html 迭代、末轮只回纯文本），则回退到模型在 agent 循环里用 `write_file`/`edit_file`/`render_html` **最后写/改的那个 `.html` 文件**（从 trace 的工具调用里取 path，经 `PathMapper` 映射回 sandbox 读取）；结果在 `vision2web.html_source` 记 `pred` / `file:<name>` / `none`。
    - **聚合**：`runtime/results.py` 用 `is_vision2web_source(source_dataset)` 判定为连续分（不再靠「score>1」启发式，全≤1 的 run 也能正确识别），输出 `avg_score`、`correct=#(score≥50)`，`avg_k>1` 时 `accuracy_avg_k=各题 score_avg 均值`。
    - **rejudge**：`rejudge.py::_rejudge_vision2web` 复用已渲染的 `generated_*.png` + 同一个 `judge_vision2web_viewports`（而非误走通用 YES/NO judge）；加 `--rerender` 则对「末轮没 HTML」的 sample 触发上面的回退：恢复模型最后写的 `.html` → 重渲染（覆盖该 sample 的 `generated.html`/`generated_*.png`）→ 重判，其余 sample 仍复用旧渲染。

- `gui`：GUI grounding（如 ScreenSpot-Pro），**不调用 LLM**，几何判定点击是否落入 GT bbox（`scoring/gui_grounding.py`）。是否 grounding 由**数据驱动**（`is_grounding_item`：`answer` 是 `[x1,y1,x2,y2]` bbox 且带 `img_size`），无需配置开关。两条打分路径（`eval.py` 按模型选）：
  - **Qwen2.5-VL 系**（原生 + Thyme/DeepEyesV2 code-agent，`cfg.is_qwen25vl() or cfg.is_code_agent()`）：模型给 smart-resize 输入空间的**绝对像素 `(x,y)`**，`score_grounding_pyautogui` 按 `grounding_max_pixels`/`grounding_min_pixels` 的 `smart_resize`（来自 `core/image.py`）比例还原到原图像素再判点（默认对齐 vLLM 的 `16384*28*28`/`4*28*28`）。
  - **其他模型**（如 Qwen3-VL）：给 0–1000 归一化 `computer_use` 坐标，走 `score_grounding`。
  - `eval_mode=="gui"` 时 `build_registry` 才注册 `ComputerUseTool`，且仅对**非 Qwen2.5-VL** 注册（`include_computer_use=not is_qwen25`）；Qwen2.5-VL 完全不走 `computer_use` 通道，base Qwen2.5-VL grounding 走 `agents/grounding.py::run_qwen25_grounding` 的官方单轮 recipe。配套 `configs/general_config_screenspot_pro.json`。

**两种 agent_mode**（被测模型 config 优先，否则 general config，默认 `function_call`）：
- `function_call`：默认，`agents/function_call.py::run` 原生 OpenAI tool-calling 循环。
- `thyme` / `deepeyesv2`：代码即工具的 baseline（均基于 Qwen2.5-VL-7B）。模型在 `<code>```python ... ```</code>` 文本块里写代码而非发 `tool_calls`，由 `agents/code_agent.py::run_code_agent` 这个文本协议循环驱动；prompt 与协议逐字复刻自各自仓库。代码在 `harness/tools/code_kernel.py`（独立子进程内核，跨步保持状态）里执行，`harness/tools/code_sandbox.py::CodeSandboxTool` 包一层把 stdout+图渲染成多模态 content block 回灌。`eval.py::_run_one_sample` 据 `cfg.agent_mode` 分流，code-agent 模式下不建 registry/path_mapper，沿用同一套 extractor+judge 对最终 `<answer>` 打分。DeepEyesV2 的联网搜索在离线框架里返回 "unavailable"，python 工具完整可用。
  - **GUI grounding（ScreenSpot-Pro，`eval_mode=gui`）**：code-agent 不走 0–1000 归一化的 `computer_use` 通道，而是用各自仓库的 GUI-agent 约定输出绝对像素 `(x, y)`（共享 `agents/grounding.py::computer_use_tools_block` / `grounding_click_instruction`，对喂入的图先 `resize_grounding_images` 预 resize）。打分与原生 Qwen2.5-VL 一致走上面 `gui` 模式的「绝对像素 + `score_grounding_pyautogui`」路径（code-agent ⊂ Qwen2.5-VL 系）。

**avg@k**：`avg_k > 1` 时每题跑 k 次，`setup_sample_sandbox` 为每个 sample 建独立子 sandbox，输出 `score_avg` + `samples` 列表，`meta.json` 额外含 `accuracy_avg_k`。

### rejudge 的配置解析与评分器选择

`rejudge.py` 只重判分、不重跑 agent，但要和原 run 用**同一套打分语义**，因此 `_resolve_config` 这样取配置：

- **run 定义字段**（`bench_path` / `eval_mode` / `avg_k` / `grounding_*_pixels`）从 `<run_dir>/general_config.json` **冻结快照**读 —— 保证 bench 查表和 scorer 选择与原 run 一致，即使外部 `configs/` 已改。
- **judge / extractor 端点 + `n_votes`** 从命令行 `--general-config`（默认 `configs/general_config.json`，即**当前**文件）读 —— 这正是 rejudge 的用途：换/改 judge 重新打分。
- **被测模型族**（是否 Qwen2.5-VL / code-agent）从 `<run_dir>/model_config.json` 冻结快照读，决定 gui 模式用 `score_grounding_pyautogui` 还是 `score_grounding`（与 `eval.py` 一致）。
- 旧 run 缺快照时回退到纯 live config（兼容）。

`_rejudge_one` 对 `avg_k==1`（无 `samples` 列表）的 run 也会重算 `rec["score_avg"]`，否则 `accuracy_avg_k` 会停留在旧值、与刷新后的 `accuracy` 不一致。旧值统一保留为 `*_prev`（`score_prev` / `extracted_prev` / `score_avg_prev` / `vision2web_prev`）。

### 工具系统

`harness/tools/` 下所有工具继承 `harness/tools/base.py::Tool`，由 `harness/tools/registry.py::ToolRegistry` 动态注册，`to_schema()` 导出 OpenAI function-calling 格式。新增工具：写一个 `Tool` 子类，在 `harness/runtime/sandbox.py::build_registry` 里 register。`harness/tools/` 内部自洽，只 import 自身、不反依赖 `harness` 的其它子包。

- `harness/tools/filesystem.py` `harness/tools/shell.py`（`ExecTool`，构造需 `exec_timeout`，单独注册）：sandbox 内文件 / shell。
- `harness/tools/vision/`：`color` / `enhance` / `draw` / `transform` / `feature` / `in_range_color` / `render_html` 等视觉工具，`_common.py` 是共享辅助。
- `harness/tools/vision/computer_use.py`：仅 gui 模式且非 Qwen2.5-VL 时注册。

**工具白名单（`allowed_tools`）**：general config 里可选填 `"allowed_tools": ["crop", ...]`，`build_registry` 则只注册名单内的 workspace/exec 工具（`computer_use` 不受名单约束，由 `eval_mode=="gui"` 单独控制）。不填（=`null`/缺省）时注册全部工具。gui 模式下提示词也会随名单自适应：`agents/prompts.py::build_gui_grounding_prompt(allowed_tools)` 只提示模型它实际拥有的工具（用于 crop / crop+draw_bbox / crop+draw_circle 这类工具集消融）。

工具对文件的读写都限定在该 task 的 sandbox 目录内。`core/io.py` 负责多模态 content 的 base64 编解码，并在写 trace 时 strip 掉 base64 防止 `trace.jsonl` 爆炸。

### 配置

`core/config.py` 加载 `configs/general_config.json`（全局 + judge/extractor 端点）和 `configs/<model>_config.json`（被测模型端点）。`configs/*.json` 含 api_key，已 gitignore。`name` 字段决定输出目录 `workspace/<name>/`。

`--resume` 时 config 从 `<run_dir>/general_config.json` + `model_config.json` 的冻结快照读取（CLI 覆盖如 `--bench`/`--eval-mode`/`--avg-k` 已在原 run 启动时折进快照），外部 `configs/` 改动不影响该 run。

### 输出布局

```
workspace/<model>/run_<timestamp>/
├── meta.json / results.jsonl                  ← 聚合结果
├── general_config.json / model_config.json    ← 冻结快照（resume / rejudge 据此还原 run 语义）
└── task_<id>/
    ├── result.json    ← 单任务 (id, gt, pred, extracted, score, score_avg, ...)
    ├── trace.jsonl    ← 完整 LLM 对话轨迹（base64 已 strip）
    └── *.png          ← 输入图 + 工具产出的中间图（Vision2Web 另有 generated_<device>.png）
```

`workspace/` 已 gitignore。bench 数据在 `dataset/*.jsonl`（如 `CharXiv_val.jsonl`、`Chart_tool_bench_union.jsonl`、`Vision2Web/Vision2Web-webpage.jsonl`、`ScreenSpot-Pro_full.jsonl`）。
