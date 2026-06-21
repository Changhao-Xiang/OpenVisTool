# AGENTS.md

This file provides guidance to coding agents working in this repository.

## 仓库定位

这是 `nanobot_for_distill`：在 `nanobot`（一个轻量 AI assistant 框架）之上做**视觉/工具调用蒸馏**的 fork。`distill/` 是该 fork 的核心改造，用 nanobot agent 在多模态数据集上批量 rollout，把 thinking + tool-call 轨迹保存下来，过滤后转成 SFT 样本喂给 `ms-swift`。

合成出的工具调用轨迹数据集对外命名为 **OpenVisTool**，后续计划开源数据并发布技术报告。产物落在 `dataset/OpenVisTool*/` 下，按消融分三个变体（见下文「蒸馏管线」末尾），按领域（Chart / GUI-Grounding / VisualSearch / Table / VinciCoder 等）切分。涉及对外命名、数据卡、报告口径时统一用 OpenVisTool。技术报告的latex项目在 `OpenVisTool_paper/` 下，主要的实验结果在 `OpenVisTool_paper/result.md` 的表格里。

`flash-attention/` 和 `ms-swift/` 是子仓库（被复制进来的训练栈），不在 `nanobot` / `distill` 主开发路径上 —— 改动它们前先确认意图。

## 论文写作约定（OpenVisTool_paper/）

编辑 `OpenVisTool_paper/sections/*.tex` 的正文时，**一行一段**：每个自然段排成一行（段间用空行分隔），不要在段内手动换行。`\paragraph{}` 标题单独占一行、其正文另起一行；`itemize` / `enumerate` 里每个 `\item` 占一行。新写或重排章节时参照 `introduction.tex` / `related_work.tex` 的排版。

## 环境

- 运行任何项目脚本（包括 pytest）前，先 `conda activate nanobot`。`cv2` 等第三方依赖只在该环境里能加载。运行 `which conda` 的结果是 `/mnt/afs_share/miniconda3/condabin/conda`。
- Python ≥ 3.11。`pip install -e ".[dev]"` 安装；`nanobot` 控制台脚本是 `nanobot.cli.commands:app`。
- 顶层 `srun.sh` / `submit.sh` 是 SCO 集群提交脚本，本地开发请勿误执行 —— `submit.sh` 会调集群 API。

## 蒸馏管线（distill/）

`filter/` 跑各类评测/打分，`select/` 据分数挑样本，`run/` 做教师 rollout，`utils/` 转 SFT 并组装 OpenVisTool。典型流程：

1. **难度过滤** — `distill/filter/filter_difficulty.sh` → `filter_difficulty.py`：用基模型（Qwen3.5-9B）对每条样本做 no-tool avg@k（k=4），**所有领域统一只保留 avg@4 ≤ 0.5 的样本**，去掉纯文本推理就能直接解决的题。全量分数写在 `*.progress.jsonl`（字段 `avg_k`），通过阈值的样本写 `*_difficulty_filtered_*.jsonl`。
2. **难度区间选择** — `distill/select/select_difficulty_range.sh` → `select_difficulty_range.py`：从 `.progress.jsonl` 里挑 avg@k 落在 `[min,max]` 的样本（CoSyn-400K 同样取 `[0, 0.5]`，与统一阈值一致），输出文件名带区间标签（如 `table_difficulty_range_0_0.5.jsonl`）。
3. **教师 rollout** — `distill/run/run_qwen35_plus.sh`（或 `run_gemini.sh`）→ `distill/run.py`：用教师模型对过滤后的数据集批量 rollout，轨迹写到 `<workspace>/session_<id>/`。任务特化 prompt 在 `distill/run/task_specific_toolcall_instructions/`（chart/table/GUI_Grounding/visual_search/html_code_generation），由 `--custom-instructions-override` 注入。`--num-workers` 控制并发。
4. **正确性 + 工具调用质量过滤** — `distill/filter/filter_acc_toolcall.sh` → `filter_acc_toolcall.py`：检查最终答案正确性 + 工具调用启发式规则（拒绝 host 绝对路径、缺图等），产出 correctness index（`*_filtered_index.jsonl`）。答案判定后端 `--accuracy-backend` 三选一：`rule`（确定性匹配，`--match-mode` generic/point/bbox）、`judge`（JudgeLLM）、`html_vlm`（VinciCoder 截图→HTML，渲染后 VLM 打分）；分别由 `evaluate_answer_rule.py` / `evaluate_answer_judge.py` / `evaluate_html_vlm_judge.py` 实现。
5. **工具增益（tool-gain）打分 + 选择**：
   - `distill/filter/filter_tool_gain_with_prefix.sh` → `filter_tool_gain_with_prefix.py`：把教师的 tool_response 前缀喂给基模型再跑 avg@k，只算指标、不过滤，写 `*_tool_gain_prefix.jsonl`。
   - `distill/select/select_tool_gain.sh` → `select_tool_gain.py`：对比 no-tool avg@k 与 with-prefix avg@k，增益超过 `--gain-threshold`（默认 0.1）的样本判为「工具有用」。`select_tool_gain_openvistool.sh` 批量跑全部子领域并汇总到 `dataset/OpenVisTool_toolgain_only/tool-gain-report/`。
   - `distill/select/select_correct_and_gain.sh` → `select_correct_and_gain.py`：把 correctness index 与 tool-gain index 取交集。
6. **转 SFT** — `distill/utils/convert_to_swift.sh` → `convert_to_swift.py`：按 index 文件把 sessions 合成 ms-swift 训练 jsonl，`--tools` 必须列全本次启用的工具名（工具 schema 由 `distill/utils/tool_loader.py` 直接从 `nanobot.agent.tools` 的 Python 类自动抽取，不再维护静态 json）。

**组装 OpenVisTool（三个消融变体，`distill/utils/build_openvistool_*.sh`）**：`build_openvistool_acc_only.sh`（仅正确性）、`build_openvistool_toolgain_only.sh`（仅工具增益）、`build_openvistool_both.sh`（正确性 ∩ 工具增益，主数据集）。三者各自 convert_to_swift 后按领域 merge，分别落到 `dataset/OpenVisTool_acc_only/`、`dataset/OpenVisTool_toolgain_only/`、`dataset/OpenVisTool/`。

可视化轨迹：`distill/utils/visualize.sh <sessions_dir>`。统计工具调用分布：`distill/utils/stat_tool_call_distribution.sh`（输出饼图到 `distill/utils/stat_tool_call_distribution/`）。

## Config 入口

蒸馏运行不读 `~/.nanobot/config.json`，而是 `--config` 指定 JSON。模板见 `workspaces/config_example.json`，关键字段：

- `agents.defaults.workspace` — 该次跑的 session 输出根目录。`distill/run.py` 的 `--workspace-override` 会覆盖。
- `agents.defaults.customInstructions` — 任务特化 prompt 文件路径（如 `distill/run/task_specific_toolcall_instructions/chart.md`）。`--custom-instructions-override` 覆盖。
- `tools.enabledTools` / `enabledSkills` — 显式开关；nanobot 据此动态裁剪 system prompt 与可用工具。视觉工具集（`crop/draw_bbox/in_range_color/computer_use/render_html/...`）就是这里启停。

## 架构骨架

`nanobot/` 是 framework；改 distill 行为通常只需碰 `distill/` 与下面这几处 nanobot 文件：

- `nanobot/agent/loop.py::AgentLoop` — 中心控制器。`_run_agent_loop` 是 thinking → tool-call 主循环；`_save_turn` 决定 session 文件的写入格式（distill 输出依赖它）。`set_tool_workspace` 用于把每个 item 隔离到自己的 sub-workspace + media_dir。
- `nanobot/agent/tools/` — 内建工具实现。视觉工具在 `tools/vision/`（`crop/draw/color/transform/feature`，以及 `computer_use.py`、`render_html.py`），文件系统/shell/web 各一个文件。`registry.py` 只是 `ToolRegistry` 容器；新增工具实际是在 `nanobot/agent/loop.py` 的 `tool_map` 里加工厂（name → 构造 lambda），并加到 config schema 的 `enabledTools`。
- `nanobot/agent/context.py` — 拼 system prompt，根据 `enabled_tools/enabled_skills` 动态裁剪。改 prompt 模板看这里。
- `nanobot/agent/skills.py` + `nanobot/skills/<name>/SKILL.md` — skill 是 markdown 形式的"额外 instruction + 可调脚本"组合，从 OpenClaw 改的格式。
- `nanobot/providers/` — `gpt_proxy_provider.py` / `custom_provider.py` / `litellm_provider.py` 三套；`custom` 直连 OpenAI 兼容端点（绕过 LiteLLM），ptu / 自部署 vLLM 走这条。
- `nanobot/config/schema.py` — pydantic Config 模型，所有 config.json 字段定义在这里；`loader.py` 负责加载与缓存 path。
- `nanobot/session/manager.py` — session 文件读写。
- `distill/config.py::build_agent_from_config` — 把 Config 组装成 `AgentLoop`，是 distill 与 nanobot 的衔接点。
- `distill/run.py::run_batch` — 多线程批量 rollout、重试可恢复 API 错误（见 `RETRYABLE_API_ERROR_PATTERNS`）、按 `--continue-on-error` 决定失败处理策略。

## 行为约定

- 改 system prompt / tool schema 时，记得同步检查 `nanobot/agent/context.py` 与 `nanobot/templates/` 模板是否一致。
- 视觉工具调用结果会写到 session 目录，路径靠 `PathMapper`（`nanobot/agent/path_mapper.py`）做相对化；新增产生文件的工具时要走它，否则蒸馏导出会拿到绝对路径。
