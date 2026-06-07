# agentic-vision-eval

工具调用 vision agent 的离线评测框架。

跑一个 chart QA benchmark，让被测模型循环调用一组视觉工具
（crop / enhance / sample_color / ... 共 16 个）回答问题；
答案先经 LLM extractor 抽取，再经 LLM judge 多次投票打分。

---

## 1. 环境

依赖 `uv`（[安装方式](https://docs.astral.sh/uv/)），其余 Python 包由 `uv sync` 自动装。

```bash
git clone <repo>
cd agentic_vision_eval
uv sync           # 在 .venv/ 里准备好所有依赖（首次约 30s）
```

`uv` **只用来管 venv**，运行时通过 `./eval.sh` 包装 `uv run python -u eval.py`，
源码改动立即生效，不需要 reinstall。

---

## 2. 配置文件

所有参数和 API 凭据都在 `configs/*.json` 里。模板：

```bash
cp configs/general_config.json.example      configs/general_config.json
cp configs/qwen35_9b_config.json.example    configs/qwen35_9b_config.json
# 编辑两个文件，填入真实 api_key / base_url
```

### `configs/general_config.json` — 全局 + judge / extractor 端点

```json
{
  "workspace_root": "workspace",
  "bench_path": "dataset/table/chart_tool_benchmark_union.jsonl",
  "workers": 8,                  // 并发任务数
  "max_steps": 50,               // 单任务最大 LLM 轮数
  "exec_timeout": 60,            // exec 工具超时
  "tools_enabled": true,
  "request_timeout_s": 3600,     // 单次 LLM 请求超时
  "eval_mode": "judge",          // "judge" = LLM extractor+judge (chart/QA)；
                                 // "gui"   = GUI grounding 点-落入-bbox 评测
                                 //           (从模型 computer_use tool_call 抽
                                 //            coordinate，按 img_size 还原到像素，
                                 //            判定是否落入 GT bbox；不调用 LLM)

  "retry": {
    "max_attempts": 3,
    "base_delay": 2.0,
    "max_delay": 30.0
  },

  "judge": {
    "model": "gpt-5.4-mini-2026-03-17",
    "base_url": "https://YOUR_ENDPOINT/v1",
    "api_key": "<YOUR_API_KEY>",
    "n_votes": 3,                // 投票次数
    "vote_temperature": 0.6
  },

  "extractor": {
    "enabled": true,
    "model": "gpt-5.4-mini-2026-03-17",
    "base_url": "https://YOUR_ENDPOINT/v1",
    "api_key": "<YOUR_API_KEY>",
    "temperature": 0
  }
}
```

### `configs/<model_name>_config.json` — 被测模型端点

```json
{
  "name": "qwen35_9b",                 // 输出目录名 (workspace/qwen35_9b/...)
  "model": "ppt_reward_grpo_qwen35_9b",
  "base_url": "https://YOUR_ENDPOINT/v1",
  "api_key": "<YOUR_API_KEY>",

  "temperature": 0,
  "top_p": 1.0,
  "max_tokens": 32768
}
```

每个模型配一个文件。`name` 字段决定输出子目录。

### 代码即工具的 baseline（DeepEyesV2 / Thyme）

DeepEyesV2 和 Thyme 这两个 agentic VLM（均基于 Qwen2.5-VL-7B 训练）不走原生
function-calling，而是把 **Python 代码当成统一工具接口**：模型在 `<code>```python
... ```</code>` 文本块里写代码，由一个沙箱内核执行，stdout / 生成的图再回灌给模型。
框架用 `agent_mode` 字段切换驱动：

```json
{
  "name": "thyme_7b",
  "model": "Thyme-RL",
  "base_url": "https://YOUR_ENDPOINT/v1",
  "api_key": "EMPTY",
  "agent_mode": "thyme",          // "function_call"(默认) | "thyme" | "deepeyesv2"
  "temperature": 0.01, "top_p": 0.001, "top_k": 1, "max_tokens": 2048
}
```

- `agent_mode` 写在被测模型 config 里（也可放 general config，模型 config 优先）。
- 这两个 baseline 的 system / user prompt 与代码协议**逐字复刻**自各自仓库
  （`harness/agents/code_agent.py`），沙箱内核见 `harness/tools/code_kernel.py`（Thyme 暴露
  `image_path`/`temp_output_dir`，DeepEyesV2 预加载 `image_1..N` 并捕获
  `plt.show()`，跨步保持内核状态）。
- DeepEyesV2 的 `<tool_call>` 联网搜索在本离线框架里不可用，会回一句
  "search unavailable" 引导模型改用代码或直接作答；**python 工具完整可用**。
- 现成 config：`configs/thyme_7b.json`、`configs/deepeyesv2_7b.json`，配套
  general config `configs/general_config_agentic_baseline.json`（按 code 执行调高了
  `exec_timeout`/降低了 `max_steps`）。跑法：

```bash
./eval.sh --general-config configs/general_config_agentic_baseline.json \
          --model-config configs/thyme_7b.json
./eval.sh --general-config configs/general_config_agentic_baseline.json \
          --model-config configs/deepeyesv2_7b.json --bench dataset/Chart_tool_bench_union.jsonl
```

> ⚠️ 用 vLLM 部署的 Qwen2.5-VL（Thyme/DeepEyesV2）在**贪心解码**下几乎不发 `<code>`
> （HF↔vLLM 贪心发散），工具调用率会塌成 0。要体现工具增益，请用带温度的采样
> （如 `temperature 0.7, top_p 0.95`）。

#### GUI Grounding（ScreenSpot-Pro）

code-agent baseline 在 `eval_mode=gui` 下不走 0–1000 归一化的 `computer_use` 通道，而是
按各自仓库约定输出 **`pyautogui.click(x=, y=)` 绝对像素**。打分会用 Qwen2.5-VL 的
`smart_resize` 把坐标从模型实际输入分辨率还原到原图像素再判 point-in-bbox，缩放预算由
`grounding_max_pixels` / `grounding_min_pixels` 控制（默认 `16384*28*28` / `4*28*28`，
对齐 vLLM Qwen2.5-VL 默认；若你部署时调小了 `max_pixels`，这里要一并改）。

```bash
./eval.sh --general-config configs/general_config_screenspot_pro.json \
          --model-config configs/thyme_7b.json
```

---

## 3. 跑评测

### 标准跑法

```bash
./eval.sh --model-config configs/qwen35_9b_config.json
```

执行后会创建：

```
workspace/qwen35_9b/run_20260425_123456/
├── meta.json                  ← run 元信息（accuracy / tokens / errors / ...）
├── results.jsonl              ← 每行一个 task 的最终结果
├── general_config.json        ← 启动时冻结的全局 config（用于 --resume）
├── model_config.json          ← 启动时冻结的模型 config
└── task_<id>/
    ├── result.json            ← 单任务结果 (id, gt, pred, extracted, score, ...)
    ├── trace.jsonl            ← 完整 LLM 对话轨迹（每行一轮 + 工具结果）
    └── <input_image>.png      ← 输入图 symlink + 工具产出的 crop/enhance 中间图
```

### 常用参数

```bash
./eval.sh --model-config configs/qwen35_9b_config.json -n 5         # 只跑前 5 题
./eval.sh --model-config configs/qwen35_9b_config.json --no-tools   # 关掉工具基线
./eval.sh --model-config configs/qwen35_9b_config.json --workers 16 --max-steps 30
WORKERS=16 MAX_STEPS=30 ./eval.sh --model-config configs/xxx.json   # 环境变量也行
```

### 实时进度

如果终端是 TTY（直接交互式跑），会显示动态表格：

```
┃    id ┃ question                             ┃ state    ┃ rounds ┃ tools ┃ score ┃    time ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━┩
│     6 │ Based on the ratings for "The...     │ DONE     │      4 │     3 │  ✓ 1  │     32s │
│     9 │ How many colors were always...       │ RUNNING  │      2 │     2 │   —   │         │
│    10 │ In what field are the number of...   │ DONE     │      1 │     0 │  ✗ 0  │      6s │
└───────┴──────────────────────────────────────┴──────────┴────────┴───────┴───────┴─────────┘
done 12/92  running 8  correct 5  acc 41.7%  elapsed 145s
```

被管道 / 重定向时（`./eval.sh ... > log.txt`）dashboard 自动关闭，只输出末尾汇总。
错误信息走 stderr，单行如 `[id=42] agent error: APITimeoutError: ...`。

### 跑完的总结

```
▶ accuracy: 18/92 = 0.196  errors=2 (timeouts=2)  total_tokens=6_069_087  total_elapsed=2185.7s
  error ids:   [421, 666]
  results  -> workspace/qwen35_9b/run_20260425_xxx/results.jsonl
```

`errors / timeouts` 也会写进 `meta.json`，方便事后追查。

---

## 4. 续跑（resume）

如果有 task 因 timeout / 网络断 / Ctrl-C 中断 → state=FAILED，不需要从头重来：

```bash
./eval.sh --resume workspace/qwen35_9b/run_20260425_xxx
```

行为：

1. 扫描 run dir，跳过 `error == null && score 已写` 的任务
2. 重跑所有 `error != null` 或缺 `result.json` 的任务
3. 跑完 **重新聚合 results.jsonl + meta.json**（含被跳过的部分）
4. `meta.json` 多一个 `resumes` 数组，记录每次续跑的 ids 和耗时

`--resume` 时 config 从 `<run_dir>/general_config.json` 和 `<run_dir>/model_config.json`
**冻结快照**读取，外面 `configs/` 即使被改也不影响这个 run 的语义。

减并发再续跑（端点过载场景）：

```bash
WORKERS=4 ./eval.sh --resume workspace/qwen35_9b/run_xxx
```

---

## 5. 重判（rejudge）

只改 judge prompt / 换 judge 模型，不重跑 agent，对一个已有 run 重新打分：

```bash
./rejudge.sh workspace/qwen35_9b/run_20260425_xxx
./rejudge.sh workspace/qwen35_9b/run_xxx --workers 16
./rejudge.sh workspace/<model>/run_xxx --rerender        # 仅 Vision2Web：恢复末轮没回 HTML 的样本再渲再判
```

会：

- **judge / extractor 端点**用当前命令行 `--general-config`（默认 `configs/general_config.json`，即现在这份）读 —— 这正是重判的用途：换/改 judge。
- **run 定义字段**（`bench_path` / `eval_mode` / `avg_k` / grounding 像素预算）和**被测模型族**从 `<run_dir>` 里的**冻结快照**（`general_config.json` / `model_config.json`）读，保证 bench 查表、judge/grounding 评分器选择都跟原 run 一致，外部 `configs/` 改了也不影响。旧 run 缺快照时回退到 live config。
- 旧 score 保留为 `score_prev`，旧 extracted 保留为 `extracted_prev`（Vision2Web 另存 `vision2web_prev`、`score_avg_prev`）。
- `results.jsonl` / `meta.json` 原子重写，`accuracy` / `accuracy_avg_k` / `avg_score` 刷新且把原值留作 `*_prev`。

---

## 6. 看结果

```bash
# 准确率 + token 总量
jq '{accuracy, correct, total, errors, timeouts, total_tokens}' \
   workspace/qwen35_9b/run_xxx/meta.json

# 错题 + extracted
jq -r 'select(.score==0) | "\(.id)  gt=\(.gt | tostring[0:30])  ext=\(.extracted | tostring[0:50])"' \
   workspace/qwen35_9b/run_xxx/results.jsonl

# 工具调用次数分布
jq -r '.n_tool_calls' workspace/.../results.jsonl | sort -n | uniq -c

# 看某条任务的完整对话轨迹
jq . workspace/.../task_9/trace.jsonl | less
```

---

## 7. 常见排查

| 症状 | 原因 / 解决 |
|---|---|
| `error: uv not found` | 装 uv 或改 `UV_BIN` 环境变量指向其路径 |
| FAILED 任务数很多 | 端点过载 / 超时；`./eval.sh --resume <dir>` 续跑，可降 workers |
| 准确率低、`n_tool_calls=0` 一堆 | 模型没用工具；试着把测试模型 temperature 改成 0.6+ |
| 总 token 超大 | reasoning 回灌 + 多轮 image 累积，正常现象 |
| 内网 base_url 但代理把请求劫了 | `eval.sh` / `rejudge.sh` 会自动 detect 内网 IP 并 unset 代理 |
| 结果稳定性差 | vLLM continuous batching 在 temp>0 + 高并发下不可避免有 ±2pp 浮动；多跑几次取平均 |

---

## 8. 项目结构

```
agentic_vision_eval/
├── eval.py                    # 评测入口
├── rejudge.py                 # 重判入口
├── eval.sh / rejudge.sh       # 包装脚本（处理 uv + 内网代理）
├── pyproject.toml             # 依赖（不打包项目本身）
│
├── configs/                   # JSON 配置（含 api_key，已 gitignore）
│   ├── general_config.json
│   └── <model>_config.json
│
├── harness/                   # 库代码（四个职责子包 + tools/，依赖自上而下单向）
│   ├── core/                  # config / llm / benchmark / io / path_mapper / image
│   ├── agents/                # prompts + 三种 rollout 驱动（function_call / code_agent / grounding）
│   ├── scoring/               # extractor / judge / gui_grounding / vision2web
│   ├── runtime/               # sandbox / results / dashboard
│   └── tools/                 # agent 调用的工具实现（内部自洽）
│       ├── filesystem.py / shell.py            # sandbox 内文件 / exec
│       ├── code_kernel.py / code_sandbox.py    # code-agent baseline 的子进程内核
│       └── vision/            # crop / enhance / sample_color / draw / render_html / computer_use / ...
│
├── dataset/                   # benchmark 数据（*.jsonl + images/）
│   ├── Chart_tool_bench_union.jsonl
│   ├── Vision2Web/Vision2Web-webpage.jsonl
│   └── ScreenSpot-Pro_full.jsonl ...
│
└── workspace/                 # 跑出来的 run（gitignore）
    └── <model_name>/run_<timestamp>/...
```

> 库代码组织：`core` 是无业务依赖的地基（配置、LLM 客户端、bench loader、I/O、路径映射、Qwen2.5-VL 几何）；`agents` 是把「模型+工具」跑成 `pred` 的循环；`scoring` 把 `pred` 变成分数；`runtime` 管 sandbox / 聚合 / 实时表格。`eval.py` 按子包导入（`from harness.agents import run`），`harness/__init__.py` 另有扁平 facade。

---

## 9. 一句话用法

```bash
uv sync                                                            # 装环境
cp configs/*.json.example <对应名字>.json && vim ...                # 填 API
./eval.sh --model-config configs/qwen35_9b_config.json             # 跑
./eval.sh --resume workspace/qwen35_9b/run_xxx                     # 中断后续跑
./rejudge.sh workspace/qwen35_9b/run_xxx                           # 改了 judge 后重判
```
