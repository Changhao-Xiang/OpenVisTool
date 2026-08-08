# Data construction

The `distill/` package implements the OpenVisTool construction pipeline. All
commands below are run from the repository root with a local copy of
`configs/distill.example.json` whose credentials are supplied privately.

1. `filter/filter_difficulty.py` runs no-tool avg@k and records `avg_k` in a
   progress JSONL.
2. `select/select_difficulty_range.py` selects a target difficulty interval.
3. `run.py` rolls out a teacher agent with domain-specific prompts from
   `run/task_specific_toolcall_instructions/`.
4. `filter/filter_acc_toolcall.py` checks final-answer correctness and rejects
   malformed or unsafe tool traces.
5. `filter/filter_tool_gain_with_prefix.py` reruns the base model with the
   teacher's tool observations; `select/select_tool_gain.py` computes gain.
6. `select/select_correct_and_gain.py` intersects the correctness and gain
   selections.
7. `utils/convert_to_swift.py` converts selected sessions to ms-swift JSONL.

Each program exposes its complete interface through `--help`. A typical
teacher rollout is:

```bash
cp configs/distill.example.json configs/distill.local.json
python distill/run.py input.jsonl \
  --query-field query \
  --media-field images \
  --media-dir /path/to/media \
  --config configs/distill.local.json \
  --custom-instructions-override \
    distill/run/task_specific_toolcall_instructions/chart.md \
  --num-workers 8
```

Input records require a stable `id`, a query field, and optionally a list of
image paths. Relative media paths are resolved against `--media-dir`. Sessions
are isolated per item, and generated files are mapped through
`nanobot.agent.path_mapper.PathMapper` before export.

API keys and service URLs are deliberately absent from this release. Use the
`custom` OpenAI-compatible provider in the example config or another public
provider supported by nanobot.

For GUI Grounding rollouts, enable `computer_use` and use the GUI-specific
instruction file; `computer_use` is the format-only terminator that returns
the final click coordinate. For Web-to-HTML rollouts, enable `render_html`
explicitly.

The filter/select scripts are intentionally parameterized CLIs rather than
cluster launch wrappers. Run `python <script> --help` for the exact arguments;
the pipeline order and selection semantics above are the stable public API.

## End-to-end command template

The following credential-free template shows how the outputs connect. Replace
the dataset/model paths and choose the domain-specific accuracy backend. The
thresholds are examples; record the values used for a published experiment.

```bash
export RAW_DATA=/path/to/source.jsonl
export MEDIA_DIR=/path/to/images
export BASE_MODEL=Qwen/Qwen3.5-9B
export BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
mkdir -p artifacts

# 1) No-tool avg@k. This writes both the filtered file and
#    artifacts/difficulty_filtered.progress.jsonl for all attempted rows.
python -m distill.filter.filter_difficulty \
  --dataset "$RAW_DATA" --media-dir "$MEDIA_DIR" \
  --output artifacts/difficulty_filtered.jsonl \
  --model "$BASE_MODEL" --rollout-api-base "$BASE_URL" \
  --k 4 --threshold 0.75 --accuracy-backend rule --num-workers 8

# 2) Optional bounded difficulty interval from the all-row progress file.
python -m distill.select.select_difficulty_range \
  --dataset "$RAW_DATA" \
  --progress artifacts/difficulty_filtered.progress.jsonl \
  --min-avg 0.25 --max-avg 0.75 \
  --output artifacts/candidates.jsonl

# 3) Teacher rollout. Choose the matching instruction file per domain.
python distill/run.py artifacts/candidates.jsonl \
  --query-field query --media-field images --media-dir "$MEDIA_DIR" \
  --config configs/distill.local.json \
  --workspace-override artifacts/teacher_sessions \
  --custom-instructions-override \
    distill/run/task_specific_toolcall_instructions/chart.md \
  --num-workers 8 --continue-on-error

# 4) Correctness and trajectory-quality index.
python -m distill.filter.filter_acc_toolcall \
  --sessions-dir artifacts/teacher_sessions \
  --dataset artifacts/candidates.jsonl \
  --media-dir "$MEDIA_DIR" --output artifacts/correctness.jsonl \
  --accuracy-backend rule --reject-host-paths --reject-missing-images \
  --workers 8

# 5) Base-model accuracy with the teacher's tool-response prefix.
python -m distill.filter.filter_tool_gain_with_prefix \
  --dataset artifacts/candidates.jsonl \
  --sessions-dir artifacts/teacher_sessions --media-dir "$MEDIA_DIR" \
  --output artifacts/with_prefix.jsonl \
  --model "$BASE_MODEL" --rollout-api-base "$BASE_URL" \
  --k 4 --accuracy-backend rule --num-workers 8

# 6) Select positive tool gain, then intersect it with correctness.
python -m distill.select.select_tool_gain \
  --notool-jsonl artifacts/difficulty_filtered.progress.jsonl \
  --with-prefix-jsonl artifacts/with_prefix.jsonl \
  --gain-threshold 0.1 --output artifacts/tool_gain.jsonl \
  --report artifacts/tool_gain_report.json
python -m distill.select.select_correct_and_gain \
  --correctness-index artifacts/correctness.jsonl \
  --tool-gain-index artifacts/tool_gain.jsonl \
  --output artifacts/correct_and_gain.jsonl \
  --report artifacts/correct_and_gain_report.json

# 7) Convert the selected sessions to ms-swift agent SFT JSONL.
#    ENABLED_TOOLS must enumerate every tool enabled during rollout.
export ENABLED_TOOLS='read_file,write_file,edit_file,list_dir,exec,crop,draw_bbox'
python -m distill.utils.convert_to_swift \
  --sessions-dir artifacts/teacher_sessions \
  --index-file artifacts/correct_and_gain.jsonl \
  --output artifacts/openvistool_sft.jsonl \
  --tools "$ENABLED_TOOLS" --workers 8
```

For judge-based QA, add `--accuracy-backend judge --judge-model ...
--judge-api-base ...` consistently in steps 1, 4, and 5. GUI grounding uses
`--accuracy-backend rule --match-mode bbox`; Web-to-HTML uses
`--accuracy-backend html_vlm` plus the judge and rendering arguments. Do not
mix no-tool and with-prefix scores produced by different answer backends.
