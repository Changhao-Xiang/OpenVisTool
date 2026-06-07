#!/usr/bin/env bash
# Sequentially evaluate multiple benchmarks (each with its own eval_mode)
# in a single invocation. Every bench gets its own fresh run dir under
# workspace/<model>/run_<timestamp>/.
#
# Usage:
#   ./eval_multi.sh --model-config configs/qwen35_9b.json
#   ./eval_multi.sh --model-config configs/xxx.json -n 5 --no-tools
#   WORKERS=16 MAX_STEPS=30 ./eval_multi.sh --model-config configs/xxx.json
#
# Extra flags (-n / --no-tools / --workers / etc) are forwarded as-is to
# each underlying ./eval.sh call.
#
# Edit BENCHES below to change the bench list / order / eval_mode mapping.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# (bench_path, eval_mode) pairs — run in this order.
BENCHES=(
    "dataset/Chart_tool_bench_union.jsonl|judge"
    "dataset/GUI_tool_bench_union.jsonl|gui"
    "dataset/Table_tool_bench_union.jsonl|judge"
    "dataset/VisualProbe_Hard.jsonl|judge"
    # "dataset/Vision2Web/Vision2Web-webpage.jsonl|judge"
)

# Forwarded flags
FWD_ARGS=("$@")

failed=()
declare -A run_dirs
declare -a order   # preserve bench order for the summary
for entry in "${BENCHES[@]}"; do
    bench="${entry%%|*}"
    mode="${entry##*|}"
    name="$(basename "$bench" .jsonl)"
    order+=("$name")

    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "▶ [$name]  bench=$bench  eval_mode=$mode"
    echo "════════════════════════════════════════════════════════════════════"

    if [ ! -f "$bench" ]; then
        echo "✗ bench not found: $bench — skipping" >&2
        failed+=("$name (missing file)")
        continue
    fi

    # Each bench is a fresh run — don't pass --resume here.
    # We *don't* pipe eval.sh's stdout (it would disable the rich.Live
    # dashboard, which requires a TTY). Instead, snapshot the newest
    # workspace/<model>/run_* dir before and after to recover the run dir.
    before=$(ls -dt workspace/*/run_* 2>/dev/null | head -1 || true)

    set +e
    ./eval.sh --bench "$bench" --eval-mode "$mode" "${FWD_ARGS[@]}"
    rc=$?
    set -e

    after=$(ls -dt workspace/*/run_* 2>/dev/null | head -1 || true)
    if [ -n "$after" ] && [ "$after" != "$before" ]; then
        run_dirs[$name]="$after"
    fi

    if [ "$rc" -ne 0 ]; then
        echo "✗ [$name] eval.sh exited with $rc — continuing with next bench" >&2
        failed+=("$name (exit=$rc)")
    fi
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "▶ multi-bench eval done"

# ---- summary across benches ---------------------------------------------
summary_pairs=()
for name in "${order[@]}"; do
    summary_pairs+=("$name=${run_dirs[$name]:-}")
done

python3 - "${summary_pairs[@]}" <<'PY'
import json, sys
from pathlib import Path

rows = []
for arg in sys.argv[1:]:
    name, _, rd = arg.partition("=")
    row = {"name": name, "acc": None, "acc_avg_k": None, "avg_k": None,
           "n": None, "avg_tools": None, "run_dir": rd}
    if rd and (Path(rd) / "meta.json").exists():
        meta = json.loads((Path(rd) / "meta.json").read_text())
        row["acc"]       = meta.get("accuracy")
        row["acc_avg_k"] = meta.get("accuracy_avg_k")
        row["avg_k"]     = meta.get("avg_k")
        row["n"]         = meta.get("total")

        # average tool-call count across all samples (or primary if no avg@k)
        tools, n = 0, 0
        for rj in Path(rd).glob("task_*/result.json"):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            samples = r.get("samples")
            if samples:
                for s in samples:
                    tools += s.get("n_tool_calls", 0); n += 1
            else:
                tools += r.get("n_tool_calls", 0); n += 1
        row["avg_tools"] = (tools / n) if n else None
    rows.append(row)

def fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "-"

hdr = f"{'bench':<32}  {'n':>5}  {'acc':>7}  {'avg@k':>7}  {'k':>3}  {'avg_tools':>10}"
print(hdr)
print("─" * len(hdr))
for r in rows:
    print(f"{r['name']:<32}  "
          f"{fmt(r['n'], 'd' if isinstance(r['n'], int) else ''):>5}  "
          f"{fmt(r['acc'], '.4f'):>7}  "
          f"{fmt(r['acc_avg_k'], '.4f'):>7}  "
          f"{fmt(r['avg_k'], 'd' if isinstance(r['avg_k'], int) else ''):>3}  "
          f"{fmt(r['avg_tools'], '.2f'):>10}")

ok = [r for r in rows if r["acc"] is not None]
if len(ok) >= 2:
    def mean(key):
        xs = [r[key] for r in ok if isinstance(r[key], (int, float))]
        return sum(xs) / len(xs) if xs else None
    print("─" * len(hdr))
    print(f"{'mean':<32}  "
          f"{'':>5}  "
          f"{fmt(mean('acc'), '.4f'):>7}  "
          f"{fmt(mean('acc_avg_k'), '.4f'):>7}  "
          f"{'':>3}  "
          f"{fmt(mean('avg_tools'), '.2f'):>10}")
PY

if [ "${#failed[@]}" -gt 0 ]; then
    echo
    echo "  failed:"
    for f in "${failed[@]}"; do echo "    - $f"; done
    exit 1
fi
echo "  all $((${#BENCHES[@]})) benches finished"
