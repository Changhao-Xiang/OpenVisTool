"""Strip the no-tool user-message file-path hint from OpenVisTool_notool swift.

Older no-tool rollouts injected a "[Image file paths — use these in scripts]"
block into the user message (since fixed in nanobot/agent/context.py). Re-rolled
samples no longer carry it; this normalizes the already-good samples so the whole
no-tool dataset is consistent. Operates in place on the merged domain JSONLs.
"""
import json
import re
import sys
from pathlib import Path

# Matches the injected block: a "[... file paths ...]" header line, the bullet
# path line(s), and the trailing blank line separating it from the query.
HINT_RE = re.compile(
    r"\[(?:Image file paths — use these in scripts|File paths — read these using tools such as read_file)\]\n"
    r"(?:- .*\n)+\n"
)


def strip_content(c):
    if not isinstance(c, str):
        return c, False
    new = HINT_RE.sub("", c)
    return new, (new != c)


def main(paths: list[str]) -> None:
    for p in paths:
        path = Path(p)
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        changed = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            hit = False
            for m in d.get("messages", []):
                if m.get("role") == "user":
                    nc, ch = strip_content(m.get("content"))
                    if ch:
                        m["content"] = nc
                        hit = True
            if hit:
                changed += 1
            out.append(json.dumps(d, ensure_ascii=False))
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{path.name}: lines={len(out)} stripped_in={changed}")


if __name__ == "__main__":
    main(sys.argv[1:])
