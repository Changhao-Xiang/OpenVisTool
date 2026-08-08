import json
import re
from pathlib import Path


def task_key(item: dict) -> str:
    """Stable per-item key used for task directories and resume bookkeeping."""
    source = str(item.get("source_dataset") or "dataset")
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "dataset"
    return f"{safe_source}_{item['id']}"


def load_benchmark(jsonl_path: str, images_dir: str | None = None) -> list[dict]:
    jsonl_path = Path(jsonl_path)
    images_root = Path(images_dir) if images_dir else jsonl_path.parent / "images"
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            src_dataset = d.get("source_dataset", "")
            img_dir = images_root / src_dataset if src_dataset else images_root
            d["image_paths"] = [str(img_dir / img) for img in d.get("images", [])]
            items.append(d)
    return items


if __name__ == "__main__":
    items = load_benchmark("dataset/Vision2Web/Vision2Web-webpage.jsonl")
    print(f"loaded {len(items)} items")
    print("sample:", {k: items[0][k] for k in ("id", "query", "answer", "image_paths")})
