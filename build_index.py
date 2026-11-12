# -*- coding: utf-8 -*-
"""扫描配置里的源文件 → 分块 → 向量化 → 增量写入 ChromaDB。

用法：
    python build_index.py               # 读同目录 sources.json
    python build_index.py -c my.json    # 指定配置
    python build_index.py --reset       # 忽略进度记录，全部重建

配置格式见 examples/sources.example.json。
"""
import argparse
import hashlib
import json
from pathlib import Path

from kb_lib import BASE_DIR, chunk_text, embed, get_collection

STATE_FILE = BASE_DIR / "data" / ".ingest_state.json"


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def expand(entry):
    """把一条配置展开成文件列表。path 可以是文件、目录或 glob。"""
    raw = entry.get("path") or ""
    if not raw:
        return []
    p = Path(raw).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        pattern = entry.get("glob", "**/*.md")
        return sorted(x for x in p.glob(pattern) if x.is_file())
    # 当成 glob 处理
    base = Path(p.anchor or ".").expanduser()
    return sorted(x for x in base.glob(str(p)[len(p.anchor):]) if x.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(BASE_DIR / "sources.json"))
    ap.add_argument("--reset", action="store_true", help="忽略进度，全部重建")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(
            f"找不到配置 {cfg_path}，可以从 examples/sources.example.json 复制一份"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    state = {} if args.reset else load_state()

    added = skipped = 0
    for coll_name, entry in cfg.items():
        kind = entry.get("kind", "note")
        files = expand(entry)
        if not files:
            print(f"[{coll_name}] 没有匹配到文件：{entry.get('path')}")
            continue
        col = get_collection(coll_name)
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            key = f"{coll_name}::{f}"
            if state.get(key) == digest:
                skipped += 1
                continue
            chunks = chunk_text(text, str(f), kind)
            if not chunks:
                continue
            ids = [
                hashlib.sha1(f"{key}#{i}".encode("utf-8")).hexdigest()
                for i in range(len(chunks))
            ]
            col.upsert(
                ids=ids,
                documents=[c for c, _ in chunks],
                embeddings=embed([c for c, _ in chunks]),
                metadatas=[{**meta, "source": str(f)} for _, meta in chunks],
            )
            state[key] = digest
            added += 1
            print(f"[{coll_name}] {f.name}: {len(chunks)} 块")

    save_state(state)
    print(f"完成：更新 {added} 个文件，跳过 {skipped} 个未变文件")


if __name__ == "__main__":
    main()
