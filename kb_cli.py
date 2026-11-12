# -*- coding: utf-8 -*-
"""知识库命令行入口：输出 JSON，方便被其他程序（脚本 / Agent / 桥接服务）调用。

用法：
  kb_cli.py search <query> [top_k] [collection]
  kb_cli.py add <collection> <source>      # 正文从 stdin 读
  kb_cli.py list
输出 JSON（UTF-8），方便桥那侧解析。
"""
import json
import sys
import uuid

from kb_lib import COLLECTIONS, embed, get_collection


def _score(d):
    return max(0.0, min(1.0, 1.0 - float(d)))


def search(query: str, top_k: int = 5, collection: str = "") -> list[dict]:
    qv = embed([query])[0]
    colls = [collection] if collection else list(COLLECTIONS.keys())
    hits = []
    for c in colls:
        if c not in COLLECTIONS:
            continue
        col = get_collection(c)
        r = col.query(
            query_embeddings=[qv],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(r["ids"][0])):
            meta = (r["metadatas"][0][i] or {})
            hits.append({
                "collection": c,
                "label": COLLECTIONS[c],
                "source": meta.get("source", ""),
                "score": _score(r["distances"][0][i]),
                "text": r["documents"][0][i],
            })
    hits.sort(key=lambda x: -x["score"])
    return hits[:top_k]


def add(collection: str, content: str, source: str = "") -> dict:
    content = content.strip()
    if not content:
        return {"ok": False, "error": "empty content"}
    col = get_collection(collection)
    cid = uuid.uuid4().hex
    col.add(
        ids=[cid],
        documents=[content],
        embeddings=embed([content]),
        metadatas=[{"source": source}],
    )
    return {"ok": True, "id": cid, "count": col.count()}


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(json.dumps({"ok": False, "error": "no command"}))
        return 1
    cmd = args[0]
    if cmd == "search":
        query = args[1] if len(args) > 1 else ""
        top_k = int(args[2]) if len(args) > 2 else 5
        collection = args[3] if len(args) > 3 else ""
        print(json.dumps(search(query, top_k, collection), ensure_ascii=False))
        return 0
    if cmd == "add":
        collection = args[1] if len(args) > 1 else "domain"
        source = args[2] if len(args) > 2 else ""
        content = sys.stdin.read()
        print(json.dumps(add(collection, content, source), ensure_ascii=False))
        return 0
    if cmd == "list":
        out = []
        for c, label in COLLECTIONS.items():
            try:
                out.append({"collection": c, "label": label, "count": get_collection(c).count()})
            except Exception as e:  # noqa: BLE001
                out.append({"collection": c, "label": label, "error": str(e)})
        print(json.dumps(out, ensure_ascii=False))
        return 0
    print(json.dumps({"ok": False, "error": f"unknown command {cmd}"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
