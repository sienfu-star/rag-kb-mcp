# -*- coding: utf-8 -*-
"""本地知识库 MCP server（stdio）：提供语义检索工具给 Claude。"""
from mcp.server.mcpserver import MCPServer

from kb_lib import COLLECTIONS, embed, get_collection

mcp = MCPServer("knowledge-base")


def _fmt_score(d):
    # chromadb 用 cosine 距离，相似度 = 1 - distance
    s = 1.0 - float(d)
    return max(0.0, min(1.0, s))


@mcp.tool()
def search(query: str, top_k: int = 5, collection: str = "") -> str:
    """语义检索本地知识库，返回最相关的经验/源码片段。

    query: 自然语言问题或关键词
    top_k: 返回条数（默认 5）
    collection: 限定某类，可选 pitfalls/code/domain/project；留空则全库搜
    """
    qv = embed([query])[0]
    colls = [collection] if collection else list(COLLECTIONS.keys())

    hits = []
    for c in colls:
        if c not in COLLECTIONS:
            continue
        try:
            col = get_collection(c)
        except Exception:
            continue
        r = col.query(
            query_embeddings=[qv],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(r["ids"][0])):
            hits.append({
                "collection": c,
                "label": COLLECTIONS[c],
                "source": r["metadatas"][0][i].get("source", ""),
                "score": _fmt_score(r["distances"][0][i]),
                "text": r["documents"][0][i],
            })

    hits.sort(key=lambda x: -x["score"])
    hits = hits[:top_k]

    if not hits:
        return "没找到相关内容哦。"

    lines = []
    for j, h in enumerate(hits, 1):
        lines.append(
            f"### {j}. [{h['label']}] {h['source']}（相关度 {h['score']:.2f}）"
        )
        lines.append(h["text"][:1200])
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def list_collections() -> str:
    """列出知识库各分类及其 chunk 数。"""
    out = []
    for c, label in COLLECTIONS.items():
        try:
            col = get_collection(c)
            out.append(f"- {label}（{c}）：{col.count()} 块")
        except Exception as e:
            out.append(f"- {label}（{c}）：读取失败 {e}")
    return "\n".join(out) if out else "知识库为空"


if __name__ == "__main__":
    mcp.run()
