# -*- coding: utf-8 -*-
"""本地知识库共享逻辑：embedding 模型、ChromaDB 存取、分块。"""
import os
from pathlib import Path

# 本机访问 HuggingFace 走镜像（须在 import fastembed 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None
_store = None

# 各类内容 → collection 名（英文，稳妥）
COLLECTIONS = {
    "pitfalls": "踩坑",
    "code": "源码",
    "domain": "领域",
    "project": "项目",
    "english": "英语",
}


def get_model():
    """懒加载 embedding 模型（单例，离线加载本地模型）。"""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(BASE_DIR / "models"),
            local_files_only=True,
        )
    return _model


def embed(texts):
    """把若干段文本编码成向量（list[list[float]]，转成 Python float）。"""
    model = get_model()
    return [[float(x) for x in v] for v in model.embed(list(texts))]


def get_store():
    """懒加载 ChromaDB 持久化客户端（单例）。"""
    global _store
    if _store is None:
        import chromadb
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _store = chromadb.PersistentClient(path=str(DATA_DIR))
    return _store


def get_collection(name):
    """按英文 collection 名取（不存在则建，cosine 距离）。"""
    store = get_store()
    return store.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def _split_long(text, max_chars):
    """超长文本按段落边界切，尽量不超过 max_chars。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    paras = [p for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = cur + "\n" + p if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _split_bullets(body, per_group=2):
    """把正文里的「- 」列表按每 per_group 条一组切开；非列表行归入首组。

    用途：词汇/搭配这类**长列表**文件。整段 17 条挤成一个向量会被稀释，
    查单个词时用短查询很难命中。实测（查询「eligible 后面接什么」）：

    - 17 条一整块：相关块 0.403
    - 4 条一组：  相关块 0.459
    - 单条独立：  相关块 0.561

    所以按每 2 条一组切。每块仍带同一段标题（如「[二、形容词 + 介词]」），
    上下文不丢。列表不足两组时返回 None，交给调用方走原来的逻辑。
    """
    groups, cur, head = [], [], []
    seen_bullet = False
    for ln in body.split("\n"):
        if ln.lstrip().startswith("- "):
            seen_bullet = True
            cur.append(ln)
            if len(cur) >= per_group:
                groups.append(cur)
                cur = []
        elif not seen_bullet:
            head.append(ln)
        else:
            cur.append(ln)  # 列表中夹着的续表/说明行，跟着当前组走
    if cur:
        groups.append(cur)
    if len(groups) <= 1:
        return None
    out = []
    for i, g in enumerate(groups):
        pre = "\n".join(head).strip() if i == 0 else ""
        out.append("\n".join(x for x in (pre, "\n".join(g)) if x.strip()))
    return out


def _split_by_heading(text):
    """按 markdown 标题（#/##）切块，返回 [(标题, 正文)]。"""
    lines = text.split("\n")
    sections, cur_title, cur_body = [], "", []
    for ln in lines:
        if ln.startswith("#"):
            if cur_title or cur_body:
                sections.append((cur_title, "\n".join(cur_body)))
            cur_title = ln.lstrip("#").strip()
            cur_body = []
        else:
            cur_body.append(ln)
    if cur_title or cur_body:
        sections.append((cur_title, "\n".join(cur_body)))
    return sections


def chunk_text(text, source, kind, max_chars=1500):
    """把一段源文本切成块，返回 [(chunk_text, extra_meta)]。

    kind: note（笔记，按标题切）/ code（源码，按文件切，超长再分）
          / vocab（词汇列表，按标题切后再把长列表拆成小组）。
    """
    out = []
    if kind == "vocab":
        sections = _split_by_heading(text) or [("", text)]
        for title, body in sections:
            head = f"[{title}]\n" if title else ""
            groups = _split_bullets(body) or _split_long(body, max_chars)
            for i, g in enumerate(groups):
                out.append((head + g, {"part": i}))
        return out
    if kind == "code":
        header = f"[文件] {source}\n"
        for i, part in enumerate(_split_long(text, max_chars)):
            out.append((header + part, {"part": i}))
    else:
        sections = _split_by_heading(text)
        if not sections:
            sections = [("", text)]
        for title, body in sections:
            head = f"[{title}]\n" if title else ""
            parts = _split_long(body, max_chars)
            for i, part in enumerate(parts):
                out.append((head + part, {"part": i}))
    return out
