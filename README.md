# 本地 RAG 知识库（MCP 服务）

一个**全程离线**的个人知识库：把散落的笔记、源码、项目文档切块向量化，通过 **MCP（Model Context Protocol）** 暴露成检索工具，Agent 可以直接调用。

不依赖任何云端 embedding API，模型在本地跑；断网可用。

## 特性

- **离线向量化**：`fastembed` 加载 `BAAI/bge-small-zh-v1.5`（ONNX，512 维），首次下载后全程本地推理
- **多集合隔离**：按内容类型分成多个 collection（踩坑 / 源码 / 领域 / 项目 / 词汇），检索时可按类限定
- **增量索引**：按文件内容哈希记录进度，内容没变的文件第二次建库直接跳过
- **三种切块策略**：笔记按标题切、源码按文件切、词汇列表按「标题 + 每 N 条一组」切
- **两种接入方式**：MCP stdio 服务（给 Agent）+ 命令行 JSON 输出（给脚本）

## 架构

```
源文件 (md / py)
      │  build_index.py   扫描 → 切块 → embed
      ▼
   ChromaDB (data/)      持久化向量 + 原文 + 元数据
      │
      ├── kb_server.py    MCP stdio 服务  ──► Agent 调用 search / list_collections
      └── kb_cli.py       命令行          ──► 其他程序调用，输出 JSON
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp examples/sources.example.json sources.json   # 改成你自己的路径
python build_index.py
```

`sources.json` 的每一项是一类语料：

```json
{
  "notes": { "kind": "note", "path": "~/notes", "glob": "**/*.md" },
  "code":  { "kind": "code", "path": "~/projects/my-app/src", "glob": "**/*.py" }
}
```

`kind` 决定切块方式：`note`（按标题）/ `code`（按文件）/ `vocab`（按标题后拆列表）。

## 接入 Agent（MCP）

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "C:/path/to/.venv/Scripts/python.exe",
      "args": ["C:/path/to/kb_server.py"]
    }
  }
}
```

暴露两个工具：

| 工具 | 作用 |
|---|---|
| `search(query, top_k=5, collection="")` | 语义检索，返回原文片段 + 来源 + 相关度 |
| `list_collections()` | 列出各分类及块数 |

## 命令行

```bash
python kb_cli.py search "怎么解决 Node 内存损坏" 5
python kb_cli.py list
```

输出是 JSON，方便被其他进程解析。

## 切块策略：为什么词汇要拆

长列表整段塞进一个向量会被稀释——查询单个词时，整块 17 条的平均语义离任何单条都远。实测（查询「eligible 后面接什么」，相关块余弦相似度）：

| 切法 | 相关块相关度 |
|---|---|
| 17 条一整块 | 0.403 |
| 每 4 条一组 | 0.459 |
| 单条独立 | 0.561 |

最终取每 2 条一组：再细会丢上下文（每组都复制一遍标题），再粗就命不中。

## 一个真实的坑：集合路由

中英混排语料混在同一个集合里时，中文文档会把英文查询的 top-k 全占掉。这个项目的处理是**给集合加路由**——查英文必须显式指定 `collection="english"`，否则召回会被中文噪声顶掉。

如果你的语料是纯中文或者纯英文，可以忽略这一条。

## 目录结构

```
kb_lib.py          向量化 / 存储 / 切块（共享逻辑）
build_index.py     建库入口（读 sources.json，增量）
kb_server.py       MCP stdio 服务
kb_cli.py          命令行 / JSON 输出
examples/          配置模板
```

## 说明

- `data/`（向量库）和 `models/`（ONNX 模型）不入版本库，见 `.gitignore`；换机器重新建库即可
- 首次运行会从 HuggingFace 拉模型；国内网络可在 `kb_lib.py` 顶部通过 `HF_ENDPOINT` 走镜像（已默认设置）

## License

MIT
