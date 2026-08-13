# 企业内部知识库问答机器人（rag-kf-support）

基于企业自有知识库的 RAG 智能问答系统：员工就「规章制度 / 产品知识 / 业务流程 / FAQ」等内部知识提问，系统检索知识库并生成有据可依的回答。

核心能力 = RAG 问答引擎，通过独立网页提供问答服务。

技术栈：**FastAPI + LangChain（LCEL） + Chroma + sentence-transformers + DeepSeek（可选）**

## 快速开始

```bash
# 安装依赖（含 RAG 与开发分组）
uv sync --extra rag --extra dev

# 构建向量索引（将 data/docs 下的文档切分、向量化、入库）
uv run python -m scripts.build_index

# 启动服务
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# 打开网页聊天
# 浏览器访问 http://localhost:8000
```

## 配置 LLM（可选）

未配置 LLM Key 时，系统自动走**降级模式**（返回命中的知识片段原文 + 出处）。

配置 DeepSeek Key 后升级为生成式回答：

```bash
# 环境变量方式
set LLM_API_KEY=sk-xxxxx    # Windows
# 或写入 .env 文件
```

## 测试

```bash
uv run pytest -v
```

## 目录结构

```
src/
├── knowledge/     # 文档加载、切分、Embedding、向量入库
├── rag/           # LCEL 链：检索、Prompt、LLM 生成 / 降级
├── compliance/    # 敏感词过滤、脱敏
├── chat/          # 会话管理
├── web/           # 网页前端
├── config.py      # 配置加载
├── logging_conf.py
└── main.py        # FastAPI 入口（/chat /healthz）
scripts/           # 索引构建等工具
data/docs/         # 原始文档（示例）
configs/           # config.yaml
```

## 技术方案

详见 `企业内部知识库问答机器人技术方案.md`。
