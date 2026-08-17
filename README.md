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

## 管理员角色隔离（重要）

系统区分**管理员**与**普通用户**两类角色，核心是「管理员负责全量分片入库，用户只拿到消化后的答案、看不到知识库原文」。

### 三层隔离

| 层 | 机制 | 效果 |
|---|---|---|
| **接口鉴权** | `/kb/*`（上传/分片/入库/检索预览）需 `X-Admin-Token` 校验 | 普通用户无法触达管理操作，越权返回 403 |
| **原文不回传** | `/chat` 的 `sources` 仅含 `index/source/score/chunk_count`，**不含 `content`** | 用户点击引用 `[1]` 只看到文件名，原文永不离开服务器 |
| **明文不落盘** | `/kb/split-to-file` 默认**不落盘**；仅 `?save=true` 且管理员鉴权通过才写 `data/chunks/` | 服务器磁盘默认不留原文明文 |

### 配置管理员令牌

```bash
# 环境变量（推荐）或写入 .env
set ADMIN_TOKEN=替换为强随机值    # Windows
```

- 为空 = **关闭鉴权**（仅本地开发允许，生产务必设置）。
- 令牌通过请求头 `X-Admin-Token: <token>` 传递。

### 管理员检索预览（含原文）

仅管理员可调用，用于核查检索质量：

```bash
curl -H "X-Admin-Token: <token>" "http://localhost:8000/kb/search?query=年假"
# 返回 sources 含原文明文（content 字段）
```

### 前端表现

- 「🔒 知识库管理」Tab 默认需输入管理员令牌才解锁；令牌仅存内存（刷新即失效）。
- 用户问答页点击引用 `[1]` 提示「原文不对外开放，仅管理员可在后台预览」。

## 测试

```bash
uv run pytest -v
```

## 目录结构

```
src/
├── knowledge/     # 文档加载、切分、Embedding、向量入库、/kb 管理接口
├── rag/           # LCEL 链：检索、Prompt、LLM 生成 / 降级（流式）
├── compliance/    # 敏感词过滤、脱敏
├── chat/          # 会话管理、意图分类（闲聊分流）
├── web/           # 网页前端
├── auth.py        # 管理员令牌鉴权（require_admin）
├── config.py      # 配置加载
├── logging_conf.py
└── main.py        # FastAPI 入口（/chat /healthz）
scripts/           # 索引构建、模型下载等工具
data/docs/         # 原始文档（示例）
configs/           # config.yaml
```

## 技术方案

详见 `企业内部知识库问答机器人技术方案.md`。
