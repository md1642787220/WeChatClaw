# 对外智能客服系统（rag-kf-support）

企业自有知识库 + 微信客服对外智能客服系统：客户通过「微信客服」渠道自助咨询，释放人工咨询量。

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"
pip install pyyaml

# 启动服务
uvicorn src.main:app --host 0.0.0.0 --port 8000

# 健康检查
curl http://localhost:8000/healthz
```

## 测试

```bash
pytest -v
```

## 目录结构

```
src/
├── wecom_kf/    # 微信客服接入（M1）
├── compliance/  # 安全合规（M4）
├── knowledge/   # 知识库管道（M2）
├── rag/         # RAG 引擎（M3）
├── chat/        # 会话管理（M3）
├── config.py    # 配置加载
├── logging_conf.py
└── main.py      # FastAPI 入口
```

详细技术方案见 `企业微信RAG智能客服技术方案.md`，实施计划见 `实施方案.md`。
