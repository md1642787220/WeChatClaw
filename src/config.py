"""配置加载：基于 pydantic-settings，支持 YAML 文件 + 环境变量覆盖。

加载优先级：环境变量 > .env 文件 > YAML 配置文件。
敏感凭证（API Key、管理员令牌）建议通过环境变量或 .env 注入，勿明文提交。

Author: MADENG
Reviewer: Li Rongdong
"""
import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# 应用基础配置。
#
# 属性：
#     name: 应用名。
#     env: 运行环境（dev/prod）。
#     log_level: 日志级别。
#     host: 监听地址。
#     port: 监听端口。
class AppConfig(BaseSettings):
    name: str = "rag-kf-support"
    env: str = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


# 向量库配置。
#
# 属性：
#     type: 向量库类型（当前固定 chroma）。
#     host: 远程向量库地址（未使用）。
#     port: 远程向量库端口（未使用）。
#     collection: 集合名。
#     persist_dir: 持久化目录。
class VectorDBConfig(BaseSettings):
    type: str = "chroma"
    host: str = ""
    port: int = 0
    collection: str = "kf_knowledge"
    persist_dir: str = "data/chroma"


# Embedding 模型配置。
#
# 属性：
#     provider: 提供方（当前固定 local）。
#     model: 模型名。
#     dimension: 向量维度。
#     device: 计算设备（cpu/cuda）。
#
# 注意：
#     采用本地 sentence-transformers 模型（bge-small-zh-v1.5：约 95MB、512 维，
#     轻量中文模型，CPU 可跑），避免依赖外部 API。
class EmbeddingConfig(BaseSettings):
    provider: str = "local"  # local
    model: str = "BAAI/bge-small-zh-v1.5"
    dimension: int = 512
    device: str = "cpu"  # cpu / cuda


# LLM 配置（OpenAI 兼容协议）。
#
# 属性：
#     provider: 提供方（deepseek/openai/qwen/hunyuan）。
#     api_key: API Key（环境变量 LLM_API_KEY；为空则走降级兜底）。
#     base_url: API base URL。
#     model: 模型名。
#     temperature: 采样温度。
#     timeout_seconds: 请求超时。
#     system_role: 系统角色名（调用 LLM 时作为 messages[0].role="system"）。
class LLMConfig(BaseSettings):
    provider: str = "deepseek"  # deepseek / qwen / hunyuan
    api_key: str = ""  # 环境变量 LLM_API_KEY；为空则走降级兜底
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    timeout_seconds: int = 30
    # 遵守 OpenAI 兼容 messages 规范，详见 README「LLM 调用 role 规范」一节。
    system_role: str = "知识库助手"

    # 有没有配上能用的 LLM（没配 Key 时走降级）。
    @property
    def available(self):
        api_key_value = self.api_key
        if not api_key_value:
            return False
        return True


# 检索配置。
#
# 属性：
#     top_k: 检索返回的最大文档数。
#     threshold: 相似度阈值，低于这个值算没命中。
class RetrievalConfig(BaseSettings):
    top_k: int = 5
    threshold: float = 0.5


# 合规配置。
#
# 属性：
#     sensitive_words: 敏感词列表。
#     transfer_keywords: 转人工关键词。
#     max_history_rounds: 保留的最大历史轮数。
class ComplianceConfig(BaseSettings):
    sensitive_words: list = []
    transfer_keywords: list = ["人工", "转人工", "投诉"]
    max_history_rounds: int = 10


# 管理员鉴权配置。
#
# 属性：
#     token: 管理员令牌（通过环境变量 ADMIN_TOKEN 或 .env 注入，勿明文写入 YAML）。
#
# 注意：
#     令牌用来保护 /kb/* 管理接口（上传/分片/入库/预览原文）。
#     为空表示关闭鉴权（仅建议本地开发使用）。
class AdminConfig(BaseSettings):
    token: str = ""


# 全局配置汇总。
#
# 属性：
#     app: 应用基础配置。
#     vector_db: 向量库配置。
#     embedding: Embedding 配置。
#     llm: LLM 配置。
#     retrieval: 检索配置。
#     compliance: 合规配置。
#     admin: 管理员鉴权配置。
class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    admin: AdminConfig = AdminConfig()

    model_config = SettingsConfigDict(env_nested_delimiter="__")


# 读 YAML 配置文件成字典。
#
# 参数：
#     file_path: 配置文件路径。
#
# 返回：
#     解析后的字典；文件不存在或内容为空时返回空字典。
def _load_yaml(file_path):
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    if data is None:
        return {}
    return data


# 嵌套环境变量映射：env key -> (section, field)
# 用来把 LLM__TEMPERATURE 这类环境变量塞进 Pydantic Settings 的嵌套结构里。
_NESTED_ENV_KEYS = {
    "LLM__PROVIDER": ("llm", "provider"),
    "LLM__API_KEY": ("llm", "api_key"),
    "LLM__BASE_URL": ("llm", "base_url"),
    "LLM__MODEL": ("llm", "model"),
    "LLM__TEMPERATURE": ("llm", "temperature"),
    "LLM__TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "LLM__SYSTEM_ROLE": ("llm", "system_role"),
    "EMBEDDING__PROVIDER": ("embedding", "provider"),
    "EMBEDDING__MODEL": ("embedding", "model"),
    "EMBEDDING__DIMENSION": ("embedding", "dimension"),
    "EMBEDDING__DEVICE": ("embedding", "device"),
    "RETRIEVAL__TOP_K": ("retrieval", "top_k"),
    "RETRIEVAL__THRESHOLD": ("retrieval", "threshold"),
    "VECTOR_DB__TYPE": ("vector_db", "type"),
    "VECTOR_DB__HOST": ("vector_db", "host"),
    "VECTOR_DB__PORT": ("vector_db", "port"),
    "VECTOR_DB__COLLECTION": ("vector_db", "collection"),
    "VECTOR_DB__PERSIST_DIR": ("vector_db", "persist_dir"),
    "ADMIN_TOKEN": ("admin", "token"),
    # 兼容旧式单层键：LLM_API_KEY（不依赖嵌套分隔符）
    "LLM_API_KEY": ("llm", "api_key"),
}

# 数值字段：要做 int/float 类型转换
_INT_FIELDS = {"timeout_seconds", "dimension", "port", "top_k"}
_FLOAT_FIELDS = {"temperature", "threshold"}


# 把环境变量字符串值转成目标字段类型。
#
# 参数：
#     raw_value: 原始字符串值。
#     field_name: 字段名，用来判断目标类型。
#
# 返回：
#     按字段类型转换后的值（int / float / str）。
#
# 注意：
#     数值字段类型由 _INT_FIELDS 和 _FLOAT_FIELDS 集合决定。
def _coerce_value(raw_value, field_name):
    if field_name in _INT_FIELDS:
        return int(raw_value)
    if field_name in _FLOAT_FIELDS:
        return float(raw_value)
    return raw_value


# 加载配置：YAML 文件打底，环境变量 + .env 覆盖敏感项。
#
# 参数：
#     config_path: 可选配置文件路径，默认 configs/config.yaml。
#
# 返回：
#     合并后的全局配置实例。
#
# 注意：
#     每次调用都重读（不缓存），这样 PATCH /kb/config 后下个请求就能拿到最新值。
def read_settings(config_path=None):
    if config_path is None:
        path = Path("configs/config.yaml")
    else:
        path = Path(config_path)

    yaml_data = _load_yaml(path)

    # 合并 .env 文件（简单解析，不引依赖）
    dotenv_data = _load_dotenv()

    # 合并 env var + dotenv：env var 优先
    merged = {}
    for dotenv_key, dotenv_value in dotenv_data.items():
        merged[dotenv_key] = dotenv_value
    for env_key, env_value in os.environ.items():
        merged[env_key] = env_value

    # 把白名单嵌套键合并到 Pydantic Settings 结构
    settings_dict = dict(yaml_data)
    for env_key, mapping in _NESTED_ENV_KEYS.items():
        section_name = mapping[0]
        field_name = mapping[1]
        env_value = merged.get(env_key)
        if env_value is None:
            continue
        if section_name not in settings_dict:
            settings_dict[section_name] = {}
        section_dict = settings_dict[section_name]
        section_dict[field_name] = _coerce_value(env_value, field_name)

    return Settings(**settings_dict)


# 读 .env 文件（简单解析，不引入额外依赖）。
#
# 参数：
#     file_path: 可选 .env 文件路径，默认项目根目录 .env。
#
# 返回：
#     键值字典；文件不存在时返回空字典。
#
# 注意：
#     优先级：环境变量 > .env 文件 > YAML 配置。
def _load_dotenv(file_path=None):
    if file_path is None:
        dotenv_path = Path(".env")
    else:
        dotenv_path = Path(file_path)

    if not dotenv_path.exists():
        return {}

    result = {}
    file_text = dotenv_path.read_text(encoding="utf-8")
    lines = file_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        # 按第一个 "=" 分割，避免 value 里出现 "=" 时被截断
        partition_index = stripped.index("=")
        key = stripped[:partition_index].strip()
        value = stripped[partition_index + 1:].strip()
        # 去掉首尾成对引号
        value = value.strip("'\"")
        if not key:
            continue
        result[key] = value
    return result
