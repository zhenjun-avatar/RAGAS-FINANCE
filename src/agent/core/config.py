# -*- coding: utf-8 -*-
"""应用配置（Node-centric RAG stack）。"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_config_dir = Path(__file__).parent
_possible_env_paths = [
    _config_dir.parent / ".env",
    _config_dir.parent.parent / ".env",
    _config_dir.parent.parent.parent / ".env",
]

_env_loaded = False
for env_path in _possible_env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        _env_loaded = True
        break
if not _env_loaded:
    load_dotenv()


class Config(BaseSettings):
    """extra='ignore'：兼容旧 .env 里已删除功能的变量，避免 ValidationError。"""

    model_config = SettingsConfigDict(
        env_file=str(_possible_env_paths[0]) if _possible_env_paths else ".env",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    qwen_api_key: Optional[str] = Field(default=None, env="QWEN_API_KEY")

    default_model: str = Field(default="deepseek/deepseek-chat", env="DEFAULT_MODEL")

    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    # Relative paths resolve from src/agent. Empty / unset: stderr only (see runtime_logging).
    log_file: str = Field(default="logs/agent.log", env="LOG_FILE")

    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    db_user: str = Field(default="postgres", env="DB_USER")
    db_password: str = Field(default="postgres", env="DB_PASSWORD")
    db_name: str = Field(default="rag", env="DB_NAME")
    db_host: str = Field(default="127.0.0.1", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")

    qdrant_host: str = Field(default="127.0.0.1", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="rag_nodes", env="QDRANT_COLLECTION")
    dense_backend: str = Field(default="qdrant", env="DENSE_BACKEND")
    sparse_backend: str = Field(default="postgres", env="SPARSE_BACKEND")

    opensearch_host: str = Field(default="127.0.0.1", env="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, env="OPENSEARCH_PORT")
    opensearch_user: Optional[str] = Field(default=None, env="OPENSEARCH_USER")
    opensearch_password: Optional[str] = Field(default=None, env="OPENSEARCH_PASSWORD")
    opensearch_use_ssl: bool = Field(default=False, env="OPENSEARCH_USE_SSL")
    opensearch_verify_certs: bool = Field(default=True, env="OPENSEARCH_VERIFY_CERTS")
    opensearch_timeout_seconds: float = Field(
        default=15.0, env="OPENSEARCH_TIMEOUT_SECONDS"
    )
    opensearch_sparse_index: str = Field(
        default="rag_nodes_sparse", env="OPENSEARCH_SPARSE_INDEX"
    )
    opensearch_sparse_index_finance: Optional[str] = Field(
        default=None, env="OPENSEARCH_SPARSE_INDEX_FINANCE"
    )
    # sparse：finance=仅财务索引；all=财务索引 + OPENSEARCH_SPARSE_INDEX（默认兜底，便于扩展第二域时再拆分）
    opensearch_sparse_search_scope: str = Field(
        default="finance", env="OPENSEARCH_SPARSE_SEARCH_SCOPE"
    )
    opensearch_sparse_analyzer: Optional[str] = Field(
        default=None, env="OPENSEARCH_SPARSE_ANALYZER"
    )
    opensearch_sparse_search_analyzer: Optional[str] = Field(
        default=None, env="OPENSEARCH_SPARSE_SEARCH_ANALYZER"
    )

    embedding_model: str = Field(default="text-embedding-v3", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    # OpenAI: can be 64–200. DashScope text-embedding-v3 compatible API allows ~6 texts per request.
    embedding_batch_size: int = Field(default=64, env="EMBEDDING_BATCH_SIZE")
    # auto | qwen | openai — use openai if DashScope (dashscope.aliyuncs.com) is blocked or unstable.
    embedding_provider: str = Field(default="auto", env="EMBEDDING_PROVIDER")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )

    rag_pipeline_version: str = Field(default="node-rag-v1", env="RAG_PIPELINE_VERSION")
    hierarchical_group_size: int = Field(default=6, env="HIERARCHICAL_GROUP_SIZE")
    retrieve_top_k: int = Field(default=12, env="RETRIEVE_TOP_K")
    retrieve_candidate_k: int = Field(default=40, env="RETRIEVE_CANDIDATE_K")
    sparse_top_k: int = Field(default=30, env="SPARSE_TOP_K")
    # pipeline_trace 里 retrieval_sparse_hits 每条的正文预览长度上限
    pipeline_trace_sparse_text_chars: int = Field(default=800, env="PIPELINE_TRACE_SPARSE_TEXT_CHARS")
    dense_top_k: int = Field(default=30, env="DENSE_TOP_K")
    context_neighbor_radius: int = Field(default=1, env="CONTEXT_NEIGHBOR_RADIUS")

    bocha_reranker_url: Optional[str] = Field(default=None, env="BOCHA_RERANKER_URL")
    bocha_api_key: Optional[str] = Field(default=None, env="BOCHA_API_KEY")
    bocha_reranker_model: str = Field(
        default="bocha-semantic-reranker-cn",
        env="BOCHA_RERANKER_MODEL",
    )
    bocha_timeout_seconds: float = Field(default=8.0, env="BOCHA_TIMEOUT_SECONDS")
    bocha_top_n: int = Field(default=12, env="BOCHA_TOP_N")

    langfuse_public_key: Optional[str] = Field(default=None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, env="LANGFUSE_SECRET_KEY")
    langfuse_base_url: Optional[str] = Field(default=None, env="LANGFUSE_BASE_URL")
    langfuse_host: Optional[str] = Field(default=None, env="LANGFUSE_HOST")
    langfuse_enabled: bool = Field(default=False, env="LANGFUSE_ENABLED")

    enable_langgraph_planner: bool = Field(default=False, env="ENABLE_LANGGRAPH_PLANNER")

    # SEC data.sec.gov / www.sec.gov HTTP：须为可识别 User-Agent（含联系邮箱），见 https://www.sec.gov/os/webmaster-faq#code-support
    sec_http_user_agent: str = Field(
        default="rag-edgar-sync/1.0 (replace-with-your-email@example.com)",
        env="SEC_HTTP_USER_AGENT",
    )
    # EDGAR 主 HTML：优先 Unstructured 分区并在同目录写 .unstructured.json，再从该 JSON 组装正文入库
    edgar_html_use_unstructured: bool = Field(default=True, env="EDGAR_HTML_USE_UNSTRUCTURED")

    # Finance / SEC companyfacts: rule-first routing; LLM disambiguates when keywords tie.
    finance_sql_routing_enabled: bool = Field(default=True, env="FINANCE_SQL_ROUTING_ENABLED")
    finance_sql_routing_llm_fallback: bool = Field(default=True, env="FINANCE_SQL_ROUTING_LLM_FALLBACK")
    # True: bypass rule router; let LLM directly decide whether SQL is needed.
    finance_llm_route_only: bool = Field(default=False, env="FINANCE_LLM_ROUTE_ONLY")
    # Hybrid finance ask: reorder RAG hits using accn / taxonomy.metric_key from SQL rows.
    finance_sql_narrow_rag_enabled: bool = Field(default=True, env="FINANCE_SQL_NARROW_RAG_ENABLED")
    # True: keep only SQL-matched nodes until top_k; pad with highest-score non-matched if needed.
    finance_sql_narrow_rag_strict: bool = Field(default=True, env="FINANCE_SQL_NARROW_RAG_STRICT")
    # LLM refines sql/rag when rules fire both signals (reduces narrative questions pulling huge facts).
    finance_llm_hybrid_refine: bool = Field(default=True, env="FINANCE_LLM_HYBRID_REFINE")
    # When True, build FinanceQueryPlan via LLM (fallback to heuristics on failure).
    finance_llm_sql_planner_enabled: bool = Field(default=True, env="FINANCE_LLM_SQL_PLANNER_ENABLED")

    ragas_enabled: bool = Field(default=False, env="RAGAS_ENABLED")
    ragas_llm_model: str = Field(default="deepseek/deepseek-chat", env="RAGAS_LLM_MODEL")
    ragas_batch_size: int = Field(default=20, env="RAGAS_BATCH_SIZE")

    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        env="CORS_ALLOWED_ORIGINS",
    )

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def effective_langfuse_base_url(self) -> Optional[str]:
        return self.langfuse_base_url or self.langfuse_host


def validate_api_keys(cfg: Config) -> dict:
    out = {}
    keys = {
        "QWEN_API_KEY": cfg.qwen_api_key,
        "DEEPSEEK_API_KEY": cfg.deepseek_api_key,
        "OPENAI_API_KEY": cfg.openai_api_key,
        "ANTHROPIC_API_KEY": cfg.anthropic_api_key,
        "BOCHA_API_KEY": cfg.bocha_api_key,
        "LANGFUSE_PUBLIC_KEY": cfg.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": cfg.langfuse_secret_key,
    }
    for name, val in keys.items():
        out[name] = "已配置" if val else "未配置"

    dm = cfg.default_model
    if dm.startswith("qwen/") and not cfg.qwen_api_key:
        out["DEFAULT_MODEL_KEY"] = "默认模型为 Qwen，但未配置 QWEN_API_KEY"
    elif dm.startswith("deepseek/") and not cfg.deepseek_api_key:
        out["DEFAULT_MODEL_KEY"] = "默认模型为 DeepSeek，但未配置 DEEPSEEK_API_KEY"
    elif dm.startswith("openai/") and not cfg.openai_api_key:
        out["DEFAULT_MODEL_KEY"] = "默认模型为 OpenAI，但未配置 OPENAI_API_KEY"
    elif dm.startswith("anthropic/") and not cfg.anthropic_api_key:
        out["DEFAULT_MODEL_KEY"] = "默认模型为 Anthropic，但未配置 ANTHROPIC_API_KEY"
    else:
        out["DEFAULT_MODEL_KEY"] = "默认模型与密钥匹配"
    return out


config = Config()

if __name__ == "__main__":
    for k, v in validate_api_keys(config).items():
        print(f"{k}: {v}")
