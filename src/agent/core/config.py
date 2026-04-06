# -*- coding: utf-8 -*-
"""应用配置（Node-centric RAG stack）。"""

import os
from pathlib import Path
from typing import Literal, Optional

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
    # ask_detail_*.json：是否在 rag.evidence 中写入 text_full（从 DB 拉取节点全文）；0 表示不截断
    report_detail_full_evidence: bool = Field(default=False, env="REPORT_DETAIL_FULL_EVIDENCE")
    report_detail_full_evidence_max_chars: int = Field(
        default=0,
        env="REPORT_DETAIL_FULL_EVIDENCE_MAX_CHARS",
    )
    # ask_result_*.json：写入磁盘前压缩 response.pipeline_trace 等（详情 JSON 仍用完整 payload）
    report_ask_slim: bool = Field(default=True, env="REPORT_ASK_SLIM")

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
    # POST /api/ask/generate (and vector search) when request body omits top_k; batch script default if --top-k not passed.
    ask_default_top_k: int = Field(default=3, ge=1, le=50, env="ASK_DEFAULT_TOP_K")
    ask_vector_search_default_top_k: int = Field(default=3, ge=1, le=20, env="ASK_VECTOR_SEARCH_DEFAULT_TOP_K")
    retrieve_top_k: int = Field(default=12, env="RETRIEVE_TOP_K")
    retrieve_candidate_k: int = Field(default=40, env="RETRIEVE_CANDIDATE_K")
    sparse_top_k: int = Field(default=30, env="SPARSE_TOP_K")
    # pipeline_trace 里 retrieval_sparse_hits 每条的正文预览长度上限
    pipeline_trace_sparse_text_chars: int = Field(default=800, env="PIPELINE_TRACE_SPARSE_TEXT_CHARS")
    dense_top_k: int = Field(default=30, env="DENSE_TOP_K")
    context_neighbor_radius: int = Field(default=1, env="CONTEXT_NEIGHBOR_RADIUS")
    # Section tree: search all section levels from 1 up to this depth.
    # Covers typical SEC filing structure: Part (1) → Item (2) → Sub-section (3) → Sub-sub (4).
    section_tree_search_depth: int = Field(default=4, env="SECTION_TREE_SEARCH_DEPTH")
    # Sibling context expansion: only expand the top-N ranked seeds (avoids over-fetching from
    # low-quality candidates).  Set to 0 to disable sibling expansion entirely.
    context_sibling_seed_count: int = Field(default=3, env="CONTEXT_SIBLING_SEED_COUNT")
    # Max siblings fetched per seed node.  Caps context size for large sections (e.g. full MD&A).
    context_sibling_limit: int = Field(default=8, env="CONTEXT_SIBLING_LIMIT")
    # Score decay applied to siblings that were not directly reranked.
    # Inherited score = seed_rerank_score * decay, keeping siblings eligible for top-k context.
    context_sibling_score_decay: float = Field(default=0.8, env="CONTEXT_SIBLING_SCORE_DECAY")
    # Context window budget (characters).  When > 0, replaces fixed top_k truncation:
    # nodes are admitted greedily by rerank_score until the budget is exhausted.
    # Mandatory slots (matching narrative_targets) are guaranteed before score-based fill.
    # Set to 0 to disable budget mode and fall back to fixed top_k.
    context_char_budget: int = Field(default=6000, env="CONTEXT_CHAR_BUDGET")

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
    # EDGAR HTML：使用 BS4 element-aware parser（edgar_htm_parser + enricher），精确保留财报表边界。
    # 优先级高于 edgar_html_use_unstructured；设为 False 回退到旧路径。
    edgar_html_use_bs4_parser: bool = Field(default=True, env="EDGAR_HTML_USE_BS4_PARSER")

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
    # mixed_narrative_first only: evidence pipeline order — sql_before_rag (default) or rag_before_sql (RAG first, then SQL from top RAG filings).
    mixed_narrative_evidence_order: Literal["sql_before_rag", "rag_before_sql"] = Field(
        default="sql_before_rag",
        env="MIXED_NARRATIVE_EVIDENCE_ORDER",
    )
    # mixed_narrative_first + sql_before_rag: after first RAG, re-fetch SQL rows with preferred_accns from RAG top hits (union with prior SQL accns for coverage).
    mixed_narrative_sql_align_to_rag_filings: bool = Field(
        default=True,
        env="MIXED_NARRATIVE_SQL_ALIGN_TO_RAG_FILINGS",
    )
    # Narrative retrieval: bypass section-role hard rejection and treat section policy as soft signal.
    narrative_disable_section_hard_filter: bool = Field(
        default=False, env="NARRATIVE_DISABLE_SECTION_HARD_FILTER"
    )
    # Narrative retrieval: do not filter or score by section_role (field still stored on nodes / in traces).
    narrative_ignore_section_role: bool = Field(
        default=True, env="NARRATIVE_IGNORE_SECTION_ROLE"
    )
    # Narrative retrieval: do not use leaf_role (or section_role where paired) for fallback thresholds,
    # answerability blocks/bonuses, or coverage slot predicates.
    narrative_ignore_leaf_role: bool = Field(
        default=False, env="NARRATIVE_IGNORE_LEAF_ROLE"
    )
    # Narrative retrieval: skip answerability-pass hard gate; keep rerank-driven candidates.
    narrative_disable_answerability_gate: bool = Field(
        default=False, env="NARRATIVE_DISABLE_ANSWERABILITY_GATE"
    )
    # Narrative retrieval: enable post-rerank citationability + near-duplicate selector.
    narrative_post_rerank_selector_enabled: bool = Field(
        default=True, env="NARRATIVE_POST_RERANK_SELECTOR_ENABLED"
    )
    # Narrative retrieval: drop obvious non-citable narrative chunks after rerank.
    narrative_citationability_filter_enabled: bool = Field(
        default=True, env="NARRATIVE_CITATIONABILITY_FILTER_ENABLED"
    )
    # Narrative retrieval: avoid repeated template chunks occupying top slots.
    narrative_dedupe_enabled: bool = Field(
        default=True, env="NARRATIVE_DEDUPE_ENABLED"
    )
    narrative_dedupe_similarity_threshold: float = Field(
        default=0.84, env="NARRATIVE_DEDUPE_SIMILARITY_THRESHOLD"
    )
    # Narrative: run Bocha rerank with multiple facet queries (e.g. RF + distribution + receivables) and merge by max score.
    narrative_multi_rerank_enabled: bool = Field(default=True, env="NARRATIVE_MULTI_RERANK_ENABLED")
    narrative_multi_rerank_max_queries: int = Field(default=3, env="NARRATIVE_MULTI_RERANK_MAX_QUERIES")
    # Narrative leaf dense/sparse: True = do not restrict by finance_accns (all filings in document_ids); False = prefer filing-scoped leaf when finance_accns is set.
    narrative_leaf_all_filings: bool = Field(default=True, env="NARRATIVE_LEAF_ALL_FILINGS")
    # Comma-separated rag_documents.id values removed from each ask's retrieval scope (e.g. 9500 XBRL companyfacts).
    rag_ask_excluded_document_ids: str = Field(default="", env="RAG_ASK_EXCLUDED_DOCUMENT_IDS")

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

    @property
    def rag_ask_excluded_document_id_set(self) -> frozenset[int]:
        raw = (self.rag_ask_excluded_document_ids or "").strip()
        if not raw:
            return frozenset()
        out: set[int] = set()
        for part in raw.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.add(int(p))
            except ValueError:
                continue
        return frozenset(out)


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
