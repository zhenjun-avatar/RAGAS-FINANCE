"""统一构造 LangChain Chat 模型（OpenAI 兼容 / DeepSeek / Qwen）。"""

import os

from langchain_openai import ChatOpenAI

from core.config import config

from .ragas_llm import RagasSanitizingChatOpenAI


def get_llm(
    model_name: str | None = None,
    temperature: float = 0.7,
    *,
    ragas_strip_json_fence: bool = False,
) -> ChatOpenAI:
    name = (model_name or config.default_model or "").strip()
    if not name:
        name = "deepseek/deepseek-chat"

    lower = name.lower()
    max_tokens = 8192
    _cls = RagasSanitizingChatOpenAI if ragas_strip_json_fence else ChatOpenAI

    if "deepseek" in lower:
        return _cls(
            model=name.replace("deepseek/", ""),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if "qwen" in lower:
        return _cls(
            model=name.replace("qwen/", ""),
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if "openai" in lower or "gpt" in lower:
        return _cls(
            model=name.replace("openai/", ""),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _cls(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
    )
