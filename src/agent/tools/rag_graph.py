"""Optional LangGraph planner for multi-step retrieval orchestration."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llamaindex_retrieval import retrieval_service
from .llm import get_llm


class GraphState(TypedDict, total=False):
    question: str
    retrieval_query: str
    retrieval_metadata_filters: dict[str, list[str]]
    retrieval_soft_hints: dict[str, list[str]]
    document_ids: list[int]
    detail_level: str
    top_k: int
    planned_queries: list[str]
    retrieved_nodes: list[dict[str, Any]]


async def _plan_queries(state: GraphState) -> GraphState:
    question = state["question"]
    llm = get_llm(temperature=0.0)
    prompt = (
        "你是检索规划器。把用户问题拆成 1 到 3 个可检索子查询。"
        "若问题本身已足够明确，则只返回原问题。"
        '输出 JSON: {"queries": ["..."]}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)]
        )
        text = response.content if hasattr(response, "content") else str(response)
        import json
        import re

        try:
            parsed = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {"queries": [question]}
        queries = [item.strip() for item in parsed.get("queries", []) if isinstance(item, str) and item.strip()]
    except Exception:
        queries = [question]
    return {"planned_queries": queries[:3] or [question]}


async def _retrieve(state: GraphState) -> GraphState:
    all_nodes: list[dict[str, Any]] = []
    seen = set()
    planned = state.get("planned_queries") or [state["question"]]
    rq = (state.get("retrieval_query") or "").strip()
    q0 = (state["question"] or "").strip()
    if rq and rq != q0 and rq not in planned:
        planned = [rq, *planned]
    for query in planned:
        result = await retrieval_service.retrieve(
            query=query,
            document_ids=state["document_ids"],
            metadata_filters=state.get("retrieval_metadata_filters"),
            retrieval_soft_hints=state.get("retrieval_soft_hints"),
        )
        for node in result["nodes"]:
            node_id = node["node_id"]
            if node_id in seen:
                continue
            seen.add(node_id)
            all_nodes.append(node)
    ranked = sorted(all_nodes, key=lambda item: item.get("score", 0.0), reverse=True)
    return {"retrieved_nodes": ranked[: max(state["top_k"] * 2, 8)]}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("plan_queries", _plan_queries)
    graph.add_node("retrieve", _retrieve)
    graph.set_entry_point("plan_queries")
    graph.add_edge("plan_queries", "retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()


graph_app = build_graph()
