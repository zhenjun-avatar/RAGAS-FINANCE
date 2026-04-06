from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .part_time_graduate_leads import extract_part_time_graduate_leads

router = APIRouter(prefix="/api/leads", tags=["Leads Extraction"])


class ExtractPartTimeGraduateLeadsRequest(BaseModel):
    # 基础检索 query（覆盖“在职研究生/在职考研/非全日制”等）
    query: str = Field(
        default="在职研究生 报考 咨询 经验贴 在职考研 非全日制 备考 评论 非机构 非广告",
        min_length=1,
        max_length=400,
    )
    # web / ai / both
    search_mode: str = Field(default="both", pattern="^(web|ai|both)$")
    freshness: str = Field(
        default="oneYear",
        description="bocha 搜索 freshness：oneDay/oneWeek/oneMonth/oneYear/noLimit",
    )
    # 每种模式的数量
    count_per_mode: int = Field(default=10, ge=1, le=50)
    # 可选：站点范围（provider-specific schema，示例：\"zhihu.com|xiaohongshu.com\"）
    include_domains: Optional[str] = Field(default=None)
    # 给 LLM 用的最多来源条数
    max_sources_for_llm: int = Field(default=12, ge=1, le=50)
    # 过滤疑似机构广告/营销内容（包含联系方式/包过等强营销信号）
    exclude_ads: bool = Field(default=True)
    # 可选：覆盖 LLM 模型名
    llm_model_name: Optional[str] = Field(default=None)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.5)


class LeadCandidate(BaseModel):
    intent_score: float = Field(default=0.0, ge=0.0, le=1.0)
    intent_label: str = ""
    lead_type: str = ""
    target_keywords: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    evidence_urls: List[str] = Field(default_factory=list)
    likely_target_majors_or_programs: List[str] = Field(default_factory=list)
    timeframe_hint: str = ""
    notes: str = ""


class ExtractPartTimeGraduateLeadsResponse(BaseModel):
    ok: bool
    query: str
    search_mode: str
    sources_found: int = 0
    sources_found_pre_ad_filter: int = 0
    ads_filtered_out: int = 0
    web_sources: int = 0
    ai_sources: int = 0
    leads: List[LeadCandidate] = Field(default_factory=list)
    error: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


@router.post("/extract-part-time-graduate-leads", response_model=ExtractPartTimeGraduateLeadsResponse)
async def extract_part_time_graduate_leads_api(
    request: ExtractPartTimeGraduateLeadsRequest,
):
    try:
        out = await extract_part_time_graduate_leads(
            query=request.query,
            search_mode=request.search_mode,
            freshness=request.freshness,
            count_per_mode=request.count_per_mode,
            include_domains=request.include_domains,
            max_sources_for_llm=request.max_sources_for_llm,
            exclude_ads=request.exclude_ads,
            llm_model_name=request.llm_model_name,
            llm_temperature=request.llm_temperature,
        )
        # ensure response shape
        return ExtractPartTimeGraduateLeadsResponse(
            ok=bool(out.get("ok")),
            query=str(out.get("query") or request.query),
            search_mode=str(out.get("search_mode") or request.search_mode),
            sources_found=int(out.get("sources_found") or 0),
            sources_found_pre_ad_filter=int(out.get("sources_found_pre_ad_filter") or 0),
            ads_filtered_out=int(out.get("ads_filtered_out") or 0),
            web_sources=int(out.get("web_sources") or 0),
            ai_sources=int(out.get("ai_sources") or 0),
            leads=out.get("leads") or [],
            error=out.get("error"),
            debug=out.get("debug"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

