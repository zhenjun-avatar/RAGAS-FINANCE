"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { askGenerate, fetchDocumentCatalog, fetchDocumentGroups, fetchReportDetail, fetchReportList, saveAskReport } from "@/lib/api";
import { DocumentScope, type DocumentScopeMode } from "@/components/DocumentScope";
import type {
  AskResponse,
  HistoryItem,
  NarrativeCard,
  PersistedReportSummary,
  DocumentCatalogItem,
  ReportLocale,
  ResolvedLocale,
  StructuredFactCard
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

function i18n(locale: ResolvedLocale) {
  if (locale === "en") {
    return {
      title: "Finance Q&A",
      placeholder: "Ask a question about the selected filings...",
      send: "Ask",
      history: "History",
      question: "Question",
      conclusion: "Conclusion",
      narrative: "Key textual evidence",
      facts: "Key financial facts",
      filings: "Filings referenced",
      empty: "None"
    };
  }
  return {
    title: "财务问答",
    placeholder: "请输入财务问题...",
    send: "提问",
    history: "历史记录",
    question: "问题",
    conclusion: "结论",
    narrative: "关键文字证据",
    facts: "关键财务事实",
      filings: "涉及披露",
      empty: "暂无"
  };
}

function localeFromResult(item?: AskResponse): ResolvedLocale {
  return item?.report_locale === "en" ? "en" : "zh";
}

export default function HomePage() {
  const [documentCatalog, setDocumentCatalog] = useState<DocumentCatalogItem[]>([]);
  const [question, setQuestion] = useState("");
  const [language, setLanguage] = useState<ReportLocale>("auto");
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([9002]);
  const [docInput, setDocInput] = useState("9002");
  const [scopeMode, setScopeMode] = useState<DocumentScopeMode>("documents");
  const [groupsMap, setGroupsMap] = useState<Record<string, number[]> | null>(null);
  const [groupsLoading, setGroupsLoading] = useState(true);
  const [groupsMissing, setGroupsMissing] = useState(false);
  const [selectedGroupKey, setSelectedGroupKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const active = useMemo(() => history.find((h) => h.id === activeId) ?? history[0], [activeId, history]);
  const activeLocale = localeFromResult(active?.result);
  const t = i18n(activeLocale);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const catalog = await fetchDocumentCatalog(1000);
        if (cancelled) return;
        setDocumentCatalog(catalog);
        const ids = catalog.map((it) => it.document_id).filter((v) => Number.isInteger(v));
        if (ids.length > 0) {
          const next = [ids[0]];
          setSelectedDocIds(next);
          setDocInput(next.join(","));
        }
      } catch {
        if (cancelled) return;
        setDocumentCatalog([]);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        setGroupsLoading(true);
        const payload = await fetchDocumentGroups();
        if (cancelled) return;
        setGroupsMap(payload.groups || {});
        setGroupsMissing(Boolean(payload.missing));
        const keys = Object.keys(payload.groups || {}).sort();
        if (keys.length > 0) {
          setSelectedGroupKey((prev) => (prev && keys.includes(prev) ? prev : keys[0]!));
        }
      } catch {
        if (cancelled) return;
        setGroupsMap({});
        setGroupsMissing(true);
      } finally {
        if (!cancelled) setGroupsLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadHistory = async () => {
      try {
        const items = await fetchReportList(80);
        if (cancelled || items.length === 0) return;
        const mapped: HistoryItem[] = items.map((it: PersistedReportSummary) => ({
          id: it.trace_id || crypto.randomUUID(),
          traceId: it.trace_id,
          createdAt: Date.parse(it.created_at || "") || Date.now(),
          locale: it.report_locale === "en" ? "en" : "zh",
          question: String(it.question || ""),
          result: {
            question: String(it.question || ""),
            answer: String(it.answer_preview || ""),
            report_locale: it.report_locale === "en" ? "en" : "zh",
            trace_id: it.trace_id
          }
        }));
        setHistory(mapped);
        setActiveId(mapped[0].id);
      } catch {
        // best-effort restore only
      }
    };
    void loadHistory();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!activeTraceId) return;
    let cancelled = false;
    const loadDetail = async () => {
      try {
        const detail = await fetchReportDetail(activeTraceId);
        if (cancelled) return;
        const full = detail.response;
        if (!full) return;
        setHistory((prev) =>
          prev.map((it) =>
            it.traceId === activeTraceId
              ? {
                  ...it,
                  question: String(full.question || it.question),
                  locale: full.report_locale === "en" ? "en" : "zh",
                  result: full
                }
              : it
          )
        );
      } catch {
        // keep preview-only item
      } finally {
        if (!cancelled) setActiveTraceId(null);
      }
    };
    void loadDetail();
    return () => {
      cancelled = true;
    };
  }, [activeTraceId]);

  const toggleDocument = (docId: number) => {
    setSelectedDocIds((prev) => {
      const next = prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId].sort((a, b) => a - b);
      setDocInput(next.join(","));
      return next;
    });
  };

  const applyDocInput = (value: string) => {
    setDocInput(value);
    const parsed = Array.from(
      new Set(
        value
          .split(",")
          .map((part) => Number(part.trim()))
          .filter((n) => Number.isInteger(n) && n > 0)
      )
    );
    setSelectedDocIds(parsed);
  };

  const handleScopeModeChange = useCallback(
    (next: DocumentScopeMode) => {
      if (next === "documents" && scopeMode === "group" && groupsMap && selectedGroupKey) {
        const ids = groupsMap[selectedGroupKey];
        if (ids?.length) {
          setSelectedDocIds(ids);
          setDocInput(ids.join(","));
        }
      }
      setScopeMode(next);
    },
    [scopeMode, groupsMap, selectedGroupKey]
  );

  const effectiveDocumentIds = useMemo(() => {
    if (scopeMode === "group" && groupsMap && selectedGroupKey) {
      return groupsMap[selectedGroupKey] ?? [];
    }
    return selectedDocIds;
  }, [scopeMode, groupsMap, selectedGroupKey, selectedDocIds]);

  const submit = async () => {
    if (!question.trim() || loading) return;
    if (effectiveDocumentIds.length === 0) {
      setError(
        activeLocale === "en"
          ? "Please select a non-empty document set (group or documents)."
          : "请选择有效的文档范围（分组或文档列表不能为空）。"
      );
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const requestPayload = {
        question: question.trim(),
        detail_level: "detailed",
        top_k: 8,
        document_ids: effectiveDocumentIds,
        report_locale: language
      } as const;
      const result = await askGenerate(requestPayload);
      void saveAskReport({
        request: {
          question: requestPayload.question,
          detail_level: requestPayload.detail_level,
          top_k: requestPayload.top_k,
          document_ids: requestPayload.document_ids,
          report_locale: requestPayload.report_locale
        },
        response: result,
        source: "frontend"
      });
      const item: HistoryItem = {
        id: result.trace_id || crypto.randomUUID(),
        traceId: result.trace_id,
        createdAt: Date.now(),
        locale: localeFromResult(result),
        question: result.question,
        result
      };
      setHistory((prev) => [item, ...prev]);
      setActiveId(item.id);
      setQuestion("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const narrativeCards: NarrativeCard[] = active?.result.evidence_ui?.evidence?.narrative_cards ?? [];
  const factCards: StructuredFactCard[] = active?.result.evidence_ui?.evidence?.structured_fact_cards ?? [];
  const filings: string[] = active?.result.external_evaluation?.filings_observed ?? [];

  return (
    <main className="h-screen bg-zinc-100">
      <div className="mx-auto flex h-full max-w-[1400px] gap-4 p-4">
        <aside className="flex w-[280px] flex-col rounded-md border border-zinc-200 bg-white">
          <div className="px-4 py-3">
            <h2 className="text-sm font-semibold text-zinc-700">{t.history}</h2>
          </div>
          <Separator />
          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            {history.length === 0 ? (
              <p className="px-2 py-3 text-sm text-zinc-500">{t.empty}</p>
            ) : (
              history.map((item) => (
                <button
                  key={item.id}
                  className={`mb-2 w-full rounded-md border px-3 py-2 text-left ${
                    active?.id === item.id ? "border-zinc-400 bg-zinc-50" : "border-zinc-200 bg-white hover:bg-zinc-50"
                  }`}
                  onClick={() => {
                    setActiveId(item.id);
                    if (item.traceId) setActiveTraceId(item.traceId);
                  }}
                >
                  <div className="mb-1 flex items-center justify-between">
                    <Badge>{item.locale.toUpperCase()}</Badge>
                    <span className="text-xs text-zinc-500">{new Date(item.createdAt).toLocaleTimeString()}</span>
                  </div>
                  <p className="max-h-10 overflow-hidden text-sm text-zinc-700">{item.question}</p>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col gap-4">
          <Card>
            <CardContent className="pt-5">
              <div className="mb-3 flex items-center justify-between">
                <h1 className="text-base font-semibold text-zinc-900">{t.title}</h1>
                <div className="flex items-center gap-2">
                  <Button variant={language === "en" ? "default" : "outline"} size="sm" onClick={() => setLanguage("en")}>
                    EN
                  </Button>
                  <Button variant={language === "zh" ? "default" : "outline"} size="sm" onClick={() => setLanguage("zh")}>
                    中文
                  </Button>
                  <Button variant={language === "auto" ? "default" : "outline"} size="sm" onClick={() => setLanguage("auto")}>
                    AUTO
                  </Button>
                </div>
              </div>
              <div className="flex gap-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={t.placeholder}
                  className="h-10 flex-1 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-500"
                />
                <Button onClick={submit} disabled={loading}>
                  {loading ? "..." : t.send}
                </Button>
              </div>
              {error ? <p className="mt-2 text-sm text-red-600">{error}</p> : null}
            </CardContent>
          </Card>

          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
            <Card>
              <CardHeader>
                <CardTitle>{t.question}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="whitespace-pre-wrap text-sm">{active?.result.question || t.empty}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t.conclusion}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="whitespace-pre-wrap text-sm">{active?.result.answer || t.empty}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t.narrative}</CardTitle>
              </CardHeader>
              <CardContent>
                {narrativeCards.length === 0 ? (
                  <p className="text-sm text-zinc-500">{t.empty}</p>
                ) : (
                  <div className="space-y-3">
                    {narrativeCards.map((card, i) => (
                      <div key={card.card_id || i} className="border-t border-zinc-200 pt-3 first:border-t-0 first:pt-0">
                        <p className="mb-1 text-sm font-medium">{card.title || `Evidence ${i + 1}`}</p>
                        <p className="whitespace-pre-wrap text-sm text-zinc-700">{card.body || t.empty}</p>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t.facts}</CardTitle>
              </CardHeader>
              <CardContent>
                {factCards.length === 0 ? (
                  <p className="text-sm text-zinc-500">{t.empty}</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-sm">
                      <thead>
                        <tr className="bg-zinc-50">
                          <th className="border border-zinc-200 px-3 py-2 text-left font-medium">Metric</th>
                          <th className="border border-zinc-200 px-3 py-2 text-left font-medium">Value</th>
                          <th className="border border-zinc-200 px-3 py-2 text-left font-medium">Filing / Context</th>
                        </tr>
                      </thead>
                      <tbody>
                        {factCards.map((row, i) => (
                          <tr key={row.card_id || i}>
                            <td className="border border-zinc-200 px-3 py-2">{row.title || "-"}</td>
                            <td className="border border-zinc-200 px-3 py-2">{row.body || "-"}</td>
                            <td className="border border-zinc-200 px-3 py-2">{row.subtitle || "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t.filings}</CardTitle>
              </CardHeader>
              <CardContent>
                {filings.length === 0 ? (
                  <p className="text-sm text-zinc-500">{t.empty}</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-sm">
                      <thead>
                        <tr className="bg-zinc-50">
                          <th className="border border-zinc-200 px-3 py-2 text-left font-medium">#</th>
                          <th className="border border-zinc-200 px-3 py-2 text-left font-medium">Filing</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filings.map((f, i) => (
                          <tr key={`${f}-${i}`}>
                            <td className="border border-zinc-200 px-3 py-2">{i + 1}</td>
                            <td className="border border-zinc-200 px-3 py-2">{f}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </section>

        <DocumentScope
          locale={activeLocale}
          mode={scopeMode}
          onModeChange={handleScopeModeChange}
          groups={groupsMap}
          groupsLoading={groupsLoading}
          groupsMissing={groupsMissing}
          selectedGroupKey={selectedGroupKey}
          onGroupKeyChange={setSelectedGroupKey}
          docInput={docInput}
          onDocInputChange={applyDocInput}
          selectedDocIds={selectedDocIds}
          documentCatalog={documentCatalog}
          onToggleDocument={toggleDocument}
        />
      </div>
    </main>
  );
}
