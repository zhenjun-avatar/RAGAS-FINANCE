"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  askGenerate,
  deleteReports,
  fetchDocumentCatalog,
  fetchDocumentGroups,
  fetchReportDetail,
  fetchReportList,
  saveAskReport
} from "@/lib/api";
import { DocumentScope, type DocumentScopeMode } from "@/components/DocumentScope";
import type {
  AskResponse,
  HistoryItem,
  NarrativeCard,
  PersistedReportSummary,
  DocumentCatalogItem,
  ReportLocale,
  ResolvedLocale
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

/** Build-time env; align with agent ASK_DEFAULT_TOP_K when omitted in API body. */
const DEFAULT_ASK_TOP_K = (() => {
  const raw = process.env.NEXT_PUBLIC_ASK_DEFAULT_TOP_K;
  if (raw === undefined || raw === "") return 3;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 1) return 3;
  return Math.min(50, n);
})();

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
      filings: "Filings referenced",
      empty: "None",
      selectAll: "Select all",
      deselectAll: "Deselect all",
      deleteSelected: "Delete selected",
      deleteHistoryConfirm: (n: number) => `Delete ${n} record(s) from history and disk?`
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
    filings: "涉及披露",
    empty: "暂无",
    selectAll: "全选",
    deselectAll: "取消全选",
    deleteSelected: "删除所选",
    deleteHistoryConfirm: (n: number) => `确定删除所选的 ${n} 条记录（含服务端文件）？`
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
  const [historySelection, setHistorySelection] = useState<Set<string>>(() => new Set());
  const [historyDeleting, setHistoryDeleting] = useState(false);

  const active = useMemo(() => history.find((h) => h.id === activeId) ?? history[0], [activeId, history]);
  const activeLocale = localeFromResult(active?.result);
  /** UI chrome follows EN/中文/AUTO; AUTO matches the active answer's report_locale. */
  const uiLocale: ResolvedLocale = useMemo(() => {
    if (language === "zh") return "zh";
    if (language === "en") return "en";
    return activeLocale;
  }, [language, activeLocale]);
  const t = i18n(uiLocale);

  const allHistorySelected = useMemo(
    () => history.length > 0 && history.every((h) => historySelection.has(h.id)),
    [history, historySelection]
  );

  const toggleHistorySelect = useCallback((id: string) => {
    setHistorySelection((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAllHistory = useCallback(() => {
    setHistorySelection((prev) => {
      if (history.length === 0) return new Set();
      const allOn = history.every((h) => prev.has(h.id));
      if (allOn) return new Set();
      return new Set(history.map((h) => h.id));
    });
  }, [history]);

  const deleteSelectedHistory = useCallback(async () => {
    if (historySelection.size === 0 || historyDeleting) return;
    const selectedIds = new Set(historySelection);
    const n = selectedIds.size;
    const confirmMsg = i18n(uiLocale).deleteHistoryConfirm(n);
    if (typeof window !== "undefined" && !window.confirm(confirmMsg)) return;
    const traceIds = Array.from(
      new Set(
        history
          .filter((h) => selectedIds.has(h.id))
          .map((h) => h.traceId || h.id)
          .filter((x): x is string => Boolean(x))
      )
    );
    setHistoryDeleting(true);
    setError(null);
    try {
      if (traceIds.length > 0) await deleteReports(traceIds);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
      setHistoryDeleting(false);
      return;
    }
    const remaining = history.filter((h) => !selectedIds.has(h.id));
    setHistory(remaining);
    setHistorySelection(new Set());
    setActiveId((prev) => {
      if (prev && selectedIds.has(prev)) return remaining[0]?.id ?? null;
      if (prev && remaining.some((h) => h.id === prev)) return prev;
      return remaining[0]?.id ?? null;
    });
    setHistoryDeleting(false);
  }, [history, historySelection, historyDeleting, uiLocale]);

  useEffect(() => {
    const valid = new Set(history.map((h) => h.id));
    setHistorySelection((prev) => {
      const next = new Set<string>();
      let dirty = false;
      for (const id of prev) {
        if (valid.has(id)) next.add(id);
        else dirty = true;
      }
      return dirty ? next : prev;
    });
  }, [history]);

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
        uiLocale === "en"
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
        top_k: DEFAULT_ASK_TOP_K,
        include_pipeline_trace: true,
        document_ids: effectiveDocumentIds,
        report_locale: language
      } as const;
      const result = await askGenerate(requestPayload);
      void saveAskReport({
        request: {
          question: requestPayload.question,
          detail_level: requestPayload.detail_level,
          top_k: requestPayload.top_k,
          include_pipeline_trace: requestPayload.include_pipeline_trace,
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
  const filings: string[] = active?.result.external_evaluation?.filings_observed ?? [];

  return (
    <main className="h-screen bg-zinc-100">
      <div className="mx-auto flex h-full max-w-[1400px] gap-4 p-4">
        <aside className="flex h-full min-h-0 w-[280px] shrink-0 flex-col rounded-md border border-zinc-200 bg-white">
          <div className="shrink-0 px-4 py-3">
            <h2 className="text-sm font-semibold text-zinc-700">{t.history}</h2>
            {history.length > 0 ? (
              <div className="mt-2 flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs"
                  disabled={historyDeleting}
                  onClick={toggleSelectAllHistory}
                >
                  {allHistorySelected ? t.deselectAll : t.selectAll}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 border-red-200 text-xs text-red-700 hover:bg-red-50"
                  disabled={historySelection.size === 0 || historyDeleting}
                  onClick={() => void deleteSelectedHistory()}
                >
                  {historyDeleting ? "…" : t.deleteSelected}
                </Button>
              </div>
            ) : null}
          </div>
          <Separator className="shrink-0" />
          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            {history.length === 0 ? (
              <p className="px-2 py-3 text-sm text-zinc-500">{t.empty}</p>
            ) : (
              history.map((item) => (
                <div
                  key={item.id}
                  className={`mb-2 flex overflow-hidden rounded-md border ${
                    active?.id === item.id ? "border-zinc-400 bg-zinc-50" : "border-zinc-200 bg-white"
                  }`}
                >
                  <label
                    className="flex shrink-0 cursor-pointer items-start border-r border-zinc-200/80 py-2 pl-2 pr-1"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      type="checkbox"
                      className="mt-1 h-3.5 w-3.5 rounded border-zinc-300 text-zinc-900"
                      checked={historySelection.has(item.id)}
                      onChange={() => toggleHistorySelect(item.id)}
                      aria-label={uiLocale === "en" ? "Select record" : "选择记录"}
                      disabled={historyDeleting}
                    />
                  </label>
                  <button
                    type="button"
                    className="min-w-0 flex-1 px-3 py-2 text-left hover:bg-zinc-50/90"
                    onClick={() => {
                      setActiveId(item.id);
                      if (item.traceId) setActiveTraceId(item.traceId);
                    }}
                  >
                    <div className="mb-1 flex items-center justify-between gap-1">
                      <Badge>{item.locale.toUpperCase()}</Badge>
                      <span className="shrink-0 text-xs text-zinc-500">{new Date(item.createdAt).toLocaleTimeString()}</span>
                    </div>
                    <p className="max-h-10 overflow-hidden text-sm text-zinc-700">{item.question}</p>
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
          <Card className="shrink-0">
            <CardContent className="py-4">
              <div className="flex items-center justify-between">
                <h1 className="text-base font-semibold text-zinc-900">{t.title}</h1>
                <div className="flex items-center gap-2">
                  <Button variant={language === "en" ? "default" : "outline"} size="sm" onClick={() => setLanguage("en")}>
                    英文
                  </Button>
                  <Button variant={language === "zh" ? "default" : "outline"} size="sm" onClick={() => setLanguage("zh")}>
                    中文
                  </Button>
                  <Button variant={language === "auto" ? "default" : "outline"} size="sm" onClick={() => setLanguage("auto")}>
                    自动
                  </Button>
                </div>
              </div>
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

          <Card className="shrink-0">
            <CardContent className="py-4">
              <div className="flex gap-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void submit();
                    }
                  }}
                  placeholder={t.placeholder}
                  className="h-10 flex-1 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-500"
                />
                <Button onClick={() => void submit()} disabled={loading}>
                  {loading ? "..." : t.send}
                </Button>
              </div>
              {error ? <p className="mt-2 text-sm text-red-600">{error}</p> : null}
            </CardContent>
          </Card>
        </section>

        <DocumentScope
          locale={uiLocale}
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
