"use client";

import { useMemo } from "react";
import type { DocumentCatalogItem, ResolvedLocale } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

export type DocumentScopeMode = "group" | "documents";

type Labels = {
  title: string;
  byGroup: string;
  byDocument: string;
  selectGroup: string;
  groupIdsHint: string;
  noGroups: string;
  selectedDocuments: string;
  loading: string;
};

function scopeLabels(locale: ResolvedLocale): Labels {
  if (locale === "en") {
    return {
      title: "Scope",
      byGroup: "By group",
      byDocument: "By document",
      selectGroup: "Group",
      groupIdsHint: "Document IDs in this group:",
      noGroups: "No groups file or empty. Add document_groups.json on the API server.",
      selectedDocuments: "Selected IDs",
      loading: "Loading…"
    };
  }
  return {
    title: "文档范围",
    byGroup: "按分组",
    byDocument: "按文档",
    selectGroup: "分组",
    groupIdsHint: "本组文档 ID：",
    noGroups: "未配置分组或文件为空。请在 agent 侧添加 tools/data/document_groups.json。",
    selectedDocuments: "已选文档 ID",
    loading: "加载中…"
  };
}

export type DocumentScopeProps = {
  locale: ResolvedLocale;
  mode: DocumentScopeMode;
  onModeChange: (mode: DocumentScopeMode) => void;
  groups: Record<string, number[]> | null;
  groupsLoading: boolean;
  groupsMissing: boolean;
  selectedGroupKey: string;
  onGroupKeyChange: (key: string) => void;
  docInput: string;
  onDocInputChange: (value: string) => void;
  selectedDocIds: number[];
  documentCatalog: DocumentCatalogItem[];
  onToggleDocument: (docId: number) => void;
};

export function DocumentScope(props: DocumentScopeProps) {
  const t = scopeLabels(props.locale);
  const keys = props.groups ? Object.keys(props.groups).sort() : [];
  const ids =
    props.selectedGroupKey && props.groups?.[props.selectedGroupKey] ? props.groups[props.selectedGroupKey] : [];

  const catalogById = useMemo(() => {
    const m = new Map<number, DocumentCatalogItem>();
    for (const row of props.documentCatalog) {
      m.set(row.document_id, row);
    }
    return m;
  }, [props.documentCatalog]);

  return (
    <aside className="flex h-full min-h-0 w-[260px] shrink-0 flex-col rounded-md border border-zinc-200 bg-white">
      <div className="shrink-0 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-700">{t.title}</h2>
      </div>
      <Separator className="shrink-0" />
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden p-3">
        <div className="flex shrink-0 gap-1 rounded-md border border-zinc-200 p-0.5">
          <Button
            type="button"
            variant={props.mode === "group" ? "default" : "ghost"}
            size="sm"
            className="h-8 flex-1 text-xs"
            onClick={() => props.onModeChange("group")}
          >
            {t.byGroup}
          </Button>
          <Button
            type="button"
            variant={props.mode === "documents" ? "default" : "ghost"}
            size="sm"
            className="h-8 flex-1 text-xs"
            onClick={() => props.onModeChange("documents")}
          >
            {t.byDocument}
          </Button>
        </div>

        {props.mode === "group" ? (
          <div className="flex min-h-0 flex-1 flex-col gap-2">
            {props.groupsLoading ? (
              <p className="text-xs text-zinc-500">{t.loading}</p>
            ) : props.groupsMissing || keys.length === 0 ? (
              <p className="text-xs text-zinc-500">{t.noGroups}</p>
            ) : (
              <>
                <label className="block shrink-0 text-xs text-zinc-500" htmlFor="doc-group-select">
                  {t.selectGroup}
                </label>
                <select
                  id="doc-group-select"
                  value={props.selectedGroupKey}
                  onChange={(e) => props.onGroupKeyChange(e.target.value)}
                  className="h-9 w-full shrink-0 rounded-md border border-zinc-300 bg-white px-2 text-sm outline-none focus:border-zinc-500"
                >
                  {keys.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
                <div className="flex min-h-0 flex-1 flex-col gap-1 text-xs text-zinc-500">
                  <p className="shrink-0">{t.groupIdsHint}</p>
                  {ids.length === 0 ? (
                    <p className="shrink-0 font-mono text-zinc-700">—</p>
                  ) : (
                    <ul className="min-h-0 flex-1 space-y-1 overflow-y-auto rounded border border-zinc-100 bg-zinc-50/80 p-2">
                      {ids.map((id) => {
                        const row = catalogById.get(id);
                        const label = row?.display_name ?? `Document ${id}`;
                        return (
                          <li key={id} className="text-zinc-700">
                            <span className="font-medium">{label}</span>
                            <span className="ml-1 font-mono text-[11px] text-zinc-500">ID {id}</span>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-2">
            <p className="shrink-0 text-xs text-zinc-500">{t.selectedDocuments}</p>
            <input
              value={props.docInput}
              onChange={(e) => props.onDocInputChange(e.target.value)}
              placeholder="9002,9100"
              className="h-9 w-full shrink-0 rounded-md border border-zinc-300 px-2 text-sm outline-none focus:border-zinc-500"
            />
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto">
              {(props.documentCatalog.length > 0
                ? props.documentCatalog
                : props.selectedDocIds.map((id) => ({
                    document_id: id,
                    display_name: `Document ${id}`,
                    subtitle: `ID ${id}`,
                    accn: undefined
                  }))
              ).map((doc) => {
                const docId = doc.document_id;
                const activeDoc = props.selectedDocIds.includes(docId);
                const sub = doc.subtitle ?? `ID ${docId}`;
                return (
                  <Button
                    key={docId}
                    size="sm"
                    variant={activeDoc ? "default" : "outline"}
                    onClick={() => props.onToggleDocument(docId)}
                    className="h-auto w-full justify-start py-2 text-left"
                  >
                    <div>
                      <div className="text-xs font-semibold">{doc.display_name}</div>
                      <div className="text-[11px] opacity-80">{sub}</div>
                    </div>
                  </Button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
