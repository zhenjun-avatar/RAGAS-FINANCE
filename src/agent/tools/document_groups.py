"""Load ``document_groups.json`` (same format as CLI ``--group-file``)."""

from __future__ import annotations

import json
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[1]


def default_document_groups_path() -> Path:
    return _AGENT_ROOT / "tools" / "data" / "document_groups.json"


def load_document_groups(path: Path) -> dict[str, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("分组文件顶层必须是 JSON 对象")
    out: dict[str, list[int]] = {}
    for k, v in raw.items():
        name = str(k).strip()
        if not name:
            continue
        if isinstance(v, list):
            ids = [int(x) for x in v]
        elif isinstance(v, str):
            ids = [int(x.strip()) for x in v.split(",") if x.strip()]
        else:
            raise ValueError(f"分组 {name!r} 的值必须是整数数组或逗号分隔字符串")
        if not ids:
            raise ValueError(f"分组 {name!r} 不能为空")
        out[name] = ids
    return out
