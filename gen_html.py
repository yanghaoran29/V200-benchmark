#!/usr/bin/env python3
"""Parse V200-benchmark/README.md and emit a self-contained interactive index.html."""

from __future__ import annotations

import html
import json
import re
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
OUT = ROOT / "index.html"
ASSETS = ROOT / "assets"

_TOOLS = ROOT.parent / "artifacts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from swimlane_timing_table import (  # noqa: E402
    BUCKETS_MIX_FINE_TAIL,
    _merge_kernel_name,
    analyze_leaf,
    collect_records,
    hist_counts,
    load_task_kinds,
)

RECOMMENDED = {
    "lt_batch16",
    "ht_batch80",
    "lt_batch12_mtp7",
    "ht_batch60_mtp3",
}
BUCKETS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50+"]
BUCKETS_FINE = [b[0] for b in BUCKETS_MIX_FINE_TAIL]

# Fixed topology: rows of parallel nodes (label, kind, kernel name fallbacks).
# Counts = sum(block_num) over matching deps tasks for the first kernel that hits.
FLASH_DEP_ROWS: list[list[tuple[str, str, list[str]]]] = [
    [("hc_pre", "aic", ["hc_pre_linear"])],
    [("rms_norm", "aiv", ["rms_norm"])],
    [
        ("qr_proj", "aic", ["qr_proj_matmul"]),
        ("qproj", "aic", ["qproj_matmul"]),
        ("kv_proj", "aic", ["kv_proj_matmul"]),
    ],
    [("idx_qr_proj", "aic", ["idx_qr_proj_matmul"])],
    [("qr_hadamard", "aic", ["qr_hadamard_matmul"])],
    # score → topk → qk_plan → qk_pv are sequential in the full graph; never co-row.
    [("score", "mix", ["score"])],
    [("qk_pv", "mix", ["qk_pv"])],
    [("merge_norm", "aiv", ["merge_norm"])],
    [("proj_a_mm", "aic", ["proj_a_mm"])],
    [("proj_b_mm", "aic", ["proj_b_mm"])],
    [("hc_post", "aiv", ["hc_post"])],
]

QWEN_DEP_ROWS: list[list[tuple[str, str, list[str]]]] = [
    [("rms", "aiv", ["residual_rms_cast", "x_gamma0"])],
    [
        ("q_proj", "aic", ["q_proj"]),
        ("k_proj", "aic", ["k_proj"]),
        ("v_proj", "aic", ["v_proj"]),
    ],
    [("attn_swpipe", "mix", ["attn_swpipe"])],
    [("out_proj", "aic", ["out_proj"])],
    [("post_rms", "aiv", ["post_rms_reduce"])],
    [("gate_proj", "aic", ["gate_proj"]), ("up_proj", "aic", ["up_proj"])],
    [("silu", "aiv", ["silu"])],
    [("down_proj", "aic", ["down_proj"])],
]


def strip_md(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def parse_tables(md: str) -> list[dict]:
    lines = md.splitlines()
    tables: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) < len(header):
                    cells += [""] * (len(header) - len(cells))
                rows.append(cells[: len(header)])
                i += 1
            tables.append({"headers": header, "rows": rows})
            continue
        i += 1
    return tables


def link_display(cell: str) -> str:
    m = re.search(r"\[([^\]]+)\]\(([^)]+)\)", cell)
    if m:
        return strip_md(m.group(1))
    return strip_md(cell)


def parse_float(cell: str) -> float | None:
    s = strip_md(cell).replace(",", "").strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def is_qwen_path(path: str) -> bool:
    return "qwen3_decode_layer" in path


def is_flash_path(path: str) -> bool:
    return "deepseek_v4_flash_csa" in path


def leaf_name(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def _leaf_swim_dir(leaf_dir: Path) -> Path:
    if (leaf_dir / "chip_swimlane_records.json").exists():
        return leaf_dir
    dfx = leaf_dir / "dfx_outputs"
    if (dfx / "chip_swimlane_records.json").exists():
        return dfx
    return leaf_dir


def _tid_merged_names(swim_dir: Path) -> dict[int, str]:
    tid_name: dict[int, str] = {}
    try:
        for r in collect_records(swim_dir):
            tid = int(r.get("task_id", -1))
            if tid < 0:
                continue
            tid_name.setdefault(tid, _merge_kernel_name(str(r.get("kernel", ""))))
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        pass
    return tid_name


def load_kernel_block_nums(leaf_dir: Path) -> dict[str, int]:
    """Sum deps.json block_num per merged kernel name for a leaf."""
    deps_path = leaf_dir / "deps.json"
    if not deps_path.exists():
        return {}
    try:
        payload = json.loads(deps_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    tid_name = _tid_merged_names(_leaf_swim_dir(leaf_dir))
    out: dict[str, int] = {}
    for t in payload.get("tasks") or []:
        try:
            tid = int(t["task_id"])
            bn = int(t.get("block_num") or 0)
        except (TypeError, ValueError, KeyError):
            continue
        name = tid_name.get(tid)
        if not name:
            continue
        out[name] = out.get(name, 0) + bn
    return out


# Display-only: force sibling kernels onto the same full-graph row.
DEP_SAME_ROW_GROUPS: list[tuple[str, ...]] = [
    ("q_proj", "k_proj", "v_proj"),
    ("qr_proj_matmul", "qproj_matmul", "kv_proj_matmul"),
    ("gate_proj", "up_proj"),
]


def load_dep_full_layers(leaf_dir: Path) -> list[list[tuple[str, str, int]]]:
    """Topo-layered full dep graph: each row is [(name, kind, sum_block_num), ...]."""
    deps_path = leaf_dir / "deps.json"
    if not deps_path.exists():
        return []
    try:
        payload = json.loads(deps_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    tid_name = _tid_merged_names(_leaf_swim_dir(leaf_dir))
    try:
        tid_kind = load_task_kinds(deps_path)
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        tid_kind = {}

    counts: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for t in payload.get("tasks") or []:
        try:
            tid = int(t["task_id"])
            bn = int(t.get("block_num") or 0)
        except (TypeError, ValueError, KeyError):
            continue
        name = tid_name.get(tid)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + bn
        k = tid_kind.get(tid, "aic")
        prev = kinds.get(name)
        if prev is None:
            kinds[name] = k
        elif prev != k:
            kinds[name] = "mix"

    if not counts:
        return []

    preds: dict[str, set[str]] = {n: set() for n in counts}
    succs: dict[str, set[str]] = {n: set() for n in counts}
    for e in payload.get("edges") or []:
        try:
            a = tid_name.get(int(e["pred"]))
            b = tid_name.get(int(e["succ"]))
        except (TypeError, ValueError, KeyError):
            continue
        if not a or not b or a == b or a not in counts or b not in counts:
            continue
        succs[a].add(b)
        preds[b].add(a)

    # Topo order, then longest-path level.
    indeg = {n: len(preds[n]) for n in counts}
    q = deque(sorted(n for n, d in indeg.items() if d == 0))
    order: list[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for n in counts:
        if n not in order:
            order.append(n)

    level: dict[str, int] = {}
    for n in order:
        if preds[n]:
            level[n] = 1 + max(level.get(p, 0) for p in preds[n])
        else:
            level[n] = 0

    # Pull declared siblings onto one row (max level among members present).
    for group in DEP_SAME_ROW_GROUPS:
        members = [n for n in group if n in level]
        if len(members) < 2:
            continue
        # Only coalesce when all members share the same AIC/AIV/MIX kind.
        member_kinds = {kinds.get(n, "aic") for n in members}
        if len(member_kinds) != 1:
            continue
        tgt = max(level[n] for n in members)
        for n in members:
            level[n] = tgt

    # Preferred left-to-right order within a forced group.
    group_rank: dict[str, int] = {}
    for gi, group in enumerate(DEP_SAME_ROW_GROUPS):
        for ri, name in enumerate(group):
            group_rank[name] = gi * 100 + ri

    kind_rank = {"aic": 0, "aiv": 1, "mix": 2}
    # One display row = one topo level × one kind (never mix AIC/AIV/MIX).
    buckets: dict[tuple[int, str], list[tuple[str, str, int]]] = {}
    for n in sorted(
        counts,
        key=lambda x: (
            level[x],
            kind_rank.get(kinds.get(x, "aic"), 9),
            group_rank.get(x, 10_000),
            -counts[x],
            x,
        ),
    ):
        k = kinds.get(n, "aic")
        key = (level[n], k)
        buckets.setdefault(key, []).append((n, k, counts[n]))

    rows: list[list[tuple[str, str, int]]] = []
    max_l = max(level.values()) if level else 0
    for lv in range(max_l + 1):
        for k in ("aic", "aiv", "mix"):
            chunk = buckets.get((lv, k))
            if chunk:
                rows.append(chunk)
    return rows


def resolve_node_count(block_nums: dict[str, int], kernels: list[str]) -> int | None:
    for k in kernels:
        if k in block_nums:
            return block_nums[k]
    return None


def resolve_node_mean(means: dict[str, float], kernels: list[str]) -> int | None:
    for k in kernels:
        if k in means:
            return int(round(means[k]))
    return None


def kernel_means_from_table(kernel: dict | None) -> dict[str, float]:
    """Map merged kernel name -> mean us from the leaf kernel timing table."""
    if not kernel:
        return {}
    headers = [strip_md(h) for h in kernel["headers"]]
    try:
        name_i = next(i for i, h in enumerate(headers) if h in ("算子", "Kernel"))
        mean_i = next(i for i, h in enumerate(headers) if "平均" in h)
    except StopIteration:
        return {}
    out: dict[str, float] = {}
    for row in kernel["rows"]:
        if name_i >= len(row) or mean_i >= len(row):
            continue
        raw = strip_md(row[name_i]).strip().strip("`")
        m = parse_float(row[mean_i])
        if not raw or m is None:
            continue
        out[raw] = m
        out[_merge_kernel_name(raw)] = m
    return out


def parse_leaf_blocks(md: str) -> list[dict]:
    pattern = re.compile(r"^#### `([^`]+)`（([^）]+)）\s*$", re.MULTILINE)
    matches = list(pattern.finditer(md))
    leaves = []
    for idx, m in enumerate(matches):
        path = m.group(1)
        shape = m.group(2)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md)
        next_h2 = re.search(r"^## ", md[start:end], re.MULTILINE)
        if next_h2:
            end = start + next_h2.start()
        body = md[start:end]
        tables = parse_tables(body)
        kernel = summary = bucket = None
        for t in tables:
            headers = t["headers"]
            if "算子" in headers or "Kernel" in headers:
                kernel = t
            elif headers and headers[0] == "类型" and any(h in ("0-5", "合计") for h in headers):
                bucket = t
            elif headers and headers[0] == "类型":
                summary = t

        meta = []
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("逻辑任务") or s.startswith("AIC 逻辑"):
                meta.append(s)

        counts = {"AIC": [], "AIV": [], "MIX": []}
        means = {"AIC": None, "AIV": None, "MIX": None}
        if summary:
            for row in summary["rows"]:
                kind = strip_md(row[0])
                if kind in means and len(row) > 2:
                    means[kind] = parse_float(row[2])
        if bucket:
            for row in bucket["rows"]:
                label = strip_md(row[0])
                m2 = re.match(r"(AIC|AIV|MIX)\s*count", label)
                if m2:
                    kind = m2.group(1)
                    counts[kind] = [int(parse_float(c) or 0) for c in row[1:8]]

        hist_buckets = list(BUCKETS)
        mix_arr = counts.get("MIX") or []
        # Only when MIX has 50+ samples: split tail into 50-100 / 100-150 / 150+.
        if len(mix_arr) >= 7 and mix_arr[6] > 0:
            leaf_dir = ROOT / path
            swim_ok = (leaf_dir / "chip_swimlane_records.json").exists() or (
                leaf_dir / "dfx_outputs" / "chip_swimlane_records.json"
            ).exists()
            if swim_ok:
                try:
                    summary_st = analyze_leaf(leaf_dir)
                    durs = summary_st.get("durations") or {}
                    mix_fine = hist_counts(durs.get("mix") or [], BUCKETS_MIX_FINE_TAIL)

                    def _aic_aiv_fine(arr: list[int]) -> list[int]:
                        # Fine split is MIX-only; roll AIC/AIV 50+ into 50-100.
                        base = list(arr[:6]) if len(arr) >= 6 else list(arr) + [0] * (6 - len(arr))
                        tail = arr[6] if len(arr) > 6 else 0
                        return base + [tail, 0, 0]

                    counts = {
                        "AIC": _aic_aiv_fine(counts.get("AIC") or []),
                        "AIV": _aic_aiv_fine(counts.get("AIV") or []),
                        "MIX": mix_fine,
                    }
                    hist_buckets = list(BUCKETS_FINE)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    pass

        conc = {}
        conc_path = ROOT / path / "concurrency_analysis.json"
        if conc_path.exists():
            try:
                raw = json.loads(conc_path.read_text(encoding="utf-8"))
                aic = raw.get("resources", {}).get("aic", {})
                aiv = raw.get("resources", {}).get("aiv", {})
                conc = {
                    "aic_peak": aic.get("peak"),
                    "aiv_peak": aiv.get("peak"),
                    "aic_util": aic.get("average_utilization"),
                    "aiv_util": aiv.get("average_utilization"),
                    "span": raw.get("captured_test_span_us"),
                    "envelope": (raw.get("aic_logical_envelope_peak_demand") or {}).get("demand"),
                }
            except (json.JSONDecodeError, OSError):
                conc = {}

        leaf_dir = ROOT / path
        dep_counts = load_kernel_block_nums(leaf_dir)
        dep_full_layers = load_dep_full_layers(leaf_dir)
        dep_means = kernel_means_from_table(kernel)

        leaves.append(
            {
                "path": path,
                "leaf": leaf_name(path),
                "shape": shape,
                "model": "qwen" if is_qwen_path(path) else "flash",
                "kernel": kernel,
                "summary": summary,
                "bucket": bucket,
                "counts": counts,
                "hist_buckets": hist_buckets,
                "means": means,
                "meta": meta,
                "recommended": leaf_name(path) in RECOMMENDED,
                "conc": conc,
                "dep_counts": dep_counts,
                "dep_full_layers": dep_full_layers,
                "dep_means": dep_means,
            }
        )
    return leaves


def extract_section_table(md: str, heading: str) -> dict | None:
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if heading in line and line.startswith("#"):
            chunk = "\n".join(lines[i : i + 40])
            tables = parse_tables(chunk)
            if tables:
                return tables[0]
    return None


def extract_comparison(md: str) -> dict:
    tables = parse_tables(md)
    for t in tables:
        if t["headers"] and t["headers"][0] == "样例" and "AIC均值" in t["headers"]:
            return t
    return tables[0]


def cell_html(text: str, *, rec: bool = False, col_type: str = "fixed") -> str:
    """Render a table cell. Relative markdown links become local <a href>."""
    m = re.search(r"\[([^\]]+)\]\(([^)]+)\)", text)
    cls = ["rec-hot"] if rec else []
    attr = f' class="{" ".join(cls)}"' if cls else ""
    if m:
        label, href = m.group(1), m.group(2).strip()
        # Allow relative in-repo paths only (no scheme / protocol-relative).
        if href and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href) and not href.startswith("//"):
            inner = f'<a href="{esc(href)}">{esc(strip_md(label))}</a>'
            return f'<td data-col-type="{esc(col_type)}"{attr}>{inner}</td>'
    display = strip_md(text)
    inner = esc(display)
    return f'<td data-col-type="{esc(col_type)}"{attr}>{inner}</td>'


def type_filter_bar(table_id: str) -> str:
    return f"""
<div class="filter-bar" data-for="{esc(table_id)}">
  <span class="filter-label">显示</span>
  <button type="button" class="fbtn active" data-filter="ALL">全部</button>
  <button type="button" class="fbtn" data-filter="AIC">AIC</button>
  <button type="button" class="fbtn" data-filter="AIV">AIV</button>
  <button type="button" class="fbtn" data-filter="MIX">MIX</button>
</div>"""


def render_kernel_table(leaf: dict, tid: str) -> str:
    t = leaf["kernel"]
    if not t:
        return ""
    type_idx = t["headers"].index("类型") if "类型" in t["headers"] else 1
    thead = "".join(f"<th>{esc(h)}</th>" for h in t["headers"])
    rows_html = []
    for row in t["rows"]:
        kind = strip_md(row[type_idx])
        cells = "".join(f"<td>{esc(strip_md(c))}</td>" for c in row)
        rows_html.append(f'<tr data-type="{esc(kind)}">{cells}</tr>')
    return f"""
{type_filter_bar(tid)}
<div class="table-wrap">
<table id="{esc(tid)}" class="data filter-rows">
<thead><tr>{thead}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
</div>"""


def render_summary_table(leaf: dict, tid: str) -> str:
    t = leaf["summary"]
    if not t:
        return ""
    thead = "".join(f"<th>{esc(h)}</th>" for h in t["headers"])
    rows_html = []
    for row in t["rows"]:
        kind = strip_md(row[0])
        cells = "".join(f"<td>{esc(strip_md(c))}</td>" for c in row)
        rows_html.append(f'<tr data-type="{esc(kind)}">{cells}</tr>')
    return f"""
{type_filter_bar(tid)}
<div class="table-wrap">
<table id="{esc(tid)}" class="data filter-rows">
<thead><tr>{thead}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
</div>"""


def render_bucket_table(leaf: dict, tid: str) -> str:
    t = leaf["bucket"]
    if not t:
        return ""
    thead = "".join(f"<th>{esc(h)}</th>" for h in t["headers"])
    rows_html = []
    for row in t["rows"]:
        label = strip_md(row[0])
        m = re.match(r"(AIC|AIV|MIX)", label)
        kind = m.group(1) if m else "ALL"
        cells = "".join(f"<td>{esc(strip_md(c))}</td>" for c in row)
        rows_html.append(f'<tr data-type="{esc(kind)}">{cells}</tr>')
    return f"""
{type_filter_bar(tid)}
<div class="table-wrap">
<table id="{esc(tid)}" class="data filter-rows">
<thead><tr>{thead}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
</div>"""


def render_hist(leaf: dict, tid: str) -> str:
    counts = leaf["counts"]
    means = leaf["means"]
    buckets = leaf.get("hist_buckets") or BUCKETS
    fine = len(buckets) > 7
    mean_parts = []
    for kind in ("AIC", "AIV", "MIX"):
        v = means.get(kind)
        if v is None:
            continue
        mean_parts.append(f'<span class="mean-chip {kind.lower()}">{kind} {v:.2f} μs</span>')

    n_bins = len(buckets)
    groups = []
    for bi, bucket in enumerate(buckets):
        bars = []
        for kind in ("AIC", "AIV", "MIX"):
            arr = counts.get(kind) or [0] * n_bins
            total = sum(arr) or 1
            c = arr[bi] if bi < len(arr) else 0
            pct = 100.0 * c / total
            bars.append(
                f'<div class="bar {kind.lower()}" data-type="{kind}" '
                f'style="height:{pct:.2f}%" title="{kind} {bucket}: {c} ({pct:.1f}%)">'
                f'<span class="bar-tip">{c}</span></div>'
            )
        groups.append(
            f'<div class="bucket">'
            f'<div class="bars">{"".join(bars)}</div>'
            f'<div class="blabel">{esc(bucket)}</div>'
            f"</div>"
        )

    note = (
        ""
        if fine
        else "桶宽不均：0–20 每 5μs，其后 10 / 20 / ∞"
    )
    return f"""
<div class="hist" id="{esc(tid)}" data-hist="1">
  <div class="hist-head">
    <div class="hist-title">任务粒度分布（占比 %，同类型内归一）</div>
    <div class="mean-row">{"".join(mean_parts)}</div>
  </div>
  {type_filter_bar(tid + "-filter")}
  <div class="hist-chart">
    <div class="y-axis"><span>100%</span><span>50%</span><span>0%</span></div>
    <div class="buckets">{"".join(groups)}</div>
  </div>
  <div class="legend">
    <span class="lg aic">AIC</span>
    <span class="lg aiv">AIV</span>
    <span class="lg mix">MIX</span>
    <span class="note">{esc(note)}</span>
  </div>
</div>"""


# Dep-graph labels whose SPMD width pins a variable-length worklist
# (grid-stride over n_items); each lane takes data as needed.
# `score` is pinned only on Flash LT/HT (not basic_batch4_mtp1).
PINNED_SPMD_ALWAYS = frozenset({"qk_pv", "attn_swpipe"})
PIN_MARKER = "📌"


def _is_pinned_label(label: str, *, model: str, leaf_name: str) -> bool:
    if label in PINNED_SPMD_ALWAYS:
        return True
    if (
        label == "score"
        and model == "flash"
        and (leaf_name.startswith("lt_batch") or leaf_name.startswith("ht_batch"))
    ):
        return True
    return False


def _render_dep_rows(
    rows: list[list[tuple[str, str, int | None, int | None]]],
    *,
    full: bool = False,
    model: str = "",
    leaf_name: str = "",
) -> str:
    """rows: each item is (label, kind, count|None, mean_us|None)."""
    row_html: list[str] = []
    graph_cls = "dep-graph full" if full else "dep-graph"
    for row in rows:
        nodes_html: list[str] = []
        total = 0
        known = True
        for label, kind, n, mean_us in row:
            pinned = _is_pinned_label(label, model=model, leaf_name=leaf_name)
            if n is None:
                known = False
                count_txt = "—"
            else:
                total += int(n)
                count_txt = str(n)
            if pinned and n is not None:
                count_txt = f"{count_txt}{PIN_MARKER}"
            mean_txt = "—" if mean_us is None else f"{int(mean_us)}us"
            node_cls = f"dep-node {esc(kind)}" + (" compact" if full else "")
            title = f"{label} · avg={mean_txt} · SPMD={count_txt}"
            if pinned:
                title += " · 不定长工作集钉死为固定 SPMD，lane 内按实际取数"
            nodes_html.append(
                f'<div class="{node_cls}" title="{esc(title)}">'
                f'<span class="dep-label">{esc(label)}</span>'
                f'<span class="dep-mean">{esc(mean_txt)}</span>'
                f'<span class="dep-count">{esc(count_txt)}</span>'
                f"</div>"
            )
        total_txt = str(total) if known else "—"
        row_html.append(
            f'<div class="dep-graph-row">'
            f'<div class="dep-nodes">{"".join(nodes_html)}</div>'
            f'<div class="dep-row-total" title="该行 SPMD 合计">{esc(total_txt)}</div>'
            f"</div>"
        )
    return f'<div class="{graph_cls}">{"".join(row_html)}</div>'


def _full_graph_levels(full_layers: list) -> dict[str, int]:
    """Map kernel/display name → topo layer index from the full dep graph."""
    levels: dict[str, int] = {}
    for i, layer in enumerate(full_layers or []):
        for name, _kind, _count in layer:
            levels[name] = i
    return levels


def _node_full_level(label: str, kernels: list[str], levels: dict[str, int]) -> int | None:
    for key in (label, *kernels):
        if key in levels:
            return levels[key]
    return None


def _split_main_row_by_full_levels(
    row_spec: list[tuple[str, str, list[str]]],
    levels: dict[str, int],
) -> list[list[tuple[str, str, list[str]]]]:
    """Keep only true peers on one main-path row.

    Rule: if two nodes are not on the same layer in the full graph, they must
    not share a main-path row (even if FLASH/QWEN_DEP_ROWS listed them together).
    """
    if not levels or len(row_spec) <= 1:
        return [row_spec]
    buckets: dict[int, list[tuple[str, str, list[str]]]] = {}
    order: list[int] = []
    unknown: list[tuple[str, str, list[str]]] = []
    for item in row_spec:
        label, kind, kernels = item
        lv = _node_full_level(label, kernels, levels)
        if lv is None:
            unknown.append(item)
            continue
        if lv not in buckets:
            buckets[lv] = []
            order.append(lv)
        buckets[lv].append(item)
    out = [buckets[lv] for lv in sorted(order)]
    for item in unknown:
        out.append([item])
    return out or [row_spec]


def render_dep_graph_html(leaf: dict) -> str:
    """HTML dependency graph with 主路径 / 完整图 toggle."""
    model = leaf["model"]
    leaf_name = leaf["leaf"]
    block_nums = leaf.get("dep_counts") or {}
    means = leaf.get("dep_means") or {}
    rows_spec = FLASH_DEP_ROWS if model == "flash" else QWEN_DEP_ROWS
    title = "Flash CSA" if model == "flash" else "Qwen decode"

    full_layers = leaf.get("dep_full_layers") or []
    levels = _full_graph_levels(full_layers)

    main_rows: list[list[tuple[str, str, int | None, int | None]]] = []
    for row in rows_spec:
        for sub in _split_main_row_by_full_levels(row, levels):
            main_rows.append(
                [
                    (
                        label,
                        kind,
                        resolve_node_count(block_nums, kernels),
                        resolve_node_mean(means, kernels),
                    )
                    for label, kind, kernels in sub
                ]
            )

    full_rows: list[list[tuple[str, str, int | None, int | None]]] = [
        [
            (
                name,
                kind,
                count,
                int(round(means[name])) if name in means else resolve_node_mean(means, [name]),
            )
            for name, kind, count in layer
        ]
        for layer in full_layers
    ]
    if not full_rows:
        flat = sorted(
            ((n, "aic", c) for n, c in block_nums.items()),
            key=lambda x: (-x[2], x[0]),
        )
        full_rows = [
            [
                (
                    n,
                    k,
                    c,
                    int(round(means[n])) if n in means else None,
                )
                for n, k, c in flat
            ]
        ] if flat else main_rows

    n_full = sum(len(r) for r in full_rows)
    n_main = sum(len(r) for r in main_rows)

    return f"""
<div class="fig dep-fig" data-dep-fig="1">
  <div class="fig-title">{esc(title)} 依赖图</div>
  <div class="dep-mode-bar" role="tablist" aria-label="依赖图模式">
    <button type="button" class="fbtn active" data-dep-mode="main">主路径（{n_main}）</button>
    <button type="button" class="fbtn" data-dep-mode="full">完整图（{n_full}）</button>
  </div>
  <p class="dep-legend">红=AIC · 蓝=AIV · 紫=MIX；左下角=平均时间(us) · 右下角=SPMD 数；右侧=该行 SPMD 合计；📌=不定长工作集钉死为固定 SPMD</p>
  <div class="dep-view" data-dep-mode="main">
    {_render_dep_rows(main_rows, full=False, model=model, leaf_name=leaf_name)}
  </div>
  <div class="dep-view" data-dep-mode="full" hidden>
    {_render_dep_rows(full_rows, full=True, model=model, leaf_name=leaf_name)}
  </div>
</div>"""


def render_leaf_panel(leaf: dict, default_leaf: str) -> str:
    path = leaf["path"]
    lid = f"leaf-{leaf['model']}-{leaf['leaf']}"
    is_default = leaf["leaf"] == default_leaf
    hidden_attr = "" if is_default else " hidden"
    rec = ' <span class="badge-rec">推荐</span>' if leaf["recommended"] else ""
    title_cls = "rec-hot" if leaf["recommended"] else ""
    return f"""
<section class="leaf-panel" id="{esc(lid)}" data-model="{esc(leaf['model'])}" data-leaf="{esc(leaf['leaf'])}"{hidden_attr}>
  <h4 class="{title_cls}"><code>{esc(path)}</code> — {esc(leaf['shape'])}{rec}</h4>
  <h5>依赖图（120 核 SPMD 并行）</h5>
  {render_dep_graph_html(leaf)}
  <h5>各 kernel 执行时间（同类合并）</h5>
  {render_kernel_table(leaf, f"{lid}-kernel")}
  <h5>类型汇总</h5>
  {render_summary_table(leaf, f"{lid}-summary")}
  <h5>任务粒度分桶</h5>
  {render_bucket_table(leaf, f"{lid}-bucket")}
  <h5>直方图</h5>
  {render_hist(leaf, f"{lid}-hist")}
</section>"""


def render_comparison_table(t: dict) -> str:
    tid = "tbl-comparison"
    headers = t["headers"]
    col_meta = []
    for h in headers:
        if "AIC" in h:
            col_meta.append("AIC")
        elif "AIV" in h:
            col_meta.append("AIV")
        elif "MIX" in h:
            col_meta.append("MIX")
        else:
            col_meta.append("fixed")

    thead_cells = [f'<th data-col-type="{col_meta[i]}">{esc(h)}</th>' for i, h in enumerate(headers)]
    rows_html = []
    for row in t["rows"]:
        display = link_display(row[0])
        path = display
        is_main = display.startswith("主线")
        leaf = leaf_name(path) if not is_main else ""
        rec = (not is_main) and leaf in RECOMMENDED
        cells = []
        for i, c in enumerate(row):
            kind = col_meta[i]
            # only name/shape get rec-hot
            cells.append(cell_html(c, rec=rec and kind == "fixed", col_type=kind))
        row_cls = "rec-row" if rec else ("main-row" if is_main else "")
        rows_html.append(f'<tr class="{row_cls}">{"".join(cells)}</tr>')

    return f"""
<p class="legend-line">推荐方案样例名标红（★ 按钮）。</p>
{type_filter_bar(tid)}
<div class="table-wrap">
<table id="{esc(tid)}" class="data filter-cols">
<thead><tr>{"".join(thead_cells)}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
</div>"""


def render_means_col_table(t: dict, tid: str, *, show_filter: bool = True) -> str:
    if not t:
        return ""
    headers = t["headers"]
    col_meta = []
    for h in headers:
        if "AIC/AIV/MIX" in h or "记录 AIC" in h:
            col_meta.append("COMBINED")
        elif re.search(r"\bAIC\b", h) or h.startswith("AIC"):
            col_meta.append("AIC")
        elif re.search(r"\bAIV\b", h) or h.startswith("AIV"):
            col_meta.append("AIV")
        elif re.search(r"\bMIX\b", h) or h.startswith("MIX"):
            col_meta.append("MIX")
        else:
            col_meta.append("fixed")

    thead = "".join(f'<th data-col-type="{col_meta[i]}">{esc(h)}</th>' for i, h in enumerate(headers))
    rows_html = []
    for row in t["rows"]:
        display = link_display(row[0])
        leaf = leaf_name(display)
        rec = leaf in RECOMMENDED
        cells = []
        for i, c in enumerate(row):
            kind = col_meta[i]
            cells.append(cell_html(c, rec=rec and i == 0, col_type=kind))
        row_cls = "rec-row" if rec else ""
        rows_html.append(f'<tr class="{row_cls}">{"".join(cells)}</tr>')

    filter_html = type_filter_bar(tid) if show_filter else ""
    return f"""
{filter_html}
<div class="table-wrap">
<table id="{esc(tid)}" class="data filter-cols">
<thead><tr>{thead}</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
</div>"""


def config_buttons(leaves: list[dict], model: str, default_leaf: str) -> str:
    btns = []
    for leaf in leaves:
        if leaf["model"] != model:
            continue
        cls = ["cfg-btn"]
        if leaf["recommended"]:
            cls.append("rec-btn")
        if leaf["leaf"] == default_leaf:
            cls.append("active")
        star = " ★" if leaf["recommended"] else ""
        btns.append(
            f'<button type="button" class="{" ".join(cls)}" '
            f'data-model="{esc(model)}" data-leaf="{esc(leaf["leaf"])}">'
            f'{esc(leaf["leaf"])}{star}</button>'
        )
    return f'<div class="cfg-bar" data-model="{esc(model)}">{"".join(btns)}</div>'


CSS = r"""
:root {
  --bg: #f4f6f9;
  --panel: #ffffff;
  --border: #d0d7e2;
  --text: #1a2332;
  --muted: #5a6a7e;
  --aic: #4c8bf5;
  --aiv: #2fa36b;
  --mix: #d4890a;
  --rec: #c62828;
  --btn: #eef2f7;
  --btn-active: #d6e4f5;
  --sidebar-w: 220px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  font-size: 14px;
}
a { color: #1a5fb4; }
.layout { display: flex; min-height: 100vh; }
#sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  background: var(--panel);
  border-right: 1px solid var(--border);
  padding: 16px 12px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  transition: margin-left 0.2s, width 0.2s;
}
body.sidebar-collapsed #sidebar {
  margin-left: calc(-1 * var(--sidebar-w) + 36px);
}
#sidebar .side-title { font-weight: 700; margin: 0 0 8px; font-size: 0.95rem; }
#sidebar ul { list-style: none; padding: 0; margin: 0; }
#sidebar li { margin: 6px 0; }
#sidebar a { text-decoration: none; color: var(--text); font-size: 0.88rem; }
#sidebar a:hover { color: #1a5fb4; }
#side-toggle {
  position: fixed;
  left: calc(var(--sidebar-w) - 4px);
  top: 12px;
  z-index: 20;
  width: 28px; height: 28px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--panel);
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  transition: left 0.2s;
}
body.sidebar-collapsed #side-toggle { left: 8px; }
main.content {
  flex: 1;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 28px 80px;
}
h1 { font-size: 1.65rem; margin-top: 0; }
h2 { margin-top: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }
h3 { margin-top: 1.4rem; }
h4 { margin-top: 0.35rem; }
h5 { margin: 0.9rem 0 0.35rem; color: var(--muted); font-weight: 600; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #eef2f7;
  padding: 0.1em 0.35em;
  border-radius: 4px;
  font-size: 0.92em;
}
.rec-hot, .rec-hot a, .rec-hot code { color: var(--rec) !important; font-weight: 700; }
.badge-rec {
  display: inline-block;
  margin-left: 8px;
  padding: 1px 8px;
  border-radius: 999px;
  background: #fdecea;
  border: 1px solid var(--rec);
  color: var(--rec);
  font-size: 0.75rem;
  vertical-align: middle;
}
.legend-line { color: var(--muted); font-size: 0.88rem; margin: 0.4rem 0 0.6rem; }
.note-block {
  color: var(--text);
  font-size: 0.88rem;
  line-height: 1.55;
  margin: 0.6rem 0 1.2rem;
  padding: 10px 12px;
  border-left: 3px solid var(--border);
  background: #f7f9fb;
}
.cfg-bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
.cfg-btn {
  background: var(--btn);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 12px;
  cursor: pointer;
  font-size: 0.9rem;
}
.cfg-btn:hover { border-color: #9ab; }
.cfg-btn.active {
  background: var(--btn-active);
  border-color: #6a90c0;
  box-shadow: 0 0 0 1px #6a90c0 inset;
}
.cfg-btn.rec-btn { border-color: var(--rec); color: var(--rec); font-weight: 600; }
.cfg-btn.rec-btn.active {
  background: #fdecea;
  box-shadow: 0 0 0 1px var(--rec) inset;
}
.filter-bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 8px 0;
}
.filter-label { color: var(--muted); font-size: 0.85rem; margin-right: 4px; }
.fbtn {
  background: var(--btn);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 0.82rem;
}
.fbtn.active { background: var(--btn-active); border-color: #6a90c0; }
.table-wrap { overflow-x: auto; margin-bottom: 12px; }
table.data {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.86rem;
  background: var(--panel);
  border: 1px solid var(--border);
}
table.data th, table.data td {
  border: 1px solid var(--border);
  padding: 5px 8px;
  text-align: left;
  white-space: nowrap;
}
table.data th { background: #e8eef5; }
table.data tbody tr:hover { background: #f0f4fa; }
table.data .main-row { background: #f7f8fa; color: var(--muted); }
.leaf-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px 20px;
  margin-bottom: 12px;
}
.leaf-panel .meta { color: var(--muted); font-size: 0.88rem; }
.fig {
  background: #fafbfc;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
  margin: 8px 0 14px;
  overflow-x: auto;
}
.fig-title { font-weight: 600; margin-bottom: 8px; }
.dep-legend { color: var(--muted); font-size: 0.78rem; margin: 0 0 10px; }
.dep-mode-bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 10px; }
.dep-graph {
  display: flex;
  flex-direction: column;
  gap: 0;
  min-width: 520px;
  max-width: 720px;
}
.dep-graph.full {
  max-width: none;
  min-width: 640px;
}
.dep-graph-row {
  display: grid;
  grid-template-columns: 1fr 72px;
  align-items: center;
  gap: 12px;
  position: relative;
  padding: 6px 0 14px;
}
.dep-graph-row:not(:last-child)::after {
  content: "";
  position: absolute;
  left: calc(50% - 36px);
  bottom: 0;
  width: 0;
  height: 10px;
  border-left: 2px solid #9aa8b8;
}
.dep-nodes {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  min-height: 40px;
}
.dep-node {
  position: relative;
  min-width: 110px;
  max-width: 160px;
  padding: 10px 28px 18px 28px;
  border-radius: 8px;
  border: 1.5px solid #333;
  font-size: 0.82rem;
  font-weight: 600;
  text-align: center;
  color: #1a2332;
  line-height: 1.2;
}
.dep-node.compact {
  min-width: 96px;
  max-width: 150px;
  padding: 8px 24px 16px 24px;
  font-size: 0.72rem;
}
.dep-node.aic { background: #e57373; }
.dep-node.aiv { background: #64b5f6; }
.dep-node.mix { background: #ba68c8; color: #1a1030; }
.dep-label { display: block; word-break: break-word; }
.dep-mean {
  position: absolute;
  left: 6px;
  bottom: 3px;
  font-size: 0.72rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  opacity: 0.95;
}
.dep-count {
  position: absolute;
  right: 6px;
  bottom: 3px;
  font-size: 0.72rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  opacity: 0.95;
}
.dep-row-total {
  text-align: right;
  font-size: 0.95rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: #1a2332;
  padding-right: 4px;
}
.hist {
  background: #fafbfc;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px 16px;
  margin-top: 8px;
}
.hist-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; }
.mean-row { display: flex; flex-wrap: wrap; gap: 8px; }
.mean-chip {
  font-size: 0.82rem;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
}
.mean-chip.aic { border-color: var(--aic); }
.mean-chip.aiv { border-color: var(--aiv); }
.mean-chip.mix { border-color: var(--mix); }
.hist-chart { display: flex; gap: 8px; margin-top: 12px; height: 220px; }
.y-axis {
  display: flex; flex-direction: column; justify-content: space-between;
  color: var(--muted); font-size: 0.72rem; width: 36px; text-align: right; padding-bottom: 22px;
}
.buckets {
  flex: 1; display: flex; gap: 6px;
  border-left: 1px solid var(--border); border-bottom: 1px solid var(--border); padding: 0 4px;
}
.bucket { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.bars {
  flex: 1; display: flex; align-items: flex-end; justify-content: center; gap: 3px; padding-top: 8px;
}
.bar {
  width: 28%; max-width: 18px; border-radius: 3px 3px 0 0; position: relative;
}
.bar.aic { background: var(--aic); }
.bar.aiv { background: var(--aiv); }
.bar.mix { background: var(--mix); }
.bar[hidden] { display: none; }
.bar-tip {
  position: absolute; top: -16px; left: 50%; transform: translateX(-50%);
  font-size: 0.65rem; color: var(--muted); white-space: nowrap; opacity: 0;
}
.bar:hover .bar-tip { opacity: 1; }
.blabel { text-align: center; font-size: 0.72rem; color: var(--muted); padding-top: 6px; height: 22px; }
.legend {
  display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
  margin-top: 10px; font-size: 0.82rem; color: var(--muted);
}
.lg::before {
  content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 6px; vertical-align: -1px;
}
.lg.aic::before { background: var(--aic); }
.lg.aiv::before { background: var(--aiv); }
.lg.mix::before { background: var(--mix); }
.legend .note { margin-left: auto; font-size: 0.75rem; }
.hidden-col { display: none !important; }
tr[hidden], .leaf-panel[hidden] { display: none !important; }
@media (max-width: 800px) {
  #sidebar { position: fixed; z-index: 15; box-shadow: 2px 0 12px rgba(0,0,0,.08); }
  main.content { padding: 20px 14px 60px; }
}
"""

JS = r"""
(function () {
  var toggle = document.getElementById('side-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      document.body.classList.toggle('sidebar-collapsed');
      toggle.textContent = document.body.classList.contains('sidebar-collapsed') ? '»' : '«';
    });
  }

  function setConfig(model, leaf) {
    document.querySelectorAll('.cfg-bar[data-model="' + model + '"] .cfg-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.leaf === leaf);
    });
    document.querySelectorAll('.leaf-panel[data-model="' + model + '"]').forEach(function (p) {
      if (p.dataset.leaf === leaf) p.removeAttribute('hidden');
      else p.setAttribute('hidden', '');
    });
  }

  document.querySelectorAll('.cfg-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      setConfig(btn.dataset.model, btn.dataset.leaf);
    });
  });

  document.querySelectorAll('.dep-fig').forEach(function (fig) {
    var bar = fig.querySelector('.dep-mode-bar');
    if (!bar) return;
    bar.addEventListener('click', function (e) {
      var btn = e.target.closest('[data-dep-mode]');
      if (!btn || !bar.contains(btn)) return;
      var mode = btn.getAttribute('data-dep-mode');
      bar.querySelectorAll('.fbtn').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      fig.querySelectorAll('.dep-view').forEach(function (v) {
        if (v.getAttribute('data-dep-mode') === mode) v.removeAttribute('hidden');
        else v.setAttribute('hidden', '');
      });
    });
  });

  function applyRowFilter(table, filter) {
    table.querySelectorAll('tbody tr').forEach(function (tr) {
      var t = tr.getAttribute('data-type') || 'ALL';
      if (filter === 'ALL' || t === filter) tr.removeAttribute('hidden');
      else tr.setAttribute('hidden', '');
    });
  }

  function applyColFilter(table, filter) {
    var show = function (kind) {
      if (filter === 'ALL') return true;
      if (kind === 'fixed' || kind === 'COMBINED') return true;
      return kind === filter;
    };
    table.querySelectorAll('th[data-col-type], td[data-col-type]').forEach(function (el) {
      if (show(el.getAttribute('data-col-type'))) el.classList.remove('hidden-col');
      else el.classList.add('hidden-col');
    });
  }

  function applyHistFilter(hist, filter) {
    hist.querySelectorAll('.bar').forEach(function (bar) {
      var t = bar.getAttribute('data-type');
      if (filter === 'ALL' || t === filter) bar.removeAttribute('hidden');
      else bar.setAttribute('hidden', '');
    });
  }

  document.querySelectorAll('.filter-bar').forEach(function (bar) {
    bar.addEventListener('click', function (e) {
      var btn = e.target.closest('.fbtn');
      if (!btn) return;
      bar.querySelectorAll('.fbtn').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      var filter = btn.dataset.filter;
      var table = document.getElementById(bar.getAttribute('data-for'));
      if (table) {
        if (table.classList.contains('filter-rows')) applyRowFilter(table, filter);
        if (table.classList.contains('filter-cols')) applyColFilter(table, filter);
      }
      var hist = bar.closest('.hist');
      if (hist) applyHistFilter(hist, filter);
    });
  });

  setConfig('qwen', 'lt_batch16');
  setConfig('flash', 'lt_batch12_mtp7');
})();
"""


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    md = README.read_text(encoding="utf-8")
    leaves = parse_leaf_blocks(md)
    comparison = extract_comparison(md)
    qwen_cmp = extract_section_table(md, "## 1.3 六档对照")
    flash_cmp = extract_section_table(md, "## 2.3 八档对照") or extract_section_table(md, "## 2.3 七档对照")

    qwen_leaves = [L for L in leaves if L["model"] == "qwen"]
    flash_leaves = [L for L in leaves if L["model"] == "flash"]

    body = f"""
<div class="layout">
<aside id="sidebar">
  <p class="side-title">目录</p>
  <ul>
    <li><a href="#comparison">总对比表</a></li>
    <li><a href="#qwen">Qwen3 decode</a></li>
    <li><a href="#flash">Flash CSA</a></li>
  </ul>
</aside>
<button type="button" id="side-toggle" title="收起/展开目录">«</button>
<main class="content">
  <h1>V200-benchmark 样例说明</h1>
  <p class="legend-line">单卡 decode 服务负载 · AIC / AIV / MIX 任务粒度。推荐方案标红。</p>

  <h2 id="comparison">总对比表</h2>
  {render_comparison_table(comparison)}
  <p class="note-block"><strong>注释（主线 Qwen3 vs basic_batch16）</strong>：同形 batch=16 / ATTN=24，AIC 已对齐（375 条，33.45 vs 33.00）。<code>basic_batch16</code> 去掉了核内 <code>syncall</code>，把 Phase-0 从 MIX 的 <code>attn_swpipe</code> 里拆成独立 <code>attn_phase0</code>（+32 条 AIV），故 AIV 均值上升（6.35→8.33），MIX 均值下降（180.68→156.94；实例数仍为 24）。</p>

  <h2 id="qwen">1. Qwen3 decode layer</h2>
  <h3>各配置</h3>
  <p class="legend-line">默认 <code>lt_batch16</code>（推荐）。依赖图见各配置面板（拓扑固定，SPMD 个数随参数变化）。按钮切换 kernel / 分桶 / 直方图。</p>
  {config_buttons(leaves, "qwen", "lt_batch16")}
  {"".join(render_leaf_panel(L, "lt_batch16") for L in qwen_leaves)}
  <h3>六档对照</h3>
  {render_means_col_table(qwen_cmp, "tbl-qwen-cmp")}

  <h2 id="flash">2. Flash CSA（DeepSeek）</h2>
  <h3>各配置</h3>
  <p class="legend-line">默认 <code>lt_batch12_mtp7</code>（推荐，B=12 / SPMD=96）。依赖图见各配置面板（拓扑固定，个数随参数变化）。</p>
  {config_buttons(leaves, "flash", "lt_batch12_mtp7")}
  {"".join(render_leaf_panel(L, "lt_batch12_mtp7") for L in flash_leaves)}
  <h3>八档对照</h3>
  {render_means_col_table(flash_cmp, "tbl-flash-cmp")}
</main>
</div>
"""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>V200-benchmark 样例说明</title>
<style>
{CSS}
</style>
</head>
<body>
{body}
<script>
{JS}
</script>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT} ({len(leaves)} leaves, {OUT.stat().st_size} bytes)")
    flash_tags = [L["leaf"] for L in flash_leaves]
    print("flash leaves:", ", ".join(flash_tags))


if __name__ == "__main__":
    main()
