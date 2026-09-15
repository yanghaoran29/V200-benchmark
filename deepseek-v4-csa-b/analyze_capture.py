"""Analyze this paired capture; requires the pinned runtime and NetworkX."""
import json
from collections import defaultdict
from pathlib import Path
from simpler_setup.tools.swimlane_converter import read_perf_data

import sys
base=Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
raw_path=base/"Chip_swimlane_records.json"
deps=json.loads((base/"deps.json").read_text())
name_path=base/"name_map.json"
names=json.loads(name_path.read_text())["callable_id_to_name"]
perf=read_perf_data(raw_path)
rows=perf["tasks"]
raw=json.loads(raw_path.read_text())
caps={typ:raw["metadata"]["core_types"].count(typ) for typ in ("aic","aiv")}

def peak_info(xs):
    events=[]
    for r in xs:
        events.append((r["start_time_us"],1,r))
        events.append((r["end_time_us"],-1,r))
    active={}
    peak=0; peak_t=None; peak_rows=[]
    for tm,delta,r in sorted(events,key=lambda x:(x[0],x[1])):
        key=id(r)
        if delta<0: active.pop(key,None)
        else: active[key]=r
        if len(active)>peak:
            peak=len(active); peak_t=tm; peak_rows=list(active.values())
    return peak,peak_t,peak_rows

resources={}
for typ in ("aic","aiv"):
    xs=[r for r in rows if r["core_type"]==typ]
    peak,t,_=peak_info(xs)
    start=min(r["start_time_us"] for r in xs); end=max(r["end_time_us"] for r in xs)
    busy=sum(r["duration_us"] for r in xs)
    resources[typ]={
      "rows":len(xs),"peak":peak,"peak_time_us":t,"busy_sum_us":round(busy,2),
      "span_us":round(end-start,2),"average_utilization":round(busy/(caps[typ]*(end-start)),4)
    }

dep_by_id={int(t["task_id"]):t for t in deps["tasks"]}
physical_by_id=defaultdict(list)
for r in rows: physical_by_id[r["task_id"]].append(r)

logical=[]
for tid,t in dep_by_id.items():
    kid=t["kernel_ids"][0]
    if kid<0: continue
    xs=[r for r in physical_by_id[tid] if r["core_type"]=="aic"]
    if not xs: continue
    logical.append({
      "task_id":tid,"name":names[str(kid)],"block_num":t["block_num"],
      "start_us":min(r["start_time_us"] for r in xs),
      "end_us":max(r["end_time_us"] for r in xs),
      "exec_sum_us":sum(r["duration_us"] for r in xs),
      "rows":len(xs),
    })

def logical_peak(metric):
    events=[]
    for x in logical:
        events += [(x["start_us"],1,x),(x["end_us"],-1,x)]
    active={}; best=-1; bt=None; ba=[]
    for tm,d,x in sorted(events,key=lambda z:(z[0],z[1])):
        if d<0: active.pop(x["task_id"],None)
        else: active[x["task_id"]]=x
        val=sum(v["block_num"] for v in active.values()) if metric=="demand" else len(active)
        if val>best: best=val;bt=tm;ba=list(active.values())
    return best,bt,ba

demand,dt,da=logical_peak("demand")
count,ct,ca=logical_peak("count")
def compact(xs): return [{"name":x["name"],"block_num":x["block_num"]} for x in xs]

selected={}
for base_name in ("qr_proj_matmul","qproj_matmul","kv_proj_matmul","kv_score_proj","idx_qr_proj_matmul","qr_hadamard_matmul","score_aic","qk_pv_aic","proj_a_mm"):
    xs=[x for x in logical if x["name"]==base_name or x["name"].startswith(base_name+"_")]
    if not xs: continue
    selected[base_name]={
      "occurrences":len(xs),"declared_blocks":sum(x["block_num"] for x in xs),
      "rows":sum(x["rows"] for x in xs),"start_us":min(x["start_us"] for x in xs),
      "end_us":max(x["end_us"] for x in xs),"exec_sum_us":round(sum(x["exec_sum_us"] for x in xs),2)
    }

expected=sum(t["block_num"]*sum(k>=0 for k in t["kernel_ids"]) for t in deps["tasks"])
result={
 "level":perf["chip_swimlane_level"],"caps":caps,
 "dependency_tasks":len(deps["tasks"]),
 "early_dispatch_true":sum(t.get("early_dispatch") is True for t in deps["tasks"]),
 "early_dispatch_false":sum(t.get("early_dispatch") is False for t in deps["tasks"]),
 "expected_physical_rows":expected,"joined_rows":len(rows),
 "captured_test_span_us":round(max(r["finish_time_us"] for r in rows)-min(r["dispatch_time_us"] for r in rows),2),
 "resources":resources,
 "aic_logical_envelope_peak_demand":{"demand":demand,"time_us":dt,"logical_tasks":len(da),"active":compact(da)},
 "aic_logical_envelope_peak_count":{"count":count,"time_us":ct,"demand":sum(x["block_num"] for x in ca),"active":compact(ca)},
 "selected":selected,
}

# A weighted antichain is a set of tasks with no dependency path between them.
# Weighted Dilworth via a capacitated bipartite flow gives an exact static bound.
import networkx as nx

graph = nx.DiGraph()
graph.add_nodes_from(dep_by_id)
for edge in deps['edges']:
    if 'wait' in edge.get('flags', []):
        graph.add_edge(int(edge['pred']), int(edge['succ']))
assert nx.is_directed_acyclic_graph(graph)
reach = nx.transitive_closure_dag(graph)

def antichain(weights):
    flow = nx.DiGraph()
    total = sum(weights.values())
    for tid, weight in weights.items():
        flow.add_edge('source', ('left', tid), capacity=weight)
        flow.add_edge(('right', tid), 'sink', capacity=weight)
    for a, b in reach.edges:
        if a in weights and b in weights:
            flow.add_edge(('left', a), ('right', b), capacity=total + 1)
    value, (left, right) = nx.minimum_cut(flow, 'source', 'sink')
    selected = [tid for tid in weights if ('left', tid) in left and ('right', tid) in right]
    assert sum(weights[tid] for tid in selected) == total - value
    assert all(not reach.has_edge(a, b) for a in selected for b in selected if a != b)
    return dict(value=total-value, tasks=[dict(task_id=tid, name=names[str(next(k for k in dep_by_id[tid]['kernel_ids'] if k >= 0))], block_num=dep_by_id[tid]['block_num']) for tid in selected])

result['static_dag'] = {
    'definition': 'Exact maximum weighted antichain of wait dependencies; a potential ready set, not measured occupancy.',
    'logical_task_width': antichain({tid: 1 for tid in dep_by_id}),
    'aic_block_width': antichain({tid: t['block_num'] for tid, t in dep_by_id.items() if t['kernel_ids'][0] >= 0}),
    'aiv_block_width': antichain({tid: t['block_num'] * sum(k >= 0 for k in t['kernel_ids'][1:]) for tid, t in dep_by_id.items() if any(k >= 0 for k in t['kernel_ids'][1:])}),
}
result['target_aic_cores'] = 120
result['target_120_aic_measured'] = False
result['target_capacity_waves'] = {'score': [100], 'qk_pv': [100], 'qproj': [120, 8]}

def percentile(values, q):
    at = (len(values) - 1) * q
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)

by_kernel = defaultdict(list)
for row in rows:
    task = dep_by_id[row['task_id']]
    kid = task['kernel_ids'][0] if row['core_type'] == 'aic' else next(k for k in task['kernel_ids'][1:] if k >= 0)
    by_kernel[(names[str(kid)], row['core_type'])].append(row)
result['kernel_statistics'] = []
for (name, core_type), records in by_kernel.items():
    durations = sorted(r['duration_us'] for r in records)
    result['kernel_statistics'].append(dict(
        name=name, core_type=core_type, records=len(records),
        logical_tasks=len({r['task_id'] for r in records}),
        total_us=sum(durations), mean_us=sum(durations)/len(durations),
        min_us=durations[0], max_us=durations[-1],
        median_us=percentile(durations, .5), p90_us=percentile(durations, .9), p99_us=percentile(durations, .99)))
(base/'concurrency_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('resources', 'static_dag')}, indent=2))
