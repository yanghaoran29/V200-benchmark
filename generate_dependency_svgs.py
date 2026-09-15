"""Render grouped captured dependencies; requires NetworkX and Graphviz neato."""
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import re
import subprocess

import networkx as nx

ROOT = Path(__file__).resolve().parent
CASES = {
    'qwen3-decode-layer': ('Qwen3 decode layer', 'B=16 | Attention=120 blocks'),
    'deepseek-v4-csa': ('DeepSeek V4 CSA - A', 'B=4, S=2 | start_pos=[8192,0,2,3]'),
    'deepseek-v4-csa-b': ('DeepSeek V4 CSA - B', 'B=20, S=2 | start_pos=8192 for all requests'),
}
COLORS = {'aic': '#F8BFC2', 'aiv': '#9EDDF6', 'mix': '#D4B5EF'}


def generate(case, title, subtitle):
    folder = ROOT / case
    deps, names, capture = [json.loads((folder / f).read_text()) for f in
                            ('deps.json', 'name_map.json', 'Chip_swimlane_records.json')]
    names = names['callable_id_to_name']
    tasks = {int(t['task_id']): t for t in deps['tasks']}
    rows = defaultdict(Counter)
    identities = set()
    for row in capture['aicore_tasks']:
        core, tid, register = row[:3]
        assert tid in tasks and (core, register) not in identities
        identities.add((core, register))
        rows[tid][capture['metadata']['core_types'][core]] += 1
    groups, counts, call_ids = {}, defaultdict(Counter), defaultdict(list)
    physical_tasks = Counter()
    for tid, task in tasks.items():
        kids = task['kernel_ids']
        assert sum(rows[tid].values()) == task['block_num'] * sum(k >= 0 for k in kids)
        name = names[str(next(k for k in kids if k >= 0))]
        if case == 'qwen3-decode-layer':
            name = re.sub(r'_\d+$', '', name)
        name = {'attn_swpipe_aic': 'Attention (QK / softmax / PV)',
                'score_aic': 'Indexer score', 'qk_pv_aic': 'QK / PV'}.get(name, name)
        # Keep main and indexer compressor branches distinct even with similar kernels.
        if case != 'qwen3-decode-layer':
            name = {'kv_score_proj': 'main: kv_score_proj',
                    'kv_score_proj_0': 'indexer: kv_score_proj',
                    'scatter_softmax_pool': 'main: scatter_softmax_pool',
                    'scatter_softmax_pool_0': 'indexer: scatter_softmax_pool',
                    'rope_interleave': 'Q: rope_interleave',
                    'rope_interleave_0': 'compressed: rope_interleave'}.get(name, name)
        groups[tid] = name
        counts[name].update(rows[tid])
        mixed = kids[0] >= 0 and any(k >= 0 for k in kids[1:])
        physical_tasks[name] += task["block_num"] if mixed else sum(rows[tid].values())
        call_ids[name].append(str(tid))
    graph = nx.DiGraph()
    graph.add_nodes_from(tasks)
    for edge in deps['edges']:
        if 'wait' in edge.get('flags', []):
            graph.add_edge(int(edge['pred']), int(edge['succ']))
    assert nx.is_directed_acyclic_graph(graph)
    # Contract nonexecuting tensor-creator nodes, preserving dependency paths.
    for node in list(nx.topological_sort(graph)):
        if node not in tasks:
            graph.add_edges_from((a, b) for a in list(graph.predecessors(node))
                                 for b in list(graph.successors(node)))
            graph.remove_node(node)
    grouped = nx.DiGraph()
    grouped.add_nodes_from(counts)
    grouped.add_edges_from((groups[a], groups[b]) for a, b in graph.edges if groups[a] != groups[b])
    assert nx.is_directed_acyclic_graph(grouped), 'Grouping would create a false cycle'
    reduced = nx.transitive_reduction(grouped)
    assert set(nx.transitive_closure_dag(grouped).edges) == set(nx.transitive_closure_dag(reduced).edges)
    for a, b in list(reduced.edges):
        reduced.remove_edge(a, b)
        assert not nx.has_path(reduced, a, b), "Redundant transitive edge"
        reduced.add_edge(a, b)
    total = sum(sum(c.values()) for c in counts.values())
    assert total == len(capture['aicore_tasks'])
    totals = Counter()
    for c in counts.values():
        totals.update(c)
    header = title
    lines = ['digraph G {', 'graph [rankdir=TB, overlap=true, bgcolor="white", pad=0.35, nodesep=0.35, ranksep=0.48, splines=true, fontname="DejaVu Sans", labelloc=t, fontsize=24, label=' + json.dumps(header) + '];',
             'node [shape=plain, fontname="DejaVu Sans"];',
             'edge [color="#566579", arrowsize=0.65, penwidth=1.15];']
    ids = {name: f'n{i}' for i, name in enumerate(counts)}
    if case == 'qwen3-decode-layer':
        positions = {
            'copy_hidden': (0, 0), 'x_gamma0': (0, 1),
            'q_seed': (-1, 2), 'kv_seed': (1, 2),
            'q_proj': (-1, 3), 'k_proj': (0, 3), 'v_proj': (1, 3),
            'rms_recip': (-2, 3), 'mlp_out_seed': (2, 3),
            'attn_phase0': (0, 4), 'attn_out_seed': (1, 4),
            'Attention (QK / softmax / PV)': (0, 5), 'out_proj': (0, 6),
            'residual_rms_cast': (0, 7), 'post_rms_reduce': (1, 7),
            'gate_proj': (-0.7, 8), 'up_proj': (0.7, 8),
            'silu': (0, 9), 'down_proj': (0, 10),
            'dcr_xgamma': (0, 11), 'copy_out': (0, 12),
        }
    else:
        positions = {
            'hc_pre_rms': (-0.6, 0), 'hc_pre_linear': (0.6, 0),
            'hc_pre_linear_reduce': (0.6, 1),
            'split_pre_post': (0, 2), 'comb_sinkhorn': (3, 2),
            'mix_x': (0, 3), 'rms_norm': (0, 4),
            'qr_proj_seed': (0, 5), 'kv_proj_seed': (-2, 5),
            'csa_rope_step': (-3, 6), 'csa_cmp_rope': (3, 6),
            'qr_proj_matmul': (0, 6), 'kv_proj_matmul': (-2, 6),
            'weights_proj': (-1, 6), 'indexer: kv_score_proj': (1, 6),
            'main: kv_score_proj': (2, 6),
            'qr_rms_norm_quant': (0, 7),
            'main: scatter_softmax_pool': (2, 7),
            'compressed: rope_interleave': (3, 7),
            'qproj_matmul': (-1, 8), 'idx_qr_proj_matmul': (0, 8),
            'indexer: scatter_softmax_pool': (1.3, 8),
            'rmsnorm_rope_cache_write': (2.6, 9),
            'q_rope_prepare': (-2, 9), 'idx_qr_proj_dequant': (-0.5, 9),
            'Q: rope_interleave': (0.5, 9),
            'qproj_dequant_rms_nope_rope': (-1.5, 10),
            'kv_rms_norm_rope': (-2.6, 10),
            'qr_rope_swap_idx': (-0.5, 10), 'rmsnorm_rope': (1.3, 10),
            'qr_rope': (0, 11), 'csa_cache_writeback': (-2.6, 11),
            'qr_hadamard_matmul': (0, 12), 'kv_hadamard': (1.3, 11),
            'qr_hadamard_quant': (0, 13), 'kv_and_cache_write': (1.3, 12),
            'weights_proj_reduce': (-1, 13),
            'Indexer score': (0, 14), 'topk': (0, 15),
            'csa_slots_build_valid_qk_plan': (0, 16),
            'kv_touch': (-2.6, 16), 'QK / PV': (0, 17),
            'rope_cs': (-1.3, 17), 'merge_norm': (0, 18),
            'proj_a_mm': (0, 19), 'quant': (0, 20), 'proj_b_mm': (0, 21),
            'proj_b_act': (0, 22), 'hc_post': (0, 23),
        }
    assert set(positions) == set(counts)

    for name, count in counts.items():
        typ = 'mix' if count['aic'] and count['aiv'] else ('aic' if count['aic'] else 'aiv')
        detail = typ.upper()
        label = (f'<<TABLE BORDER="1" WIDTH="280" HEIGHT="58" CELLBORDER="0" CELLSPACING="0" CELLPADDING="7" COLOR="#566579" BGCOLOR="{COLORS[typ]}">'
                 f'<TR><TD WIDTH="280" ALIGN="CENTER"><FONT POINT-SIZE="14">{escape(name)}</FONT></TD></TR>'
                 f'<TR><TD WIDTH="280" ALIGN="RIGHT"><FONT POINT-SIZE="10">{detail}  |  x {physical_tasks[name]}</FONT></TD></TR></TABLE>>')
        x, y = positions[name]
        lines.append(f'{ids[name]} [pos="{x * 330},{-y * 125}!", label={label}];')
    for a, b in sorted(reduced.edges):
        lines.append(f'{ids[a]} -> {ids[b]} [tailport=s, headport=n];')
    lines.append('}')
    subprocess.run(['neato', '-n2', '-Tsvg', '-o', str(folder / 'dependency_graph.svg')],
                   input='\n'.join(lines), text=True, check=True)
    data = {'physical_tasks': sum(physical_tasks.values()),
            'counting_rule': 'AIC/AIV: execution records; MIX: one task per mixed SPMD block',
            'physical_records': total, 'resources': dict(totals),
            'nodes': [{'name': n, 'physical_tasks': physical_tasks[n], 'physical_records': sum(c.values()), 'resources': dict(c),
                       'logical_task_ids': call_ids[n]} for n, c in counts.items()],
            'grouped_edges': sorted(grouped.edges), 'displayed_edges': sorted(reduced.edges)}
    (folder / 'dependency_graph_counts.json').write_text(json.dumps(data, indent=2) + '\n')
    print(case, len(counts), 'nodes;', sum(physical_tasks.values()), 'physical tasks;', total, 'records;', len(reduced.edges), 'displayed edges')


if __name__ == '__main__':
    for case, (title, subtitle) in CASES.items():
        generate(case, title, subtitle)
