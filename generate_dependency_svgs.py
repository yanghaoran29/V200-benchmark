"""Render grouped captured dependencies; requires NetworkX and Graphviz dot."""
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
    total = sum(sum(c.values()) for c in counts.values())
    assert total == len(capture['aicore_tasks'])
    totals = Counter()
    for c in counts.values():
        totals.update(c)
    legend = ('AIC = red | AIV = blue | MIX = purple',
              'x N = physical records; MIX counts AIC + AIV',
              'Grouped operator dependencies; transitive edges omitted.',
              'An edge denotes a captured dependency between members, not an all-block barrier.')
    header = f'{title}\n{subtitle}\n{total} records = {totals["aic"]} AIC + {totals["aiv"]} AIV\n' + '\n'.join(legend)
    lines = ['digraph G {', 'graph [rankdir=TB, bgcolor="white", pad=0.35, nodesep=0.35, ranksep=0.48, splines=polyline, fontname="DejaVu Sans", labelloc=t, fontsize=15, label=' + json.dumps(header) + '];',
             'node [shape=plain, fontname="DejaVu Sans"];',
             'edge [color="#566579", arrowsize=0.65, penwidth=1.15];']
    ids = {name: f'n{i}' for i, name in enumerate(counts)}
    for name, count in counts.items():
        typ = 'mix' if count['aic'] and count['aiv'] else ('aic' if count['aic'] else 'aiv')
        detail = f'{count["aic"]} AIC + {count["aiv"]} AIV' if typ == 'mix' else typ.upper()
        label = (f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="7" COLOR="#566579" BGCOLOR="{COLORS[typ]}">'
                 f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="14">{escape(name)}</FONT></TD></TR>'
                 f'<TR><TD ALIGN="RIGHT"><FONT POINT-SIZE="10">{detail}  |  x {sum(count.values())}</FONT></TD></TR></TABLE>>')
        lines.append(f'{ids[name]} [label={label}];')
    for a, b in sorted(reduced.edges):
        lines.append(f'{ids[a]} -> {ids[b]};')
    lines.append('}')
    subprocess.run(['dot', '-Tsvg', '-o', str(folder / 'dependency_graph.svg')],
                   input='\n'.join(lines), text=True, check=True)
    data = {'physical_records': total, 'resources': dict(totals),
            'nodes': [{'name': n, 'physical_records': sum(c.values()), 'resources': dict(c),
                       'logical_task_ids': call_ids[n]} for n, c in counts.items()],
            'grouped_edges': sorted(grouped.edges), 'displayed_edges': sorted(reduced.edges)}
    (folder / 'dependency_graph_counts.json').write_text(json.dumps(data, indent=2) + '\n')
    print(case, len(counts), 'nodes;', total, 'records;', len(reduced.edges), 'displayed edges')


if __name__ == '__main__':
    for case, (title, subtitle) in CASES.items():
        generate(case, title, subtitle)
