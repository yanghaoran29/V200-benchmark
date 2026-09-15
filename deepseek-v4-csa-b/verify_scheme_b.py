"""Check Scheme B work coverage, fixture ownership and captured dispatch invariants."""
import ast
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'pypto-lib-operator'))


def main():
    import torch
    from decode_csa import B, S, build_tensor_specs
    from decode_indexer import REDUCE_NSPLIT, REDUCE_TILE, SCORE_LEN, SCORE_BLOCKS
    from decode_sparse_attn_csa import NUM_QK_CORES, QK_ITEMS, SPARSE_BLOCKS

    assert (B, S, REDUCE_NSPLIT, NUM_QK_CORES, QK_ITEMS, SPARSE_BLOCKS) == (20, 2, 5, 100, 200, 5)
    score_items = [[unit for unit in range(lane, B * S * REDUCE_NSPLIT, SCORE_BLOCKS)] for lane in range(SCORE_BLOCKS)]
    assert SCORE_BLOCKS == 100 and all(len(work) == 2 for work in score_items)
    assert sorted(unit for work in score_items for unit in work) == list(range(B * S * REDUCE_NSPLIT))
    # Every visible score element has exactly one owner, including empty/tail pages.
    for length in range(SCORE_LEN + 1):
        pages = (length + REDUCE_TILE - 1) // REDUCE_TILE
        ownership = [page for lane in range(5) for page in range(lane, pages, 5)]
        assert sorted(ownership) == list(range(pages)), length
    items = [[lane + it * NUM_QK_CORES for it in range(
        (QK_ITEMS - lane + NUM_QK_CORES - 1) // NUM_QK_CORES)] for lane in range(NUM_QK_CORES)]
    assert all(len(work) == 2 for work in items)
    assert sorted(item for work in items for item in work) == list(range(QK_ITEMS))
    # Actual fixture tables must be dense in the physical pool and disjoint across requests.
    specs = {s.name: s for s in build_tensor_specs()}
    pools = {"kv_cache": specs["kv_cache"].shape[0]}
    for table_name, pool_name in [('compress_state_block_table', 'compress_state'),
                                  ('inner_compress_state_block_table', 'inner_compress_state'),
                                  ('cmp_block_table', 'cmp_kv'), ('idx_block_table', 'idx_kv_cache')]:
        table = specs[table_name].init_value()
        pages = table[table >= 0].tolist()
        assert len(pages) == len(set(pages)) == specs[pool_name].shape[0], table_name
        assert sorted(pages) == list(range(len(pages))), table_name
        pools[pool_name] = len(pages)
    positions = specs['position_ids'].init_value().reshape(B, S)
    assert torch.equal(positions, torch.tensor([[8192, 8193]] * B, dtype=positions.dtype))
    for slot_name, pool_name, page_size in [('state_slot_mapping', 'compress_state', 4),
                                           ('inner_state_slot_mapping', 'inner_compress_state', 4),
                                           ('ori_slot_mapping', 'kv_cache', 128),
                                           ('cmp_slot_mapping', 'cmp_kv', 32),
                                           ('idx_slot_mapping', 'idx_kv_cache', 32)]:
        slots = specs[slot_name].init_value()
        valid = slots[slots >= 0]
        assert torch.all(valid < pools[pool_name] * page_size), slot_name
        # A compressed slot can repeat within a request; it cannot alias another request.
        per_request = [set(row[row >= 0].tolist()) for row in slots.reshape(B, S)]
        assert sum(map(len, per_request)) == len(set().union(*per_request)), slot_name
    def inventory(folder):
        deps = json.loads((folder / 'deps.json').read_text())
        names = json.loads((folder / 'name_map.json').read_text())['callable_id_to_name']
        tasks = Counter((names[str(next(k for k in t['kernel_ids'] if k >= 0))], t['block_num']) for t in deps['tasks'])
        return deps, tasks
    deps, actual = inventory(HERE)
    _, expected = inventory(HERE.parent / 'deepseek-v4-csa')
    expected[('score_aic', 80)] -= 1
    expected[('score_aic', 100)] += 1
    expected[('qk_pv_aic', 40)] -= 1
    expected[('qk_pv_aic', 100)] += 1
    assert actual == +expected, (actual - expected, expected - actual)
    assert len(deps['tasks']) == 70
    assert all(t.get('early_dispatch') is False for t in deps['tasks'])
    assert all(any(k >= 0 for k in t['kernel_ids']) for t in deps['tasks'])
    physical = [sum(t['block_num'] for t in deps['tasks'] if t['kernel_ids'][slot] >= 0) for slot in range(3)]
    assert (physical[0], sum(physical[1:])) == (825, 597), physical
    for path in (HERE / 'pypto-lib-operator').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not ast.unparse(node.func).lower().endswith('syncall'), path
                assert not any(k.arg in ('allow_early_resolve', 'sync_start') and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords), path
    report = dict(status='PASS', score_lengths_checked=SCORE_LEN + 1,
                  qk_work_items=QK_ITEMS, qk_items_per_block=2,
                  batch=B, tokens=B*S, seed=1234,
                  start_positions=positions[:, 0].tolist(), physical_pools=pools,
                  logical_tasks=70, aic_records=825, aiv_records=597, score_blocks=SCORE_BLOCKS, score_items_per_block=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
