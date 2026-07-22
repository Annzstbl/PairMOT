"""0719_01 pair-consensus Liquid with task-preserving set diversity."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


_sampler = model['backbone']['liquid_sampler']
_sampler.update(
    # One jointly generated route is sampled once and used by both frames.
    pair_sampler_router=None,
    pair_consensus_router=dict(
        hidden_dims=64,
        zero_init=True),
    # Set transport is a confidence-gated preference, never a hard ban on
    # repeated sets. Hard groups remain independently selected.
    hard_group_unique_sets=False,
    use_soft_context_after_hard=True,
    soft_group_set_transport=dict(
        initial_strength=0.0,
        num_iters=16,
        temperature=1.0,
        confidence_gated=True,
        margin_threshold=0.35,
        margin_temperature=0.1,
        min_gate=0.05,
        apply_to_independent_hard=True),
    pair_aligned_compact_detail_enhancement=dict(hidden_dims=16))

_fusion = _sampler['liquid_aware_fusion']
_fusion.update(
    # Shared routes give group g an exact temporal counterpart, so coverage
    # matching is replaced by direct same-index token coupling.
    pair_transport=None,
    pair_aligned_coupling=dict(
        hidden_dims=128,
        init_std=1e-3,
        zero_init=True,
        relation_mode='pair_diff_product'))

for _hook in custom_hooks:
    if _hook.get('type') == 'LiquidSamplerAnnealHook':
        _hook.update(
            set_transport_start=0.0,
            set_transport_end=0.25,
            set_transport_anneal_epochs=12)

optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'backbone.stem.0.liquid_sampler.pair_consensus_router': dict(
        lr_mult=1.0),
    'backbone.stem.0.liquid_aware_fusion.pair_aligned_coupling': dict(
        lr_mult=1.0),
    'backbone.stem.0.pair_aligned_compact_detail_enhancement': dict(
        lr_mult=1.0),
})

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0719_01_paper_liquid_pairconsensus_relaxedset_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
