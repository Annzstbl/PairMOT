# Pair-Detection Two-Stage Audit

Date: 2026-06-30

This note summarizes the current two-stage implementation in the Pair RT-DETR
baseline and lists the main concerns for discussion.

## Context

The current PairMOT pipeline uses a pair-wise rotated RT-DETR detector as the
main modeling component. The tracker is only a downstream consumer of pair
detections. Therefore, the next innovation stage should focus on improving the
pair-detection model itself.

The current baseline is functional and useful, but its pair-detection design is
still a rough adaptation of single-frame RT-DETR. The most important open issue
is whether the two-stage proposal mechanism is appropriate for pair-wise
detection.

## Relevant Files

- Pair model:
  `projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr.py`
- Pair decoder:
  `projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_layers.py`
- Pair head:
  `projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_head.py`
- Pair config:
  `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn.py`
- Original single-frame RT-DETR:
  `projects/rotated_rtdetr/rotated_rtdetr/rotated_rtdetr.py`

## Current Pair Two-Stage Implementation

The pair config still inherits RT-DETR two-stage behavior:

```python
as_two_stage=True
query_init='dual_topk'
```

However, in pair mode the model does not call the original single-frame
`pre_decoder()` path. Instead, it uses a custom pair query initialization
function: `_topk_pair_queries()`.

Current data flow:

1. The input tensor has shape `(B, 2, C, H, W)`.
2. It is flattened to `(2B, C, H, W)`, with all previous frames first and all
   current frames second.
3. Both frames pass through the same backbone, neck, and RT-DETR hybrid encoder.
4. The encoder memory is split by batch:

```python
memory_prev = memory[:B]
memory_curr = memory[B:]
```

5. `_topk_pair_queries()` generates pair decoder initialization:

```python
output_memory_p, output_proposals_p = gen_encoder_output_proposals(memory_prev)
enc_cls_p = cls_branch(output_memory_p)
topk_idx_p = topk(enc_cls_p.max(-1), k=num_queries)

query = gather(output_memory_p, topk_idx_p)

ref_prev = reg_prev(query) + gather(output_proposals_p, topk_idx_p)

output_memory_c, output_proposals_c = gen_encoder_output_proposals(memory_curr)
ref_curr = reg_curr(query) + gather(output_proposals_c, topk_idx_p)
```

Therefore, the current pair proposal is **prev-guided top-k**:

- top-k selection is based only on the previous frame encoder classification
  score;
- the query content comes only from previous-frame encoder memory;
- previous reference boxes are initialized from previous-frame proposals;
- current reference boxes use the same spatial proposal indices as previous
  top-k, but applied to current-frame proposal grids.

## Difference From Original Single-Frame RT-DETR

Original single-frame RT-DETR uses encoder proposals as follows:

1. Generate dense encoder proposals from feature grids.
2. Predict encoder classification scores.
3. Select top-k proposals.
4. Use top-k proposal boxes as decoder reference points.
5. During training, pass `enc_outputs_class` and `enc_outputs_coord` to the head
   for encoder proposal loss.

In the current pair head:

```python
def loss_by_feat(..., enc_cls_scores=None, enc_bbox_preds=None, ...):
    del enc_cls_scores, enc_bbox_preds
```

So the pair model currently **does not use encoder proposal loss**. The
two-stage mechanism only initializes decoder queries and references. It is not
explicitly trained as a pair-aware proposal generator.

## Main Problems

### 1. The Proposal Is Not a Pair Proposal

The current proposal is not a true pair unit:

```text
(prev_box, curr_box, association_score)
```

Instead, it is closer to:

```text
prev single-frame top-k proposal + current-frame proposal at the same feature index
```

This is a weak assumption for moving objects.

### 2. Top-K Selection Depends Only on the Previous Frame

If an object is weak, occluded, tiny, or low-confidence in the previous frame,
it may never enter the pair decoder queries, even if it is clear in the current
frame.

This may hurt:

- newborn handling;
- recovery after temporary degradation;
- objects that become clearer in the current frame;
- gap > 1 settings.

### 3. Current Reference Initialization Assumes Same Spatial Index

The current-frame reference uses the same top-k feature index selected from the
previous frame. This means the model initially assumes the current object is
near the previous object's feature-grid location.

This may be acceptable for same-frame or very small motion, but is less
reasonable for:

- sequential frames with noticeable motion;
- gap=3 or gap=5 training;
- low-altitude scenes with fast apparent motion;
- small objects whose feature-grid localization is unstable.

### 4. Query Content Is Previous-Frame Only

The decoder query content is gathered from `output_memory_p` only. Current-frame
memory does not contribute to the initial query content.

Although the pair decoder later attends to both frames, the decoder starts from
a previous-frame-biased hypothesis.

### 5. No Encoder Proposal Supervision

The pair head discards `enc_cls_scores` and `enc_bbox_preds`, so there is no
direct loss forcing encoder proposals to become good pair proposals.

The encoder proposal quality is therefore only indirectly influenced through
decoder losses.

### 6. Presence/Birth/Death Are Not Part of Two-Stage Proposal

The pair head predicts `presence_prev` and `presence_curr` after the decoder.
However, the two-stage proposal selection does not consider:

- whether the object exists in the previous frame;
- whether the object exists in the current frame;
- whether this is a matched pair;
- whether this is a birth/death case.

This is a mismatch between pair-detection semantics and the proposal mechanism.

## Current Decoder Design Related to Two-Stage

The pair decoder uses:

- one shared pair query;
- two oriented references: `reference_prev` and `reference_curr`;
- one self-attention block over pair queries;
- separate deformable cross-attention to previous-frame memory and current-frame
  memory;
- linear fusion of previous/current cross-attention outputs;
- separate reference refinement branches for previous and current boxes.

This part is already more pair-aware than the proposal stage. The largest gap is
that the decoder is initialized by a weak prev-guided proposal rather than a
true pair proposal.

## Candidate Redesign Directions

### Option A: Pair-Aware Encoder Proposal Loss

Minimal change:

1. Keep current `dual_topk` query initialization.
2. Return encoder proposal outputs for both frames:
   - `enc_cls_prev`
   - `enc_bbox_prev`
   - `enc_bbox_curr`
3. Add pair encoder loss in `PairRotatedRTDETRHead`.
4. Supervise selected top-k encoder outputs with pair GT.

Potential benefit:

- keeps the architecture close to current baseline;
- directly trains proposal quality;
- easier to implement and ablate.

Risk:

- if top-k selection is still previous-only, the structural weakness remains.

### Option B: Dual-Frame Top-K Selection

Instead of selecting top-k only from previous-frame scores, compute a joint score:

```text
score_pair = f(score_prev, score_curr, feature_similarity, motion_prior)
```

Examples:

```text
score_pair = sqrt(score_prev * score_curr)
score_pair = max(score_prev, score_curr)
score_pair = score_prev + score_curr
score_pair = score_prev + score_curr + local_cross_similarity
```

Potential benefit:

- current-frame evidence can promote useful queries;
- less biased toward previous-frame visibility.

Risk:

- simple score fusion still does not solve association explicitly.

### Option C: Prev-Guided Current Search

Use previous-frame top-k proposals as anchors, then search the current-frame
feature map locally or globally for corresponding current proposals.

Possible design:

1. Select previous-frame top-k proposals.
2. For each previous proposal, perform cross-attention or correlation against
   current-frame features.
3. Predict current reference from the attended current feature, not from the same
   spatial index.

Potential benefit:

- naturally models motion;
- preserves RT-DETR top-k efficiency;
- matches the pair-detection task better.

Risk:

- implementation complexity is moderate;
- requires careful control of memory and speed.

### Option D: True Pair Proposal Generator

Generate dense pair proposals directly:

```text
(prev_ref, curr_ref, class_score, presence_prev, presence_curr, pair_score)
```

This can be implemented by a small pair proposal head over previous/current
encoder memory.

Potential benefit:

- most semantically correct;
- proposal stage becomes aligned with pair detection and MOT.

Risk:

- largest architectural change;
- needs new matching and loss design;
- harder to load from single-frame pretraining.

### Option E: Motion-Delta Proposal

Use previous proposal plus predicted motion:

```text
curr_ref = prev_ref + delta
```

The proposal stage predicts:

```text
prev_box, delta_box, pair_score
```

Potential benefit:

- compact and motion-aware;
- may be especially useful for gap=1/3/5 curriculum.

Risk:

- rotated box delta parameterization must be robust;
- large or nonlinear motion may be difficult.

## Recommended First Experiment

The first innovation should be conservative and diagnostic:

### Pair-Aware Two-Stage v1

Implement a new query init mode, for example:

```python
query_init='pair_topk_v1'
```

Design:

1. Compute encoder proposal outputs for both frames.
2. Use a joint top-k score instead of previous-only score.
3. Initialize query content from fused previous/current encoder memory:

```python
query = fuse(memory_prev_topk, memory_curr_topk)
```

4. Initialize:

```python
reference_prev = prev_reg(query_prev_or_fused) + prev_proposal
reference_curr = curr_reg(query_curr_or_fused) + curr_proposal
```

5. Add optional encoder pair loss after verifying overfit stability.

Suggested ablations:

| Variant | Top-k score | Query content | Curr reference |
|---|---|---|---|
| baseline dual_topk | prev only | prev memory | same index in curr |
| v1-score | prev+curr score | prev memory | same index in curr |
| v1-fusion | prev+curr score | fused prev/curr memory | same index in curr |
| v2-search | prev score | fused prev + searched curr | searched curr |

## Discussion Questions

1. Is previous-frame top-k a reasonable default for MOT pair detection, or
   should current-frame evidence participate from the beginning?
2. Should the two-stage proposal represent a detection hypothesis, a pair
   hypothesis, or a tracklet hypothesis?
3. Should encoder proposal supervision be restored for pair detection?
4. Should presence prediction be introduced at the proposal stage?
5. Is same-index current proposal acceptable for gap=1 only, or should it be
   replaced before experimenting with larger gaps?
6. What is the best first change that preserves pretrained RT-DETR weights while
   making the proposal more pair-aware?

## Preliminary Conclusion

The current pair two-stage design is a useful baseline but not a principled
pair-detection proposal mechanism. It is best described as:

```text
previous-frame single-object top-k proposal initialization for a pair decoder
```

The most promising next innovation is to redesign two-stage query initialization
so that proposal selection and reference initialization are pair-aware.
