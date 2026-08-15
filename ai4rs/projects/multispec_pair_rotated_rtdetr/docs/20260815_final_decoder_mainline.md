# PairMOT final Decoder code mainline

This branch uses the final R18 product-tangent Decoder implementation as its
code base.  The paper configurations retained on the mainline are:

| Paper row | Experiment | Configuration |
| --- | --- | --- |
| Base | `0719_05` | `configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_rerun_coco_full_1200x900_bf16_178.py` |
| Base + Liquid | `0723_01` | `configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99.py` |
| Base + Liquid + Encoder | `0727_01` | `configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py` |
| Base + Liquid + Encoder + Decoder | `0812_05` | `configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0812_05_iterative_cls_terminal_transport_product_tangent_wsd4_44_cos24_decoder_252.py` |
| Base + Liquid + Encoder + pure ellipse | Encoder c04 | `configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pure_ellipse_c04_178.py` |

The pure-ellipse configuration is evaluation-only and uses the locked c04
parameters: sim/geometry/score/spectral weights `0.1/0.6/0.3/0.0`, affinity
rank weight `0.3`, aspect limit `1.6`, and isotropic area threshold `0.4e-3`.

R34 and R50 scale experiments are intentionally outside this branch's scope.
