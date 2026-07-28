# 20260729 Tri-State Decoder Initialization Audit

## Scope

This note audits the initialization used by the historical tri-state decoder
experiments and defines the evidence gate for the corrected successor.

## Root cause

`PairRotatedRTDETRTransformerDecoder.init_weights()` intended to initialize:

- `query_to_prev`, `query_to_curr`, and `query_to_pointer` as identity maps;
- the first block of `pointer_init_fusion` as identity and the rest as zero;
- `pointer_to_prev`, `pointer_to_curr`, and `pointer_update` as zero when
  `tristate_zero_init_coupling=True`.

However, `RotatedRTDETR.init_weights()` subsequently applied Xavier uniform
initialization to every matrix in `self.decoder.parameters()`. This later pass
overwrote both the identity initialization and the optional zero coupling
initialization.

The bug was fixed in commit `71c69b4` by separating the Pair structural
initialization into an idempotent hook and reapplying it after detector-level
initialization. Commit `4e6ea7b` adds an isolated four-iteration smoke and a
post-smoke checkpoint gate.

## Historical interpretation

The measured metrics in `20260709_decoder_experiment_report.md` remain valid
observations of those runs, but their mechanism attribution must be revised.

- `0708_03` and `0708_04` did not actually test zero-initialized recurrent
  coupling. Their coupling matrices started at Xavier scale.
- `0708_01` and `0708_02` also did not retain the intended tri-state identity
  maps.
- Consequently, the difference between `0708_01` and `0708_03`, or between
  `0708_02` and `0708_04`, cannot be cited as evidence for the effect of
  zero initialization.
- `0728_01` was started before the fix and has the same initialization defect.
  It remains useful as a diagnostic run, but it is not the corrected decoder
  candidate.

An old four-iteration `0728_01` smoke checkpoint showed
`query_to_prev` identity maximum error `1.10767281`, which is incompatible
with an identity start. The new verifier correctly rejects this checkpoint.

## Corrected initialization evidence

After commit `71c69b4`, a full 0729 model-level initialization probe showed:

- each query transform was an exact identity matrix;
- `pointer_init_fusion` contained only the intended identity block;
- every tri-state coupling matrix was exactly zero;
- the decoder test suite passed all 19 tests.

A two-step CPU gradient probe further showed:

- `pointer_to_prev` and `pointer_to_curr` receive nonzero gradients on the
  first backward pass;
- trainable `pointer_update` layers receive nonzero gradients on the second
  backward pass after frame coupling begins to move;
- the zero start therefore delays recurrent feedback by one optimizer update
  but does not create a permanently dead path.

## Successor experiment

`0729_01` is the first experiment that combines:

- the corrected tri-state structural initialization;
- the Dual-Evidence encoder;
- easy/hard-positive PairDN.

Before formal training, run:

```bash
projects/multispec_pair_rotated_rtdetr/tools/launch_smoke_0729_01_encoder_dualevidence_decoder0708_03_easyhardpositive_initfix_197.sh
```

The script uses a fresh `initfix71c69b4` work directory and automatically runs
`verify_tristate_decoder_smoke_init.py`. Formal training is allowed only when
the smoke log contains `TRISTATE_SMOKE_INIT_OK`, all losses and gradients are
finite, and there is no traceback or out-of-memory error.

The conservative primary target remains a single checkpoint satisfying both:

- Cls HOTA greater than `54.673`;
- Det HOTA greater than `62.393`.
