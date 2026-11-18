# Time-conditioned GeoCLIP implementation plan

**Goal:** Add a runnable, jointly trained image/year and GPS/year dual encoder on the latest Project 2 branch, with no full training in this session.

**Architecture:** Reuse unnormalized cached 768-dimensional CLIP features and GeoCLIP pretrained image MLP/location encoder. Learn small residual time adapters on both towers. For each query, all candidate GPS embeddings use that query's date interval, never the candidate photograph's date. Preserve existing experiment scripts.

**Tech stack:** Python, PyTorch, NumPy, pandas, PyArrow, YAML, pytest; existing GeoCLIP package for pretrained backbones.

**Approved design:** User approved the dual-encoder design in this conversation and requested implementation and a personal GitHub branch. Original checkout stays unchanged; use the independent clone at `/Users/zhangyang/Documents/Codex/ML_project_1_time`, branch `codex/yang-geoclip-time`, based on `36df4f5`.

## Global constraints

- No images, feature arrays, checkpoints, credentials, or full training outputs in Git.
- Preserve missing dates and encode intervals as normalized midpoint, half-width, known mask; do not silently filter them out.
- Training uses exactly `train`; model selection uses exactly `val`; never choose checkpoints on test sets.
- Model inference uses the same query-conditioned scoring as training. Cache only time-independent GPS base features.
- Freeze CLIP by consuming cached features. Preserve fixed RFF matrices when fine-tuning location layers.
- New code must be usable with `--help` and data validation without loading full CLIP weights or creating default server directories.
- Existing scripts/configurations remain unchanged; new train/predict entrypoints and configs are separate.

## Task 1: validated temporal data loading

Files: `lib/temporal_data.py`, `tests/test_temporal_data.py`.

Interfaces:
```python
encode_date_intervals(frame, *, year_center=1950.0, year_scale=100.0,
                      min_year=1800, max_year=2026) -> (np.ndarray, dict)
load_temporal_data(data_root, split_csv, *, feature_name="clip_vitl14",
                   time_config=None) -> TemporalData
# TemporalData: features (mmap), table with feat_idx and id/split/GPS,
# times (aligned to table), audit dict. Table is in feature-cache order.
```

- [x] Write failing behavioral tests: shuffled ID joins retain correct GPS/date/feature associations; duplicate IDs and inconsistent lengths fail; interval/missing/out-of-range/one-sided dates have explicit handling.
- [x] Implement validated one-to-one ID joins, finite valid GPS checks, coverage counts by split, and interval quality counts. Keep partial cache coverage visible in audit; callers explicitly choose whether to permit it.
- [x] Run `/private/tmp/geoclip-time-env/bin/python -m pytest tests/test_temporal_data.py -q`.

## Task 2: temporal model, objective and retrieval

Files: `lib/temporal_model.py`, `lib/temporal_ops.py`, `tests/test_temporal_model.py`.

Interfaces:
```python
TemporalGeoCLIP(image_mlp, location_encoder, *, embed_dim=512,
                time_dim=3, hidden_dim=128, mode="both", logit_scale=None)
# methods: encode_image(feats, times), encode_location_base(gps),
# condition_locations(base, times) -> [B,C,D], score_candidates(feats,times,gps)
temporal_loss(model, feats, gps, times, negative_gps, *, soft_tau_km=None)
predict_temporal(model, feats, times, gallery, *, top_k=1,
                 query_chunk=16, gallery_chunk=256)
# returns (predicted_gps, top_indices, top_logits, logsumexp), numpy arrays.
```

- [x] Write failing tests that detect using candidate dates instead of query dates, prove time-only representations cannot distinguish candidates, verify both trainable towers receive gradients, chunked retrieval equals a hand-checked score calculation, duplicate true coordinates are not false negatives.
- [x] Implement residual time fusion; `mode=none` bypasses all time fusion for a controlled baseline. Compute GPS bases once per candidate set. Use single-direction contrastive loss, handling same-location positives; support existing distance soft targets optionally.
- [x] Run `/private/tmp/geoclip-time-env/bin/python -m pytest tests/test_temporal_model.py -q`.

## Task 3: executable training, restore and prediction

Files: `lib/temporal_factory.py`, `lib/temporal_training.py`, `scripts/train_time.py`, `scripts/predict_time.py`, `configs/T0_no_time.yaml`, `configs/T1_dual_time.yaml`, `tests/test_temporal_training.py`.

- [x] Test end-to-end with small real torch modules and synthetic local cache/parquet/split/grid fixtures: one short training pass, checkpoint reload, identical predictions, split isolation, metadata-order alignment, final incomplete batch, output collision protection.
- [x] Construct only GeoCLIP image projection and LocationEncoder, load packaged pretrained weights without downloading CLIP. Save enough model/time configuration to reconstruct checkpoint without original pretrained weights.
- [x] Implement configurable paths, CPU/GPU batches from mmap, training/validation filtering, deterministic validation subset, best-validation checkpoint, history and data audit. Separate prediction CLI produces `id, split, lat_pred, lon_pred` parquet and split metrics. Expose `--check-data` without model creation.
- [x] Default to a modest batch/negative count and chunked retrieval. Require explicit consent via configuration for incomplete train/val feature coverage. Never overwrite existing named run artifacts implicitly.
- [x] Run `/private/tmp/geoclip-time-env/bin/python -m pytest tests -q` and both CLI `--help` checks.

## Task 4: documentation, review and publish

Files: `doc/time_geoclip.md`, README link.

- [x] Document model equation, shared-query-time negatives, date policy, data prerequisites, baseline and temporal training commands, prediction, artifact paths, costs and absence of measured accuracy improvements.
- [x] Run existing `scripts/check_geoclip_ops.py` with a temporary P2_DATA; run new tests, synthetic CLI flow, and actual lightweight GeoCLIP factory/restore smoke if dependencies are available.
- [ ] Review diff for data leakage, shape/order issues, memory behavior, accidental source checkout changes and secrets.
- [ ] Commit only scoped source/config/test/docs changes and push new branch via user's GitHub SSH configuration; verify remote commit. No merge and no full training.
