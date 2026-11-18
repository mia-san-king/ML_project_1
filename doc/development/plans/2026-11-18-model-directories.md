# Model directory reorganization

**Goal:** Keep each model's code, experiment configuration, launchers, and tests together while retaining shared data and reporting infrastructure.

**Design:** `models/geoclip/` owns the existing GeoCLIP experiments, `models/geoclip_time/` owns the time-conditioned dual tower, and `models/baselines/` owns constant and image-nearest-neighbor baselines. Top-level `lib/` and `scripts/` contain reusable data, feature, geography, and evaluation utilities. `doc/`, `report/`, `notebooks/`, and `data/splits/` remain shared. Development plans are consolidated under `doc/development/`; DHLAB setup tools go under `infrastructure/dhlab/`.

**Constraints:** This is a structural refactor, not a new training method. Preserve experiment names, configuration values, checkpoint state-dictionary keys, data locations, saved notebook outputs, model math, and existing evaluation behavior. Do not modify the user's original checkout or run a full training job. External `geoclip` package imports must retain their original names. Commands are documented from the repository root; directly executed nested Python entry points must bootstrap that root correctly. Existing commands acquire new paths; documentation and shell launchers must use those paths.

## Tasks

- [x] Move `lib/temporal_*.py`, temporal scripts, T0/T1 configs, and temporal tests into `models/geoclip_time/{lib,scripts,configs,tests}`. Update imports, repository-root resolution, CLI integration tests, and `doc/time_geoclip.md`.
- [x] Move GeoCLIP operations and retrieval helpers into `models/geoclip/lib`, existing model launchers into its `scripts`, and A3/A4 configs into its `configs`. Move B0/B2 to `models/baselines`; keep B1/B2/B3 batch orchestration available in GeoCLIP scripts. Extract reusable feature-table loading and result serialization into top-level `lib/feature_tables.py`; move the unchanged nearest-neighbor function into the baselines package.
- [x] Move only DHLAB setup scripts to `infrastructure/dhlab`, put their human-facing guide in `doc/`, and consolidate previous development plans. Update root README, model README files, tutorial references, shell command targets, and development plan paths. Preserve shared data and report paths.
- [x] Verify all temporal tests using `/private/tmp/geoclip-time-env/bin/python -m pytest -q`. Run the existing CPU model and metrics checks with `P2_DATA` set to a fresh temporary directory. Check nested CLI help and data-check commands, both T0/T1 configs, all Python syntax and local imports, shell syntax, Markdown links, and obsolete path references. Review the diff for unintended behavior or saved-result changes.
- [ ] Obtain an independent code review, address structural regressions, commit on `Yang_Zhang`, and publish to the existing remote branch. Verify the remote commit and a clean local checkout.

## Validation contract

The moved temporal tests continue exercising date intervals, query-conditioned scoring, data alignment, training, checkpoint reload, and CLI data checking. The existing GeoCLIP smoke check exercises unchanged loss/retrieval functions, including the moved nearest-neighbor baseline. No accuracy claim follows from these checks.
