# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.3.5.1] — 2026-11-08

### Added
- Initial public release of the companion code and dataset.
- Derived tabular dataset (`data/derived/public_dataset.csv`): 1,558 rows × 483 columns.
- ML pipeline: nested cross-validation (5×3×10), 7 classifiers, bootstrap CIs,
  paired permutation tests, and BH-FDR correction over two families (28 + 112 comparisons).
- CLI with 7 commands: `reproduce`, `analyze`, `fdr`, `validate-data`, `smoke`, `run`, `audit`.
- Paper configuration (`configs/paper.yaml`): seed=2026, outer=5, inner=3, repeats=10.
- 28 automated dataset integrity checks.
- 48 unit and integration tests.
- Column dictionary (`data/metadata/data_dictionary.md`): 483 columns described.
- CI workflow (GitHub Actions): lint + tests on every push.
- Documentation: `docs/SCIENTIFIC_CONTRACT.md`, `docs/REPRODUCIBILITY.md`,
  `docs/OUTPUTS.md`, `docs/METHODOLOGY_SCOPE.md`.
