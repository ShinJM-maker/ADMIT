# ADMIT: Support-Gated Memory-Write Admission for Document QA Agents

Reference admission code and recomputation artifacts for the NeurIPS 2026 paper.

ADMIT separates answer delivery from memory admission. It checks each proposed
answer value against the exact OCR/text supplied to the reader, permits one
RepairRead per candidate, and withholds values that still fail the check.
The original answer can remain visible as `UNVERIFIED`; automatic memory reuse
receives only admitted values. This contract guarantees lexical substring
presence, while answer performance, memory availability, and reader cost are
measured separately.

## Contents

- `code/admit_admission_engine.py`: model-independent admission interface,
  strict lexical check, typed statuses, and repair-failure policies.
- `code/answer_metrics.py`: refusal-aware answer scoring.
- `code/recompute_*.py`: recomputation of the saved experimental aggregates.
- `code/validate_*.py`: checks for the admission contract, primary metrics,
  and all eighteen upstream-pipeline comparisons.
- `configs/`: execution settings, exact prompt templates, and hashes.
- `results/`: compact per-question records and aggregate summaries.
- `examples/quickstart.py`: a small, model-free admission example.

The release supports reuse of the admission module and verification of saved
results. It does not include the complete retriever/model-serving pipeline or
redistribute reader weights and source documents. Fresh end-to-end generation
is outside this release's scope. Dataset and model licenses remain those of
their original providers; see `LICENSE`.

## Setup and quick start

Python 3.10 or newer. The admission engine uses only the Python standard library;
NumPy is used by some recomputation scripts.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/quickstart.py
```

A caller supplies the exact post-truncation reader-visible text and an optional
repair callback. See the example for successful repair and failed repair.

## Verify the released results

Run from the repository root:

```bash
python code/validate_bundle.py
python code/validate_strict_portability.py
python code/verify_review_diagnostics.py
```

Selected analyses can be recomputed without a GPU or model call:

```bash
python code/recompute_window_control.py
python code/recompute_shared_memory.py
python code/recompute_injected_value_replay.py
```

These scripts check stored summaries by default; scripts exposing `--write`
can also regenerate them. Bootstrap-based analyses take longer than the basic
validators.

## Interpretation and provenance

- Zero contamination follows from the lexical admission and consumer contract.
- Support-span controls compare excerpts around an admitted value with
  equal-length excerpts from the start of the checked text. Their accuracy
  gains concern retained source context, separately from admission.
- The earlier exact-reader audit contains 2,508 proposed writes, including
  900 terminal writes. Candidate and repair fields belong to one write event.
  The primary typed-memory comparison is a separate execution with 2,176 ADMIT
  writes; the counts are not interchangeable.
- The `LLM chunking` baseline retains the internal artifact key `metachunker`.
  It uses prompted topic boundaries, rather than the uncertainty-based
  Meta-Chunking algorithm or MoC. Metric records retain their original keys.
- Benchmark questions reused during debugging are documented in the execution
  settings. These comparisons should not be treated as fresh holdout tests.

The paper and its supplementary material provide the full experimental
interpretation, limits, and dataset citations.

## License

Apache License 2.0 for the released code, configurations, and original artifacts.
Third-party dataset/model terms are described in `LICENSE`.
