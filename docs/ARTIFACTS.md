# Artifact details

This document describes the released evaluation artifacts.
For the definitive contract, every admitted
non-refusal value must be a normalized substring of the exact reader-visible,
post-truncation OCR/text packet. Memory records are typed by `memory_status`;
downstream state consumers ignore `REFUSAL` and `NO_COMMIT`. No semantic or
binary-answer admission bypass is active. The repair generation prompt retains
a yes/no permission; every generated value still faces the strict gate.

ADMIT makes answer delivery and memory admission separate decisions. The
released results show the answer and memory costs of this separation:
ADMIT is 0.67 pp below fail-open and 5.56 pp above hard refusal in substring EM
(`results/strict_comparator_per_qid.jsonl`). It admits no lexically unsupported
value under the typed consumer contract. In a source-blocked injection replay,
final answers overlap with injected values that differ from the main-question
reference in up to 43.40% of selected fail-open cases and none of ADMIT's
(`results/injected_value_replay_*`); this label does not establish the
semantic correctness of each sub-question answer. Multi-hop final-answer
overlap with an unsupported stored value is structurally zero under ADMIT and
reaches 25.80% under fail-open (`results/multihop_final_check_*`); the overlap
measure does not establish causal reliance. Adding located support spans
improves final answer scores in separate experiments
(`results/support_span_memory_*`, `results/anytime_*`).

## Contents

Model-running code (needs vLLM and the M3DocVQA OCR/text index; see README):

- `harness/agents.py`: ReAct, PlanRAG and AutoGen loops over BM25 top-10 section cards,
  typed memory records (`public_memory_record`), the ADMIT write hook (`cavec_mfc_hook`),
  the fail-open and hard-refusal hooks (`cavec_hook`), final-output repair (`_post_hoc_arms`),
  and the vLLM reader (`ReaderVLLM`). `harness/run_reader_qwen25vl.py` and
  `harness/retrieve_hikey_stage2.py` provide evidence packing and BM25.
- `experiments/dynamic/run_dynamic.py`, `summarize_dynamic.py`: Table 1 policies and component
  arms; the summarizer reproduces the released Table 1 EM from the saved trajectories.
- `experiments/static/run_reader_vllm.py`, `evaluate_static.py`: static reader passes and the
  static evaluation, including the leakage-free SupportRouter refit (LightGBM lambdarank fitted,
  early-stopped and thresholded on the calibration partition only).
- `experiments/portability/`: frozen-packet terminal replay over upstream pipelines and typed PlanRAG
  on DUDE, TAT-QA and MMLongBench-Doc with the official scorers (`native_scoring.py`).
- `experiments/learned_alternatives/`: learned refusal selector and second-chance verifier (App. D.3).
- `inputs/static/`: frozen retrieval for both static partitions (document and card IDs with
  scores, no document text) and the router's candidate answers (`qid`, `kind`, `answer`).
- `configs/dynamic_qids.json`: the 300 dynamic M3DocVQA questions.

Recompute and validation code (no model inference):

- `code/admit_admission_engine.py`: self-contained reference implementation of
  structured-value extraction, the strict lexical gate, one bounded
  `RepairRead`, non-reusable refusal records, and the MFC/OFO, hard-refuse, and
  diagnostic fail-open actions. A repair refusal, like an unsupported repaired
  non-refusal, is a failed recovery rather than a new reuse-eligible write.
  Missing or non-string `answer_value` fields also reach the repair path.
  Callers pass the original reader output as `fallback_output` when parsing
  fails; if omitted, the fallback is empty. An explicitly empty string value
  remains a refusal under the paper's predicate. The definitive policy requires
  a `repair_read` callback with `repair_selected=True`; disabling repair is
  supported for the gate-only diagnostic.
- `code/validate_bundle.py`: validates the schema, contract, metric rows, and
  the strict admission module.
- `configs/prompts.md`: exact experiment-specific prompt templates and assembly rules.
- `configs/frozen_prompt_templates.json` and `frozen_prompt_manifest.json`: literal
  templates and SHA-256 hashes; static best-guess rereads are distinguished from
  dynamic copy-focused repairs.
- `code/answer_metrics.py`: refusal-aware substring EM and the No/refusal regression check.
- `configs/execution_contract.json`: reader, packet, decoding, metric, and
  artifact-scope definitions used by the revised paper.
- `results/results_registry_public.json`: public source of truth for the
  definitive static answer--cost rows and strict dynamic comparator/component
  rows. Scores were corrected on 2026-10-20 so fixed refusals receive zero.
  Its dynamic entries preserve identifiers and hashes for the upstream
  full typed trajectories, which are not redistributed; a separate provenance
  block hashes the compact per-qid files included here and used for public
  recomputation. Historical and supplementary diagnostic audits are scoped
  in the manuscript rather than represented as definitive registry entries.
  Internal execution identifiers are omitted.
- `results/static_deployable_router_results.json`: aggregate clean static audit,
  including the 2,324-qid candidate-complete scope, disjoint
  calibration/reporting contract, five deployable features, hyperparameters,
  paired intervals, and the descriptive non-dominating learned-router result. The
  released validation file has 2,441 qids; 117 incomplete-candidate qids are
  outside this static artifact.
- `results/static_deployable_router_per_qid.jsonl`: reporting-partition
  predictions, route decisions, and per-policy metrics used to verify the
  static table. It contains no evidence text or local execution path.
- `results/strict_dynamic_summary_public.json`: strict dynamic comparator and
  component summaries.
- `results/write_scope_per_qid.jsonl`: compact audit for all 6,300 comparator
  and component trajectories. Prior contamination and typed terminal-record
  indicators are separate. Final-support uses the canonical Vanilla-memory
  indicators below; the other arms retain the terminal-write union correction.
- `results/final_support_memory_contract_20261020.jsonl`: all 900 final-only
  trajectories under one storage rule, preserving every Vanilla memory write.
- `results/final_support_contract_revision_20261020.json`: normalization rule,
  source hashes, canonical-artifact hash, and counts for both input formats.
- `results/word_boundary_sensitivity_20261020.json`: secondary answer-matching
  sensitivity on the same 300 qids and three agents, with paired intervals.
- `results/strict_component_per_qid.jsonl`: compact per-qid metrics for the
  three strict component arms; it excludes questions, answers, evidence text,
  and internal execution metadata. Rows include raw EM, contamination,
  `NO_COMMIT`, calls/q, and final-refusal indicators.
- `results/strict_comparator_per_qid.jsonl`: compact per-qid metrics for the
  four policy-comparison arms under the same typed-memory runtime contract.
- `MANIFEST.sha256`: checksums for every other file in this bundle.

The public registry distinguishes the static answer--cost evaluation from the
dynamic state-contamination evaluation. Static and dynamic EM values use the
metric conventions recorded in `configs/execution_contract.json` and should
not be compared across surfaces.

## Quick validation

From the bundle root, run:

```bash
python code/validate_bundle.py
```

The validator uses only the Python standard library. It checks that:

1. the public contract selects strict normalized-substring admission;
2. no bypass is enabled;
3. refusals and `NO_COMMIT` are status records excluded from automatic reuse;
4. every compact component and comparator row has the required metric fields;
5. all strict component rows have zero lexical contamination;
6. the static reporting artifact has 1,162 unique qids and no evidence text;
7. every static aggregate is recomputed from the included per-qid rows, the
   static registry matches those aggregates, and the reported ADMIT/final-only
   answer-surface parity holds per qid;
8. every public verification file matches the hash recorded in the registry
   and dynamic summary;
9. the contract records the 2,441/2,324/117 coverage accounting, the
   dynamic-aligned static EM convention, and no untouched-test claim; and
10. the dynamic contract records typed FACT/STATUS memory and FACT-only reuse-eligible
   consumption;
11. every dynamic aggregate and every reported paired point estimate is
    recomputed from the included compact per-qid rows;
12. every dynamic registry row matches the frozen typed-runtime summary and
    retains its upstream full-trajectory provenance identifier and hash;
13. private execution identifiers and local filesystem paths are absent; and
14. the manifest matches the included files.

## Scope

The reference module implements the public memory-write boundary, not the
upstream document retriever or model server. Public dataset releases and the
reader named in the execution contract are required to regenerate model
outputs. The supplied aggregate registry and compact per-qid metrics permit
recomputation of the reported strict-contract tables and paired point
estimates without redistributing document evidence, full typed trajectories,
or private execution metadata. Regenerating reader outputs remains outside this compact bundle and requires
the public datasets and model named in the execution contract.

## Paper notation and historical archive (2026-09-27)

The paper calls reuse-eligible records `ADMITTED` and their projection `ReuseEligible(M)`. These are notation aliases for the frozen runtime `FACT` tag and its existing projection. No result, runtime decision, or serialized trace was rewritten for this naming change.

Policy labels in the per-qid files keep their frozen runtime names. Comparator
labels map to the paper as follows: `cavec_mfc` is ADMIT (MFC/OFO),
`cavec_paper` is fail-open, `cavec_h1_hard_refuse` is hard refusal, and
`final_support_retry` is Final-support repair. The `h1` in the hard-refusal
label is part of the historical artifact name; H1 is disabled in every final
comparator run. Component labels `gate_only`, `gate_repair`, and `full_admit`
are Gate-only, Gate + RepairRead, and the archived Full ADMIT reconstruction.
In the portability files, pipeline label `vega_kg` is the PILAR row of the
paper, `colpali_ocr` is ColPali, and `raptor` and `late_chunking` are the
local RAPTOR and Late Chunking reproductions.


## Terminal-write scope and matching sensitivity (2026-10-20; interim final-support row)

The typed trajectory aggregator omitted `final_memory_record` from
contamination, although it included that record for `NO_COMMIT`. The corrected
metric takes the union with unsupported terminal FACT records. Existing
historical reuse labels already included their terminal memory state and are
retained. The correction adds 21 contaminated fail-open trajectories
(102 to 123 of 900; 11.33% to 13.67%) and 20 final-support trajectories
(148 to 168 of 900; 16.44% to 18.67%). ADMIT, hard refusal, and component
contamination remain zero. No generated output or memory action was changed.
The final-support figure in this interim correction is superseded below.

The compact write audit records the prior indicator and terminal-record
indicator separately. Original full-trajectory files retain their original
hashes; the revised public metrics supersede the historical embedded scores.
`python code/validate_bundle.py` checks these aggregates without model inference.

The primary refusal-aware substring EM remains unchanged. A
Unicode word-boundary sensitivity check excludes within-word matches while
retaining the same normalization and bidirectional containment. It may also
exclude valid inflections, so it is not a semantic correctness estimate.
The ADMIT-minus-fail-open difference is -0.44 pp (95% qid-cluster interval
[-1.00, 0.00]); ADMIT minus hard refusal is +5.33 pp [3.33, 7.67]. These are
secondary results on the existing sample, not new independent evaluations.

## Unified final-support memory contract (2026-10-20)

Final-support retry now has one explicit control rule across all 900 cases:
checking and RepairRead affect only the user-facing output. All Vanilla
memory writes, including the pre-repair terminal value, remain unchanged.
The repaired answer does not replace that memory value. The manuscript calls
this dynamic control Final-support repair. The static final-support control
is a separate terminal replay, applying repair before its single storage decision.

The 422 typed records are normalized from the saved original final candidate
and its lexical-check branch. The 478 reused records use the archived
Vanilla writes and exact-reader support annotations. Binary values without
reader-text support are counted as contamination, without the historical
yes/no exception. Eleven typed terminal metadata records change, and three
legacy binary terminals enter the strict count. Nine trajectory labels change
relative to the interim union correction: final-support contamination is
now **177/900 (19.67%)**: ReAct 37/300, PlanRAG 101/300, AutoGen 39/300.

Raw trajectory files and hashes remain preserved. A separate canonical
900-row artifact records the normalized memory state; the registry points to
its hash and separately retains the original generation-source hash. Original
outputs, primary answer scores, reader calls, and intermediate memory values
are unchanged. The ADMIT, fail-open, and hard-refusal policies and results
are unchanged. The canonical final-memory indicators are used directly in
recomputation, replacing the mixed historical aggregate for this comparator.


## Additional saved-output diagnostics (2026-10-22)

`python code/verify_saved_diagnostics.py` verifies additional aggregates and
10,000-draw qid-cluster intervals from compact records without model calls.
The original definitive result files and generations are unchanged.

- `results/final_write_diagnostics_per_qid_20261022.jsonl` and its aggregate
  JSON separate intermediate/terminal contamination, count common write
  outcomes, and report strict normalized EM, token F1, and Yes/No accuracy.
- `results/gate_span_per_write_20261022.jsonl` and its summary audit the
  admission gate separately from answer scoring. Coverage is 1,318 of 1,321
  admitted writes: 323 writes whose audited text matches the recorded packet
  hash (78 saved texts plus 245 card-ID reconstructions; the latter keep
  `packet_source` "reconstructed from recorded card IDs") and 995 other
  card-ID reconstructions; three unreproduced recorded hashes are excluded.
  The character-substring gate and every saved memory action are unchanged.
- `results/interior_match_examples_20261022.json` contains short attributed
  excerpts showing actual interior-only matches. Normalized offsets refer to
  the full normalized packet, not the excerpt.
- `results/representative_write_consumer_20261022.json` connects one local
  question, candidate, repair, matching span, action, and full PlanRAG consumer
  question. It includes a short Wikipedia-derived excerpt, not the full packet.
- `results/historical_pool_lineage_20261022.json` records archived five-fold
  memberships and candidate-pool counts. FRAMES is a 760-question
  candidate-complete subset of the official 824-question test split, not an
  official training split. Historical fitting is distinct from the corrected
  calibration-only router diagnostic.
- `results/answer_disagreements_20261022.json` exposes all eight
  ADMIT/fail-open primary-match disagreements, with predictions and references.
- `results/reconstruction_differences_20261022.json` identifies the 21
  comparator/component pairs with differing EM, NO_COMMIT, or reader calls.
- `results/component_write_availability_per_qid_20261022.jsonl` and its
  aggregate `component_write_availability_20261022.json` count proposed,
  admitted, NO_COMMIT, and refusal writes in the frozen Gate-only and
  Gate+RepairRead component arms. Content admission is 1,230/2,177 (56.50%)
  versus 1,315/2,180 (60.32%). These are write-level fractions, distinct from
  trajectory-level Any NO_COMMIT and from the final comparator ADMIT arm.
- `results/candidate_row_unit_audit_20261022.json` defines a historical pool
  row as one cached reader output for an evidence configuration and modality.
  It records the text/image partition and per-question row-count histogram.
  Full selectors use all cached rows (20.00/19.27/19.78 outputs per question
  for VisDoMSlide/M3DocVQA/FRAMES). The archived cascade weights 1/4/10 are
  schedule-budget units, not measured modality-specific reader-call counts.

The aggregate verifier checks arithmetic; it does not independently certify
evidence content.

## Controlled distractor audit (2026-10-25)

`python code/recompute_distractor_audit.py` recomputes the appendix counts
and Wilson 95% intervals for lexical admission of planted-distractor copies
from `results/controlled_distractor_audit_20261025.jsonl` (290 per-question
records: counterfactual stratum, distractor-copy flag, saved answer, and
lexical admission on the exact perturbed text) and checks them against
`results/controlled_distractor_audit_summary_20261025.json`; `--write`
regenerates the summary. No model call is made.

## Injected-value replay (2026-10-30)

`python code/recompute_injected_value_replay.py` recomputes the appendix rates
for the injected-value replay from `results/injected_value_replay_20261030.jsonl`
(230 M3DocVQA per-trajectory records: reader study at injection rates 0.1/0.2/0.4 with
sources blocked or available, and the second consumer agent at 0.2; per memory
policy, substring EM, refusal, whether the answer overlaps with an injected value differing from the main-question reference, and whether such a value enters the consumer's search queries) and from
`results/injected_value_replay_portability_20261030.jsonl` (900 records for
DUDE, TAT-QA, and MMLongBench-Doc, sources blocked or available, also pooled) and checks
them against `results/injected_value_replay_summary_20261030.json`; `--write`
regenerates the summary. No model call is made.

Fail-open final answers overlap with such a value in up to 43.40% of selected cases on
M3DocVQA (sources blocked) and in 17.46-22.83% pooled over the three portability
datasets; random withholding of as many writes repeats one in 8.40-9.02%, and
ADMIT in none.

## Anytime document agents (2026-10-31, 2026-11-01)

Documents arrive one per step and cannot be reopened, so memory is the only
carrier of earlier evidence (App. B.12). The released records support the
appendix's outcome measures for each agent: final (step 4) substring EM, each
dataset's official metric (DUDE ANLS, TAT-QA F1, MMLongBench-Doc typed score),
and answer accuracy, the substring EM among non-refused final answers. Storing
each admitted value with its support span (ADMIT + spans) raises all three over
fail-open:

| Agent | Final EM (pp) | Official metric (pp) | Answer accuracy (%) |
|---|---|---|---|
| Fixed reader | +7.22 [5.11, 9.33] | +6.51 [4.52, 8.54] | 19.65 to 33.10 (+13.45 [9.57, 17.45]) |
| ReAct and AutoGen | +5.56 [3.83, 7.33] | +3.17 [1.67, 4.70] | 23.58 to 33.54 (+9.96 [6.59, 13.30]) |
| PlanRAG | +1.89 [0.44, 3.33] | +2.24 [0.68, 3.82] | 27.84 to 31.87 (+4.03 [1.91, 6.24]) |

The gain comes from the spans: fail-open with spans gains as much, and a
prefix-window control with the same nominal context budget gains less (below).
Withholding keeps unsupported values out of that memory: ADMIT stores none and
no final answer repeats one, whereas fail-open stores one in 14.44-35.00% of
runs and repeats it in 3.89-21.67% of final answers. The summary also gives the
answered rate per policy.

`python code/recompute_anytime_agent.py` recomputes these figures, the three
appendix tables, and the decomposition into spans and withholding from:

- `results/anytime_document_agent_20261031.jsonl` (4,500 records: 900 DUDE,
  TAT-QA, and MMLongBench-Doc questions under five memory policies with the
  fixed reader);
- `results/anytime_react_autogen_20261031.jsonl` (9,000 records: the same
  questions and policies with ReAct and AutoGen agents, plus each step's action);
- `results/anytime_planrag_20261101.jsonl` (4,500 records: the same questions and
  policies with a PlanRAG agent, plus the number of sub-question reads per step).

Each record stores final substring EM, refusal, the official metric, per-step EM, whether an
unsupported value was stored, and whether the final answer repeats one.
Intervals are paired 4,000-draw question bootstraps (seed 0); for answer
accuracy, each draw takes the difference of the two ratios over the resampled
questions, and the ReAct and AutoGen figures keep both agents for each
resampled question. The script checks its output against
`results/anytime_document_agent_summary_20261031.json`; `--write` regenerates the
summary. No model call is made.

### Window control (2026-11-01)

The window arm uses the same admission policy as ADMIT + spans. Each admitted
value gets a prefix of the checked text, capped at 440 characters plus the
value length. The span arm retains up to 220 characters on either side of the
matched value. Spans can be shorter at packet boundaries, and ellipsis markers
also differ: this is the same nominal source-text budget, not exact character-
or token-length matching. The comparison therefore evaluates excerpt strategies
under that budget rather than isolating position at identical length. ADMIT, ADMIT + spans, and this window arm were run together
in all four settings (900 questions each). The primary contrast, its interval,
and the decision rule were recorded with file hashes before the run. Support
spans exceed the window in final EM by +3.72 pp [2.67, 4.81], averaging each
question's difference over the four settings:

| Agent | Final EM (pp) | Official metric (pp) | Answer accuracy (pp) |
|---|---|---|---|
| Fixed reader | +5.44 [3.67, 7.33] | +5.17 [3.40, 6.98] | +7.51 [4.15, 11.07] |
| ReAct | +2.44 [1.11, 3.89] | +2.56 [1.22, 3.93] | +1.84 [-0.78, 4.55] |
| AutoGen | +4.33 [2.67, 5.89] | +3.39 [1.90, 4.92] | +5.51 [1.90, 9.10] |
| PlanRAG | +2.67 [1.44, 3.89] | +2.65 [1.50, 3.87] | +2.34 [0.51, 4.15] |

`python code/recompute_window_control.py` recomputes these figures and the
per-arm EM, answered rate, answer accuracy, and official metric. Its input is
`results/anytime_window_control_20261101.jsonl` (10,800 records: 900 questions
x 4 settings x 3 arms; final substring EM, refusal, official metric, per-step
EM). It checks them against
`results/anytime_window_control_summary_20261101.json`; `--write` regenerates the
summary. No model call is made.

## Multi-hop search agents: answer accuracy and lexical memory provenance (2026-11-02)

A search agent answers multi-hop questions in hops. Each hop writes one
sub-question from the main question and the stored records, retrieves the top
six cards by BM25 over the whole corpus, reads them, and passes the answer
through the memory policy (fail-open or ADMIT MFC/OFO). The final answer is
synthesized from the records and the top six cards for the main question and
passes a fail-open output check. Each policy runs with two supports for that
check: the main-question cards (default), or those cards plus the stored values
(memory-aware). Direct-text admission requires the value in its own read
text; memory-aware admission can also use earlier stored values. In the
recursive lexical source-trace diagnostic, a stored value is grounded if it
occurs in its own read text or in a single earlier grounded value. A final
answer overlaps with an ungrounded value if it contains, or is contained in,
one; this does not establish causal reliance or semantic support. FRAMES was used for development;
for the other four settings the arms, analysis script, and file hashes were
recorded before the run.

| Setting | n | FO+m EM | ADMIT+m EM | ADMIT - FO (pp) | Overlap with ungrounded, FO / ADMIT (%) |
|---|---|---|---|---|---|
| FRAMES (development) | 824 | 29.61 | 28.28 | -1.33 [-2.67, -0.12] | 5.22 / 0 |
| 2WikiMultihopQA | 750 | 77.07 | 76.93 | -0.13 [-0.67, 0.27] | 0.13 / 0 |
| MuSiQue | 750 | 46.13 | 46.00 | -0.13 [-0.80, 0.40] | 2.67 / 0 |
| MuSiQue fresh, complete corpus* | 1,062 | 33.62 | 34.56 | +0.94 [-0.56, 2.45] | 18.36 / 0 |
| MuSiQue fresh, incomplete corpus* | 1,062 | 17.89 | 17.98 | +0.09 [-1.22, 1.41] | 25.80 / 0 |

\* Reader prompt without a refusal clause. Incomplete corpus: one supporting
article per question is removed from all of its searches (as in MuSiQue's
unanswerable contrast questions). Averaged over the four confirmation settings,
ADMIT minus fail-open is +0.19 pp [-0.44, 0.79]; the memory-aware check raises
ADMIT's EM over the default check by +5.25 pp [4.48, 6.07] (fail-open gains as
much). The pooled intervals use 4,000 question bootstrap resamples (seed 0),
with complete/incomplete MuSiQue results sampled jointly by qid. The other two
question sets are resampled independently, retaining equal weight across the
four settings. On the questions where fail-open's final answer overlaps with an
ungrounded value, fail-open is correct in 9.74% (195 questions, complete corpus)
and 6.57% (274, incomplete corpus), ADMIT in 20.51% and 12.04%; on the other
questions fail-open 38.99% and 21.83%, ADMIT 37.72% and 20.05%.

`python code/recompute_multihop_final_check.py` recomputes these figures, the
default-check EM of both policies, and the RepairRead replacements under the
default check. Its input is `results/multihop_final_check_20261102.jsonl`
(22,240 records: setting, split, reader prompt, qid, arm in {rag, failopen,
admit, failopen_memory_support, admit_memory_support}, final answer, substring
EM, refusal, final-check branch, substring EM of the synthesized answer before
the check, hops, stored memory with a flag for whether each value occurs in the
text it was read from). It checks them against
`results/multihop_final_check_summary_20261102.json`; `--write` regenerates the
summary. No model call is made.

### FRAMES direct-text memory contamination (2026-11-06)

`python code/recompute_frames_contamination.py` verifies the FRAMES entries in
the main-text cross-dataset table from the same released per-question records. The default direct-text
arms have EM 25.36/24.27 and contaminated-run counts 128/824 and 0/824
(fail-open/ADMIT), or 15.53%/0%. A run is contaminated when any stored
non-refusal value has a false direct-text support flag; the final output is
not appended to memory. This counts stored values, not final-answer overlap.
`results/frames_memory_contamination_20261106.json` also reports the
memory-aware variants separately: ADMIT has 6/824 runs (0.73%) with a value
absent from the current read's text but supported through earlier grounded
memory. This expanded admission support differs from the default direct-text
criterion used in the main-text cross-dataset table. No model calls are required.

`results/frames_admit_trace_20261106.jsonl` supplies the 824 matching ADMIT hop
traces, validated against the existing compact records. Their source and hash
are in `results/frames_admit_trace_manifest_20261106.json`. The script also
recomputes 128/824 runs with NO_COMMIT (15.53%) and 3,888 logical reader calls
(4.718/q): 3,349 initial reads, 265 hop repairs, and 274 final repairs. The
original memoized serving counter covers all arms and is not used as the
per-policy reader-call count.

For hard refusal, a matched-output replay preserves every admitted
question/value pair, unresolved subquestion, and reader output. Intermediate
failed repairs yield no reusable value under either policy, so consumer
inputs are identical. Only final failed-repair outputs change to the fixed
refusal: 178 outputs, including 63 previously correct answers. This gives
137/824 correct (16.63 EM), zero contamination, and the same logical reader
cost. This is a controlled replay, not an independently generated policy run.


## Where withholding raises answer scores (2026-11-02)

The main-text injection replay table shows both EM and injected-value
recurrence under source-blocked and source-available retrieval. Its rates are
recomputed by `python code/recompute_injected_value_replay.py`, averaging random
seeds within each record. The paired score contrasts, multi-hop figures by
question group, and appendix analysis of natural settings where withholding
lowers answer scores are recomputed by
`python code/recompute_withholding_gains.py` from the same released records.
On selected cases with sources blocked and a reference-different injected
value present, ADMIT's substring EM exceeds fail-open's by 1.97-4.76 pp and
random withholding's by 3.17-3.54 pp (every interval against random excludes
zero); with sources available it exceeds fail-open by 1.59-3.94 pp but not random
withholding. In multi-hop search, on the fresh MuSiQue questions where
fail-open's final answer rests on an ungrounded stored value, ADMIT scores 10.77
and 5.47 pp higher; on the other questions it scores 1.27 and 1.78 pp lower, and
on the 43 such FRAMES questions (development) it scores 18.60 pp lower. Intervals
are 4,000 paired resamples (seed 0). The script checks its output against
`results/withholding_gains_summary_20261102.json`; `--write` regenerates the
summary. No model call is made.

## Multi-hop search agents with an improved harness (2026-11-03)

The same agent is changed identically under fail-open and ADMIT (memory-aware
final check): stored values that occur in the text they were read from carry
their card and a 220-character span to the planner and the final reader (at most
four extra final cards), a repeated sub-question is re-planned once with the
article titles found so far, and a search also returns the best card of up to
two corpus articles named in the query. The arms, analysis script, and file
hashes were recorded before the run on questions no earlier run used: 1,000
2WikiMultihopQA validation questions outside the earlier sample and 1,000
MuSiQue-Ans training questions, strict prompt.

| Setting | n | RAG | FO base | ADMIT base | FO improved | ADMIT improved | ADMIT - FO (improved) | Rests FO / ADMIT |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 2WikiMultihopQA | 1,000 | 38.70 | 67.10 | 67.20 | 68.90 | 68.90 | 0.00 [0.00, 0.00] | 0.70 / 0 |
| MuSiQue | 1,000 | 22.90 | 45.20 | 45.20 | 47.20 | 46.60 | -0.60 [-1.10, -0.20] | 2.20 / 0 |

Averaged over the two, the improved harness raises fail-open by +1.90 pp
[0.30, 3.50] and ADMIT by +1.55 pp [0.00, 3.20]; ADMIT minus fail-open is
-0.30 pp [-0.55, -0.10].

`python code/recompute_multihop_improved_agent.py` recomputes these figures from
`results/multihop_improved_agent_20261103.jsonl` (10,000 records: setting, qid,
arm in {none, failopen_m, admit_m, fo2, admit2}, final answer, substring EM,
refusal, hops with retrieved and linked titles, reader output and stored value,
stored memory with a flag for whether each value occurs in the text it was read
from, re-plans, gold articles among the final cards) and checks them against
`results/multihop_improved_agent_summary_20261103.json`; `--write` regenerates
the summary. No model call is made.

## Memory shared across questions (2026-11-03)

Streams of development questions that share facts: MuSiQue, 200 streams of 2 to
6 questions whose decompositions use the same supporting paragraph with the same
step answer (1,100 questions, 900 later questions); TAT-QA, the 274 development
documents with their six questions each (1,644 questions, 800 later questions;
one card per table row and per paragraph). The improved multi-hop agent (strict
prompt, Qwen3-VL-8B) answers a stream in order, and each question's stored
records (sub-question, value, support span) are published to the later
questions as notes. Arms: no sharing (m0), fail-open sharing (fo_s), ADMIT
sharing (ad_s), and an oracle sharing only fail-open records that match a fact
or answer of their question (or_s). Sources available or revoked (the cards of
facts an earlier question needed are removed from the later question's
searches). EM on later questions: substring EM on MuSiQue, official EM on
TAT-QA.

| Dataset | Sources | n | None | FO | ADMIT | Oracle | ADMIT - FO | Oracle - FO | Wrong shared (FO) |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| MuSiQue | revoked | 900 | 11.22 | 28.67 | 28.11 | 33.67 | -0.56 [-2.37, 1.16] | 5.00 [2.07, 8.19] | 51.50 |
| MuSiQue | available | 900 | 32.11 | 38.44 | 37.44 | 46.22 | -1.00 [-3.29, 1.47] | 7.78 [3.61, 12.25] | 47.76 |
| TAT-QA | revoked | 800 | 7.38 | 14.62 | 14.62 | 14.75 | 0.00 [-1.01, 0.99] | 0.12 [-1.24, 1.42] | 52.27 |
| TAT-QA | available | 800 | 19.00 | 20.50 | 20.50 | 20.75 | 0.00 [-0.93, 0.98] | 0.25 [-1.10, 1.60] | 51.38 |

98.1 to 98.7% of the wrong shared fail-open records occur in their support
span, so lexical admission keeps them. A relation check (does the card state
that the value answers the sub-question; Qwen3-VL-8B and Qwen3-VL-32B) separates
step-matched wrong values on MuSiQue with AUC 0.69 to 0.72 and records repeated
by a wrong later answer with AUC 0.56 to 0.64.

`python code/recompute_shared_memory.py` recomputes these figures from
`results/shared_memory_streams_20261103.jsonl` (21,952 records: dataset,
condition, stream, qid, position, number of revoked cards, arm, final answer,
substring EM, official TAT-QA EM, refusal, number of notes, published records
with sub-question, value, support span, lexical-support flag, and correctness)
and `results/shared_memory_relation_check_20261103.jsonl` (5,262 MuSiQue
fail-open records with the matched decomposition step and the 8B and 32B
P(Yes)), and checks them against `results/shared_memory_summary_20261103.json`;
`--write` regenerates the summary. No model call is made.

## Additional upstream pipelines (2026-11-06)

`results/strict_portability_ext_20261106/` extends the terminal portability
replay of `results/strict_portability_20260928/` from eight to sixteen frozen
upstream artifacts. The eight added artifacts are public retrievers and
chunkers from frozen baseline packets: BM25/Qwen3-Embedding fusion,
gte-Qwen2-7B-instruct, and Qwen3-Embedding-0.6B over one shared section-card
index; Qwen3-Embedding over whole-page OCR/text; and BM25 top-10 retrieval over
fixed-length, semantic, LumberChunker, and Meta Chunker chunks (the last two
are local reproductions). The questions (the same 300 M3DocVQA qids), the
frozen contract file (same SHA-256 as the first replay), reader checkpoint,
greedy decoding, top-10 serialization, and 4,500-token packet are unchanged;
only the upstream packets differ. `MANIFEST.json` records the hash of each
upstream artifact and of the frozen packet file, written before inference.
`terminal_per_qid.jsonl` holds one compact row per pipeline and qid (2,400
rows) with each arm's substring EM, contamination, NO_COMMIT, reader calls,
and refusal flag (`answer_metrics.is_refusal`; no refusal receives EM, so the
2026-10-20 correction changes no row). `TERMINAL_SUMMARY.json` adds the paired
10,000-resample intervals (seed 20260928). `python
code/validate_strict_portability.py` recomputes every aggregate from the
compact rows.

## Graph-based upstream pipelines (2026-11-06)

`results/strict_portability_graph_20261106/` adds HopRAG and MoLoRAG to the
terminal portability replay (eighteen upstreams in total), on the same 300
M3DocVQA qids with the same frozen contract, reader, decoding, and top-10
serialization. Both are local reproductions of the released code applied to
this open-domain corpus. HopRAG: the released graph construction and
`bfs_node` traversal with its prompts and Qwen3-8B (thinking off, temperature
0.1); nodes are the OCR/text chunks, each question's graph covers the
documents of its dense bge-base-en-v1.5 top-10 entry nodes (as the released
per-question graphs cover each question's documents), the graph is held in
memory instead of Neo4j, and traversal runs four hops. MoLoRAG: the released
beam search (beam 3, four hops, alpha 0.4, threshold 0.3) with the
training-free Qwen2.5-VL-3B-Instruct page judge over ColPali v1.3 page scores
and page graphs of the documents of the ColPali top-20 pages; selected pages
are read through their OCR/text. `MANIFEST.json` records these settings and
the packet hashes written before inference; `terminal_per_qid.jsonl` (600
rows) and `TERMINAL_SUMMARY.json` follow the format of the extension release.
`python code/validate_strict_portability.py` recomputes every aggregate.

## Support spans as memory provenance (2026-10-31)

`python code/recompute_support_spans.py` recomputes the appendix decomposition
from `results/support_span_memory_20261031.jsonl` (2,880 records: DUDE, TAT-QA,
and MMLongBench-Doc fail-open PlanRAG trajectories at injection rates 0 and 0.2,
sources available or blocked; substring EM and refusal under fail-open, ADMIT,
fail-open with support spans, and ADMIT with support spans) with paired
4,000-draw bootstrap intervals, and checks it against
`results/support_span_memory_summary_20261031.json`; `--write` regenerates the
summary. No model call is made.

With sources available, ADMIT with spans raises pooled substring EM over
fail-open by 2.08 pp [0.56, 3.61]. The spans carry the gain (fail-open with
spans: +2.22 pp [0.83, 3.61]); withholding lowers answered-but-wrong answers,
by 0.97 pp without injection and by 5.97 pp [-8.06, -4.03] at injection rate
0.2 with sources blocked.

## Accuracy of answers withheld from memory (2026-10-25)

`python code/recompute_withheld_accuracy.py` recomputes the appendix comparison
of answers withheld from memory versus admitted non-refusal answers, with
10,000-draw qid-cluster intervals, from the released compact rows and checks it
against `results/withheld_accuracy_20261025.json`; `--write` regenerates the
file. `results/mmlongbench_not_answerable_qids_20261025.json` lists the sampled
MMLongBench-Doc qids whose public reference answer is `Not answerable`, used for
the variant that excludes them. No model call, admission decision, or score is
changed.

## Supplementary audits cited in the appendix (2026-10-25 release)

`results/supplementary_audits_20260725/` holds the inputs for the appendix
paragraphs on OCR perturbation, derived values, the exact-packet correction
scope, audit-record completeness, and structured-write prevalence:

- `ocr_perturbation.jsonl` (600 rows: 150 lexically supported terminal ADMIT
  answers, first 50 qids per agent, under four packet conditions) and its
  summary. `span_whitespace` inserts one space inside the answer span,
  `span_confusion` applies one OCR-style character confusion inside the span,
  and `global_cer_0.5pct` perturbs 0.5% of packet alphanumerics.
- `tatqa_derived_value_cases.jsonl` (145 TAT-QA arithmetic questions from the
  typed PlanRAG transfer run: official derivation, arithmetic replay, table
  mapping, and the derived-only, exact-match, withheld, and table-context
  flags) and its summary.
- `exact_packet_correction_summary.json`: admission decisions of the submitted
  ADMIT run whose outcome changes when the exact reader packet replaces the
  expanded serialization.
- `audit_log_completeness_summary.json`: field completeness of the 2,508 public
  write events and 900 terminal outputs, source-level checks that later steps
  read only admitted values, and the surface multi-value flags.
- `structured_decomposition_summary.json`: typed decomposition counts.
- `response_period_final_only_transfer_summary.json`: final-support retry and
  ADMIT arms of the supplementary PlanRAG transfer on the same 300 qids per
  dataset, with paired ADMIT-minus-final-support contrasts.

`python code/recompute_supplementary_audits.py` recomputes the OCR rates and
qid-cluster intervals with the original seeds, replays every official TAT-QA
derivation with an arithmetic-only parser, recounts the derived-value flags,
and checks the remaining counts against the appendix. No model call is made.

## Refusal repair (2026-11-06)

`results/refusal_repair_20261106/` holds the compact rows behind Table 3(b) and
App. B.15. Refusal repair gives an initial refusal one RepairRead; the repaired
value is admitted only if it is a non-refusal that passes the same lexical
check, otherwise the refusal record stays.

- `dynamic_refusal_repair_900.jsonl`: one row per (qid, agent) of the 300-qid
  dynamic sample with substring EM and reader calls under ADMIT, ADMIT+R,
  fail-open, and fail-open+R. The 335 ADMIT and 334 fail-open pairs whose
  trajectory records an initial refusal were rerun with and without the rule
  (`admit_rerun`, `failopen_rerun`, with replication flags against Table 1);
  the other pairs are identical under both arms by construction.
- `terminal_refusal_repair.jsonl`: one row per (upstream, qid) of the
  18-upstream terminal replay; the 1,766 initial refusals carry the repaired
  outcome and one extra reader call.
- `datasets_refusal_repair.jsonl`: one row per qid of the typed PlanRAG
  transfer with substring EM, reader calls, the official per-question score
  (DUDE ANLS, TAT-QA F1, MMLongBench-Doc generalized accuracy), and the
  MMLongBench-Doc Not answerable flag.

`python code/recompute_refusal_repair.py` recomputes the dynamic, terminal, and
dataset results with the paired qid bootstrap (10,000 resamples, seed 0) and
checks them against the paper. The runs themselves are
`experiments/refusal_repair/run_dynamic_rr.py` (pairs in
`configs/refusal_repair_pairs.json`) and `run_portability_rr.py`.
