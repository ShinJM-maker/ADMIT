# ADMIT: Support-Gated Memory-Write Admission for Document QA Agents

Code and evaluation artifacts for the NeurIPS 2026 paper
*ADMIT: Support-Gated Memory-Write Admission for Document QA Agents*.

Document QA agents store intermediate answers and reuse them in later steps.
ADMIT treats **showing an answer** and **storing its value for reuse** as
separate decisions. At every proposed memory write it

1. extracts `answer_value` from the agent's structured candidate;
2. records an initial refusal as a non-reusable `STATUS` record, or admits the
   value when it is a normalized substring of the exact OCR/text the reader saw;
3. otherwise calls the reader once more on the same question and evidence
   (**RepairRead**) and checks the repaired value again;
4. admits the repaired value if it passes, or takes `NO_COMMIT`: the original
   answer is still shown to the user as `UNVERIFIED`, but memory keeps only a
   non-reusable `STATUS` record.

Later agent steps read only admitted values. Under this rule, lexical memory
contamination is zero by construction. Lexical admission checks that a value
appears in the delivered text; it does not establish semantic correctness.

## Pipeline and evaluation scope

ADMIT checks and repairs proposed memory writes, controlling which values later
agent steps may reuse. Failed admission keeps the value out of reusable memory
while retaining the original tentative output under MFC/OFO. Optional support
spans attach source context; optional refusal repair retries initial refusals.
Both extensions also apply to fail-open and are evaluated separately.

The standalone engine implements the admission stage. The primary policy
comparisons use value-only ADMIT to isolate that stage. Separate provenance
controls compare the span-aware path with value-only memory and a prefix-window
control under the same nominal context budget; their records and recomputation scripts are included here.
Support spans are quoted source evidence. The lexical guarantee applies to the
admitted answer-value field.

## Repository layout

| Path | Contents |
|---|---|
| `code/admit_admission_engine.py` | Reference ADMIT admission boundary (standard library only) |
| `code/answer_metrics.py` | Refusal-aware substring EM used in the paper |
| `configs/` | Execution contract, exact prompt templates (`prompts.md`), and their SHA-256 manifest |
| `results/` | Aggregate registries and compact per-question metrics behind the paper's tables |
| `code/recompute_*.py`, `code/validate_*.py`, `code/verify_saved_diagnostics.py` | Scripts that recompute the reported numbers from `results/` |
| `examples/quickstart.py` | Minimal use of the admission engine |
| `harness/` | Agent harness: ReAct, PlanRAG and AutoGen loops, typed memory records, the ADMIT write hook (`cavec_mfc_hook`), comparators, vLLM reader, evidence packing |
| `experiments/static/` | Static reader passes and evaluation of ADMIT, comparators, and the leakage-free SupportRouter refit |
| `experiments/dynamic/` | Dynamic agent runs for all policies and the Table 1 summary |
| `experiments/portability/` | Upstream-packet and dataset portability runners (DUDE, TAT-QA, MMLongBench-Doc) with official metrics |
| `experiments/learned_alternatives/` | Learned refusal selector and second-chance verifier (App. D.3) |
| `scripts/` | Launch scripts for the static and dynamic runs |
| `inputs/static/` | Frozen retrieval (document and card IDs with scores) and the router's candidate answers |
| `docs/ARTIFACTS.md` | File-by-file description of the released artifacts |

The compact results contain metrics and identifiers only. Document evidence and
full agent trajectories are not redistributed; regenerating reader outputs
requires the public datasets and the reader model named in
`configs/execution_contract.json` (Qwen3-VL-8B-Instruct, greedy decoding).

## Quick start

Python 3.10 or later. The engine and the main validator use only the standard
library; some recompute scripts use NumPy.

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

```python
from admit_admission_engine import admit_write, is_factual_memory_record

decision = admit_write(
    question="What kind of stick is Eric Boguniecki holding?",
    commit={"answer_value": "hockey stick"},
    evidence_packet_text=reader_visible_ocr_text,   # exact post-truncation text
    repair_read=lambda q, evidence: call_reader(q, evidence, repair_prompt),
)
decision.output_value            # shown to the user (UNVERIFIED if not admitted)
decision.memory_status           # COMMIT, NO_COMMIT, or REFUSAL
is_factual_memory_record(decision)  # True only for admitted values
```

`repair_fail_policy` selects the action after a failed repair: `"mfc_ofo"`
(ADMIT, default), `"hard_refuse"`, or `"fail_open"` (diagnostic comparator).
In the released files, the runtime tag `FACT`/`COMMIT` corresponds to
`ADMITTED` in the paper.

## Reproducing the paper's numbers

No model inference is needed. From the repository root:

```bash
python code/validate_bundle.py            # main tables, contract, engine, checksums
python code/validate_strict_portability.py  # upstream and dataset portability (Tables 4-5)
python code/verify_saved_diagnostics.py   # write-level and gate-span diagnostics
```

| Paper section | Script |
|---|---|
| Sec. 6.2, injected-value replay; App. B.10 | `recompute_injected_value_replay.py` |
| Sec. 6.4, support spans; App. B.11-B.12 | `recompute_support_spans.py`, `recompute_anytime_agent.py`, `recompute_window_control.py` |
| Sec. 6.2, multi-hop search; App. B.13 | `recompute_multihop_final_check.py`, `recompute_multihop_improved_agent.py`, `recompute_frames_contamination.py` |
| App. B.8-B.9 | `recompute_withheld_accuracy.py`, `recompute_distractor_audit.py` |
| Memory shared across questions; App. B.14 | `recompute_shared_memory.py` |
| Sec. 6.4, refusal repair (Table 3b); App. B.15 | `recompute_refusal_repair.py` |
| App. E-F audits | `recompute_supplementary_audits.py` |

Each script prints the recomputed values and ends with `OK`.

## Running the experiments

Running the agents and reader needs a GPU with vLLM (`pip install -r requirements-run.txt`), the
M3DocVQA questions, and the OCR/text index the agents retrieve from (`chunks.jsonl`, `docs.jsonl`,
`sec_index_md/` per dataset). Point the code at them with environment variables:

```bash
export ADMIT_INDEX=/path/to/index          # <index>/M3DocVQA/{chunks.jsonl,docs.jsonl,sec_index_md/}
export ADMIT_DATA=/path/to/datasets        # <data>/M3DocVQA/val.json
export ADMIT_READER_MODEL=Qwen/Qwen3-VL-8B-Instruct
export ADMIT_WORK=$PWD/work                # outputs
```

| Paper result | Command |
|---|---|
| Table 1 and Sec. 6.2-6.3 (dynamic agents, 900 trajectories per policy) | `GPUS="0 1 2 3" bash scripts/run_dynamic.sh` |
| Static control (Sec. 6.1), SupportRouter and App. D.3 | `GPUS="0 1 2 3" bash scripts/run_static.sh` |
| Tables 4-5 (portability) | `experiments/portability/run_terminal.py`, `run_datasets.py` |
| Refusal repair (Table 3b, App. B.15) | `experiments/refusal_repair/run_dynamic_rr.py --family admit` and `--family failopen`; `run_portability_rr.py` |

`run_dynamic.sh` writes `work/dynamic/SUMMARY.json` with substring EM, reader calls per question,
contamination, and NO_COMMIT rate per policy and agent, plus paired bootstrap differences against
ADMIT. Policies: `admit` (ADMIT), `fail_open`, `hard_refusal`, `final_support` (Final-support repair),
and the component arms `gate_only` and `gate_repair`. The harness keeps the internal labels of the
original runs (`cavec_mfc` = ADMIT, `cavec_paper` = fail-open, `cavec_h1_hard_refuse` = hard refusal).

The portability runners read frozen text packets from the upstream pipelines; because these contain
document text they are not redistributed. `experiments/portability/prepare_packets.py` rebuilds them
from each upstream's own outputs (`ADMIT_UPSTREAM_ROOT`), and `fetch_metrics.py` downloads the official
DUDE, TAT-QA, and MMLongBench-Doc scorers at the commits pinned in `native_metrics_sources.json`.

## Citation

```bibtex
@inproceedings{shin2026admit,
  title     = {{ADMIT}: Support-Gated Memory-Write Admission for Document {QA} Agents},
  author    = {Shin, Joongmin and Shim, Gyuho and Moon, Hyeonseok and Seo, Jaehyung},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Apache License 2.0 (see `LICENSE`). Dataset excerpts in `results/` keep the
licenses of their sources (for example, Wikipedia text under CC BY-SA).
