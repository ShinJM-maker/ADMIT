# Fresh strict typed dataset transfer

MMLongBench-Doc uses the official typed scorer directly on short outputs; GPT-4o answer extraction is omitted. Not the full official leaderboard pipeline. DUDE is the train-tagged public sample; TAT-QA is dev. Scores are conditional on each frozen 300-qid scalar subset.

| Dataset (metric) | Fail-open score | ADMIT score | Hard-refuse score | FO contamination % | ADMIT contamination % | NO_COMMIT % | ADMIT calls/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| DUDE (ANLS) | 27.56 | 27.50 | 25.93 | 15.67 | 0.00 | 15.67 | 3.440 |
| TAT-QA (official F1) | 22.75 | 22.74 | 20.01 | 28.67 | 0.00 | 28.67 | 3.893 |
| MMLongBench-Doc (generalized accuracy; direct short output) | 24.76 | 25.76 | 27.76 | 16.67 | 0.00 | 16.67 | 3.353 |

Paired 95% qid-bootstrap CIs:

| Dataset | Δ task score vs FO (pp) | Δ contamination vs FO (pp) | Δ calls/q vs FO |
|---|---:|---:|---:|
| dude | -0.064 [-1.000, +0.808] | -15.667 [-19.667, -11.667] | -0.027 [-0.047, -0.010] |
| tatqa | -0.017 [-1.010, +0.970] | -28.667 [-34.000, -23.667] | +0.000 [-0.013, +0.013] |
| mmlongbench | +1.000 [+0.000, +2.333] | -16.667 [-21.000, -12.667] | -0.007 [-0.017, +0.000] |

Substring-EM diagnostics use the 2026-10-20 refusal guard; dataset-native task scores are unchanged. Current values are in DATASET_SUMMARY.json.
