# Fresh strict terminal portability: two graph-based upstream artifacts

Matched post-repair fail-open / ADMIT / hard refusal. Each setting has 300 reused qids. No full-system or downstream-effect claim. FACT is the runtime ADMITTED alias; projection is tested, not followed by a new downstream task.

| Pipeline | FO contamination % | ADMIT contamination % | FO/ADMIT raw EM % | Hard-refuse EM % | NO_COMMIT % | Reader calls/q |
|---|---:|---:|---:|---:|---:|---:|
| HopRAG (local) | 6.33 | 0.00 | 37.67 | 34.33 | 6.33 | 1.113 |
| MoLoRAG (local) | 4.67 | 0.00 | 43.00 | 40.33 | 4.67 | 1.093 |

Per-setting paired 95% CIs (ADMIT minus comparator); no pooled p-value.

| Pipeline | Δ contamination vs FO (pp) | Δ EM vs hard refusal (pp) |
|---|---:|---:|
| HopRAG (local) | -6.33 [-9.33, -3.67] | +3.33 [+1.33, +5.67] |
| MoLoRAG (local) | -4.67 [-7.33, -2.33] | +2.67 [+1.00, +4.67] |
