# Fresh strict terminal portability: eight additional upstream artifacts

Matched post-repair fail-open / ADMIT / hard refusal. Each setting has 300 reused qids. No full-system or downstream-effect claim. FACT is the runtime ADMITTED alias; projection is tested, not followed by a new downstream task.

| Pipeline | FO contamination % | ADMIT contamination % | FO/ADMIT raw EM % | Hard-refuse EM % | NO_COMMIT % | Reader calls/q |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-length chunking | 6.00 | 0.00 | 34.67 | 31.00 | 6.00 | 1.097 |
| gte-Qwen2-7B | 8.33 | 0.00 | 25.33 | 21.33 | 8.33 | 1.120 |
| BM25 + Qwen3-Embedding hybrid | 6.00 | 0.00 | 36.67 | 32.67 | 6.00 | 1.103 |
| LumberChunker | 5.33 | 0.00 | 36.00 | 32.00 | 5.33 | 1.100 |
| LLM chunking | 6.67 | 0.00 | 35.67 | 32.00 | 6.67 | 1.117 |
| Page (text) | 8.67 | 0.00 | 34.67 | 30.67 | 8.67 | 1.140 |
| Qwen3-Embedding-0.6B | 9.33 | 0.00 | 32.33 | 26.33 | 9.33 | 1.127 |
| Semantic chunking | 4.67 | 0.00 | 36.00 | 32.67 | 4.67 | 1.097 |

Per-setting paired 95% CIs (ADMIT minus comparator); no pooled p-value.

| Pipeline | Δ contamination vs FO (pp) | Δ EM vs hard refusal (pp) |
|---|---:|---:|
| Fixed-length chunking | -6.00 [-9.00, -3.33] | +3.67 [+1.67, +6.00] |
| gte-Qwen2-7B | -8.33 [-11.67, -5.33] | +4.00 [+2.00, +6.33] |
| BM25 + Qwen3-Embedding hybrid | -6.00 [-9.00, -3.33] | +4.00 [+2.00, +6.33] |
| LumberChunker | -5.33 [-8.00, -3.00] | +4.00 [+2.00, +6.33] |
| LLM chunking | -6.67 [-9.67, -4.00] | +3.67 [+1.67, +6.00] |
| Page (text) | -8.67 [-12.00, -5.67] | +4.00 [+2.00, +6.33] |
| Qwen3-Embedding-0.6B | -9.33 [-12.67, -6.33] | +6.00 [+3.33, +9.00] |
| Semantic chunking | -4.67 [-7.33, -2.33] | +3.33 [+1.33, +5.33] |
