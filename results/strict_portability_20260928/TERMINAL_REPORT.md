# Strict terminal portability — refusal-aware scores

Saved generations and admission decisions are unchanged. Substring EM excludes fixed refusals.

| Pipeline | FO contamination % | ADMIT contamination % | FO/ADMIT substring EM % | Hard-refuse EM % | NO_COMMIT % | Reader calls/q |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 7.33 | 0.00 | 35.67 | 31.00 | 7.33 | 1.103 |
| colpali_ocr | 3.33 | 0.00 | 46.33 | 44.67 | 3.33 | 1.080 |
| hikey | 5.33 | 0.00 | 36.33 | 33.00 | 5.33 | 1.090 |
| late_chunking | 7.33 | 0.00 | 35.33 | 30.33 | 7.33 | 1.120 |
| m3docdep | 6.00 | 0.00 | 36.33 | 33.00 | 6.00 | 1.097 |
| multidocfusion | 8.67 | 0.00 | 35.33 | 30.33 | 8.67 | 1.143 |
| raptor | 4.67 | 0.00 | 33.67 | 30.67 | 4.67 | 1.103 |
| vega_kg | 7.00 | 0.00 | 41.33 | 37.00 | 7.00 | 1.120 |
