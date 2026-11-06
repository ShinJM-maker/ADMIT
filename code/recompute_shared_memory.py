#!/usr/bin/env python3
"""Memory shared across the questions of a task stream (MuSiQue and TAT-QA development data).

Records: results/shared_memory_streams_20261103.jsonl, one row per question, arm, and condition.
Streams: MuSiQue, 200 streams of 2 to 6 development questions whose decompositions share a fact
(the same supporting paragraph with the same step answer), 1,100 questions; TAT-QA, the 274
development documents with their 6 questions each, 1,644 questions. Each question is run by the
improved multi-hop agent (App. multi-hop, Qwen3-VL-8B reader, prompt with a refusal clause); its
stored records (sub-question, value, support span) are then published to the later questions of
its stream, which see them as notes in the planner prompt and as a card. Arms: m0 (nothing
shared), fo_s (fail-open, all records shared), ad_s (ADMIT, all records shared), or_s (fail-open,
only records whose value matches a fact or answer of their own question). A later question needs
a fact that an earlier question of its stream needed. Conditions: available (all cards searchable)
and revoked (the cards of those facts are removed from the later question's searches).
A published record is "wrong" if its value matches no fact or answer of its question.

Relation check: results/shared_memory_relation_check_20261103.jsonl, one row per fo_s record of
MuSiQue (condition, qid, record index into that row's published list). p_yes_8b / p_yes_32b:
P(Yes) of Qwen3-VL-8B / Qwen3-VL-32B asked whether the card the value was read from states that
the value answers the sub-question. step_sub_q / step_answer: the decomposition step whose
sub-question (placeholders filled) has the largest content-word Jaccard overlap with the record's
sub-question. Labels: on-path wrong (overlap >= 0.5 and value differs from the step answer) and
harmful (a later question's non-refusal final answer repeats the value and is wrong; defined for
records some later answer repeats).

Primary metric: substring EM on MuSiQue, the official TAT-QA scorer's EM on TAT-QA. Intervals:
4,000 stream resamples, seed 0. No model call is made. By default the script checks the stored
summary; --write regenerates it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAMS = ROOT / "results" / "shared_memory_streams_20261103.jsonl"
RELATION = ROOT / "results" / "shared_memory_relation_check_20261103.jsonl"
OUTPUT = ROOT / "results" / "shared_memory_summary_20261103.json"
ARMS = ("m0", "fo_s", "ad_s", "or_s")
PAIRS = (("or_s", "fo_s"), ("ad_s", "fo_s"), ("fo_s", "m0"), ("ad_s", "m0"))


def norm(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def loose(s):
    return re.sub(r"[^a-z0-9.]+", " ", str(s or "").lower()).strip()


def match(v, g):
    v, g = norm(v), norm(g)
    return bool(v and g) and (v == g or (len(v) > 3 and len(g) > 3 and (v in g or g in v)))


def repeats(value, answer):
    v, a = norm(value), norm(answer)
    return bool(v) and (v in a or a in v)


def boot(groups, n=4000):
    rng = random.Random(0)
    flat = lambda gs: [x for g in gs for x in g]
    mean = lambda gs: sum(flat(gs)) / max(1, len(flat(gs)))
    bs = sorted(mean([rng.choice(groups) for _ in groups]) for _ in range(n))
    return [round(100 * mean(groups), 2), round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n) - 1], 2)]


def auc(pos, neg):
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    i, rs = 0, 0.0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        rs += (i + j + 1) / 2 * sum(1 for k in range(i, j) if s[k][1])
        i = j
    return round((rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 3)


def summarize():
    cell = {}
    for line in open(STREAMS):
        r = json.loads(line)
        cell[(r["dataset"], r["condition"], r["qid"], r["arm"])] = r
    out = {}
    for ds in ("MuSiQue", "TAT-QA"):
        key = "em" if ds == "MuSiQue" else "native_em"
        for cond in ("revoked", "available"):
            qids = sorted({q for d, c, q, _ in cell if d == ds and c == cond})
            meta = {q: cell[(ds, cond, q, "m0")] for q in qids}
            later = [q for q in qids if meta[q]["n_revoked"] > 0]
            by_s = defaultdict(list)
            for q in later:
                by_s[meta[q]["stream"]].append(q)
            res = {"questions": len(qids), "later": len(later), "streams": len({meta[q]["stream"] for q in qids})}
            for a in ARMS:
                rows = [cell[(ds, cond, q, a)] for q in later]
                pub = [p for q in qids for p in cell[(ds, cond, q, a)]["published"]]
                wrong = [p for p in pub if not p["correct"]]
                res[a] = {"later": round(100 * sum(r[key] for r in rows) / len(rows), 2),
                          "later_substring_em": round(100 * sum(r["em"] for r in rows) / len(rows), 2),
                          "published": len(pub), "wrong_published_pct": round(100 * len(wrong) / max(1, len(pub)), 2),
                          "wrong_with_value_in_span_pct": round(100 * sum(loose(p["value"]) in loose(p["span"]) for p in wrong)
                                                               / max(1, len(wrong)), 2)}
            for x, y in PAIRS:
                res[f"{x}_minus_{y}"] = boot([[cell[(ds, cond, q, x)][key] - cell[(ds, cond, q, y)][key] for q in L]
                                              for L in by_s.values()])
            # later fail-open answers that repeat a wrong note of their stream
            streams = defaultdict(list)
            for q in qids:
                streams[meta[q]["stream"]].append(cell[(ds, cond, q, "fo_s")])
            rep = cor = 0
            for rs in streams.values():
                rs.sort(key=lambda r: r["position"])
                notes = []
                for r in rs:
                    if r["n_revoked"] > 0 and not r["refused"] and any(not p["correct"] and repeats(p["value"], r["answer"])
                                                                       for p in notes):
                        rep += 1
                        cor += r["em"]
                    notes += r["published"]
            res["fo_s_later_answers_repeating_a_wrong_note"] = rep
            res["of_which_substring_correct"] = cor
            out[f"{ds} {cond}"] = res
    # relation check (MuSiQue, fo_s records)
    rel = [json.loads(l) for l in open(RELATION)]
    rows = {(c, q): cell[("MuSiQue", c, q, "fo_s")] for (_, c, q, a) in cell if a == "fo_s" and _ == "MuSiQue"}
    for cond in ("revoked", "available"):
        R = [x for x in rel if x["condition"] == cond]
        res = {"records": len(R)}
        for x in R:
            r = rows[(cond, x["qid"])]
            p = r["published"][x["record"]]
            assert p["value"] == x["value"]
            x["wrong"] = int(not p["correct"])
            x["onpath"] = int(not match(x["value"], x["step_answer"])) if x["step_overlap"] >= 0.5 else None
            uses = [u for (c, q), u in rows.items() if c == cond and u["stream"] == r["stream"]
                    and u["position"] > r["position"] and not u["refused"] and repeats(x["value"], u["answer"])]
            x["harm"] = int(any(not u["em"] for u in uses)) if uses else None
        for lab in ("wrong", "onpath", "harm"):
            S = [x for x in R if x[lab] is not None]
            res[f"{lab}_n"] = len(S)
            res[f"{lab}_pos"] = sum(x[lab] for x in S)
            for m in ("8b", "32b"):
                res[f"auc_{lab}_{m}"] = auc([1 - x[f"p_yes_{m}"] for x in S if x[lab]], [1 - x[f"p_yes_{m}"] for x in S if not x[lab]])
        for m in ("8b", "32b"):
            H = [x for x in R if x["harm"] is not None]
            res[f"withhold_p_lt_0.5_{m}"] = {
                "withheld_pct": round(100 * sum(x[f"p_yes_{m}"] < .5 for x in R) / len(R), 2),
                "harmful_caught": sum(x["harm"] and x[f"p_yes_{m}"] < .5 for x in H),
                "harmful": sum(x["harm"] for x in H),
                "helpful_lost": sum((not x["harm"]) and x[f"p_yes_{m}"] < .5 for x in H),
                "helpful": sum(1 - x["harm"] for x in H)}
        res["onpath_wrong_accepted_32b_p_gt_0.9"] = sum(1 for x in R if x["onpath"] == 1 and x["p_yes_32b"] > .9)
        out[f"relation check {cond}"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    got = summarize()
    for k, v in got.items():
        print(k, json.dumps(v))
    if a.write:
        OUTPUT.write_text(json.dumps(got, indent=1) + "\n")
        print(f"wrote {OUTPUT}")
        return
    want = json.loads(OUTPUT.read_text())
    assert got == want, "summary mismatch"
    print("Shared memory: OK")


if __name__ == "__main__":
    main()
