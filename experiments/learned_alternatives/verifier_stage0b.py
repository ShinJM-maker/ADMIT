"""Stage 0b: learned second-chance verifier on a pooled NO_COMMIT set (no model calls).

Sources (every row: question, NO_COMMIT candidate shown as UNVERIFIED, its RepairRead value, the exact packet
the gate checked, and a substring-EM label for the candidate):
  static      : static M3DocVQA calibration + reporting partitions (Qwen3-VL-8B, 14k-token packets);
  terminal    : terminal upstream replay, 18 upstreams x 300 M3DocVQA qids (4,500-token packets);
  dude/tatqa/mmlb : typed PlanRAG final writes on the three portability datasets.
Protocol (fixed before looking at held-out results):
  * leave-one-source-out: fit on the other sources, pick the threshold on them (precision >= 0.85 on
    grouped out-of-fold scores), evaluate once on the held-out source;
  * grouped 5-fold CV by qid over the pool as a second view;
  * deterministic rules and random selection of the same count are the comparators.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
import os
E = Path(os.environ.get("ADMIT_WORK", HERE.parents[1] / "work")) / "portability"  # terminal and dataset run outputs
spec = importlib.util.spec_from_file_location("v0", HERE / "verifier_stage0.py")
v0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v0)
rdr = v0.rdr
TARGET = 0.85


def feats(question, cand, repair, packet):
    an, pn, qn = rdr.norm(cand), rdr.norm(packet), rdr.norm(question)
    at, pt = v0.toks(cand), set(v0.toks(packet))
    recall = sum(t in pt for t in at) / max(1, len(at))
    nums = v0.numbers(cand)
    num_match = int(bool(nums) and nums <= v0.numbers(packet))
    sm = difflib.SequenceMatcher(None, an, pn, autojunk=False)
    lcs = sm.find_longest_match(0, len(an), 0, len(pn)).size / max(1, len(an))
    rn = rdr.norm(repair or "")
    agree = int(bool(rn) and (rn in an or an in rn))
    x = [recall, num_match, lcs, agree, int(an in qn), len(at), len(an), int(bool(nums)),
         int(an in ("yes", "no")), int(not rdr.is_refusal(repair or "")), min(len(pn), 60000) / 60000.0]
    return x, dict(recall=recall, num=num_match, agree=agree)


def add(rows, src, qid, question, cand, repair, packet, y):
    x, d = feats(question, cand, repair, packet)
    rows.append(dict(src=src, qid=qid.split("_")[0], y=int(y), x=x, **d))


def load():
    rows = []
    gold = rdr.load_gold()
    for part in ("calibration", "reporting"):
        s = v0.az.SRC[part]
        st, stp, gu, gup = (v0.az.rows(s[k]) for k in ("strict", "strict_pk", "guess", "guess_pk"))
        for q in sorted(st):
            a, pk, r, rpk = (st[q]["generated_answer"], stp[q]["reader_ctx_text"],
                             gu[q]["generated_answer"], gup[q]["reader_ctx_text"])
            if rdr.is_refusal(a) or rdr.lexical_support(a, pk):
                continue
            if not rdr.is_refusal(r) and rdr.lexical_support(r, rpk):
                continue
            add(rows, "static", q, st[q]["question"], a, r, pk, rdr.uniform_em(a, gold[q]))
    for d in ("strict_portability_20260928", "strict_portability_ext_20261106", "strict_portability_graph_20261106"):
        for f in sorted((E / d / "results").glob("terminal.shard*.jsonl")):
            for line in f.open():
                r = json.loads(line)
                a = r["arms"]["admit"]
                if not a.get("no_commit"):
                    continue
                h = a["hook"]
                add(rows, "terminal", r["qid"], r["question"], h["candidate"], h.get("repair_value"),
                    r["packet_text"], rdr.uniform_em(h["candidate"], r["gold"]))
    for ds, src in (("dude", "dude"), ("tatqa", "tatqa"), ("mmlongbench", "mmlb")):
        for f in sorted((E / "strict_portability_20260928/dataset_results").glob(f"{ds}.shard*.jsonl")):
            if "generations" in f.name:
                continue
            for line in f.open():
                r = json.loads(line)
                res = r["arms"]["admit"]["result"]
                rec = res.get("final_memory_record") or {}
                if rec.get("status") != "NO_COMMIT":
                    continue
                h = res.get("final_hook_log") or {}
                cand = h.get("candidate") or res.get("final_committed_answer", "")
                add(rows, src, r["qid"], r["question"], cand, h.get("repair_value"), res.get("final_pack_text", ""),
                    rdr.uniform_em(cand, r["gold"] if isinstance(r["gold"], list) else [r["gold"]]))
    return rows


def thr_for_precision(scores, y, target=TARGET, min_k=10):
    order = np.argsort(-scores)
    best = None
    for k in range(min_k, len(order) + 1):
        if y[order[:k]].mean() >= target:
            best = scores[order[k - 1]]
    return best


def oof_scores(X, y, groups, mk, seed=0):
    out = np.zeros(len(y))
    k = min(5, len(set(groups)))
    for tr, te in GroupKFold(k).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        out[te] = mk().fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return out


def evaluate_selection(sel, y, rng, B=500):
    k = int(sel.sum())
    tp = int(y[sel].sum())
    rnd = [int(y[rng.choice(len(y), k, replace=False)].sum()) for _ in range(B)] if k else [0]
    return {"admitted": k, "correct": tp, "precision": round(tp / k, 3) if k else None,
            "random_same_count_correct": round(float(np.mean(rnd)), 1)}


def main():
    rows = load()
    srcs = sorted(set(r["src"] for r in rows))
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    g = np.array([r["qid"] for r in rows])
    s = np.array([r["src"] for r in rows])
    res = {"pool": {k: {"n": int((s == k).sum()), "correct": int(y[s == k].sum())} for k in srcs},
           "total": {"n": len(y), "correct": int(y.sum())}}
    det = {"number_date_match": lambda r: r["num"] == 1, "token_recall==1": lambda r: r["recall"] >= 1.0,
           "initial==repair": lambda r: r["agree"] == 1}
    res["deterministic_by_source"] = {
        name: {k: {"admitted": sum(1 for r in rows if r["src"] == k and f(r)),
                   "precision": round(np.mean([r["y"] for r in rows if r["src"] == k and f(r)]), 3)
                   if any(r["src"] == k and f(r) for r in rows) else None} for k in srcs}
        for name, f in det.items()}
    models = {"logreg": lambda: LogisticRegression(C=0.5, max_iter=3000),
              "gbdt": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                             min_samples_leaf=15, random_state=0)}
    rng = np.random.default_rng(0)
    for name, mk in models.items():
        # pooled grouped CV view
        oof = oof_scores(X, y, g, mk)
        t = thr_for_precision(oof, y)
        res.setdefault("pooled_grouped_cv", {})[name] = {
            "auc": round(float(roc_auc_score(y, oof)), 3),
            "max_k_at_precision>=0.85": int((oof >= t).sum()) if t is not None else 0,
            "precision_top20pct": round(float(y[np.argsort(-oof)[:len(y) // 5]].mean()), 3)}
        # leave-one-source-out
        loso = {}
        for k in srcs:
            tr, te = s != k, s == k
            oof_tr = oof_scores(X[tr], y[tr], g[tr], mk)
            t = thr_for_precision(oof_tr, y[tr])
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            sc = mk().fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
            sel = sc >= t if t is not None else np.zeros(te.sum(), bool)
            loso[k] = {"held_out_n": int(te.sum()), "held_out_base_precision": round(float(y[te].mean()), 3),
                       "auc": round(float(roc_auc_score(y[te], sc)), 3) if 0 < y[te].sum() < te.sum() else None,
                       "threshold_found_on_training": t is not None,
                       **evaluate_selection(sel, y[te], rng)}
        res.setdefault("leave_one_source_out", {})[name] = loso
    res["precision_target"] = TARGET
    (HERE / "VERIFIER_STAGE0B.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
