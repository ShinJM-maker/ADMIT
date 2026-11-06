"""Learned refusal selector (App. D.3) and the repair-gain decomposition on the static partitions.

Partitions: calibration (development: fitting and every choice) and reporting (evaluated once, with the
choices frozen here). Same reader outputs, gate and raw-EM rule as run_deployable_router.py.

Policies (all keep the lexical memory guarantee; a value is stored only if it is a non-refusal substring
of its packet):
  ADMIT      : repair only unsupported non-refusal initial answers (current rule).
  ADMIT+R    : ADMIT, plus one RepairRead for every initial refusal; a supported non-refusal repair is
               admitted and shown, otherwise the refusal stays a STATUS record.
  Every-q    : repair every question.
  Select(k)  : ADMIT plus RepairRead for the k refusals a selector ranks highest (random or learned).
The selector sees only pre-repair, answer-free signals: question text, retrieval scores and packet size.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import os
REPO = Path(__file__).resolve().parents[2]
W = Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "static"  # reader outputs written by experiments/static
spec = importlib.util.spec_from_file_location("rdr", REPO / "experiments/static/evaluate_static.py")
rdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rdr)
HERE = Path(__file__).resolve().parent

SRC = {
    "calibration": dict(
        strict=W / "calibration_strict_pred.jsonl", strict_pk=W / "calibration_strict_packet.jsonl",
        guess=W / "calibration_guess_pred.jsonl", guess_pk=W / "calibration_guess_packets.jsonl",
        retr=W / "calibration_1162_retrieval.jsonl"),
    "reporting": dict(
        strict=W / "baseline_strict_pred.jsonl", strict_pk=W / "baseline_strict_packets.jsonl",
        guess=W / "repair_guess_pred.jsonl", guess_pk=W / "repair_guess_packets.jsonl",
        retr=W / "heldout_1162_retrieval.jsonl"),
}
WH = ["what", "which", "who", "when", "where", "how", "is", "are", "does", "do", "did", "was", "were", "can", "in"]


def rows(p):
    return {str(r.get("id") or r.get("qid")): r for r in map(json.loads, open(p)) if r}


def features(q, pred, retr):
    ql = q.lower().split()
    first = ql[0] if ql else ""
    sc = [d["score"] for d in retr.get("doc_candidates", [])]
    sc = sc + [0.0] * (3 - len(sc))
    f = [len(ql), int(bool(re.search(r"\d", q))), int("or" in ql), int(q.strip().endswith("?")),
         sc[0], sc[0] - sc[1], sc[0] - sc[2], len(retr.get("doc_candidates", [])),
         pred.get("n_text_units_in_pack", 0), pred.get("n_image_units_in_pack", 0),
         pred.get("prompt_used_tokens", 0) / 14000]
    f += [int(first == w) for w in WH]
    f += [int(k in q.lower()) for k in ("image", "picture", "photo", "color", "logo", "poster", "wearing", "holding",
                                         "table", "year", "date", "how many", "both", "same")]
    return f


def load(part):
    s = SRC[part]
    gold = rdr.load_gold()
    st, stp, gu, gup = rows(s["strict"]), rows(s["strict_pk"]), rows(s["guess"]), rows(s["guess_pk"])
    rt = rows(s["retr"])
    assert set(st) == set(gu) == set(stp) == set(gup) and len(st) == 1162
    out = []
    for q in sorted(st):
        a, ap = st[q]["generated_answer"], stp[q]["reader_ctx_text"]
        r, rp = gu[q]["generated_answer"], gup[q]["reader_ctx_text"]
        out.append(dict(
            qid=q, init_ref=rdr.is_refusal(a), init_sup=bool(rdr.lexical_support(a, ap)),
            rep_ok=(not rdr.is_refusal(r)) and bool(rdr.lexical_support(r, rp)),
            em_init=rdr.uniform_em(a, gold[q]), em_rep=rdr.uniform_em(r, gold[q]),
            x=features(st[q]["question"], st[q], rt[q])))
    return out


def policy(rs, repair_refusal):
    """repair_refusal: set of qids whose initial refusal also gets a RepairRead."""
    em = calls = 0
    for r in rs:
        if r["init_ref"]:
            if r["qid"] in repair_refusal:
                calls += 2
                em += r["em_rep"] if r["rep_ok"] else 0
            else:
                calls += 1
        elif r["init_sup"]:
            calls += 1; em += r["em_init"]
        else:
            calls += 2; em += r["em_rep"] if r["rep_ok"] else r["em_init"]
    n = len(rs)
    return 100 * em / n, calls / n


def every_q(rs):
    em = 0
    for r in rs:
        if r["rep_ok"]:
            em += r["em_rep"]
        elif not r["init_ref"]:
            em += r["em_init"]
    return 100 * em / len(rs), 2.0


def label(r):
    return int(r["rep_ok"] and r["em_rep"] == 1)


def paired_ci(rs, a, b, B=10000, seed=0):
    """EM difference (pp) between two repair-refusal sets, qid bootstrap."""
    def per(r, S):
        if r["init_ref"]:
            return (r["em_rep"] if r["rep_ok"] else 0) if r["qid"] in S else 0
        if r["init_sup"]:
            return r["em_init"]
        return r["em_rep"] if r["rep_ok"] else r["em_init"]
    d = np.array([per(r, a) - per(r, b) for r in rs], float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def main():
    res = {}
    data = {p: load(p) for p in SRC}
    for p, rs in data.items():
        refs = [r for r in rs if r["init_ref"]]
        res[p] = {
            "n": len(rs), "initial_refusals": len(refs),
            "refusal_repair_correct_admitted": sum(label(r) for r in refs),
            "ADMIT": policy(rs, set()), "ADMIT+R": policy(rs, {r["qid"] for r in refs}), "Every-q": every_q(rs),
            "ADMIT+R_minus_ADMIT": paired_ci(rs, {r["qid"] for r in refs}, set()),
        }
    # learned selector: fit on calibration refusals only; ranking frozen before touching reporting
    cal = [r for r in data["calibration"] if r["init_ref"]]
    rep = [r for r in data["reporting"] if r["init_ref"]]
    Xc, yc = np.array([r["x"] for r in cal], float), np.array([label(r) for r in cal])
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit((Xc - mu) / sd, yc)
    # calibration 5-fold OOF AUC (development check)
    rng = np.random.default_rng(42); idx = rng.permutation(len(cal)); oof = np.zeros(len(cal))
    for k in range(5):
        te = idx[k::5]; tr = np.setdiff1d(idx, te)
        m = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit((Xc[tr] - mu) / sd, yc[tr])
        oof[te] = m.predict_proba((Xc[te] - mu) / sd)[:, 1]
    res["selector"] = {"calibration_oof_auc": float(roc_auc_score(yc, oof)) if 0 < yc.sum() < len(yc) else None,
                       "calibration_positives": int(yc.sum()), "calibration_refusals": len(cal)}
    Xr, yr = np.array([r["x"] for r in rep], float), np.array([label(r) for r in rep])
    sr = clf.predict_proba((Xr - mu) / sd)[:, 1]
    res["selector"]["reporting_auc"] = float(roc_auc_score(yr, sr)) if 0 < yr.sum() < len(yr) else None
    order = [rep[i]["qid"] for i in np.argsort(-sr)]
    curve = []
    for frac in (0.25, 0.5, 0.75):
        k = int(round(frac * len(rep)))
        learned = set(order[:k])
        rnd = []
        for s in range(200):
            rr = random.Random(s).sample([r["qid"] for r in rep], k)
            rnd.append(policy(data["reporting"], set(rr))[0])
        curve.append({"fraction_of_refusals_repaired": frac, "k": k,
                      "learned": policy(data["reporting"], learned),
                      "random_mean_EM": float(np.mean(rnd)),
                      "learned_minus_ADMIT": paired_ci(data["reporting"], learned, set())})
    res["selector"]["reporting_curve"] = curve
    (HERE / "RESULTS.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
