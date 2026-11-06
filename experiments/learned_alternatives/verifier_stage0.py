"""Stage 0 for a learned second-chance verifier (no model calls).

Pool: static ADMIT writes that end in NO_COMMIT (non-refusal initial answer, lexical check fails, RepairRead does
not produce an admissible value). The value a verifier could admit is the shown (initial) answer; its label is
raw substring EM against the reference. Fit and every choice use the calibration partition only; the reporting
partition is evaluated once.

Compared admission rules over the pool (all admit a subset of NO_COMMIT values):
  deterministic: number/date-normalized match, token recall >= tau, initial == repaired answer, and their OR;
  learned: logistic regression and gradient boosting on answer-free cheap features;
  random: same count as the learned rule.
Headline comparison: correct values recovered at matched precision (threshold set on calibration).
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("az", HERE / "refusal_selector.py")
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)
rdr = az.rdr
TOK = re.compile(r"[a-z0-9]+")
MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
STOP = set("the a an of in on at to for and or is was by with".split())


def toks(s):
    return [t for t in TOK.findall(str(s).lower()) if t not in STOP]


def numbers(s):
    s = str(s).lower().replace(",", "")
    out = set(re.findall(r"\d+(?:\.\d+)?", s))
    for m, i in MONTHS.items():
        if m in s:
            out.add(f"m{i}")
    return out


def pool(part):
    s = az.SRC[part]
    gold = rdr.load_gold()
    st, stp, gu, gup = (az.rows(s[k]) for k in ("strict", "strict_pk", "guess", "guess_pk"))
    out = []
    for q in sorted(st):
        a, pk = st[q]["generated_answer"], stp[q]["reader_ctx_text"]
        r, rpk = gu[q]["generated_answer"], gup[q]["reader_ctx_text"]
        if rdr.is_refusal(a) or rdr.lexical_support(a, pk):
            continue
        if not rdr.is_refusal(r) and rdr.lexical_support(r, rpk):
            continue  # repaired value admitted: not a NO_COMMIT write
        an, pn, qn = rdr.norm(a), rdr.norm(pk), rdr.norm(st[q]["question"])
        at, pt = toks(a), set(toks(pk))
        recall = sum(t in pt for t in at) / max(1, len(at))
        nums = numbers(a)
        num_match = int(bool(nums) and nums <= numbers(pk))
        sm = difflib.SequenceMatcher(None, an, pn, autojunk=False)
        lcs = sm.find_longest_match(0, len(an), 0, len(pn)).size / max(1, len(an))
        agree = int(bool(rdr.norm(r)) and (rdr.norm(r) in an or an in rdr.norm(r)))
        x = [recall, num_match, lcs, agree, int(an in qn), len(at), len(an), int(bool(nums)),
             int(an in ("yes", "no")), int(not rdr.is_refusal(r)), len(pt) / 5000.0]
        out.append(dict(qid=q, y=rdr.uniform_em(a, gold[q]), x=x, recall=recall, num=num_match,
                        agree=agree, inq=int(an in qn)))
    return out


def summarize(sel, rows):
    k = len(sel)
    tp = sum(rows[i]["y"] for i in sel)
    return {"admitted": k, "correct": tp, "precision": round(tp / k, 3) if k else None}


def main():
    cal, rep = pool("calibration"), pool("reporting")
    res = {"pool": {p: {"n": len(r), "correct": sum(x["y"] for x in r)} for p, r in
                    (("calibration", cal), ("reporting", rep))}}
    Xc, yc = np.array([r["x"] for r in cal], float), np.array([r["y"] for r in cal])
    Xr, yr = np.array([r["x"] for r in rep], float), np.array([r["y"] for r in rep])

    # deterministic rules; token-recall threshold chosen on calibration for precision >= target
    target = 0.85
    det = {}
    for tau in (1.0, 0.9, 0.8, 0.75, 0.67, 0.5):
        sel = [i for i, r in enumerate(cal) if r["recall"] >= tau]
        s = summarize(sel, cal)
        det[f"recall>={tau}"] = s
    res["calibration_deterministic"] = det
    taus = [t for t in (1.0, 0.9, 0.8, 0.75, 0.67, 0.5)
            if (det[f"recall>={t}"]["precision"] or 0) >= target and det[f"recall>={t}"]["admitted"] >= 5]
    tau = min(taus) if taus else 1.0
    rules = {
        "number_date_match": lambda r: r["num"] == 1,
        f"token_recall>={tau}": lambda r: r["recall"] >= tau,
        "initial==repair": lambda r: r["agree"] == 1,
        "OR_of_rules": lambda r: r["num"] == 1 or r["recall"] >= tau or r["agree"] == 1,
    }
    res["chosen_recall_tau"] = tau
    res["deterministic"] = {name: {"calibration": summarize([i for i, r in enumerate(cal) if f(r)], cal),
                                   "reporting": summarize([i for i, r in enumerate(rep) if f(r)], rep)}
                            for name, f in rules.items()}

    learned = {}
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    models = {
        "logreg": lambda: LogisticRegression(C=0.5, max_iter=3000),
        "gbdt": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                                       min_samples_leaf=10, random_state=0),
    }
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, mk in models.items():
        oof = np.zeros(len(cal))
        for tr, te in skf.split(Xc, yc):
            m = mk().fit((Xc[tr] - mu) / sd, yc[tr])
            oof[te] = m.predict_proba((Xc[te] - mu) / sd)[:, 1]
        # threshold on calibration OOF: lowest score with precision >= target among admitted
        order = np.argsort(-oof)
        thr = None
        for k in range(5, len(order) + 1):
            if yc[order[:k]].mean() >= target:
                thr = oof[order[k - 1]]
        m = mk().fit((Xc - mu) / sd, yc)
        sr = m.predict_proba((Xr - mu) / sd)[:, 1]
        sel = [i for i in range(len(rep)) if thr is not None and sr[i] >= thr]
        rng = np.random.default_rng(0)
        rnd = [summarize(list(rng.choice(len(rep), len(sel), replace=False)), rep)["correct"]
               for _ in range(500)] if sel else [0]
        learned[name] = {"calibration_oof_auc": round(float(roc_auc_score(yc, oof)), 3),
                         "reporting_auc": round(float(roc_auc_score(yr, sr)), 3),
                         "threshold_from_calibration": None if thr is None else round(float(thr), 4),
                         "reporting": summarize(sel, rep),
                         "random_same_count_mean_correct": round(float(np.mean(rnd)), 1)}
    res["learned"] = learned
    res["precision_target"] = target
    (HERE / "VERIFIER_STAGE0.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
