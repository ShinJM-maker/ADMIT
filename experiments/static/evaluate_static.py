#!/usr/bin/env python3
"""Strict deployment-contract static SupportRouter evaluation.

The calibration half alone supplies training labels, early-stopping validation,
and the routing threshold.  The reporting half is scored once by a final model
trained on calibration qids.  Runtime scores use only five features computable
from one candidate and its exact reader-visible OCR/text packet.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np


REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "static"
HERE = WORK
SOURCE = WORK  # calibration_strict_{pred,packet}.jsonl from run_reader_vllm.py
REPORT_SOURCE = WORK  # baseline_strict_* and repair_guess_* for the reporting partition
HARNESS = WORK  # cands_with_ocr_lp.jsonl: candidate pool (router diagnostic only)
GOLD_PATH = Path(os.environ.get("ADMIT_DATA", REPO / "data/datasets")) / "M3DocVQA/val.json"

FEATURES = [
    "f_S_str",
    "f_S_tok",
    "f_S_ent",
    "f_S_non",
    "f_H_unsupported",
]
BASE_HP = dict(
    objective="lambdarank",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    min_data_in_leaf=20,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
N_FOLDS = 5
THRESHOLD_PERCENTILE = 50
BOOTSTRAP_DRAWS = 10_000

_TOK_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[가-힣]+", re.UNICODE)
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_DATE_RE = re.compile(r"\b(?:\d{4}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_PUNCT = set(string.punctuation)
_ARTICLES = {"a", "an", "the"}
_PREFIX_RE = re.compile(
    "|".join([
        r"^\s*answer\s*[:\-]\s*",
        r"^\s*the\s+answer\s+is\s*[:\-]?\s*",
        r"^\s*final\s+answer\s*[:\-]\s*",
        r"^\s*the\s+correct\s+answer\s+is\s*[:\-]?\s*",
        r"^\s*based\s+on\s+the\s+(context|document|image|evidence)[^,.]*[,.]\s*",
    ]),
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(
    r"(?:\n\s*\n|\.\s+(?=[A-Z])|\bbecause\b|\bexplanation\s*[:\-]|\breasoning\s*[:\-])",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_map(path: Path, value_field: str) -> dict[str, str]:
    return {
        str(row.get("id") or row.get("qid")): str(row.get(value_field) or "")
        for row in load_rows(path)
    }


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().strip())


def feature_tokens(value: str) -> list[str]:
    return [token.lower() for token in _TOK_RE.findall(str(value or ""))]


def is_refusal(value: str) -> bool:
    """Return the paper-contract refusal predicate.

    Refusals are typed status records, not lexically unsupported factual
    candidates.  Keep this predicate identical to ``rho`` in the manuscript
    and to the public reference engine.
    """
    value_n = norm(value)
    return not value_n or "do not have enough information" in value_n


def is_metric_refusal(value: str) -> bool:
    """Alias retained for metric code; execution and metrics share a contract."""
    return is_refusal(value)


def lexical_support(answer: str, packet: str) -> int:
    answer_n = norm(answer)
    return int(bool(answer_n) and not is_refusal(answer) and answer_n in norm(packet))


def runtime_features(answer: str, packet: str) -> dict[str, float]:
    answer_tokens = set(feature_tokens(answer))
    packet_tokens = set(feature_tokens(packet))
    answer_entities = (
        set(_NUM_RE.findall(answer))
        | set(_DATE_RE.findall(answer))
        | set(_CAP_RE.findall(answer))
    )
    packet_entities = (
        set(_NUM_RE.findall(packet))
        | set(_DATE_RE.findall(packet))
        | set(_CAP_RE.findall(packet))
    )
    s_str = float(lexical_support(answer, packet))
    s_non = float(not is_refusal(answer))
    return {
        "f_S_str": s_str,
        "f_S_tok": len(answer_tokens & packet_tokens) / max(1, len(answer_tokens)),
        "f_S_ent": len(answer_entities & packet_entities) / max(1, len(answer_entities))
        if answer_entities else 0.0,
        "f_S_non": s_non,
        "f_H_unsupported": float(s_non == 1.0 and s_str == 0.0),
    }


def short_answer(value: str) -> str:
    text = str(value or "").strip()
    for _ in range(3):
        match = _PREFIX_RE.match(text)
        if not match:
            break
        text = text[match.end():].lstrip()
    text = text.split("\n", 1)[0].strip()
    text = _EXPLAIN_RE.split(text, maxsplit=1)[0].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text[:-1].strip() if text.endswith(".") else text


def metric_norm(value: str) -> str:
    text = "".join(ch for ch in short_answer(value).lower() if ch not in _PUNCT)
    return " ".join(token for token in text.split() if token not in _ARTICLES).strip()


def gold_values(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value or "")]


def uniform_em(answer: str, golds: list[str]) -> int:
    """Match the frozen dynamic raw-EM convention exactly.

    Raw EM deliberately uses only lowercase/whitespace normalization and
    bidirectional substring containment.  Short-answer extraction and
    punctuation/article removal remain complementary F1/ANLS preprocessing;
    they are not part of raw EM.
    """
    if is_metric_refusal(answer):
        return 0
    answer_n = norm(answer)
    return int(any(
        answer_n and (answer_n in norm(gold) or norm(gold) in answer_n)
        for gold in golds if norm(gold)
    ))


def f1(answer: str, golds: list[str]) -> float:
    pred = metric_norm(answer).split()
    best = 0.0
    for gold in golds:
        target = metric_norm(gold).split()
        if not pred and not target:
            best = max(best, 1.0)
            continue
        if not pred or not target:
            continue
        overlap = sum((Counter(pred) & Counter(target)).values())
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(target)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index] + [0] * len(right)
        for right_index, right_char in enumerate(right, 1):
            current[right_index] = min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            )
        previous = current
    return previous[-1]


def anls(answer: str, golds: list[str]) -> float:
    pred = metric_norm(answer)
    best = 0.0
    for gold in golds:
        target = metric_norm(gold)
        if not pred and not target:
            best = max(best, 1.0)
            continue
        if not pred or not target:
            continue
        score = 1.0 - levenshtein(pred, target) / max(len(pred), len(target))
        best = max(best, score if score >= 0.5 else 0.0)
    return best


def load_gold() -> dict[str, list[str]]:
    rows = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return {
        str(row.get("id") or row.get("qid")): gold_values(row.get("answers") or row.get("answer") or "")
        for row in rows
    }


def partition_inputs() -> tuple[dict, dict, dict, dict]:
    cal_answers = load_map(SOURCE / "calibration_strict_pred.jsonl", "generated_answer")
    cal_packets = load_map(SOURCE / "calibration_strict_packet.jsonl", "reader_ctx_text")
    report_answers = load_map(REPORT_SOURCE / "baseline_strict_pred.jsonl", "generated_answer")
    report_packets = load_map(REPORT_SOURCE / "baseline_strict_packets.jsonl", "reader_ctx_text")
    if len(cal_answers) != 1162 or len(report_answers) != 1162:
        raise RuntimeError("expected disjoint 1,162-qid calibration and reporting partitions")
    if set(cal_answers) != set(cal_packets) or set(report_answers) != set(report_packets):
        raise RuntimeError("answer/packet linkage mismatch")
    if set(cal_answers) & set(report_answers):
        raise RuntimeError("calibration and reporting qids overlap")
    return cal_answers, cal_packets, report_answers, report_packets


def make_folds(qids: list[str]) -> list[list[str]]:
    ordered = sorted(qids, key=lambda qid: hashlib.sha256(qid.encode()).hexdigest())
    return [ordered[index::N_FOLDS] for index in range(N_FOLDS)]


def build_training_rows(
    calibration_answers: dict[str, str],
    calibration_packets: dict[str, str],
    gold: dict[str, list[str]],
) -> dict[str, list[dict]]:
    historical = defaultdict(list)
    for row in load_rows(HARNESS / "cands_with_ocr_lp.jsonl"):
        qid = str(row["qid"])
        if qid in calibration_answers:
            historical[qid].append(row)
    if set(historical) != set(calibration_answers):
        raise RuntimeError("missing calibration candidate group")

    by_qid = {}
    for qid, source_rows in historical.items():
        packet = calibration_packets[qid]
        rebuilt = []
        for source_row in source_rows:
            answer = (
                calibration_answers[qid]
                if source_row.get("kind") == "baseline_text"
                else str(source_row.get("answer") or "")
            )
            features = runtime_features(answer, packet)
            em = uniform_em(answer, gold[qid])
            answer_f1 = f1(answer, gold[qid])
            supported = int(features["f_S_str"])
            label = (
                4 if em and supported else
                3 if em else
                2 if answer_f1 >= 0.5 and supported else
                1 if is_refusal(answer) else 0
            )
            rebuilt.append({
                "qid": qid,
                "kind": source_row.get("kind"),
                "answer": answer,
                "label": label,
                **features,
            })
        by_qid[qid] = rebuilt
    return by_qid


def ordered_rows(by_qid: dict[str, list[dict]], qids: list[str]) -> list[dict]:
    return [row for qid in qids for row in by_qid[qid]]


def fit_with_inner_validation(
    by_qid: dict[str, list[dict]], train_qids: list[str], valid_qids: list[str]
) -> int:
    train_rows = ordered_rows(by_qid, train_qids)
    valid_rows = ordered_rows(by_qid, valid_qids)
    model = lgb.LGBMRanker(**BASE_HP)
    model.fit(
        np.asarray([[row[name] for name in FEATURES] for row in train_rows], dtype=np.float32),
        np.asarray([row["label"] for row in train_rows], dtype=np.int32),
        group=[len(by_qid[qid]) for qid in train_qids],
        eval_set=[(
            np.asarray([[row[name] for name in FEATURES] for row in valid_rows], dtype=np.float32),
            np.asarray([row["label"] for row in valid_rows], dtype=np.int32),
        )],
        eval_group=[[len(by_qid[qid]) for qid in valid_qids]],
        eval_at=[1, 3],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return max(1, int(model.best_iteration_ or BASE_HP["n_estimators"]))


def fit_fixed(by_qid: dict[str, list[dict]], qids: list[str], iterations: int):
    rows = ordered_rows(by_qid, qids)
    params = {**BASE_HP, "n_estimators": iterations}
    model = lgb.LGBMRanker(**params)
    model.fit(
        np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float32),
        np.asarray([row["label"] for row in rows], dtype=np.int32),
        group=[len(by_qid[qid]) for qid in qids],
    )
    return model


def baseline_matrix(answers: dict[str, str], packets: dict[str, str], qids: list[str]) -> np.ndarray:
    return np.asarray([
        [runtime_features(answers[qid], packets[qid])[name] for name in FEATURES]
        for qid in qids
    ], dtype=np.float32)


def train_router(
    by_qid: dict[str, list[dict]],
    calibration_answers: dict[str, str],
    calibration_packets: dict[str, str],
    reporting_answers: dict[str, str],
    reporting_packets: dict[str, str],
) -> tuple[dict[str, float], dict]:
    calibration_qids = sorted(calibration_answers)
    folds = make_folds(calibration_qids)
    oof_scores = {}
    fold_records = []
    for outer_index, outer_test in enumerate(folds):
        inner_valid = folds[(outer_index + 1) % N_FOLDS]
        inner_train = [
            qid for fold_index, fold in enumerate(folds)
            if fold_index not in {outer_index, (outer_index + 1) % N_FOLDS}
            for qid in fold
        ]
        outer_train = [qid for fold_index, fold in enumerate(folds) if fold_index != outer_index for qid in fold]
        best_iteration = fit_with_inner_validation(by_qid, inner_train, inner_valid)
        model = fit_fixed(by_qid, outer_train, best_iteration)
        predictions = model.predict(baseline_matrix(
            calibration_answers, calibration_packets, outer_test
        ))
        oof_scores.update({qid: float(score) for qid, score in zip(outer_test, predictions)})
        fold_records.append({
            "outer_fold": outer_index,
            "outer_train_qids": len(outer_train),
            "outer_test_qids": len(outer_test),
            "inner_train_qids": len(inner_train),
            "inner_validation_qids": len(inner_valid),
            "selected_iterations": best_iteration,
        })
    if set(oof_scores) != set(calibration_qids):
        raise RuntimeError("calibration OOF coverage incomplete")

    threshold = float(np.percentile(list(oof_scores.values()), THRESHOLD_PERCENTILE))
    final_iterations = int(round(float(np.median([
        row["selected_iterations"] for row in fold_records
    ]))))
    final_model = fit_fixed(by_qid, calibration_qids, final_iterations)
    reporting_qids = sorted(reporting_answers)
    reporting_predictions = final_model.predict(baseline_matrix(
        reporting_answers, reporting_packets, reporting_qids
    ))
    scores = {qid: float(score) for qid, score in zip(reporting_qids, reporting_predictions)}
    metadata = {
        "features": FEATURES,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_source": "calibration baseline-row OOF scores only",
        "threshold": threshold,
        "final_iterations": final_iterations,
        "folds": fold_records,
        "feature_gain": {
            name: float(gain) for name, gain in zip(
                FEATURES, final_model.booster_.feature_importance(importance_type="gain")
            )
        },
    }
    return scores, metadata


def metric_row(
    qid: str,
    answer: str,
    packet: str,
    golds: list[str],
    calls: float,
    route: str,
    no_commit: bool = False,
) -> dict:
    em = uniform_em(answer, golds)
    supported = lexical_support(answer, packet)
    non_refusal = int(not is_metric_refusal(answer))
    return {
        "qid": qid,
        "answer": answer,
        "route": route,
        "EM": em,
        "F1": f1(answer, golds),
        "ANLS": anls(answer, golds),
        "support": supported,
        "unsupNR_strict": non_refusal * (1 - supported),
        "value_supported_EM": em * supported,
        "calls": float(calls),
        "no_commit": int(no_commit),
    }


def evaluate(
    scores: dict[str, float],
    threshold: float,
    answers: dict[str, str],
    packets: dict[str, str],
    gold: dict[str, list[str]],
) -> dict[str, list[dict]]:
    repair_answers = load_map(REPORT_SOURCE / "repair_guess_pred.jsonl", "generated_answer")
    repair_packets = load_map(REPORT_SOURCE / "repair_guess_packets.jsonl", "reader_ctx_text")
    qids = sorted(answers)
    if set(repair_answers) != set(qids) or set(repair_packets) != set(qids):
        raise RuntimeError("reporting RepairRead coverage mismatch")
    policies = defaultdict(list)
    for qid in qids:
        initial, initial_packet = answers[qid], packets[qid]
        repair, repair_packet = repair_answers[qid], repair_packets[qid]
        initial_refusal = is_refusal(initial)
        repair_refusal = is_refusal(repair)
        initial_supported = lexical_support(initial, initial_packet)
        repair_supported = lexical_support(repair, repair_packet)
        routed = not initial_refusal and (
            scores[qid] < threshold or not initial_supported
        )

        policies["Vanilla single-pass"].append(metric_row(
            qid, initial, initial_packet, gold[qid], 1, "single_pass"
        ))
        if initial_refusal:
            gate_only = metric_row(
                qid, initial, initial_packet, gold[qid], 1, "initial_refusal"
            )
        elif initial_supported:
            gate_only = metric_row(
                qid, initial, initial_packet, gold[qid], 1, "initial_admit"
            )
        else:
            gate_only = metric_row(
                qid, initial, initial_packet, gold[qid], 1,
                "no_commit_output_fallback", no_commit=True,
            )
        policies["Gate-only"].append(gate_only)

        if initial_refusal:
            gate_repair = metric_row(
                qid, initial, initial_packet, gold[qid], 1, "initial_refusal"
            )
        elif initial_supported:
            gate_repair = metric_row(qid, initial, initial_packet, gold[qid], 1, "initial_admit")
        elif not repair_refusal and repair_supported:
            gate_repair = metric_row(qid, repair, repair_packet, gold[qid], 2, "repair_admit")
        else:
            gate_repair = metric_row(
                qid, initial, initial_packet, gold[qid], 2,
                "no_commit_output_fallback", no_commit=True,
            )
        policies["ADMIT"].append(gate_repair)

        if initial_refusal:
            learned = metric_row(
                qid, initial, initial_packet, gold[qid], 1, "initial_refusal"
            )
        elif not routed:
            learned = metric_row(qid, initial, initial_packet, gold[qid], 1, "initial_admit")
        elif not repair_refusal and repair_supported:
            learned = metric_row(qid, repair, repair_packet, gold[qid], 2, "repair_admit")
        else:
            learned = metric_row(
                qid, initial, initial_packet, gold[qid], 2,
                "no_commit_output_fallback", no_commit=True,
            )
        policies["Learned-router diagnostic"].append(learned)

        if initial_refusal:
            final_support = metric_row(
                qid, initial, initial_packet, gold[qid], 1, "initial_refusal"
            )
        elif initial_supported:
            final_support = metric_row(qid, initial, initial_packet, gold[qid], 1, "initial_supported")
        elif not repair_refusal and repair_supported:
            final_support = metric_row(qid, repair, repair_packet, gold[qid], 2, "repair_admit")
        else:
            final_support = metric_row(qid, initial, initial_packet, gold[qid], 2, "fallback_initial")
        policies["Final-support + retry"].append(final_support)

        if not repair_refusal and repair_supported:
            every = metric_row(qid, repair, repair_packet, gold[qid], 2, "repair_admit")
        elif initial_refusal:
            every = metric_row(qid, initial, initial_packet, gold[qid], 2, "initial_refusal_fallback")
        else:
            every = metric_row(
                qid, initial, initial_packet, gold[qid], 2,
                "no_commit_output_fallback", no_commit=True,
            )
        policies["Every-q repair"].append(every)
    return dict(policies)


def aggregate(rows: list[dict]) -> dict:
    keys = ("EM", "F1", "ANLS", "unsupNR_strict", "value_supported_EM", "calls", "no_commit")
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return {"n": len(rows), **result, "routes": dict(Counter(row["route"] for row in rows))}


def bootstrap(left: list[dict], right: list[dict], key: str) -> dict:
    left_by, right_by = {row["qid"]: row for row in left}, {row["qid"]: row for row in right}
    qids = sorted(set(left_by) & set(right_by))
    delta = np.asarray([left_by[qid][key] - right_by[qid][key] for qid in qids], dtype=float)
    rng = np.random.default_rng(42)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sample = rng.integers(0, len(delta), len(delta))
        draws[index] = delta[sample].mean()
    return {
        "n": len(qids),
        "mean": float(delta.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "draws": BOOTSTRAP_DRAWS,
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    cal_answers, cal_packets, report_answers, report_packets = partition_inputs()
    candidate_complete_qids = set(cal_answers) | set(report_answers)
    if len(gold) != 2441:
        raise RuntimeError(f"expected 2,441 released validation qids, observed {len(gold)}")
    if len(candidate_complete_qids) != 2324:
        raise RuntimeError(
            f"expected 2,324 candidate-complete qids, observed {len(candidate_complete_qids)}"
        )
    if not candidate_complete_qids <= set(gold):
        raise RuntimeError("candidate-complete qids are not a subset of released validation")
    excluded_incomplete_qids = set(gold) - candidate_complete_qids
    if len(excluded_incomplete_qids) != 117:
        raise RuntimeError(
            f"expected 117 excluded incomplete-candidate qids, observed {len(excluded_incomplete_qids)}"
        )
    training_rows = build_training_rows(cal_answers, cal_packets, gold)
    scores, router = train_router(
        training_rows, cal_answers, cal_packets, report_answers, report_packets
    )
    policies = evaluate(scores, router["threshold"], report_answers, report_packets, gold)
    metrics = {name: aggregate(rows) for name, rows in policies.items()}
    comparisons = {}
    for comparator in ("Vanilla single-pass", "Final-support + retry", "Every-q repair"):
        comparisons[f"ADMIT_minus_{comparator}"] = {
            key: bootstrap(policies["ADMIT"], policies[comparator], key)
            for key in ("EM", "F1", "ANLS", "value_supported_EM", "calls", "no_commit")
        }
    comparisons["Learned-router_diagnostic_minus_ADMIT"] = {
        key: bootstrap(policies["Learned-router diagnostic"], policies["ADMIT"], key)
        for key in ("EM", "F1", "ANLS", "value_supported_EM", "calls", "no_commit")
    }

    per_qid_path = HERE / "deployable_router_per_qid.jsonl"
    policy_by_qid = {
        name: {row["qid"]: row for row in rows} for name, rows in policies.items()
    }
    with per_qid_path.open("w", encoding="utf-8") as handle:
        for qid in sorted(report_answers):
            handle.write(json.dumps({
                "qid": qid,
                "initial_packet_sha256": hashlib.sha256(report_packets[qid].encode()).hexdigest(),
                "router_score": scores[qid],
                "router_triggered": not is_refusal(report_answers[qid]) and (
                    scores[qid] < router["threshold"] or not lexical_support(
                        report_answers[qid], report_packets[qid]
                    )
                ),
                "policies": {name: rows[qid] for name, rows in policy_by_qid.items()},
            }, ensure_ascii=False) + "\n")

    result = {
        "status": "PASS",
        "contract": {
            "dataset": "M3DocVQA",
            "released_validation_qids": len(gold),
            "candidate_complete_qids": len(candidate_complete_qids),
            "excluded_incomplete_candidate_qids": len(excluded_incomplete_qids),
            "population_scope": "candidate-complete subset; 117 incomplete-candidate qids excluded before splitting",
            "calibration_qids": len(cal_answers),
            "reporting_qids": len(report_answers),
            "partition_overlap": 0,
            "reader": "Qwen3-VL-8B-Instruct for initial read and RepairRead",
            "packet": "exact post-truncation reader-visible OCR/text",
            "runtime_features": FEATURES,
            "runtime_counterfactual_outputs": 0,
            "training_augmentation": "counterfactual candidate answers from calibration qids only; every feature recomputed against the exact calibration packet",
            "model_selection": "five outer calibration folds; early stopping only on an inner calibration fold; outer-test labels never enter fitting or stopping",
            "threshold": "median of calibration baseline-row OOF scores",
            "reporting_labels_used_for_training_or_threshold": False,
            "metric": "dynamic-aligned bidirectional lowercase/whitespace-normalized substring EM for every output; fixed refusals are typed STATUS records and do not trigger RepairRead; complementary F1 and ANLS use short-answer extraction plus punctuation/article removal",
        },
        "router": router,
        "metrics": metrics,
        "paired_deltas": comparisons,
        "artifact_sha256": {
            "calibration_predictions": sha256(SOURCE / "calibration_strict_pred.jsonl"),
            "calibration_packets": sha256(SOURCE / "calibration_strict_packet.jsonl"),
            "reporting_initial_predictions": sha256(REPORT_SOURCE / "baseline_strict_pred.jsonl"),
            "reporting_initial_packets": sha256(REPORT_SOURCE / "baseline_strict_packets.jsonl"),
            "reporting_repair_predictions": sha256(REPORT_SOURCE / "repair_guess_pred.jsonl"),
            "reporting_repair_packets": sha256(REPORT_SOURCE / "repair_guess_packets.jsonl"),
            "per_qid": sha256(per_qid_path),
        },
    }
    result_path = HERE / "deployable_router_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for name, row in metrics.items():
        print(
            f"{name:24s} EM={100 * row['EM']:6.2f} F1={100 * row['F1']:6.2f} "
            f"ANLS={100 * row['ANLS']:6.2f} vEM={100 * row['value_supported_EM']:6.2f} "
            f"calls={row['calls']:.3f} NC={100 * row['no_commit']:6.2f}"
        )
    print(f"threshold={router['threshold']:.9f} iterations={router['final_iterations']}")
    print(result_path)


if __name__ == "__main__":
    main()
