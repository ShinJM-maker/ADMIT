#!/usr/bin/env python3
"""Phase 30 - CAVEC-MFC (Memory-Fail-Closed / Output-Fail-Open).

Extends Phase 29 (6 arms) by adding a 7th arm `cavec_mfc` that decouples the
memory-write fallback from the final-output fallback at every CAVEC commit
point. When the substring gate AND repair both fail and the baseline candidate
is also unsupported:
  - Paper-policy CAVEC (`cavec_paper`) writes the unsupported baseline raw
    into public memory (contamination chain remains).
  - CAVEC-MFC writes a quarantine sentinel "[NO_COMMIT]" into public memory
    (memory-fail-closed) but the FINAL output may still fall back to the
    vanilla baseline (output-fail-open) so raw EM is preserved.

Arms (7):
  1. vanilla
  2. vanilla_hard_refuse  (post-hoc derived from vanilla)
  3. final_support_retry  (paper Table 4 final-surface fail-open)
  4. cavec_paper          (CAVEC dynamic fail-open +H1, commit-time substring)
  5. cavec_h1_hard_refuse (CAVEC fail-closed)
  6. retrained_router_commit_time  (Phase 26 LightGBM router @ commit-time)
  7. cavec_mfc            (CAVEC Memory-Fail-Closed / Output-Fail-Open, NEW)

Critical pairwise comparison: cavec_paper vs retrained_router_commit_time on
intermediate-memory contamination metric and on strict-grounded EM.

This script is sharded — one process per GPU, each loads its own vLLM
Qwen3-VL-8B-Instruct (TP=1) and processes its slice of the qid set. The
parent launch.sh launches 4 shards in parallel.

Outputs (per shard):
    react_vanilla_pred.shard{i}.jsonl
    react_cavec_pred.shard{i}.jsonl
    react_per_qid.shard{i}.jsonl
    planrag_vanilla_pred.shard{i}.jsonl
    planrag_cavec_pred.shard{i}.jsonl
    planrag_per_qid.shard{i}.jsonl
    autogen_vanilla_pred.shard{i}.jsonl
    autogen_cavec_pred.shard{i}.jsonl
    autogen_per_qid.shard{i}.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, re, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HIKEY_CODE = str(Path(__file__).resolve().parent)  # reader/BM25 helpers shipped next to this file
sys.path.insert(0, HIKEY_CODE)

from retrieve_hikey_stage2 import BM25Index  # noqa: E402
from run_reader_qwen25vl import (  # noqa: E402
    PROMPT_TAIL, PROMPT_VARIANTS,
    build_chunks_index, build_docs_index, build_cards_index,
    build_parent_to_cards, build_evidence_pack,
)


# ----------------------------- Constants -----------------------------
HW = os.environ.get("ADMIT_WORK", str(Path(__file__).resolve().parents[1] / "work"))
INDEX_ROOT = os.environ.get("ADMIT_INDEX", str(Path(__file__).resolve().parents[1] / "data/index"))
SEC_DIR = f"{INDEX_ROOT}/M3DocVQA/sec_index_md"
DATA = os.environ.get("ADMIT_DATA", str(Path(__file__).resolve().parents[1] / "data/datasets"))
PHASE21 = f"{HW}/cavec_v4/phase21_qwen3vl32b_20260503"
PHASE25 = f"{HW}/dynamic"

REFUSAL_PHRASE = "do not have enough information"

BM25_TOK = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\uac00-\ud7a3]+", re.UNICODE)


# ----------------------------- Utilities -----------------------------
def tok(s: str) -> List[str]:
    return [t.lower() for t in BM25_TOK.findall(s or "")]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower().strip())


def substr_em(pred: str, gold: str) -> int:
    if is_refusal(pred):
        return 0
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    if not pred_n or not gold_n:
        return 0
    return 1 if gold_n in pred_n or pred_n in gold_n else 0


def is_refusal(ans: str) -> int:
    n = normalize(ans)
    if not n:
        return 1
    return 1 if REFUSAL_PHRASE in n else 0


def support_in_pack(ans: str, pack_text: str) -> int:
    if not ans:
        return 0
    n = normalize(ans)
    if not n or REFUSAL_PHRASE in n:
        return 0
    return 1 if n in normalize(pack_text) else 0


def evidence_packet_sha256(reader_ctx: str) -> str:
    return hashlib.sha256((reader_ctx or "").encode("utf-8")).hexdigest()


# ----------------------------- Data loading -----------------------------
def load_chunk_text() -> Dict[Tuple[str, int], List[str]]:
    chunk_text = defaultdict(list)
    for line in open(f"{INDEX_ROOT}/M3DocVQA/chunks.jsonl"):
        d = json.loads(line)
        doc = d.get("doc_id"); ps = d.get("page_span")
        if not doc:
            continue
        if isinstance(ps, dict):
            s = int(ps.get("start", 1)); e = int(ps.get("end", s))
        elif isinstance(ps, list) and len(ps) >= 2:
            s = int(ps[0]); e = int(ps[1])
        elif isinstance(ps, list) and len(ps) == 1:
            s = e = int(ps[0])
        else:
            s = e = 1
        t = d.get("text") or ""
        for p in range(s, e + 1):
            chunk_text[(doc, p)].append(t)
    return chunk_text


def load_bm25_and_cards():
    bm25_path = f"{SEC_DIR}/sectioncard_bm25_index.pkl"
    cards_path = f"{SEC_DIR}/section_cards_index.jsonl"
    sys.modules["__main__"].BM25Index = BM25Index
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    cards = []
    for line in open(cards_path):
        if line.strip():
            cards.append(json.loads(line))
    return bm25, cards


def build_pack_for_query(q: str, bm25, cards, _chunk_text, top_k: int = 10):
    """BM25 retrieval -> section-card and document candidates.

    Evidence text is serialized only by ``ReaderVLLM.reader_call`` after packing
    and truncation. Keeping retrieval free of a second text serialization makes
    it impossible for admission to consume a wider, reader-invisible packet.
    """
    scores = bm25.get_scores(tok(q))
    top_idx = scores.argsort()[::-1][:top_k]
    sec_cands = []
    for rank, idx in enumerate(top_idx):
        c = cards[idx]
        sec_cands.append({
            "card_id": c["card_id"],
            "chunk_id": c["card_id"],
            "doc_id": c["doc_id"],
            "card_type": c.get("card_type"),
            "path_str": c.get("path_str"),
            "page_span": c.get("page_span"),
            "chunk_ids": c.get("chunk_ids"),
            "text": c.get("embed_text") or c.get("bm25_text") or "",
            "final": float(scores[idx]),
        })
    doc_ids_seen = []
    for sc in sec_cands:
        if sc["doc_id"] not in doc_ids_seen:
            doc_ids_seen.append(sc["doc_id"])
    doc_cands = [{"doc_id": d, "score": 1.0/(i+1), "rank": i+1}
                 for i, d in enumerate(doc_ids_seen[:10])]

    return sec_cands, doc_cands


# ----------------------------- vLLM Reader wrapper -----------------------------
class ReaderVLLM:
    """Single-GPU vLLM Qwen3-VL-8B-Instruct text-only reader with per-call generate.

    All calls use temperature=0, max_new_tokens=64, max_images_total=0 (text-only).
    """

    def __init__(self, model_path: str, max_model_len: int = 16384,
                 token_budget: int = 14000, gpu_mem_util: float = 0.85):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.token_budget = token_budget
        self.max_images_total = 0  # text-only
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},  # vLLM still wants >=1 even if unused
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
        # Indexes for build_evidence_pack
        self._chunks_idx = build_chunks_index("M3DocVQA")
        self._cards_idx = build_cards_index("M3DocVQA")
        self._docs_idx = build_docs_index("M3DocVQA")
        self._parent_to_cards = build_parent_to_cards(self._cards_idx)

    def make_prompt(self, question: str, sec_cands: List[Dict], prompt_variant: str = "strict",
                    top_k_cards: int = 10) -> Tuple[str, str, int]:
        """Build the full prompt string + return (prompt_text, ctx_text, used_tokens).

        Text-only: max_images_total=0, so build_evidence_pack returns no images.
        """
        cards = (sec_cands or [])[:top_k_cards]
        ctx_text, images, used_tokens = build_evidence_pack(
            cards, self._chunks_idx, self._cards_idx, self._docs_idx,
            parent_to_cards=self._parent_to_cards,
            semantic_visual_by_doc={},
            token_budget=self.token_budget,
            image_token_cost=300,
            max_images_total=self.max_images_total,
            tokenizer=self.tokenizer,
            enable_sibling=True,
            enable_semantic=True,
        )
        prompt_head = PROMPT_VARIANTS[prompt_variant]
        text_prompt = prompt_head.format(question=question) + ctx_text + PROMPT_TAIL
        # Strip image markers since text-only
        text_prompt = text_prompt.replace("<|image|>\n", "").replace("<|image|>", "")
        # Apply chat template
        messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
        try:
            templated = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            templated = text_prompt
        return templated, ctx_text, used_tokens

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Greedy decode a batch of prompts. Returns list of answer strings."""
        if not prompts:
            return []
        request_inputs = [{"prompt": p} for p in prompts]
        try:
            outputs = self.llm.generate(request_inputs, self.sampling, use_tqdm=False)
        except (ValueError, RuntimeError) as e:
            print(f"[reader] batch generate failed ({type(e).__name__}: {e}), returning empty",
                  flush=True)
            return ["" for _ in prompts]
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

    def reader_call(self, question: str, sec_cands: List[Dict], prompt_variant: str = "strict",
                    top_k_cards: int = 10) -> Tuple[str, str, int]:
        """Single reader call. Returns (answer, ctx_text, used_tokens)."""
        prompt, ctx_text, used = self.make_prompt(question, sec_cands, prompt_variant, top_k_cards)
        ans = self.generate_batch([prompt])[0]
        return ans, ctx_text, used

    def reasoning_call(self, prompt_text: str, max_new_tokens: int = 96) -> str:
        """Free-form reasoning call (no chat template munging)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        try:
            templated = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            templated = prompt_text
        sampling = self.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        outputs = self.llm.generate([{"prompt": templated}], sampling, use_tqdm=False)
        return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""


# ----------------------------- CAVEC pre-memory-write hook -----------------------------
_YN_LEAD = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|"
    r"should|shall|may|might|am)\b",
    flags=re.IGNORECASE,
)


def is_yes_no_question(question: str) -> bool:
    if not question:
        return False
    return bool(_YN_LEAD.match(question.strip()))


def is_yes_no_value(v: str) -> bool:
    if not v:
        return False
    return v.strip().lower().rstrip(".") in {"yes", "no", "true", "false"}


def cavec_admit(commit_value: str, evidence_pack_text: str,
                question: str = "") -> Tuple[bool, str]:
    """Phase 1 of the CAVEC hook: substring-support gate, with H1 yes/no
    entailment carve-out (Phase 25 stage-D diagnosis fix).
    """
    if is_refusal(commit_value):
        return True, "refusal_admitted"
    if support_in_pack(commit_value, evidence_pack_text):
        return True, "S_str_positive"
    if (is_yes_no_question(question) and is_yes_no_value(commit_value)
            and len(evidence_pack_text or "") >= 200):
        return True, "yesno_entailment_admitted"
    return False, "S_str_zero"


# Strict support-anchored repair prompt, applied as a one-shot RepairRead at
# the public-commit boundary if S_str=0.
REPAIR_PROMPT_HEAD = (
    "[System Instruction]\n"
    "You are an AI assistant that extracts the answer from the provided documents.\n"
    "STRICT RULE: Your answer MUST be a verbatim substring of the [Context] below.\n"
    "- If the context contains the answer, copy the shortest exact phrase verbatim.\n"
    "- EXCEPTION: If the question is a yes/no question (e.g., starts with Is/Are/Was/Were/Do/Does/Did/Has/Have/Can/Could/Will/Would/Should), you MAY output exactly 'Yes' or 'No' if the context entails or contradicts the claim. Do not output Yes/No otherwise.\n"
    "- If you cannot find a verbatim answer in the context (and it is not a yes/no question), output exactly: I do not have enough information to answer.\n"
    "- Do NOT paraphrase. Do NOT add any words not in the context (the yes/no exception aside).\n"
    "- Output ONLY the answer phrase, nothing else.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)


def repair_read(reader: ReaderVLLM, question: str, sec_cands: List[Dict],
                top_k_cards: int = 10) -> Tuple[str, int]:
    """Bounded RepairRead — single reader call with the strict support-anchored
    prompt. Returns (answer, n_calls_used)."""
    cards = (sec_cands or [])[:top_k_cards]
    ctx_text, _, used_tokens = build_evidence_pack(
        cards, reader._chunks_idx, reader._cards_idx, reader._docs_idx,
        parent_to_cards=reader._parent_to_cards,
        semantic_visual_by_doc={},
        token_budget=reader.token_budget,
        image_token_cost=300,
        max_images_total=0,
        tokenizer=reader.tokenizer,
        enable_sibling=True,
        enable_semantic=True,
    )
    text_prompt = REPAIR_PROMPT_HEAD.format(question=question) + ctx_text + PROMPT_TAIL
    text_prompt = text_prompt.replace("<|image|>\n", "").replace("<|image|>", "")
    messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
    try:
        templated = reader.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        templated = text_prompt
    ans = reader.generate_batch([templated])[0]
    return ans, 1


def cavec_hook(reader: ReaderVLLM, candidate_value: str,
               question: str, sec_cands: List[Dict], pack_text: str,
               baseline_value: Optional[str] = None,
               top_k_cards: int = 10,
               fail_closed: bool = True,
               cached_repair: Optional[Tuple[str, int]] = None) -> Tuple[str, Dict]:
    """Unified CAVEC pre-memory-write hook for paper-policy / hard-refuse / final-support-retry.

    `fail_closed`: if RepairRead unsupported AND baseline unsupported, refuse (True)
                   or fall back to baseline raw (False = paper Table 4 fail-open spec).
    `cached_repair`: optional (repair_val, n_calls) tuple to share RepairRead across arms.
    """
    log = {"candidate": candidate_value, "n_repair_calls": 0, "admit_reason": None,
           "repair_value": None, "final_source": None}
    admit, reason = cavec_admit(candidate_value, pack_text, question=question)
    log["admit_reason"] = reason
    if admit:
        log["final_source"] = "candidate"
        return candidate_value, log
    # Rejected. Fire one RepairRead (or use cached).
    if cached_repair is not None:
        repair_val, n_calls = cached_repair
    else:
        repair_val, n_calls = repair_read(reader, question, sec_cands, top_k_cards)
    log["repair_value"] = repair_val
    log["n_repair_calls"] = n_calls
    repair_admit, _ = cavec_admit(repair_val, pack_text, question=question)
    if not is_refusal(repair_val) and repair_admit:
        log["final_source"] = "repair"
        return repair_val, log
    # RepairRead also unsupported -> fallback to baseline if supplied; else refuse.
    baseline_admit = False
    if baseline_value is not None:
        if is_refusal(baseline_value):
            baseline_admit = True
        else:
            baseline_admit, _ = cavec_admit(baseline_value, pack_text, question=question)
    if baseline_value is not None and baseline_admit:
        log["final_source"] = "fallback_baseline"
        return baseline_value, log
    if not fail_closed and baseline_value is not None:
        # Paper Table 4 fail-open: fall back to baseline raw even if unsupported.
        log["final_source"] = "fallback_baseline_unsupported"
        return baseline_value, log
    log["final_source"] = "fallback_refusal"
    return "I do not have enough information to answer.", log


# ----------------------------- Phase 26 retrained router (commit-time) -----------------------------
# Phase 26 LightGBM lambdarank, trained on final-row baseline_text rows, scored
# by 17-feature gbdt_extended set (group features collapse to singleton at
# deployment commit-time — matches the natural deployment contract).
import math, statistics  # noqa: E402

PHASE26_MODEL = os.path.join(HW, "router/f_phi_FA.lgb")  # optional SupportRouter checkpoint (App. D.3); unused by the final policy

# Tau prime from Phase 26 results.json — the L_4way deployment threshold.
# Lowest-score qids trigger; this is what was used in Phase 26 evaluation.
PHASE26_TAU_L4WAY = -0.055362327118893696

FEATURES_17 = [
    "f_S_str", "f_S_tok", "f_S_ent", "f_S_non", "f_S_con", "f_H_unsupported",
    "f_answer_length", "f_refusal_flag", "f_unsupported_nonrefusal_flag",
    "f_answer_agreement_top_frac", "f_answer_cluster_top_size",
    "f_group_entropy", "f_support_score",
    "f_support_rank_in_group", "f_support_z_score",
    "f_slot_swapped", "f_kind_is_swap",
]

_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_DATE_RE = re.compile(r"\b(?:\d{4}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def _exact_str(a: str, c: str) -> float:
    n = normalize(a)
    return 1.0 if n and n in normalize(c) else 0.0


def _tok_recall(a: str, c: str) -> float:
    A = set(tok(a))
    if not A:
        return 0.0
    return len(A & set(tok(c))) / len(A)


def _ent_match(a: str, c: str) -> float:
    A = set(_NUM_RE.findall(a)) | set(_DATE_RE.findall(a)) | set(_CAP_RE.findall(a))
    if not A:
        return 0.0
    C = set(_NUM_RE.findall(c)) | set(_DATE_RE.findall(c)) | set(_CAP_RE.findall(c))
    return len(A & C) / len(A)


def commit_time_features_17(candidate_value: str, pack_text: str,
                             baseline_value: str = "") -> List[float]:
    """Compute the 17 deployment-available features for a single commit
    candidate at commit-time.

    Group-relative features (rank, z-score, agreement, cluster, entropy)
    collapse to singleton-group constants since at commit-time we have ONE
    candidate. We use {candidate, baseline_value} as the local agreement group
    when baseline_value is provided AND non-trivially differs (lets us include
    baseline/commit agreement as required by the feature contract).
    f_kind_is_swap=0, f_slot_swapped=-1 (matches the baseline_text training
    convention; we are scoring the deployment baseline candidate).
    """
    ans = candidate_value or ""
    ctx = pack_text or ""
    s_str = _exact_str(ans, ctx)
    s_tok = _tok_recall(ans, ctx)
    s_ent = _ent_match(ans, ctx)
    s_non = 0.0 if is_refusal(ans) else 1.0
    n_ans = normalize(ans)

    # Build local "group" {candidate, baseline_value} for agreement metrics
    siblings = [n_ans]
    n_base = normalize(baseline_value)
    if n_base and n_base != n_ans:
        siblings.append(n_base)
    siblings = [s for s in siblings if s]
    if siblings:
        from collections import Counter as _C
        cnt = _C(siblings)
        total = sum(cnt.values())
        cluster_top = cnt.most_common(1)[0][1]
        agree_top = cluster_top / total
        entropy = -sum((c / total) * math.log(c / total + 1e-9) for c in cnt.values())
    else:
        cluster_top = 0; agree_top = 0.0; entropy = 0.0
    # s_con = fraction of siblings whose normalized answer == this candidate
    s_con = sum(1 for s in siblings if s == n_ans) / max(1, len(siblings)) if n_ans else 0.0

    # Group-relative support features (need baseline support for ranking)
    base_s_str = _exact_str(baseline_value, ctx) if baseline_value else s_str
    sup_vals = [s_str]
    if baseline_value and normalize(baseline_value) != n_ans:
        sup_vals.append(base_s_str)
    sup_mean = statistics.mean(sup_vals) if sup_vals else 0.0
    sup_std = statistics.stdev(sup_vals) if len(sup_vals) > 1 else 0.0
    # rank: 1 if this candidate has highest s_str; else 2
    rank_in_group = 1 if s_str >= max(sup_vals) - 1e-9 else 2
    z_score = (s_str - sup_mean) / (sup_std + 1e-9)

    # h_unsup: 1 if candidate matches baseline AND candidate unsupported
    h_unsup = 1.0 if (n_ans == n_base and s_str == 0.0 and n_ans) else 0.0
    refusal_flag = float(is_refusal(ans))
    unsup_nr_flag = float(s_str == 0 and not is_refusal(ans) and s_non == 1)

    feats = {
        "f_S_str": s_str,
        "f_S_tok": s_tok,
        "f_S_ent": s_ent,
        "f_S_non": s_non,
        "f_S_con": s_con,
        "f_H_unsupported": h_unsup,
        "f_answer_length": float(len(ans)),
        "f_refusal_flag": refusal_flag,
        "f_unsupported_nonrefusal_flag": unsup_nr_flag,
        "f_answer_agreement_top_frac": agree_top,
        "f_answer_cluster_top_size": float(cluster_top),
        "f_group_entropy": entropy,
        "f_support_score": s_str,
        "f_support_rank_in_group": float(rank_in_group),
        "f_support_z_score": z_score,
        "f_slot_swapped": -1.0,
        "f_kind_is_swap": 0.0,
    }
    return [feats[k] for k in FEATURES_17]


_ROUTER_MODEL = None
# Calibrated tau' (per-shard, set after a calibration pass over the first
# ~25 qids' vanilla commits to match CAVEC's natural ~10% trigger rate).
# Default: use Phase 26 OOF tau as fallback. The actual tau is overwritten
# by main() after calibration.
_ROUTER_TAU = PHASE26_TAU_L4WAY
# Target trigger rate (matches CAVEC commit-time trigger rate observed in
# Phase 25c: react=8.8%, planrag=12.7%, autogen=10.1%, mean ~10%).
ROUTER_TARGET_TRIGGER_RATE = 0.10
# Collect scores during calibration pass so we can pick tau as quantile.
_CALIB_SCORES: List[float] = []
_CALIB_MODE = False


def _load_router():
    global _ROUTER_MODEL
    if _ROUTER_MODEL is None:
        import lightgbm as lgb
        _ROUTER_MODEL = lgb.Booster(model_file=PHASE26_MODEL)
        print(f"[router] loaded {PHASE26_MODEL} ({_ROUTER_MODEL.num_feature()} feats, "
              f"{_ROUTER_MODEL.current_iteration()} iters)", flush=True)
    return _ROUTER_MODEL


def router_score(candidate_value: str, pack_text: str,
                 baseline_value: str = "") -> float:
    import numpy as _np
    m = _load_router()
    feats = commit_time_features_17(candidate_value, pack_text, baseline_value)
    arr = _np.array([feats], dtype=_np.float32)
    s = float(m.predict(arr)[0])
    if _CALIB_MODE:
        _CALIB_SCORES.append(s)
    return s


def set_router_tau(tau: float) -> None:
    global _ROUTER_TAU
    _ROUTER_TAU = tau
    print(f"[router] tau' set to {tau:.6f}", flush=True)


def get_router_tau() -> float:
    return _ROUTER_TAU


def calibrate_router_tau(target_trigger_rate: float = ROUTER_TARGET_TRIGGER_RATE) -> float:
    """Pick tau as the target_trigger_rate-quantile of collected calibration scores.
    A score is below tau iff the trigger fires; we want fraction(scores < tau)
    == target_trigger_rate, so tau = quantile(scores, target_trigger_rate).
    """
    if not _CALIB_SCORES:
        print(f"[router] no calibration scores collected; keeping default tau="
              f"{_ROUTER_TAU:.6f}", flush=True)
        return _ROUTER_TAU
    import numpy as _np
    arr = _np.array(_CALIB_SCORES, dtype=_np.float64)
    tau = float(_np.quantile(arr, target_trigger_rate))
    print(f"[router] calibration: n_scores={len(arr)}, mean={arr.mean():.4f} "
          f"std={arr.std():.4f} q10={tau:.4f} q50={_np.median(arr):.4f}", flush=True)
    set_router_tau(tau)
    return tau


def retrained_router_hook(reader: ReaderVLLM, candidate_value: str,
                           question: str, sec_cands: List[Dict], pack_text: str,
                           baseline_value: Optional[str] = None,
                           top_k_cards: int = 10,
                           tau_prime: Optional[float] = None,
                           cached_repair: Optional[Tuple[str, int]] = None
                           ) -> Tuple[str, Dict]:
    """Phase 29 retrained-router placed at commit-time.

    Logic mirrors paper-CAVEC fail-open exactly, EXCEPT the gate is the Phase
    26 LightGBM score (score < tau_prime triggers repair) instead of CAVEC's
    substring-support binary gate.

    On trigger:
      - Fire ONE bounded RepairRead.
      - If repair score >= tau_prime AND non-refusal -> commit repair.
      - Else fall back to baseline_value (fail-open mirror).
    """
    if tau_prime is None:
        tau_prime = get_router_tau()
    log = {"candidate": candidate_value, "n_repair_calls": 0,
           "router_score_cand": None, "router_score_repair": None,
           "tau_prime": tau_prime, "repair_value": None,
           "final_source": None, "admit_reason": None}

    # Refusal admits unconditionally (mirrors CAVEC behavior)
    if is_refusal(candidate_value):
        log["admit_reason"] = "refusal_admitted"
        log["final_source"] = "candidate"
        return candidate_value, log

    s_cand = router_score(candidate_value, pack_text, baseline_value or "")
    log["router_score_cand"] = s_cand

    if s_cand >= tau_prime:
        log["admit_reason"] = "router_score_passed"
        log["final_source"] = "candidate"
        return candidate_value, log

    # Trigger RepairRead
    if cached_repair is not None:
        repair_val, n_calls = cached_repair
    else:
        repair_val, n_calls = repair_read(reader, question, sec_cands, top_k_cards)
    log["repair_value"] = repair_val
    log["n_repair_calls"] = n_calls
    log["admit_reason"] = "router_score_below_tau"

    if not is_refusal(repair_val):
        s_rep = router_score(repair_val, pack_text, baseline_value or "")
        log["router_score_repair"] = s_rep
        if s_rep >= tau_prime:
            log["final_source"] = "repair"
            return repair_val, log

    # Fail-open: fall back to baseline (mirror paper-CAVEC fail-open)
    if baseline_value is not None:
        log["final_source"] = "fallback_baseline_unsupported"
        return baseline_value, log
    log["final_source"] = "fallback_refusal"
    return "I do not have enough information to answer.", log


def vanilla_hard_refuse(candidate_value: str, pack_text: str, question: str = "") -> Tuple[str, str]:
    """Post-hoc hard-refuse on vanilla output: if substring-support fails, refuse.
    Returns (committed_value, source_label)."""
    if is_refusal(candidate_value):
        return candidate_value, "refusal"
    admit, reason = cavec_admit(candidate_value, pack_text, question=question)
    if admit:
        return candidate_value, f"admit_{reason}"
    return "I do not have enough information to answer.", "post_hoc_refuse"


def final_support_retry_hook(reader: ReaderVLLM, candidate_value: str,
                              question: str, sec_cands: List[Dict], pack_text: str,
                              top_k_cards: int = 10,
                              cached_repair: Optional[Tuple[str, int]] = None
                              ) -> Tuple[str, Dict]:
    """Paper Table 4 baseline: final-answer support-check + one retry, fail-open.
    If candidate supported -> commit. Else fire ONE RepairRead. If repair supported
    and non-refusal -> commit repair. Else fall back to ORIGINAL Vanilla raw answer.
    """
    log = {"candidate": candidate_value, "n_repair_calls": 0, "repair_value": None,
           "final_source": None}
    if is_refusal(candidate_value):
        log["final_source"] = "refusal"
        return candidate_value, log
    if support_in_pack(candidate_value, pack_text):
        log["final_source"] = "candidate_supported"
        return candidate_value, log
    if cached_repair is not None:
        repair_val, n_calls = cached_repair
    else:
        repair_val, n_calls = repair_read(reader, question, sec_cands, top_k_cards)
    log["repair_value"] = repair_val
    log["n_repair_calls"] = n_calls
    if not is_refusal(repair_val) and support_in_pack(repair_val, pack_text):
        log["final_source"] = "repair_supported"
        return repair_val, log
    log["final_source"] = "fallback_vanilla_raw_unsupported"
    return candidate_value, log


# ----------------------------- CAVEC-MFC hook -----------------------------
NO_COMMIT_SENTINEL = "[NO_COMMIT]"


def public_memory_record(value: str, evidence_packet_text: Optional[str] = None) -> Dict[str, Any]:
    """Create the typed public-memory record used by downstream consumers.

    Refusal and quarantine values remain available to the audit trail, but
    only ``FACT`` records belong to the factual projection consumed by
    retrieval, planning, and synthesis.
    """
    value = str(value or "")
    if value == NO_COMMIT_SENTINEL:
        return {"record_type": "STATUS", "status": "NO_COMMIT", "value": value}
    if is_refusal(value):
        return {"record_type": "STATUS", "status": "REFUSAL", "value": value}
    record: Dict[str, Any] = {
        "record_type": "FACT", "status": "COMMIT", "value": value,
    }
    if evidence_packet_text is not None:
        record["lexically_supported"] = bool(support_in_pack(value, evidence_packet_text))
        record["evidence_packet_sha256"] = evidence_packet_sha256(evidence_packet_text)
    return record


def factual_projection(record: Dict[str, Any]) -> Optional[str]:
    """Return a value only when the typed record is factual."""
    if record.get("record_type") != "FACT":
        return None
    return str(record.get("value") or "")


def cavec_mfc_hook(reader: ReaderVLLM, candidate_value: str,
                   question: str, sec_cands: List[Dict], pack_text: str,
                   baseline_value: Optional[str] = None,
                   top_k_cards: int = 10,
                   cached_repair: Optional[Tuple[str, int]] = None
                   ) -> Tuple[str, str, Dict]:
    """CAVEC Memory-Fail-Closed / Output-Fail-Open hook.

    Mirrors `cavec_hook` (with H1 yes/no carve-out) on the *output* surface
    (output-fail-open == paper-policy fail-open: if substring fail, repair
    fail, and baseline unsupported -> output baseline raw). On the *memory*
    surface, however, the fail-open branch is blocked: instead of writing the
    unsupported baseline raw into public memory, MFC writes the quarantine
    sentinel `[NO_COMMIT]`.

    Returns (memory_value, output_value, log).
      - memory_value: what is written into the trajectory state (scratchpad /
        sub_logs / conversation `committed_answer`). For per-turn / per-sub-Q
        commits, only this matters (it gets read by downstream synthesis).
      - output_value: what is committed as the FINAL answer for grading EM.
        For per-turn commits, ignore this. For the final hook, this is the
        per-qid prediction.

    Behavior tree:
      candidate refusal -> memory=candidate, output=candidate (paper convention)
      candidate admit (S_str>0 OR yes/no carve-out) -> both = candidate
      else fire RepairRead
        if repair admit AND non-refusal -> both = repair
        else baseline admit -> both = baseline
        else (baseline unsupported, paper fail-open branch):
          memory = NO_COMMIT_SENTINEL  (MFC: refuse to contaminate)
          output = baseline_value      (output-fail-open: preserve raw EM)
    """
    log = {"candidate": candidate_value, "n_repair_calls": 0,
           "admit_reason": None, "repair_value": None,
           "final_source": None, "memory_source": None}
    admit, reason = cavec_admit(candidate_value, pack_text, question=question)
    log["admit_reason"] = reason
    if admit:
        log["final_source"] = "candidate"
        log["memory_source"] = "candidate"
        return candidate_value, candidate_value, log
    if cached_repair is not None:
        repair_val, n_calls = cached_repair
    else:
        repair_val, n_calls = repair_read(reader, question, sec_cands, top_k_cards)
    log["repair_value"] = repair_val
    log["n_repair_calls"] = n_calls
    repair_admit, _ = cavec_admit(repair_val, pack_text, question=question)
    if not is_refusal(repair_val) and repair_admit:
        log["final_source"] = "repair"
        log["memory_source"] = "repair"
        return repair_val, repair_val, log
    baseline_admit = False
    if baseline_value is not None:
        if is_refusal(baseline_value):
            baseline_admit = True
        else:
            baseline_admit, _ = cavec_admit(baseline_value, pack_text, question=question)
    if baseline_value is not None and baseline_admit:
        log["final_source"] = "fallback_baseline"
        log["memory_source"] = "fallback_baseline"
        return baseline_value, baseline_value, log
    if baseline_value is not None:
        # MFC: memory quarantine, output fail-open to baseline raw
        log["final_source"] = "fallback_baseline_unsupported"
        log["memory_source"] = "quarantine_no_commit"
        return NO_COMMIT_SENTINEL, baseline_value, log
    # No baseline supplied -> refuse on both surfaces
    log["final_source"] = "fallback_refusal"
    log["memory_source"] = "fallback_refusal"
    refusal = "I do not have enough information to answer."
    return refusal, refusal, log


# ----------------------------- Agent: ReAct -----------------------------
REACT_PLANNER_PROMPT = (
    "You are a research agent. To answer the user's question, you can issue "
    "search queries against a document corpus. After each search, you will see "
    "the retrieved evidence. Decide your next action.\n\n"
    "Question: {question}\n\n"
    "Search history so far:\n{history}\n\n"
    "Output exactly ONE line in one of these formats:\n"
    "  SEARCH: <a focused query string>\n"
    "  FINISH\n\n"
    "If you have enough evidence to answer, output FINISH. Otherwise output a "
    "SEARCH query that targets the missing information. Output:"
)


def run_react(reader: ReaderVLLM, question: str, bm25, cards, chunk_text,
              max_turns: int = 4, top_k: int = 10, with_cavec: bool = False,
              cavec_fail_closed: bool = True, with_router: bool = False,
              with_cavec_mfc: bool = False) -> Dict:
    """ReAct loop. Per-step "memory" = scratchpad of (search_query, sub_answer).

    Each turn:
      1. Planner emits SEARCH:<query> or FINISH.
      2. If SEARCH, retrieve top-k pack via BM25; reader emits a sub-answer
         conditioned on (question, pack).
      3. The sub-answer is the public commit candidate -> CAVEC hook (if enabled).
      4. Append (query, committed_value, pack) to scratchpad.
    Final answer: synthesizer call over the full scratchpad pack (concatenated).
    """
    scratchpad: List[Dict] = []  # each: {query, raw_answer, committed_answer, pack_text, hook_log}
    n_reader_calls = 0
    n_repair_calls_total = 0
    history_str = "(none)"

    for turn in range(max_turns):
        # Planner decision
        planner_in = REACT_PLANNER_PROMPT.format(question=question, history=history_str)
        if turn == 0:
            # Force a search on the first turn (the question itself).
            decision = f"SEARCH: {question}"
        else:
            decision = reader.reasoning_call(planner_in, max_new_tokens=64)
        decision = decision.strip()
        # Extract action
        if "FINISH" in decision.upper().split():
            break
        if "FINISH" in decision.upper() and "SEARCH" not in decision.upper():
            break
        if "SEARCH:" in decision.upper():
            # Take everything after first SEARCH:
            m = re.search(r"SEARCH\s*:\s*(.+)", decision, flags=re.IGNORECASE)
            query = m.group(1).strip() if m else question
            # Drop trailing "FINISH" if present in same line
            query = query.split("\n")[0].strip()
            if not query:
                query = question
        elif "FINISH" in decision.upper():
            break
        else:
            # Bad output -> fall back to question
            query = question

        sec_cands, _ = build_pack_for_query(
            query, bm25, cards, chunk_text, top_k
        )
        # Sub-reader call: ask the reader for an answer conditioned on this pack.
        # Use strict prompt -> baseline candidate.
        raw_ans, ctx_text, _ = reader.reader_call(question, sec_cands, prompt_variant="strict",
                                                   top_k_cards=top_k)
        n_reader_calls += 1
        committed = raw_ans
        hook_log = None
        if with_cavec:
            committed, hook_log = cavec_hook(
                reader, raw_ans, question, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
                fail_closed=cavec_fail_closed,
            )
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        elif with_router:
            committed, hook_log = retrained_router_hook(
                reader, raw_ans, question, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
            )
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        elif with_cavec_mfc:
            mem_val, _out_val, hook_log = cavec_mfc_hook(
                reader, raw_ans, question, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
            )
            committed = mem_val  # per-turn: memory surface only
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        memory_record = public_memory_record(committed, ctx_text)
        scratchpad.append({
            "turn": turn, "query": query, "raw_answer": raw_ans,
            "committed_answer": committed,
            "memory_record": memory_record,
            "pack_card_ids": [s["card_id"] for s in sec_cands],
            "pack_text_len": len(ctx_text),
            "evidence_packet_sha256": evidence_packet_sha256(ctx_text),
            "hook_log": hook_log,
        })
        # Build planner history (committed answers feed back into next planner step)
        history_lines = []
        for s in scratchpad:
            history_lines.append(f"  [t{s['turn']}] SEARCH: {s['query']}")
            fact_value = factual_projection(s["memory_record"])
            if fact_value is not None:
                history_lines.append(f"        committed: {fact_value[:120]}")
        history_str = "\n".join(history_lines)

    # Final synthesis: union of all queried packs, then strict reader call.
    union_card_ids = []
    seen_ids = set()
    for s in scratchpad:
        for cid in s["pack_card_ids"]:
            if cid not in seen_ids:
                union_card_ids.append(cid); seen_ids.add(cid)
    # Re-construct sec_cands from union (use their text). Easiest: cache cards by card_id.
    card_by_id = {c["card_id"]: c for c in cards}
    union_sec_cands = []
    for cid in union_card_ids[:top_k]:  # cap union at top_k for token budget
        c = card_by_id.get(cid)
        if c is None:
            continue
        union_sec_cands.append({
            "card_id": c["card_id"], "chunk_id": c["card_id"],
            "doc_id": c["doc_id"], "card_type": c.get("card_type"),
            "path_str": c.get("path_str"), "page_span": c.get("page_span"),
            "chunk_ids": c.get("chunk_ids"),
            "text": c.get("embed_text") or c.get("bm25_text") or "",
            "final": 1.0,
        })

    # If CAVEC and there was a committed answer that is supported in the union pack,
    # we could skip the final synth — but the paper convention is: the agent's
    # FINAL answer is what's returned to the user. So we do one synth call.
    final_ans, final_ctx_text, _ = reader.reader_call(
        question, union_sec_cands, prompt_variant="strict", top_k_cards=top_k
    )
    final_pack_text = final_ctx_text
    n_reader_calls += 1
    # Apply CAVEC to the final answer too (the public commit to the user)
    final_committed = final_ans
    final_hook_log = None
    if with_cavec:
        final_committed, final_hook_log = cavec_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
            fail_closed=cavec_fail_closed,
        )
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)
    elif with_router:
        final_committed, final_hook_log = retrained_router_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
        )
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)
    elif with_cavec_mfc:
        # Final hook: output_value (output-fail-open to baseline raw) is the
        # per-qid prediction. Memory_value (NO_COMMIT sentinel on fail-open) is
        # NOT used for grading but IS recorded in final_hook_log for the
        # contamination metric.
        mem_final, out_final, final_hook_log = cavec_mfc_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
        )
        final_committed = out_final
        final_hook_log["memory_value_at_final"] = mem_final
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)

    final_memory_value = (
        final_hook_log.get("memory_value_at_final", final_committed)
        if final_hook_log else final_committed
    )

    return {
        "agent": "react",
        "with_cavec": with_cavec,
        "with_router": with_router,
        "with_cavec_mfc": with_cavec_mfc,
        "scratchpad": scratchpad,
        "final_pack_card_ids": [s["card_id"] for s in union_sec_cands],
        "final_pack_text": final_pack_text,
        "final_evidence_packet_sha256": evidence_packet_sha256(final_pack_text),
        "final_raw_answer": final_ans,
        "final_committed_answer": final_committed,
        "final_memory_record": public_memory_record(final_memory_value, final_pack_text),
        "final_hook_log": final_hook_log,
        "n_reader_calls": n_reader_calls,
        "n_repair_calls": n_repair_calls_total,
        "n_turns": len(scratchpad),
    }


# ----------------------------- Agent: PlanRAG -----------------------------
PLANRAG_PROMPT = (
    "You are a planning agent. Decompose the question into 2 short, "
    "self-contained sub-questions that can be answered independently and whose "
    "answers, taken together, suffice to answer the original.\n\n"
    "Question: {question}\n\n"
    "Output exactly two lines, each starting with 'Q1:' or 'Q2:'.\n"
    "Q1:"
)


def run_planrag(reader: ReaderVLLM, question: str, bm25, cards, chunk_text,
                top_k: int = 10, with_cavec: bool = False,
                cavec_fail_closed: bool = True, with_router: bool = False,
                with_cavec_mfc: bool = False) -> Dict:
    """PlanRAG: decompose -> retrieve per sub-q -> per-sub-answer (CAVEC) -> synthesize."""
    n_reader_calls = 0
    n_repair_calls_total = 0
    # Plan
    plan_out = reader.reasoning_call(PLANRAG_PROMPT.format(question=question), max_new_tokens=128)
    # Parse — model continues from "Q1:" so first line lacks the prefix
    sub_qs = []
    lines = plan_out.split("\n")
    if lines:
        first = lines[0].strip()
        # Strip leading "Q1:" if model echoed it
        first = re.sub(r"^Q\d+\s*:\s*", "", first, flags=re.IGNORECASE).strip()
        if first:
            sub_qs.append(first)
    for line in lines[1:]:
        line = line.strip()
        m = re.match(r"^Q\d+\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            sq = m.group(1).strip()
            if sq:
                sub_qs.append(sq)
        if len(sub_qs) >= 2:
            break
    if len(sub_qs) < 2:
        # Fallback: original question + "entities mentioned in: question"
        sub_qs = [question, f"entities and details mentioned in: {question}"]
    sub_qs = sub_qs[:2]

    sub_logs = []
    union_card_ids = []
    seen_ids = set()
    for i, sq in enumerate(sub_qs):
        sec_cands, _ = build_pack_for_query(
            sq, bm25, cards, chunk_text, top_k
        )
        raw_ans, ctx_text, _ = reader.reader_call(
            sq, sec_cands, prompt_variant="strict", top_k_cards=top_k
        )
        n_reader_calls += 1
        committed = raw_ans
        hook_log = None
        if with_cavec:
            committed, hook_log = cavec_hook(
                reader, raw_ans, sq, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
                fail_closed=cavec_fail_closed,
            )
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        elif with_router:
            committed, hook_log = retrained_router_hook(
                reader, raw_ans, sq, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
            )
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        elif with_cavec_mfc:
            mem_val, _out_val, hook_log = cavec_mfc_hook(
                reader, raw_ans, sq, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
            )
            committed = mem_val
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        memory_record = public_memory_record(committed, ctx_text)
        for c in sec_cands:
            if c["card_id"] not in seen_ids:
                union_card_ids.append(c["card_id"]); seen_ids.add(c["card_id"])
        sub_logs.append({
            "sub_q": sq, "raw_answer": raw_ans, "committed_answer": committed,
            "memory_record": memory_record,
            "hook_log": hook_log,
            "pack_card_ids": [c["card_id"] for c in sec_cands],
            "pack_text_len": len(ctx_text),
            "evidence_packet_sha256": evidence_packet_sha256(ctx_text),
        })

    # Synthesize: build union pack and call reader on the original question.
    card_by_id = {c["card_id"]: c for c in cards}
    union_sec_cands = []
    for cid in union_card_ids[:top_k]:
        c = card_by_id.get(cid)
        if c is None:
            continue
        union_sec_cands.append({
            "card_id": c["card_id"], "chunk_id": c["card_id"],
            "doc_id": c["doc_id"], "card_type": c.get("card_type"),
            "path_str": c.get("path_str"), "page_span": c.get("page_span"),
            "chunk_ids": c.get("chunk_ids"),
            "text": c.get("embed_text") or c.get("bm25_text") or "",
            "final": 1.0,
        })
    # Inject committed sub-answers as a "memory" preamble in the question (synth-time)
    # so the synthesizer is conditioned on the agent's intermediate state.
    factual_findings = []
    for s in sub_logs:
        fact_value = factual_projection(s["memory_record"])
        if fact_value is not None:
            factual_findings.append(f"- Q: {s['sub_q']}\n  A: {fact_value}")
    memory_preamble = (
        "Findings from sub-questions:\n" + "\n".join(factual_findings) + "\n\n"
        if factual_findings else ""
    )
    synth_question = f"{memory_preamble}Main question: {question}"
    final_ans, final_ctx_text, _ = reader.reader_call(
        synth_question, union_sec_cands, prompt_variant="strict", top_k_cards=top_k
    )
    final_pack_text = final_ctx_text
    n_reader_calls += 1
    final_committed = final_ans
    final_hook_log = None
    if with_cavec:
        final_committed, final_hook_log = cavec_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
            fail_closed=cavec_fail_closed,
        )
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)
    elif with_router:
        final_committed, final_hook_log = retrained_router_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
        )
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)
    elif with_cavec_mfc:
        mem_final, out_final, final_hook_log = cavec_mfc_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
        )
        final_committed = out_final
        final_hook_log["memory_value_at_final"] = mem_final
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)

    final_memory_value = (
        final_hook_log.get("memory_value_at_final", final_committed)
        if final_hook_log else final_committed
    )

    return {
        "agent": "planrag",
        "with_cavec": with_cavec,
        "with_router": with_router,
        "with_cavec_mfc": with_cavec_mfc,
        "sub_questions": sub_qs,
        "sub_logs": sub_logs,
        "final_pack_card_ids": [s["card_id"] for s in union_sec_cands],
        "final_pack_text": final_pack_text,
        "final_evidence_packet_sha256": evidence_packet_sha256(final_pack_text),
        "final_raw_answer": final_ans,
        "final_committed_answer": final_committed,
        "final_memory_record": public_memory_record(final_memory_value, final_pack_text),
        "final_hook_log": final_hook_log,
        "n_reader_calls": n_reader_calls,
        "n_repair_calls": n_repair_calls_total,
        "n_turns": len(sub_logs),
    }


# ----------------------------- Agent: AutoGen-style -----------------------------
AUTOGEN_REASONER_PROMPT = (
    "You are a Reasoner agent in a multi-agent QA conversation. The Retriever "
    "agent will fetch evidence based on your queries. Decide whether to ask "
    "for more evidence or stop.\n\n"
    "Question: {question}\n\n"
    "Conversation so far:\n{conversation}\n\n"
    "Output exactly ONE line:\n"
    "  ASK_RETRIEVER: <a focused query>\n"
    "  STOP\n\n"
    "Output:"
)


def run_autogen(reader: ReaderVLLM, question: str, bm25, cards, chunk_text,
                max_turns: int = 4, top_k: int = 10, with_cavec: bool = False,
                cavec_fail_closed: bool = True, with_router: bool = False,
                with_cavec_mfc: bool = False) -> Dict:
    """AutoGen-style: Reasoner asks Retriever; Retriever fetches and Reader
    summarizes; the summary is the public commit checked by CAVEC.
    """
    conversation: List[Dict] = []
    n_reader_calls = 0
    n_repair_calls_total = 0
    union_card_ids = []
    seen_ids = set()

    for turn in range(max_turns):
        if turn == 0:
            # First turn always asks for the question
            ask = question
            conversation.append({"role": "reasoner", "content": f"ASK_RETRIEVER: {ask}"})
        else:
            visible_messages = []
            for message in conversation:
                if message.get("role") == "reader_commit":
                    fact_value = factual_projection(message["_memory_record"])
                    if fact_value is None:
                        continue
                    visible_messages.append(
                        f"  [reader_commit] {fact_value[:120]}"
                    )
                else:
                    visible_messages.append(
                        f"  [{message['role']}] {message['content'][:120]}"
                    )
            conv_text = "\n".join(visible_messages)
            decision = reader.reasoning_call(
                AUTOGEN_REASONER_PROMPT.format(question=question, conversation=conv_text),
                max_new_tokens=64,
            ).strip()
            if "STOP" in decision.upper() and "ASK_RETRIEVER" not in decision.upper():
                conversation.append({"role": "reasoner", "content": "STOP"})
                break
            m = re.search(r"ASK_RETRIEVER\s*:\s*(.+)", decision, flags=re.IGNORECASE)
            if not m:
                # Bad output -> stop
                conversation.append({"role": "reasoner", "content": "STOP"})
                break
            ask = m.group(1).strip().split("\n")[0]
            if not ask:
                ask = question
            conversation.append({"role": "reasoner", "content": f"ASK_RETRIEVER: {ask}"})

        # Retriever action
        sec_cands, _ = build_pack_for_query(
            ask, bm25, cards, chunk_text, top_k
        )
        for c in sec_cands:
            if c["card_id"] not in seen_ids:
                union_card_ids.append(c["card_id"]); seen_ids.add(c["card_id"])
        conversation.append({
            "role": "retriever",
            "content": f"Returned {len(sec_cands)} cards (top: {sec_cands[0]['card_id'] if sec_cands else 'none'})",
        })
        # Reader emits a per-step summary/answer for THIS evidence
        raw_ans, ctx_text, _ = reader.reader_call(
            question, sec_cands, prompt_variant="strict", top_k_cards=top_k
        )
        n_reader_calls += 1
        committed = raw_ans
        hook_log = None
        if with_cavec:
            committed, hook_log = cavec_hook(
                reader, raw_ans, question, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
                fail_closed=cavec_fail_closed,
            )
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        elif with_router:
            committed, hook_log = retrained_router_hook(
                reader, raw_ans, question, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
            )
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        elif with_cavec_mfc:
            mem_val, _out_val, hook_log = cavec_mfc_hook(
                reader, raw_ans, question, sec_cands, ctx_text,
                baseline_value=raw_ans, top_k_cards=top_k,
            )
            committed = mem_val
            n_repair_calls_total += hook_log.get("n_repair_calls", 0)
        memory_record = public_memory_record(committed, ctx_text)
        conversation.append({
            "role": "reader_commit", "content": committed,
            "_memory_record": memory_record,
            "_hook_log": hook_log,
            "_pack_card_ids": [c["card_id"] for c in sec_cands],
            "_pack_text_len": len(ctx_text),
            "_evidence_packet_sha256": evidence_packet_sha256(ctx_text),
        })

    # Final synthesis over union pack (with all committed values as memory).
    card_by_id = {c["card_id"]: c for c in cards}
    union_sec_cands = []
    for cid in union_card_ids[:top_k]:
        c = card_by_id.get(cid)
        if c is None:
            continue
        union_sec_cands.append({
            "card_id": c["card_id"], "chunk_id": c["card_id"],
            "doc_id": c["doc_id"], "card_type": c.get("card_type"),
            "path_str": c.get("path_str"), "page_span": c.get("page_span"),
            "chunk_ids": c.get("chunk_ids"),
            "text": c.get("embed_text") or c.get("bm25_text") or "",
            "final": 1.0,
        })
    # Inject committed memory into question
    mem_lines = []
    for message in conversation:
        if message.get("role") != "reader_commit":
            continue
        fact_value = factual_projection(message["_memory_record"])
        if fact_value is not None:
            mem_lines.append(fact_value)
    if mem_lines:
        memory_preamble = "Conversation memory:\n" + "\n".join(f"- {m}" for m in mem_lines) + "\n\n"
    else:
        memory_preamble = ""
    synth_question = f"{memory_preamble}Main question: {question}"
    final_ans, final_ctx_text, _ = reader.reader_call(
        synth_question, union_sec_cands, prompt_variant="strict", top_k_cards=top_k
    )
    final_pack_text = final_ctx_text
    n_reader_calls += 1
    final_committed = final_ans
    final_hook_log = None
    if with_cavec:
        final_committed, final_hook_log = cavec_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
            fail_closed=cavec_fail_closed,
        )
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)
    elif with_router:
        final_committed, final_hook_log = retrained_router_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
        )
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)
    elif with_cavec_mfc:
        mem_final, out_final, final_hook_log = cavec_mfc_hook(
            reader, final_ans, question, union_sec_cands, final_pack_text,
            baseline_value=final_ans, top_k_cards=top_k,
        )
        final_committed = out_final
        final_hook_log["memory_value_at_final"] = mem_final
        n_repair_calls_total += final_hook_log.get("n_repair_calls", 0)

    final_memory_value = (
        final_hook_log.get("memory_value_at_final", final_committed)
        if final_hook_log else final_committed
    )

    return {
        "agent": "autogen",
        "with_cavec": with_cavec,
        "with_router": with_router,
        "with_cavec_mfc": with_cavec_mfc,
        "conversation": conversation,
        "final_pack_card_ids": [s["card_id"] for s in union_sec_cands],
        "final_pack_text": final_pack_text,
        "final_evidence_packet_sha256": evidence_packet_sha256(final_pack_text),
        "final_raw_answer": final_ans,
        "final_committed_answer": final_committed,
        "final_memory_record": public_memory_record(final_memory_value, final_pack_text),
        "final_hook_log": final_hook_log,
        "n_reader_calls": n_reader_calls,
        "n_repair_calls": n_repair_calls_total,
        "n_turns": sum(1 for m in conversation if m["role"] == "reasoner"),
    }


# ----------------------------- Subset selection -----------------------------
def get_held_out_subset(n: int = 300, seed: int = 2026_05_04) -> List[str]:
    meta = json.loads(Path(f"{PHASE21}/split_meta.json").read_text())
    eva = list(meta["eva_qids"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eva))[:n]
    return sorted([eva[i] for i in idx])


def load_qid_meta() -> Tuple[Dict[str, str], Dict[str, str]]:
    val = json.loads(Path(f"{DATA}/M3DocVQA/val.json").read_text())
    qid2gold, qid2q = {}, {}
    for e in val:
        q = str(e.get("id") or e.get("qid"))
        a = e.get("answer") or e.get("answers") or ""
        if isinstance(a, list):
            a = a[0] if a else ""
        qid2gold[q] = a
        qid2q[q] = e.get("question", "")
    return qid2gold, qid2q


# ----------------------------- Per-qid driver -----------------------------
ARMS = ("vanilla", "vanilla_hard_refuse", "final_support_retry",
        "cavec_paper", "cavec_h1_hard_refuse", "retrained_router_commit_time",
        "cavec_mfc")


def _post_hoc_arms(reader: ReaderVLLM, vanilla_res: Dict, question: str,
                    bm25, cards, chunk_text, top_k: int) -> Tuple[Dict, Dict]:
    """Build vanilla_hard_refuse and final_support_retry from a vanilla trajectory.

    Sharing: both arms reuse the vanilla final pack and final raw answer; the
    only extra reader call is the RepairRead in final_support_retry when the
    vanilla final answer is unsupported (cached if already done).
    """
    vfinal = vanilla_res.get("final_committed_answer", "")
    vpack = vanilla_res.get("final_pack_text", "")
    vpack_hash = vanilla_res.get("final_evidence_packet_sha256") or evidence_packet_sha256(vpack)
    vfinal_card_ids = vanilla_res.get("final_pack_card_ids", [])
    # vanilla_hard_refuse: post-hoc hard-refuse on vanilla output; no extra call.
    vhr_ans, vhr_src = vanilla_hard_refuse(vfinal, vpack, question=question)
    vhr_res = {
        "agent": vanilla_res.get("agent",""),
        "arm": "vanilla_hard_refuse",
        "final_committed_answer": vhr_ans,
        "final_raw_answer": vfinal,
        "final_pack_card_ids": vfinal_card_ids,
        "final_pack_text": vpack,
        "final_evidence_packet_sha256": vpack_hash,
        "final_hook_log": {"final_source": vhr_src},
        "n_reader_calls": 0, "n_repair_calls": 0,
        "n_turns": vanilla_res.get("n_turns", 0),
    }
    # final_support_retry: paper Table 4 fail-open. Need union sec_cands to call
    # RepairRead. Reconstruct from vfinal_card_ids using card_by_id.
    card_by_id = {c["card_id"]: c for c in cards}
    fsr_sec_cands = []
    for cid in vfinal_card_ids[:top_k]:
        c = card_by_id.get(cid)
        if c is None: continue
        fsr_sec_cands.append({
            "card_id": c["card_id"], "chunk_id": c["card_id"],
            "doc_id": c["doc_id"], "card_type": c.get("card_type"),
            "path_str": c.get("path_str"), "page_span": c.get("page_span"),
            "chunk_ids": c.get("chunk_ids"),
            "text": c.get("embed_text") or c.get("bm25_text") or "",
            "final": 1.0,
        })
    fsr_committed, fsr_log = final_support_retry_hook(
        reader, vfinal, question, fsr_sec_cands, vpack, top_k_cards=top_k,
    )
    fsr_res = {
        "agent": vanilla_res.get("agent",""),
        "arm": "final_support_retry",
        "final_committed_answer": fsr_committed,
        "final_raw_answer": vfinal,
        "final_pack_card_ids": vfinal_card_ids,
        "final_pack_text": vpack,
        "final_evidence_packet_sha256": vpack_hash,
        "final_hook_log": fsr_log,
        "n_reader_calls": 0,
        "n_repair_calls": fsr_log.get("n_repair_calls", 0),
        "n_turns": vanilla_res.get("n_turns", 0),
    }
    return vhr_res, fsr_res


def collect_calib_scores_from_vanilla_traj(van_res: Dict, agent_name: str) -> int:
    """Walk vanilla trajectory and collect router scores at every commit point
    (without triggering any repair). Returns number of scores added.
    Uses pack_text reconstructed via final_pack_text for final, and
    per-step pack via the raw_answer + per-step pack reconstruction.
    For simplicity we score the final commit only here (per-step packs are not
    cached for vanilla). This still gives 1 score per qid per agent; over 15
    qids × 3 agents = 45 scores, enough for a stable q10 estimate."""
    n_added = 0
    final_ans = van_res.get("final_raw_answer", "") or ""
    final_pack = van_res.get("final_pack_text", "") or ""
    if final_ans and final_pack:
        _ = router_score(final_ans, final_pack, baseline_value=final_ans)
        n_added += 1
    return n_added


def run_one_qid(qid: str, question: str, gold: str, reader: ReaderVLLM,
                bm25, cards, chunk_text, top_k: int, max_turns: int,
                calib_only: bool = False) -> Dict:
    """Run 5 arms for one qid via 3 trajectories + 2 post-hoc derivations."""
    out = {"qid": qid, "question": question, "gold": gold}
    for agent_name, fn, kwargs in [
        ("react", run_react, {"max_turns": max_turns}),
        ("planrag", run_planrag, {}),
        ("autogen", run_autogen, {"max_turns": max_turns}),
    ]:
        # 5 trajectories: vanilla, cavec_paper (fail-open), cavec_h1_hard_refuse,
        # retrained_router_commit_time (Phase 29), cavec_mfc (Phase 30 new)
        # tuple = (arm_name, with_cavec, fail_closed, with_router, with_cavec_mfc)
        traj_specs = [
            ("vanilla", False, True, False, False),
            ("cavec_paper", True, False, False, False),
            ("cavec_h1_hard_refuse", True, True, False, False),
            ("retrained_router_commit_time", False, False, True, False),
            ("cavec_mfc", False, False, False, True),
        ]
        traj_results = {}
        for arm_name, with_cavec, fail_closed, with_router, with_cavec_mfc in traj_specs:
            t0 = time.time()
            try:
                res = fn(reader, question, bm25, cards, chunk_text,
                         top_k=top_k, with_cavec=with_cavec,
                         cavec_fail_closed=fail_closed, with_router=with_router,
                         with_cavec_mfc=with_cavec_mfc,
                         **kwargs)
                res["wall_s"] = round(time.time() - t0, 2)
                final_ans = res["final_committed_answer"]
                pack_text = res["final_pack_text"]
                res["em"] = substr_em(final_ans, gold)
                res["unsupNR_strict"] = (1 - is_refusal(final_ans)) * (1 - support_in_pack(final_ans, pack_text))
                res["calls_q"] = res["n_reader_calls"] + res["n_repair_calls"]
                res["agent"] = agent_name
                res["arm"] = arm_name
            except Exception as e:
                res = {
                    "agent": agent_name, "arm": arm_name,
                    "error": f"{type(e).__name__}: {e}",
                    "final_committed_answer": "", "final_raw_answer": "",
                    "final_pack_card_ids": [], "final_pack_text": "",
                    "em": 0, "unsupNR_strict": 0, "calls_q": 0,
                    "n_reader_calls": 0, "n_repair_calls": 0,
                    "wall_s": round(time.time() - t0, 2),
                }
            traj_results[arm_name] = res

        # Post-hoc arms derived from vanilla trajectory
        van_res = traj_results["vanilla"]
        vhr_res, fsr_res = _post_hoc_arms(reader, van_res, question, bm25, cards, chunk_text, top_k)
        # Compute metrics for derived arms
        for d_res in (vhr_res, fsr_res):
            d_ans = d_res["final_committed_answer"]
            d_pack = d_res["final_pack_text"]
            d_res["em"] = substr_em(d_ans, gold)
            d_res["unsupNR_strict"] = (1 - is_refusal(d_ans)) * (1 - support_in_pack(d_ans, d_pack))
            d_res["calls_q"] = d_res["n_reader_calls"] + d_res["n_repair_calls"]
        traj_results["vanilla_hard_refuse"] = vhr_res
        traj_results["final_support_retry"] = fsr_res

        for arm in ARMS:
            out[f"{agent_name}_{arm}"] = traj_results[arm]
    return out


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--n_subset", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026_05_04)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--model", default=os.environ.get("ADMIT_READER_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--token_budget", type=int, default=14000)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--out_dir", type=Path, default=Path(PHASE25))
    ap.add_argument("--qid_subset_file", type=Path, default=None,
                    help="Optional override: a JSON list of qids to process.")
    ap.add_argument("--smoke_n", type=int, default=0,
                    help="If >0, override n_subset and only run this many qids (for smoke).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[shard{args.shard_id}/{args.num_shards}] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    print(f"[shard{args.shard_id}] model={args.model}", flush=True)
    print(f"[shard{args.shard_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)

    # Load data
    print(f"[shard{args.shard_id}] loading subset...", flush=True)
    if args.qid_subset_file is not None:
        all_qids = json.loads(args.qid_subset_file.read_text())
    else:
        all_qids = get_held_out_subset(args.n_subset, args.seed)
    if args.smoke_n > 0:
        all_qids = all_qids[:args.smoke_n]
    # Shard
    my_qids = [q for i, q in enumerate(all_qids) if i % args.num_shards == args.shard_id]
    print(f"[shard{args.shard_id}] got {len(my_qids)} qids out of {len(all_qids)} total",
          flush=True)

    print(f"[shard{args.shard_id}] loading qid meta...", flush=True)
    qid2gold, qid2q = load_qid_meta()
    print(f"[shard{args.shard_id}] loading chunk_text...", flush=True)
    chunk_text = load_chunk_text()
    print(f"[shard{args.shard_id}] loading BM25 + cards...", flush=True)
    bm25, cards = load_bm25_and_cards()
    print(f"[shard{args.shard_id}] cards={len(cards)} chunk_pages={len(chunk_text)}",
          flush=True)

    # Load reader (vLLM)
    print(f"[shard{args.shard_id}] loading vLLM reader...", flush=True)
    t_load = time.time()
    reader = ReaderVLLM(
        model_path=args.model,
        max_model_len=args.max_model_len,
        token_budget=args.token_budget,
        gpu_mem_util=args.gpu_mem_util,
    )
    print(f"[shard{args.shard_id}] reader loaded in {time.time()-t_load:.1f}s", flush=True)

    # Output paths (5 arms per agent)
    sid = args.shard_id
    paths = {}
    for agent in ("react", "planrag", "autogen"):
        for arm in ARMS:
            paths[f"{agent}_{arm}"] = args.out_dir / f"{agent}_{arm}_pred.shard{sid}.jsonl"
        paths[f"{agent}_per_qid"] = args.out_dir / f"{agent}_per_qid.shard{sid}.jsonl"

    # Open all output files
    fhs = {k: open(v, "w") for k, v in paths.items()}

    # ---------------- Phase 29 router calibration phase --------------------
    # Pre-load router model and pre-calibrate tau' on the first ~15 qids by
    # running ONLY the vanilla trajectory (no router triggering), then scoring
    # the final commit candidate. After scores collected, set tau' as the
    # ROUTER_TARGET_TRIGGER_RATE-quantile.
    global _CALIB_MODE
    n_calib_qids = min(5, len(my_qids))
    print(f"[shard{sid}] router calibration: precomputing on {n_calib_qids} qids", flush=True)
    _CALIB_MODE = True
    _ = _load_router()
    calib_t0 = time.time()
    calib_van_results = {}  # cache vanilla results so we don't recompute for these qids
    for ci, qid in enumerate(my_qids[:n_calib_qids]):
        gold = qid2gold.get(qid, "")
        question = qid2q.get(qid, "")
        if not question:
            continue
        try:
            calib_van_results[qid] = {}
            for agent_name, fn, kwargs in [
                ("react", run_react, {"max_turns": args.max_turns}),
                ("planrag", run_planrag, {}),
                ("autogen", run_autogen, {"max_turns": args.max_turns}),
            ]:
                vres = fn(reader, question, bm25, cards, chunk_text,
                          top_k=args.top_k, with_cavec=False,
                          cavec_fail_closed=False, with_router=False, **kwargs)
                # Score this final commit (no triggering — calibration only)
                collect_calib_scores_from_vanilla_traj(vres, agent_name)
                calib_van_results[qid][agent_name] = vres
        except Exception as e:
            print(f"[shard{sid}] calib FATAL on qid={qid}: {type(e).__name__}: {e}",
                  flush=True)
    _CALIB_MODE = False
    tau_set = calibrate_router_tau(target_trigger_rate=ROUTER_TARGET_TRIGGER_RATE)
    print(f"[shard{sid}] calibration done in {time.time()-calib_t0:.1f}s, "
          f"tau'={tau_set:.4f}", flush=True)

    t_run = time.time()
    n_done = 0
    for qid in my_qids:
        gold = qid2gold.get(qid, "")
        question = qid2q.get(qid, "")
        if not question:
            print(f"[shard{sid}] skipping {qid}: no question text", flush=True)
            continue
        try:
            out = run_one_qid(qid, question, gold, reader, bm25, cards, chunk_text,
                              args.top_k, args.max_turns)
        except Exception as e:
            print(f"[shard{sid}] FATAL on qid={qid}: {type(e).__name__}: {e}", flush=True)
            continue

        for agent in ("react", "planrag", "autogen"):
            # Per-qid full log: include all 5 arms
            per_qid_row = {
                "qid": qid, "question": question, "gold": gold,
            }
            for arm in ARMS:
                per_qid_row[arm] = out[f"{agent}_{arm}"]
            # Strip pack_text fields out of per-qid log to keep file size sane
            for arm in ARMS:
                pack_text = per_qid_row[arm].get("final_pack_text", "")
                per_qid_row[arm].pop("final_pack_text", None)
                per_qid_row[arm]["final_pack_text_len"] = len(pack_text)
            fhs[f"{agent}_per_qid"].write(json.dumps(per_qid_row, ensure_ascii=False) + "\n")
            for arm in ARMS:
                key = f"{agent}_{arm}"
                r = out[key]
                pred_row = {
                    "id": qid, "qid": qid, "question": question, "gold": gold,
                    "agent": agent, "arm": arm,
                    "generated_answer": r.get("final_committed_answer", ""),
                    "raw_answer": r.get("final_raw_answer", ""),
                    "n_reader_calls": r.get("n_reader_calls", 0),
                    "n_repair_calls": r.get("n_repair_calls", 0),
                    "calls_q": r.get("calls_q", 0),
                    "em": r.get("em", 0),
                    "unsupNR_strict": r.get("unsupNR_strict", 0),
                    "n_turns": r.get("n_turns", 0),
                    "final_pack_card_ids": r.get("final_pack_card_ids", []),
                    "final_evidence_packet_sha256": r.get("final_evidence_packet_sha256"),
                    "wall_s": r.get("wall_s", 0.0),
                    "error": r.get("error"),
                }
                fhs[key].write(json.dumps(pred_row, ensure_ascii=False) + "\n")
        for f in fhs.values():
            f.flush()
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t_run
            eta = elapsed / n_done * (len(my_qids) - n_done)
            print(f"[shard{sid}] {n_done}/{len(my_qids)} qids done; "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                  f"({n_done/elapsed:.2f} q/s)", flush=True)

    for f in fhs.values():
        f.close()
    print(f"[shard{sid}] DONE: {n_done} qids in {time.time()-t_run:.1f}s", flush=True)


if __name__ == "__main__":
    main()
