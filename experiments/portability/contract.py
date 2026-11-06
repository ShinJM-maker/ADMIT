from __future__ import annotations
import re, hashlib
from types import SimpleNamespace

REFUSAL_PHRASE = "do not have enough information"

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower().strip())

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

p30=SimpleNamespace(is_refusal=is_refusal,support_in_pack=support_in_pack)
def strict_cavec_admit(commit_value: str, evidence_pack_text: str,
                       question: str = "") -> tuple[bool, str]:
    del question
    if p30.is_refusal(commit_value):
        return True, "refusal_recorded"
    if p30.support_in_pack(commit_value, evidence_pack_text):
        return True, "S_str_positive"
    return False, "S_str_zero"
cavec_admit=strict_cavec_admit

MODEL = __import__("os").environ.get("ADMIT_READER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

PACKET_TOKEN_BUDGET = 4500

BASE_PROMPT = """/no_think
You are an AI assistant that answers questions by analyzing provided documents.
Answer the [Question] using only the [Context] below.
- Give a SHORT, DIRECT answer (a few words or a short phrase).
- Do NOT explain your reasoning. Do NOT include citations.
- If unsure, output exactly: I do not have enough information to answer.

[Question]
{question}

[Context]
{context}

[Answer]
"""

def clean_answer(raw: str) -> str:
    value = str(raw or "").strip()
    value = re.sub(r"^\s*\[Answer\]\s*", "", value, flags=re.I)
    return value.splitlines()[0].strip() if value else ""

def chat_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text
PROMPT_TAIL="\n[Answer]\n"
