#!/usr/bin/env python3
"""
HiKEY Phase 5 — Qwen2.5-VL-7B-Instruct reader inference.

Inputs:
  --retrieval  legacy.jsonl produced by Stage-2 retrieval (one record per question
               with sectioncard_candidates ordered by `final` score)
  --dataset    M3DocVQA | FRAMES (resolves docs.jsonl + chunks.jsonl)
  --out        path to write predictions jsonl  (id, question, generated_answer,
               n_text_units, n_image_units, prompt_tokens)

Pipeline per question:
  1) Build evidence subgraph from top-K_sec sectioncards (default K=10).
     For now, simple ancestry-aware packing (Algorithm 1 lite):
       - For each card in rank order:
           include `# {doc_title}` + `## {section_path}` + card text body
           + (if budget remains and crops exist) up to N image crops with
             scale-aware fixed token cost.
       - Stop when token budget reached.
     Token budget defaults to 16K (paper §5).
  2) Format with paper Appendix prompt template.
  3) Greedy decode with Qwen2.5-VL-7B-Instruct (T=0.0, max_new_tokens=256).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

INDEX_ROOT = Path(os.environ.get("ADMIT_INDEX", os.environ.get("HIKEY_INDEX", str(Path(__file__).resolve().parents[1] / "data/index"))))

PROMPT_HEAD = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "- Do not answer using information that is not present in the context.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- If you cannot be confident, output exactly: I do not have enough information to answer.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_GUESS = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using the [Context] given below as your primary source.\n"
    "- Prefer information from the context, but if the context is incomplete, make your best informed guess rather than refusing.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- Always produce an answer. Never output refusals like 'I do not have enough information'.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

# Submitted-paper prompt policy.  The policy sentences below reproduce the
# NeurIPS submission's Appendix prompt wording; the [Context]/[Answer]
# delimiters are held fixed so the same frozen evidence body can be reused in
# the reader/prompt/visual robustness grid.
PROMPT_HEAD_PAPER = (
    "Answer the following question based on the provided documents. "
    "If the answer cannot be determined from the documents, respond with \"unanswerable\".\n\n"
    "Question: {question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_PAPER_NO_REFUSAL = (
    "Answer the following question based on the provided documents.\n\n"
    "Question: {question}\n\n"
    "[Context]\n"
)

# ICLR 2027 common-clause control, frozen before inference on 2026-09-06.
# This inserts the Base family's refusal clause byte-for-byte into the
# Submitted-family surrounding prompt. Deleting that exact clause recovers
# PROMPT_HEAD_PAPER_NO_REFUSAL, isolating the surrounding-prompt moderator for
# this clause. The insertion point is immediately after the task sentence.
PROMPT_HEAD_PAPER_BASE_REFUSAL = (
    "Answer the following question based on the provided documents.\n"
    "- If you cannot be confident, output exactly: I do not have enough information to answer.\n"
    "\n"
    "Question: {question}\n\n"
    "[Context]\n"
)

# ICLR 2027 external-validation correction.  These two variants copy the
# Base family's two output-format bullets byte-for-byte into the Submitted
# surrounding prompt while holding the Base abstention clause transition fixed.
PROMPT_HEAD_PAPER_FORMAT_BASE_REFUSAL = (
    "Answer the following question based on the provided documents.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- If you cannot be confident, output exactly: I do not have enough information to answer.\n"
    "\n"
    "Question: {question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_PAPER_FORMAT_NO_REFUSAL = (
    "Answer the following question based on the provided documents.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "\n"
    "Question: {question}\n\n"
    "[Context]\n"
)

# No-refusal-only: removes refusal clause but does NOT add best-guess instruction.
# Isolates the abstention-removal effect from the best-guess encouragement.
PROMPT_HEAD_NO_REFUSAL_ONLY = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "- Do not answer using information that is not present in the context.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

# Rebuttal robustness prompts.  The shared task/format instructions are held
# fixed; only the refusal-versus-answer-forcing policy clause is paraphrased.
PROMPT_HEAD_STRICT_PARAPHRASE_A = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using the [Context] given below.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- Answer only when the context provides sufficient evidence; otherwise output exactly: I do not have enough information to answer.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_STRICT_PARAPHRASE_B = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using the [Context] given below.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- Do not guess when the documents are insufficient. In that case respond exactly: I do not have enough information to answer.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_GUESS_PARAPHRASE_A = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using the [Context] given below.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- Always answer the question; if the evidence is incomplete, return the most likely answer instead of abstaining.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_GUESS_PARAPHRASE_B = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using the [Context] given below.\n"
    "- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.\n"
    "- Do NOT include any explanation, restating of the question, or the words 'the answer is'.\n"
    "- Never refuse to answer. When uncertain, provide your best supported estimate from the available context.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

# A4Pz W4 post-review pilot prompts.  Within each named family, the strict and
# guess variants share all task/format wording and differ only in the answer
# policy.  These are intentionally separate from the earlier paraphrase grid:
# they test few-shot demonstrations, a format-only reinforcement, and an
# internal chain-of-thought instruction on a prespecified n=500 subset.
PROMPT_HEAD_STRICT_FEWSHOT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Use only the supplied context. Return only the shortest final answer.\n"
    "If the context is insufficient, output exactly: I do not have enough information to answer.\n\n"
    "[Example 1]\nQuestion: In what year did Project Cedar begin?\n"
    "Context: Project Cedar began in 2019.\nAnswer: 2019\n\n"
    "[Example 2]\nQuestion: In which city is the Oriole team based?\n"
    "Context: The Oriole team is based in Canada.\n"
    "Answer: I do not have enough information to answer.\n\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_GUESS_FEWSHOT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Use the supplied context as the primary source. Return only the shortest final answer.\n"
    "If the context is incomplete, make your best informed guess and never refuse.\n\n"
    "[Example 1]\nQuestion: In what year did Project Cedar begin?\n"
    "Context: Project Cedar began in 2019.\nAnswer: 2019\n\n"
    "[Example 2]\nQuestion: What is the capital of Canada?\n"
    "Context: Canada is a country in North America.\nAnswer: Ottawa\n\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_STRICT_FORMAT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] below.\n"
    "- Return one line containing only the shortest answer; do not use bullets, labels, or explanations.\n"
    "- If the context is insufficient, output exactly: I do not have enough information to answer.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_GUESS_FORMAT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using the [Context] below as the primary source.\n"
    "- Return one line containing only the shortest answer; do not use bullets, labels, or explanations.\n"
    "- If the context is incomplete, make your best informed guess and never refuse.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_STRICT_COT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Analyze the evidence step by step internally, but do not reveal your reasoning.\n"
    "Use only the supplied context and output only the shortest final answer.\n"
    "If the context is insufficient, output exactly: I do not have enough information to answer.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_GUESS_COT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Analyze the evidence step by step internally, but do not reveal your reasoning.\n"
    "Use the supplied context as the primary source and output only the shortest final answer.\n"
    "If the context is incomplete, make your best informed guess and never refuse.\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_CITATION = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions using ONLY the provided document pack.\n"
    "Return your output in this exact format on three lines:\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "cited_chunk_id: <id of the chunk that directly supports the answer>\n"
    "evidence_quote: <one short verbatim quote from the cited chunk>\n"
    "If no chunk in [Context] directly supports the answer, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

# Decomposing what the citation prompt actually changes. `citation_required` moves three things at
# once relative to `strict`: the output becomes several labelled lines, the reader must quote evidence,
# and it must name the unit that evidence came from. Measured effect of all three together on the
# text-only arm: +1.56 EM (M3DocVQA) and +2.83 (DUDE) -- about what delivering images was worth, which
# is why the two turned out to be substitutes rather than additive. These two variants split that.
#
#   strict            1 line,  no evidence            baseline
#   struct_noground   3 lines, no evidence            + structured output
#   quote_only        2 lines, quote, no unit id      + grounding to text
#   citation_required 3 lines, quote + unit id        + grounding to a unit
#
# The abstention token is held at REFUSE across all three so the refusal policy is not a fourth moving
# part; `strict`'s longer refusal sentence is a known lever on this project's other lines.
PROMPT_HEAD_STRUCT_NOGROUND = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "Return your output in this exact format on three lines:\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "answer_type: <person | place | organization | date | number | other>\n"
    "confidence: <high | medium | low>\n"
    "If you cannot answer from the context, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_STRUCT_COMPARE = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "If the question compares candidates or asks which is first/last/most/least, you must list the\n"
    "value each candidate has BEFORE choosing, then choose by comparing those values.\n"
    "Return your output in this exact format:\n"
    "candidate_values: <name = value; name = value>   (write NONE if the question compares nothing)\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "answer_type: <person | place | organization | date | number | other>\n"
    "confidence: <high | medium | low>\n"
    "If you cannot answer from the context, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)
PROMPT_HEAD_QUOTE_ONLY = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions using ONLY the provided document pack.\n"
    "Return your output in this exact format on two lines:\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "evidence_quote: <one short verbatim quote from [Context] that directly supports the answer>\n"
    "If nothing in [Context] directly supports the answer, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

# Targeting the conversion bottleneck, not the refusal one.
#
# Among failures where the gold document is in the packet and its text carries the answer, 54.1% are
# "right document, wrong span": the reader picks a value from the correct page that is not the answer.
# Refusal (16.7%) and formatting (17.0%) are the next two, and both were already shown to be levers of
# the prompt's shape rather than of comprehension.
#
# These three are built on `struct_noground` rather than `strict` so the refusal rate stays at ~8%
# instead of 28%. Otherwise any gain would be the refusal lever again, which is what the earlier ladder
# turned out to be measuring (+2.87 EM from output shape alone, refusals 28.6% -> 7.9%).
#
# `struct_locate` is the one aimed directly at the failure: it forces the span to be written *before*
# the answer, so the answer is extracted from a committed span rather than chosen and then justified.
# `evidence_quote` in the citation prompt asked for the span *after* the answer and cost -1.84 EM.
PROMPT_HEAD_STRUCT_COT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "Work through the evidence step by step internally, but do not reveal your reasoning.\n"
    "Return your output in this exact format on three lines:\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "answer_type: <person | place | organization | date | number | other>\n"
    "confidence: <high | medium | low>\n"
    "If you cannot answer from the context, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_STRUCT_LOCATE = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "First copy the one sentence or table row from [Context] that contains the answer, then read the "
    "answer out of it.\n"
    "Return your output in this exact format on three lines:\n"
    "located_span: <the sentence or row from [Context] that contains the answer>\n"
    "answer_value: <shortest phrase / number / proper noun, taken from located_span>\n"
    "confidence: <high | medium | low>\n"
    "If you cannot answer from the context, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_HEAD_STRUCT_FEWSHOT = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "Return your output in this exact format on three lines:\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "answer_type: <person | place | organization | date | number | other>\n"
    "confidence: <high | medium | low>\n"
    "If you cannot answer from the context, output exactly: REFUSE\n\n"
    "[Example 1]\nQuestion: How many roses were in the blanket?\n"
    "Context: The garland was draped with 554 roses before the race.\n"
    "answer_value: 554\nanswer_type: number\nconfidence: high\n\n"
    "[Example 2]\nQuestion: Which album featured the single made by that artist?\n"
    "Context: Chic released several singles in 1982.\n"
    "answer_value: REFUSE\n\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

# Multimodal citation. The text-only variant asks for a `cited_chunk_id`, which an answer read off an
# image cannot supply -- the packet's crops carry ids of the form `qcrop1` and are perfectly citable, but
# the schema had no way to name them. Measured consequence: on questions whose answer was absent from the
# delivered text, the crop arm refused 2.3x more often and produced an unresolvable citation 2.9x more
# often than on questions the text already answered. That is the schema failing, not the arm.
PROMPT_HEAD_CITATION_MM = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions using ONLY the provided document pack.\n"
    "The pack contains text units (each marked with an id) and image units (marked id=qcrop1, "
    "qcrop2, ...).\n"
    "Return your output in this exact format on three lines:\n"
    "answer_value: <shortest phrase / number / proper noun>\n"
    "cited_evidence_id: <id of the text unit OR image unit that directly supports the answer>\n"
    "evidence_quote: <a short verbatim quote from the cited text unit, or the exact text you read "
    "in the cited image>\n"
    "If nothing in [Context] directly supports the answer, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)

PROMPT_TAIL = "\n[Answer]\n"

PROMPT_HEAD_LONGFORM = (
    "[System Instruction]\n"
    "You are an AI assistant that answers questions by analyzing the provided documents.\n"
    "Write an accurate answer to [Question] using only the [Context] given below.\n"
    "Answer in complete sentences, stating the relevant figures, entities and comparisons the "
    "question asks for. Do not pad the answer with material the question did not ask about.\n"
    "If you cannot answer from the context, output exactly: REFUSE\n"
    "\n"
    "[Question]\n{question}\n\n"
    "[Context]\n"
)


PROMPT_VARIANTS = {
    "strict": PROMPT_HEAD,
    "guess": PROMPT_HEAD_GUESS,
    "no_refusal_only": PROMPT_HEAD_NO_REFUSAL_ONLY,
    "paper": PROMPT_HEAD_PAPER,
    "paper_no_refusal": PROMPT_HEAD_PAPER_NO_REFUSAL,
    "paper_base_refusal": PROMPT_HEAD_PAPER_BASE_REFUSAL,
    "paper_format_no_refusal": PROMPT_HEAD_PAPER_FORMAT_NO_REFUSAL,
    "paper_format_base_refusal": PROMPT_HEAD_PAPER_FORMAT_BASE_REFUSAL,
    "citation_required": PROMPT_HEAD_CITATION,
    "struct_noground": PROMPT_HEAD_STRUCT_NOGROUND,
    # MMDocRAG gold answers run 21 words at the median and 88% are 6+ words, so a prompt demanding
    # "the shortest phrase" cannot produce a scoreable answer there. Keeps the refusal clause, which
    # P10 showed is the grounding-safest of the variants tested; only the length contract changes.
    "longform": PROMPT_HEAD_LONGFORM,
    # Comparison questions are 19.3% of the failures where the answer WAS delivered: the
    # reader picks one candidate instead of extracting each one's value and comparing.
    # `candidate_values` forces the extraction into the output before the choice is made.
    "struct_compare": PROMPT_HEAD_STRUCT_COMPARE,
    "struct_cot": PROMPT_HEAD_STRUCT_COT,
    "struct_locate": PROMPT_HEAD_STRUCT_LOCATE,
    "struct_fewshot": PROMPT_HEAD_STRUCT_FEWSHOT,
    "quote_only": PROMPT_HEAD_QUOTE_ONLY,
    "citation_multimodal": PROMPT_HEAD_CITATION_MM,
    "strict_paraphrase_a": PROMPT_HEAD_STRICT_PARAPHRASE_A,
    "strict_paraphrase_b": PROMPT_HEAD_STRICT_PARAPHRASE_B,
    "guess_paraphrase_a": PROMPT_HEAD_GUESS_PARAPHRASE_A,
    "guess_paraphrase_b": PROMPT_HEAD_GUESS_PARAPHRASE_B,
    "strict_fewshot": PROMPT_HEAD_STRICT_FEWSHOT,
    "guess_fewshot": PROMPT_HEAD_GUESS_FEWSHOT,
    "strict_format": PROMPT_HEAD_STRICT_FORMAT,
    "guess_format": PROMPT_HEAD_GUESS_FORMAT,
    "strict_cot": PROMPT_HEAD_STRICT_COT,
    "guess_cot": PROMPT_HEAD_GUESS_COT,
}


def load_jsonl(p): return [json.loads(l) for l in open(p)]


def build_chunks_index(dataset: str) -> Dict[str, Dict[str, Any]]:
    """chunk_id -> chunk record (text, section_path, media)."""
    out: Dict[str, Dict[str, Any]] = {}
    p = INDEX_ROOT / dataset / "chunks.jsonl"
    with open(p) as f:
        for line in f:
            c = json.loads(line)
            out[c["chunk_id"]] = c
    return out


def build_cards_index(dataset: str) -> Dict[str, Dict[str, Any]]:
    """card_id -> card record (chunk_ids, embed_text, bm25_text, ...)."""
    sec_dir = INDEX_ROOT / dataset / ("sec_index_md" if dataset in ("M3DocVQA", "VisDoMSlide") or dataset.startswith("VisDoMSlide_") else "sec_index")
    p = sec_dir / "section_cards_index.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    with open(p) as f:
        for line in f:
            c = json.loads(line)
            out[c["card_id"]] = c
    return out


def build_docs_index(dataset: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    p = INDEX_ROOT / dataset / "docs.jsonl"
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def serialize_section_path(c: Dict[str, Any]) -> str:
    sp = c.get("section_path") or []
    titles = []
    for node in sp:
        t = node.get("section_title") or node.get("title") or ""
        if t and t.lower() != "lead":
            titles.append(t)
    return " > ".join(titles) if titles else (c.get("heading") or "")


def cropped_pdf_region(pdf_path: str, page: int, bbox: List[float], resolution: int = 144):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        if page < 1 or page > len(pdf.pages): return None
        pi = pdf.pages[page - 1].to_image(resolution=resolution).original.convert("RGB")
        scale = resolution / 72.0
        x0, y0, x1, y1 = bbox
        px = (max(0, int(x0*scale)), max(0, int(y0*scale)),
              min(pi.width, int(x1*scale)), min(pi.height, int(y1*scale)))
        if px[2] - px[0] < 4 or px[3] - px[1] < 4: return None
        return pi.crop(px)


def resolve_card_chunk_ids(card: Dict[str, Any], cards_idx: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return chunk_ids for a candidate, resolving via sec_index when only card_id is present."""
    cid_list = card.get("chunk_ids") or []
    if cid_list: return list(cid_list)
    cand_card_id = card.get("card_id") or card.get("chunk_id")
    if cand_card_id and cand_card_id in cards_idx:
        c = cards_idx[cand_card_id]
        cid_list = c.get("chunk_ids") or []
        if not cid_list:
            for p in (c.get("paths") or []):
                cid_list.extend(p.get("chunk_ids") or [])
    return list(cid_list)


def card_parent_path(card: Dict[str, Any], cards_idx: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
    """Section path up to but not including the leaf. Used for sibling grouping."""
    cand_card_id = card.get("card_id") or card.get("chunk_id")
    c = cards_idx.get(cand_card_id or "") or {}
    path_titles = c.get("path_titles") or []
    # Drop leaf (last element) to get parent scope
    if len(path_titles) >= 1:
        return tuple(path_titles[:-1]) if len(path_titles) > 1 else tuple(path_titles)
    p = card.get("path_str") or c.get("path_str") or ""
    return tuple(p.split(" > "))


def build_parent_to_cards(cards_idx: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, Tuple[str, ...]], List[str]]:
    """(doc_id, parent_path_tuple) -> list of sibling card_ids sharing the same parent section."""
    out: Dict[Tuple[str, Tuple[str, ...]], List[str]] = {}
    for card_id, c in cards_idx.items():
        doc_id = c.get("doc_id")
        pt = c.get("path_titles") or []
        parent = tuple(pt[:-1]) if len(pt) > 1 else tuple(pt)
        key = (doc_id, parent)
        out.setdefault(key, []).append(card_id)
    return out


def build_evidence_pack(
    cards: List[Dict[str, Any]],
    chunks_idx: Dict[str, Dict[str, Any]],
    cards_idx: Dict[str, Dict[str, Any]],
    docs_idx: Dict[str, Dict[str, Any]],
    parent_to_cards: Dict[Tuple[str, Tuple[str, ...]], List[str]],
    semantic_visual_by_doc: Dict[str, List[Dict[str, Any]]],
    token_budget: int,
    image_token_cost: int,
    max_images_total: int,
    tokenizer,
    *,
    enable_sibling: bool = True,
    enable_semantic: bool = True,
    max_table_images: int = -1,
    max_figure_images: int = -1,
    text_source: str = "chunks",
    serialization_mode: str = "structured",
    hold_evidence_set: bool = False,
    evidence_budget_serialization_mode: Optional[str] = None,
    delivery_trace: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[Any], int]:
    """Algorithm 1 (Appendix §C): ancestry-aware hybrid packing.

    For each top-K anchor card in rank order:
      Phase 1 — Anchor + Ancestry (mandatory header + first text chunk + own-chunk visuals)
      Phase 2 — Sibling cards sharing the same parent section (enable_sibling)
      Phase 3 — Semantic associates: top-M visual regions from same doc by
                query-visual cosine (enable_semantic), skipping already-used chunks.
    """
    text_parts: List[str] = []
    images: List[Any] = []
    used = 0

    def tok_count(s: str) -> int:
        if not s: return 0
        try: return len(tokenizer.encode(s, add_special_tokens=False))
        except Exception: return max(1, len(s) // 4)

    def can_fit(n: int) -> bool:
        return used + n <= token_budget

    seen_chunk_ids: set = set()
    seen_visual_ids: set = set()
    n_table_placed: int = 0
    n_figure_placed: int = 0
    pending_header: str = ""
    trace_origin: str = ""
    trace_parent: str = ""

    if text_source not in {"chunks", "card"}:
        raise ValueError(f"unsupported text_source={text_source}")
    if serialization_mode not in {"structured", "concat"}:
        raise ValueError(f"unsupported serialization_mode={serialization_mode}")
    if evidence_budget_serialization_mode not in {None, "structured", "concat"}:
        raise ValueError(
            "unsupported evidence_budget_serialization_mode="
            f"{evidence_budget_serialization_mode}"
        )
    budget_serialization_mode = (
        evidence_budget_serialization_mode or serialization_mode
    )

    def emit_text_chunk(chunk: Dict[str, Any], force_id: Optional[str] = None) -> bool:
        nonlocal used, pending_header
        cid = force_id or chunk.get("chunk_id")
        if cid in seen_chunk_ids: return False
        body = (chunk.get("text") or "").strip()
        if not body: return False
        meta = (
            f"[UNIT id={cid} | Type=Text]\n"
            if serialization_mode == "structured" else ""
        )
        budget_meta = (
            f"[UNIT id={cid} | Type=Text]\n"
            if budget_serialization_mode == "structured" else ""
        )
        format_cost = tok_count(budget_meta)
        if hold_evidence_set and evidence_budget_serialization_mode is None:
            # Preserve legacy behavior for old callers. Verified paired-format
            # controls set evidence_budget_serialization_mode explicitly.
            format_cost = 0
        cost = format_cost + tok_count(body) + 1
        if not can_fit(cost): return False
        text_parts.append(meta + body + "\n")
        if delivery_trace is not None:
            delivery_trace.append({"unit_id": cid, "doc_id": chunk.get("doc_id"),
                                   "text": pending_header + meta + body + "\n",
                                   "body_text": body,
                                   "origin": trace_origin,
                                   "parent_carrier_id": trace_parent})
        pending_header = ""
        used += cost
        seen_chunk_ids.add(cid)
        return True

    def emit_visual_region(doc_id: str, chunk_id: str, region: Dict[str, Any]) -> bool:
        nonlocal used, n_table_placed, n_figure_placed
        if len(images) >= max_images_total: return False
        vid = (chunk_id, region.get("id"))
        if vid in seen_visual_ids: return False
        pdf_path = (docs_idx.get(doc_id, {}) or {}).get("pdf_path")
        if not pdf_path or not os.path.exists(pdf_path): return False
        try:
            img = cropped_pdf_region(pdf_path, int(region["page"]), region["bbox"])
        except Exception:
            return False
        if img is None: return False
        kind = "Table" if "table" in str(region.get("kind","")).lower() else "Figure"
        # Per-type cap enforcement (max_table_images, max_figure_images). -1 = unlimited (use max_images_total only).
        if kind == "Table" and max_table_images >= 0 and n_table_placed >= max_table_images:
            return False
        if kind != "Table" and max_figure_images >= 0 and n_figure_placed >= max_figure_images:
            return False
        label = f"[UNIT id={region.get('id') or 'r'} | Type={kind} | page={region.get('page')}]\n<|image|>\n"
        cost = tok_count(label) + image_token_cost
        if not can_fit(cost): return False
        images.append(img)
        text_parts.append(label)
        if delivery_trace is not None:
            rgb_payload = img.tobytes()
            delivery_trace.append({
                "unit_id": f"{chunk_id}#region#{region.get('id') or 'r'}",
                "doc_id": doc_id,
                "modality": "image",
                "text": label,
                "body_text": "",
                "origin": trace_origin,
                "parent_carrier_id": trace_parent,
                "source_path": pdf_path,
                "page": int(region["page"]),
                "bbox": [float(value) for value in region["bbox"]],
                "region_id": region.get("id"),
                "region_kind": kind,
                "render_resolution_dpi": 144,
                "payload_mode": img.mode,
                "payload_size": list(img.size),
                "payload_rgb_sha256": hashlib.sha256(rgb_payload).hexdigest(),
                "payload_rgb_bytes": len(rgb_payload),
            })
        used += cost
        seen_visual_ids.add(vid)
        if kind == "Table":
            n_table_placed += 1
        else:
            n_figure_placed += 1
        return True

    def emit_card_header(
        doc_id: str, cand_card_id: str, doc_title: str, path_str: str,
    ) -> bool:
        nonlocal used, pending_header
        header = f"\n# {doc_title}\n## {path_str}\n"
        h = tok_count(header) if budget_serialization_mode == "structured" else 0
        if hold_evidence_set and evidence_budget_serialization_mode is None:
            h = 0
        if not can_fit(h): return False
        if serialization_mode == "structured":
            text_parts.append(header)
            if delivery_trace is not None:
                # Record the header at the point where it is serialized.  A
                # card can emit an image before its first non-empty text chunk;
                # deferring this record to the text chunk would then invert the
                # actual reader-visible order in the lineage trace.
                delivery_trace.append({
                    "unit_id": f"header:{cand_card_id}",
                    "doc_id": doc_id,
                    "modality": "text",
                    "text": header,
                    "body_text": "",
                    "origin": trace_origin,
                    "parent_carrier_id": trace_parent,
                })
                pending_header = ""
            else:
                pending_header = header
        used += h
        return True

    def emit_whole_card(
        card: Dict[str, Any], *, include_ancestry: bool, origin: str,
        parent_carrier_id: Optional[str] = None,
    ) -> bool:
        """Add one card's ancestry-header (optional) + text chunks + own-chunk visuals.
        Returns True if at least the header (or text) was placed."""
        nonlocal used, pending_header, trace_origin, trace_parent
        trace_origin = origin
        doc_id = card.get("doc_id")
        doc_title = (docs_idx.get(doc_id, {}) or {}).get("title", "")
        cand_card_id = card.get("card_id") or card.get("chunk_id")
        # For retrieved anchors the carrier is its own parent.  Expansion units
        # must point to the retrieved anchor that caused their insertion, not
        # to themselves; this makes ancestry a real retrieval->delivery edge.
        trace_parent = parent_carrier_id or cand_card_id or ""
        trace_before = len(delivery_trace) if delivery_trace is not None else 0
        if text_source == "card" and cand_card_id in seen_chunk_ids:
            return False
        meta = cards_idx.get(cand_card_id or "")
        path_str = card.get("path_str") or (meta.get("path_str") if meta else "") or ""
        if include_ancestry:
            if not emit_card_header(doc_id, cand_card_id or "card", doc_title, path_str):
                return False
        placed_any = False
        cid_list = resolve_card_chunk_ids(card, cards_idx)
        if text_source == "chunks":
            for cid in cid_list:
                chunk = chunks_idx.get(cid)
                if not chunk: continue
                if emit_text_chunk(chunk, force_id=cid):
                    placed_any = True
                m = chunk.get("media") or {}
                if isinstance(m, dict):
                    for r in (m.get("image_regions") or []):
                        if not emit_visual_region(doc_id, cid, r):
                            break
        # Fallback: inline text
        if text_source == "card" or not placed_any:
            inline = (card.get("text") or "").strip()
            if not inline and meta:
                section_parts = [
                    str(path.get("section_markdown") or "").strip()
                    for path in (meta.get("paths") or [])
                    if str(path.get("section_markdown") or "").strip()
                ]
                inline = "\n".join(section_parts).strip()
            if not inline and meta:
                inline = (meta.get("embed_text") or meta.get("bm25_text") or "").strip()
            if inline:
                mline = (
                    f"[UNIT id={cand_card_id or 'card'} | Type=Text]\n"
                    if serialization_mode == "structured" else ""
                )
                budget_mline = (
                    f"[UNIT id={cand_card_id or 'card'} | Type=Text]\n"
                    if budget_serialization_mode == "structured" else ""
                )
                format_cost = tok_count(budget_mline)
                if hold_evidence_set and evidence_budget_serialization_mode is None:
                    format_cost = 0
                cost = format_cost + tok_count(inline) + 1
                if can_fit(cost):
                    text_parts.append(mline + inline + "\n")
                    if delivery_trace is not None:
                        delivery_trace.append({"unit_id": cand_card_id or f"card-{len(seen_chunk_ids)}",
                                               "doc_id": doc_id,
                                               "text": pending_header + mline + inline + "\n",
                                               "body_text": inline,
                                               "origin": trace_origin,
                                               "parent_carrier_id": trace_parent})
                    pending_header = ""
                    used += cost
                    seen_chunk_ids.add(cand_card_id or f"card-{len(seen_chunk_ids)}")
                    placed_any = True
        # The production packer may emit ancestry even when every body unit is
        # deduplicated or the next body exceeds budget.  Preserve that actual
        # delivered carrier in the trace instead of silently dropping it.
        if (delivery_trace is not None and include_ancestry and
                len(delivery_trace) == trace_before and pending_header):
            delivery_trace.append({"unit_id": f"header:{cand_card_id or trace_before}",
                                   "doc_id": doc_id, "text": pending_header,
                                   "body_text": "",
                                   "origin": origin, "parent_carrier_id": trace_parent})
            pending_header = ""
        return placed_any

    # ---- Algorithm 1 loop ----
    semantic_used_region_rowcap = 4  # cap semantic associates per anchor
    for ci, anchor in enumerate(cards):
        if used >= token_budget: break
        doc_id = anchor.get("doc_id")
        # Phase 1: anchor + ancestry
        placed = emit_whole_card(anchor, include_ancestry=True, origin="retrieved-anchor")
        if not placed:
            continue

        # Phase 2: siblings sharing the same parent section
        if enable_sibling:
            parent_key = (doc_id, card_parent_path(anchor, cards_idx))
            sibling_card_ids = parent_to_cards.get(parent_key, [])
            for sib_id in sibling_card_ids:
                if used >= token_budget: break
                if sib_id == (anchor.get("card_id") or anchor.get("chunk_id")):
                    continue
                sib_meta = cards_idx.get(sib_id)
                if not sib_meta: continue
                # Build a minimal card-like dict for reuse
                sib_card = {
                    "card_id": sib_id,
                    "doc_id": sib_meta.get("doc_id"),
                    "path_str": sib_meta.get("path_str"),
                    "chunk_ids": sib_meta.get("chunk_ids"),
                }
                # Sibling body added without repeating ancestry header
                emit_whole_card(
                    sib_card, include_ancestry=False, origin="sibling-augmentation",
                    parent_carrier_id=(anchor.get("card_id") or anchor.get("chunk_id") or ""),
                )

        # Phase 3: semantic associates (cross-section visuals from same doc)
        if enable_semantic:
            cands = (semantic_visual_by_doc or {}).get(doc_id) or []
            added = 0
            for m in cands:
                if added >= semantic_used_region_rowcap: break
                if used >= token_budget: break
                cid = m.get("chunk_id")
                rid = m.get("region_id")
                if (cid, rid) in seen_visual_ids: continue
                chunk = chunks_idx.get(cid) if cid else None
                if not chunk: continue
                media = chunk.get("media") or {}
                regs = media.get("image_regions") or []
                # find the matching region
                target = None
                for r in regs:
                    if str(r.get("id")) == str(rid):
                        target = r; break
                if target is None and regs:
                    target = regs[0]
                if target is None: continue
                if emit_visual_region(doc_id, cid, target):
                    added += 1

    return "".join(text_parts), images, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["M3DocVQA","FRAMES"])
    ap.add_argument("--retrieval", required=True, help="legacy jsonl with sectioncard_candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k_cards", type=int, default=10)
    ap.add_argument("--token_budget", type=int, default=14000)
    ap.add_argument("--image_token_cost", type=int, default=300)
    ap.add_argument("--max_images_total", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_sibling", action="store_true",
                    help="Disable Algorithm 1 Phase-2 sibling packing")
    ap.add_argument("--no_semantic", action="store_true",
                    help="Disable Algorithm 1 Phase-3 semantic-associate packing")
    args = ap.parse_args()

    os.environ.setdefault("USE_TF","0"); os.environ.setdefault("TRANSFORMERS_NO_TF","1")

    print("[load] text/chunks index", flush=True)
    chunks_idx = build_chunks_index(args.dataset)
    docs_idx = build_docs_index(args.dataset)
    cards_idx = build_cards_index(args.dataset)
    parent_to_cards = build_parent_to_cards(cards_idx)
    print(f"  chunks={len(chunks_idx)} docs={len(docs_idx)} cards={len(cards_idx)} parent_groups={len(parent_to_cards)}", flush=True)

    print("[load] retrieval", args.retrieval, flush=True)
    rrows = load_jsonl(args.retrieval)
    print(f"  questions={len(rrows)}", flush=True)

    print(f"[load] {args.model}", flush=True)
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    tokenizer = processor.tokenizer

    out_f = open(args.out, "w")
    from tqdm import tqdm
    n_done = 0
    t0 = time.time()
    for r in tqdm(rrows, desc="reader"):
        qid = r.get("id")
        question = r.get("question") or ""
        cards = (r.get("sectioncard_candidates") or [])[:args.top_k_cards]
        semantic_visual_by_doc = r.get("semantic_visual_by_doc") or {}
        ctx_text, images, used_tokens = build_evidence_pack(
            cards, chunks_idx, cards_idx, docs_idx,
            parent_to_cards=parent_to_cards,
            semantic_visual_by_doc=semantic_visual_by_doc,
            token_budget=args.token_budget,
            image_token_cost=args.image_token_cost,
            max_images_total=args.max_images_total,
            tokenizer=tokenizer,
            enable_sibling=not args.no_sibling,
            enable_semantic=not args.no_semantic,
        )
        prompt = PROMPT_HEAD.format(question=question) + ctx_text + PROMPT_TAIL

        # Build qwen-vl chat messages with images interleaved at the end (simple variant)
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_in], images=images if images else None, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0
            )
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        out_f.write(json.dumps({
            "id": qid, "question": question, "generated_answer": ans,
            "n_text_units_in_pack": ctx_text.count("Type=Text"),
            "n_image_units_in_pack": ctx_text.count("Type=Figure") + ctx_text.count("Type=Table"),
            "prompt_used_tokens": used_tokens,
        }, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 25 == 0:
            out_f.flush()
            elapsed = time.time() - t0
            print(f"  done={n_done}/{len(rrows)} rate={n_done/elapsed:.2f}q/s", flush=True)
    out_f.close()
    print(f"[done] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
