#!/usr/bin/env python3
"""DUDE PlanRAG transfer under an immutable exact reader-visible packet.

The submitted admission and agent functions are reused unchanged. Dataset-specific
retrieval is supplied by a BM25 index over the fixed DUDE stage-2 card pool, and
the M3DocVQA pack serializer is replaced with a deterministic inline serializer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO / "harness/agents.py"
STAGE2 = Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "portability/dude_stage2.jsonl"
OUT = Path(__file__).resolve().parent / "results/dude_dynamic"
MODEL = os.environ.get("ADMIT_READER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
NO_COMMIT = "[NO_COMMIT]"


def load_runner():
    spec = importlib.util.spec_from_file_location("admit_dude_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SimpleBM25:
    def __init__(self, texts: list[str], tokenizer):
        self.tokenizer = tokenizer
        tokens = [tokenizer(text) for text in texts]
        self.lengths = np.asarray([len(row) for row in tokens], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df = Counter()
        for index, row in enumerate(tokens):
            counts = Counter(row)
            for term, count in counts.items():
                self.postings[term].append((index, count))
                df[term] += 1
        n = len(texts)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        k1, b = 1.5, 0.75
        for term in query_tokens:
            for index, tf in self.postings.get(term, []):
                denom = tf + k1 * (1.0 - b + b * self.lengths[index] / self.avgdl)
                scores[index] += self.idf[term] * (tf * (k1 + 1.0) / (denom + 1e-9))
        return scores


def inline_pack(cards, *_args, token_budget, tokenizer, **_kwargs):
    units = []
    for index, card in enumerate(cards, 1):
        text = card.get("text") or card.get("embed_text") or card.get("bm25_text") or ""
        units.append(
            f"[UNIT {index} | DOC {card.get('doc_id', '')} | ID {card.get('card_id', '')}]\n{text.strip()}"
        )
    context = "\n\n".join(units)
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(token_ids) > token_budget:
        token_ids = token_ids[:token_budget]
        context = tokenizer.decode(token_ids, skip_special_tokens=True)
    return context, [], len(token_ids)


class InlineReader:
    def __init__(self, runner, model_path: str, gpu_mem_util: float):
        from vllm import LLM, SamplingParams

        self.runner = runner
        self.SamplingParams = SamplingParams
        self.token_budget = 12000
        self.max_images_total = 0
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=16384,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
        self._chunks_idx = {}
        self._cards_idx = {}
        self._docs_idx = {}
        self._parent_to_cards = {}

    def _make_prompt(self, question, sec_cands, prompt_variant="strict", top_k_cards=10):
        context, _, used = inline_pack(
            (sec_cands or [])[:top_k_cards], token_budget=self.token_budget,
            tokenizer=self.tokenizer,
        )
        text = self.runner.PROMPT_VARIANTS[prompt_variant].format(question=question)
        text += context + self.runner.PROMPT_TAIL
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = text
        return prompt, context, used

    def generate_batch(self, prompts):
        outputs = self.llm.generate([{"prompt": prompt} for prompt in prompts], self.sampling)
        return [item.outputs[0].text.strip() if item.outputs else "" for item in outputs]

    def reader_call(self, question, sec_cands, prompt_variant="strict", top_k_cards=10):
        prompt, context, used = self._make_prompt(question, sec_cands, prompt_variant, top_k_cards)
        return self.generate_batch([prompt])[0], context, used

    def reasoning_call(self, prompt_text, max_new_tokens=96):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = prompt_text
        sampling = self.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        output = self.llm.generate([{"prompt": prompt}], sampling)[0]
        return output.outputs[0].text.strip() if output.outputs else ""


def cards_from_rows(rows: list[dict]) -> list[dict]:
    cards: dict[str, dict] = {}
    for row in rows:
        for item in row["sectioncard_candidates"]:
            card_id = str(item["card_id"])
            cards.setdefault(card_id, {
                **item,
                "card_id": card_id,
                "embed_text": item.get("text", ""),
                "bm25_text": item.get("text", ""),
            })
    return [cards[key] for key in sorted(cards)]


def selected_rows(rows: list[dict], n: int = 300) -> list[dict]:
    scalar = [row for row in rows if isinstance(row.get("answer"), list) and len(row["answer"]) == 1]
    scalar.sort(key=lambda row: str(row["id"]))
    rng = np.random.default_rng(20260725)
    indices = sorted(rng.choice(len(scalar), size=min(n, len(scalar)), replace=False).tolist())
    return [scalar[index] for index in indices]


def sec_cands(card_ids: list[str], card_by_id: dict[str, dict]) -> list[dict]:
    return [card_by_id[card_id] for card_id in card_ids if card_id in card_by_id]


def final_only_from_vanilla(runner, reader, vanilla, question, cards_by_id):
    final_cards = sec_cands(vanilla.get("final_pack_card_ids", []), cards_by_id)
    output, hook = runner.final_support_retry_hook(
        reader,
        vanilla.get("final_raw_answer", ""),
        question,
        final_cards,
        vanilla.get("final_pack_text", ""),
        top_k_cards=10,
    )
    result = dict(vanilla)
    result["agent"] = "planrag"
    result["arm"] = "final_support_retry"
    result["final_committed_answer"] = output
    result["final_hook_log"] = hook
    result["memory_value_at_final"] = vanilla.get("final_raw_answer", "")
    result["n_repair_calls"] = vanilla.get("n_repair_calls", 0) + hook.get("n_repair_calls", 0)
    return result


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def em(answer: str, gold: str) -> int:
    a, g = normalize(answer), normalize(gold)
    return int(bool(a and g and (a in g or g in a)))


def pack_from_ids(reader, ids, card_by_id):
    return inline_pack(sec_cands(ids, card_by_id), token_budget=reader.token_budget, tokenizer=reader.tokenizer)[0]


def memory_values(result: dict, arm: str, reader, card_by_id):
    values = []
    sub_questions = result.get("sub_questions", [])
    for index, item in enumerate(result.get("sub_logs", [])):
        pack = pack_from_ids(reader, item.get("pack_card_ids", []), card_by_id)
        question = sub_questions[index] if index < len(sub_questions) else ""
        values.append((item.get("committed_answer", ""), pack, question))
    final_pack = result.get("final_pack_text", "")
    if arm == "admit":
        value = (result.get("final_hook_log") or {}).get(
            "memory_value_at_final", result.get("final_committed_answer", "")
        )
    elif arm == "final_support_retry":
        value = result.get("memory_value_at_final", result.get("final_raw_answer", ""))
    else:
        value = result.get("final_committed_answer", "")
    values.append((value, final_pack, None))
    return values


def annotate(runner, result: dict, arm: str, question: str, gold: str, reader, card_by_id):
    writes = memory_values(result, arm, reader, card_by_id)
    contaminated = any(
        value != NO_COMMIT and not runner.is_refusal(value)
        and not runner.cavec_admit(value, pack, write_question or question)[0]
        for value, pack, write_question in writes
    )
    no_commit = any(value == NO_COMMIT for value, _, _ in writes)
    final = result.get("final_committed_answer", "")
    final_pack = result.get("final_pack_text", "")
    return {
        "raw_em": em(final, gold),
        "lexical_trajectory_contamination": int(contaminated),
        "has_no_commit": int(no_commit),
        "output_unsup_nonrefusal": int(
            not runner.is_refusal(final) and not runner.support_in_pack(final, final_pack)
        ),
        "calls_q": result.get("n_reader_calls", 0) + result.get("n_repair_calls", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--gpu-mem-util", type=float, default=0.82)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    runner.build_evidence_pack = inline_pack
    rows = read_jsonl(STAGE2)
    cards = cards_from_rows(rows)
    card_by_id = {card["card_id"]: card for card in cards}
    chosen = selected_rows(rows, args.limit)
    mine = [row for index, row in enumerate(chosen) if index % args.num_shards == args.shard_id]
    reader = InlineReader(runner, args.model, args.gpu_mem_util)
    bm25 = SimpleBM25([card["bm25_text"] for card in cards], runner.tok)

    output_path = OUT / f"dude_planrag.shard{args.shard_id}.jsonl"
    started = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(mine, 1):
            question = row["question"]
            gold = row["answer"][0]
            vanilla = runner.run_planrag(reader, question, bm25, cards, {}, top_k=10)
            arms = {
                "final_support_retry": final_only_from_vanilla(
                    runner, reader, vanilla, question, card_by_id
                ),
                "fail_open": runner.run_planrag(
                    reader, question, bm25, cards, {}, top_k=10,
                    with_cavec=True, cavec_fail_closed=False,
                ),
                "hard_refuse": runner.run_planrag(
                    reader, question, bm25, cards, {}, top_k=10,
                    with_cavec=True, cavec_fail_closed=True,
                ),
                "admit": runner.run_planrag(
                    reader, question, bm25, cards, {}, top_k=10,
                    with_cavec_mfc=True,
                ),
            }
            record = {
                "qid": row["id"],
                "question": question,
                "gold": gold,
                "dataset": "DUDE",
                "agent": "planrag",
                "packet_contract": "immutable_inline_post_token_truncation",
                "arms": {},
            }
            for arm, result in arms.items():
                record["arms"][arm] = {
                    "metrics": annotate(runner, result, arm, question, gold, reader, card_by_id),
                    "result": result,
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{args.shard_id}] {index}/{len(mine)} {row['id']} "
                f"em=" + "/".join(str(record['arms'][arm]['metrics']['raw_em']) for arm in arms)
                + f" elapsed={time.time()-started:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
