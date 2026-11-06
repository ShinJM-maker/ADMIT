#!/usr/bin/env python3
"""
HiKEY Reader (vLLM variant) — Qwen2.5-VL-7B-Instruct via vLLM for ~5x throughput
over the HuggingFace transformers wrapper.

Drop-in alternative to run_reader_qwen25vl.py. Same CLI flags + same input
schema (legacy retrieval jsonl) + same output (predictions jsonl). Reuses
build_evidence_pack and friends from run_reader_qwen25vl.

Differences from the HF version:
  * Loads Qwen2.5-VL via vllm.LLM with single-GPU tensor_parallel_size=1
    (multi-GPU sharding via separate processes, like the HF sharded launcher).
  * Generates in batches via LLM.generate with multi_modal_data per-prompt.
  * Greedy decoding via SamplingParams(temperature=0, top_p=1, max_tokens=...).

Usage:
  python run_reader_vllm.py --dataset M3DocVQA --retrieval ... --out ... \
       --top_k_cards 10 --token_budget 14000 --max_images_total 4 \
       --max_new_tokens 64 --batch_size 16
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

INDEX_ROOT = Path(os.environ.get("ADMIT_INDEX", str(Path(__file__).resolve().parents[2] / "data/index")))
HIKEY_CODE = Path(__file__).resolve().parents[2] / "harness"
if str(HIKEY_CODE) not in sys.path:
    sys.path.insert(0, str(HIKEY_CODE))

# Reuse helpers from the HF Reader implementation
from run_reader_qwen25vl import (
    PROMPT_HEAD, PROMPT_TAIL, PROMPT_VARIANTS,
    load_jsonl, build_chunks_index, build_docs_index, build_cards_index,
    build_parent_to_cards, build_evidence_pack,
)


def _dataset_choices():
    """Return all valid dataset names: fixed baselines + any VisDoMSlide_ocr* or M3DocVQA_corrupt* index dirs."""
    base = ["M3DocVQA", "FRAMES", "VisDoMSlide",
            "VisDoMSlide_ocr25", "VisDoMSlide_ocr50", "VisDoMSlide_ocr75"]
    import re
    # Pattern-based: include any VisDoMSlide_ocr{pct}_s{seed} dirs that exist in INDEX_ROOT
    patterns = [
        re.compile(r"^VisDoMSlide_ocr\d+_s\d+$"),
        re.compile(r"^M3DocVQA_corrupt\d+$"),
    ]
    extras = sorted(
        p.name for p in INDEX_ROOT.iterdir()
        if p.is_dir() and any(pat.match(p.name) for pat in patterns)
    )
    # Deduplicate while preserving order
    seen = set(base)
    for e in extras:
        if e not in seen:
            base.append(e)
            seen.add(e)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=_dataset_choices())
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k_cards", type=int, default=10)
    ap.add_argument("--token_budget", type=int, default=14000)
    ap.add_argument("--image_token_cost", type=int, default=300)
    ap.add_argument("--max_images_total", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="vLLM tensor_parallel_size; >1 requires CUDA_VISIBLE_DEVICES to expose that many GPUs.")
    ap.add_argument("--no_sibling", action="store_true")
    ap.add_argument("--no_semantic", action="store_true")
    ap.add_argument("--prompt_variant", choices=sorted(PROMPT_VARIANTS.keys()), default="strict",
                    help="Which system-prompt template to use. 'strict' keeps the original refusal clause; "
                         "'guess' replaces it with a best-guess instruction (for FR gap diagnosis).")
    ap.add_argument("--max_table_images", type=int, default=-1,
                    help="Per-type cap for Table-kind visual regions (-1 = unlimited, use max_images_total only).")
    ap.add_argument("--max_figure_images", type=int, default=-1,
                    help="Per-type cap for Figure-kind visual regions (-1 = unlimited, use max_images_total only).")
    ap.add_argument("--logprobs", type=int, default=0,
                    help="If >0, request top-K logprobs per generated token from vLLM and emit per-token logprob arrays in output rows.")
    ap.add_argument("--evidence_packet_out", default=None,
                    help="Optional JSONL containing the exact post-packing ctx_text shown to the reader.")
    ap.add_argument("--text_source", choices=("chunks", "card"), default="chunks",
                    help="Text body source for controlled source x serialization experiments.")
    ap.add_argument("--serialization_mode", choices=("structured", "concat"), default="structured",
                    help="Structured unit metadata/ancestry headers or body-only concatenation.")
    ap.add_argument("--hold_evidence_set", action="store_true",
                    help="Hold body selection fixed across format swaps; with no explicit evidence-budget mode, preserves the legacy zero-wrapper-cost behavior.")
    ap.add_argument("--evidence_budget_serialization_mode", choices=("structured", "concat"), default=None,
                    help="Select a common evidence set using this format's wrapper cost while rendering --serialization_mode.")
    ap.add_argument("--fail_on_generation_error", action="store_true",
                    help="Abort rather than converting a failed batch into empty answers.")
    ap.add_argument("--disable_thinking", action="store_true",
                    help="Request the model chat template's non-thinking mode when supported. "
                         "This is useful for short-answer and structured-output audits.")
    args = ap.parse_args()
    prompt_head = PROMPT_VARIANTS[args.prompt_variant]

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    print("[load] indexes", flush=True)
    chunks_idx = build_chunks_index(args.dataset)
    docs_idx = build_docs_index(args.dataset)
    cards_idx = build_cards_index(args.dataset)
    parent_to_cards = build_parent_to_cards(cards_idx)
    print(f"  chunks={len(chunks_idx)} docs={len(docs_idx)} cards={len(cards_idx)}", flush=True)

    print("[load] retrieval", args.retrieval, flush=True)
    rrows = load_jsonl(args.retrieval)
    print(f"  questions={len(rrows)}", flush=True)

    print(f"[load] vLLM model {args.model}", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": max(1, args.max_images_total)},
        enforce_eager=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
    )
    tokenizer = llm.get_tokenizer()

    sp_kwargs = dict(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    if args.logprobs > 0:
        sp_kwargs["logprobs"] = int(args.logprobs)
    sampling = SamplingParams(**sp_kwargs)

    def _per_row_sampling(r):
        """If the row has _seed or _temperature, build a custom SamplingParams; else return the default."""
        t = r.get("_temperature")
        s = r.get("_seed")
        if t is None and s is None:
            return sampling
        kw = dict(
            temperature=float(t) if t is not None else 0.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            seed=int(s) if s is not None else None,
        )
        if args.logprobs > 0:
            kw["logprobs"] = int(args.logprobs)
        return SamplingParams(**kw)

    out_f = open(args.out, "w")
    packet_f = open(args.evidence_packet_out, "w") if args.evidence_packet_out else None
    t0 = time.time()
    n_done = 0
    n_total = len(rrows)

    # Prepare each question's prompt+images using the HF helper
    def make_request(r):
        question = r.get("question") or ""
        cards = (r.get("sectioncard_candidates") or [])[:args.top_k_cards]
        semantic_visual_by_doc = r.get("semantic_visual_by_doc") or {}
        delivery_trace = []
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
            max_table_images=args.max_table_images,
            max_figure_images=args.max_figure_images,
            text_source=args.text_source,
            serialization_mode=args.serialization_mode,
            hold_evidence_set=args.hold_evidence_set,
            evidence_budget_serialization_mode=args.evidence_budget_serialization_mode,
            delivery_trace=delivery_trace,
        )
        # Honor explicit image_paths (e.g., query-aligned crops) — append after evidence pack.
        extra_paths = r.get("image_paths") or []
        extra_marker = ""
        if extra_paths and len(images) < args.max_images_total:
            from PIL import Image as _PIL_Image
            for p in extra_paths:
                if len(images) >= args.max_images_total: break
                try:
                    if not p or not os.path.exists(p): continue
                    img = _PIL_Image.open(p).convert("RGB")
                    images.append(img)
                    extra_marker += f"[UNIT id=qcrop{len(images)} | Type=QueryCrop]\n<|image|>\n"
                except Exception:
                    continue
        prompt_text = prompt_head.format(question=question) + ctx_text + extra_marker + PROMPT_TAIL
        # vLLM Qwen2.5-VL chat template requires building the messages and applying it.
        # Use the model's tokenizer.apply_chat_template via processor-style.
        # Simpler: use plain text + multi_modal_data; vLLM will splice <|image_pad|> tokens.
        # Model-family-aware prompt construction
        model_lower = args.model.lower()
        is_phi = "phi" in model_lower
        is_llava = "llava" in model_lower
        is_internvl = "internvl" in model_lower
        if images and is_phi:
            # Phi-3.5-vision needs <|image_N|> placeholders in text, not content parts
            img_markers = "".join(f"<|image_{i+1}|>\n" for i in range(len(images)))
            phi_prompt = img_markers + prompt_text
            messages = [{"role": "user", "content": phi_prompt}]
        elif images and is_llava:
            # LLaVA-v1.6: build prompt directly, bypassing chat template which
            # mangles <image> tokens. Use the vicuna/llava conversation format.
            img_markers = "<image>\n" * len(images)
            llava_prompt = f"USER: {img_markers}{prompt_text}\nASSISTANT:"
            # Pre-check: estimate total tokens. LLaVA-v1.6 image can use up to
            # 5*576+576=3456 tokens (6 patches). Use 3500 as conservative estimate.
            # If text tokens + image tokens would exceed max_model_len, drop the image.
            text_token_count = len(tokenizer.encode(llava_prompt))
            est_image_tokens = args.max_model_len  # LLaVA-v1.6 images consume too many tokens to coexist with text evidence
            if text_token_count + est_image_tokens + args.max_new_tokens > args.max_model_len:
                # Drop images — fall through to text-only path below
                images = []
            else:
                # Skip apply_chat_template for LLaVA with images
                req = {"prompt": llava_prompt, "stat": (ctx_text.count("Type=Text"),
                                                         ctx_text.count("Type=Figure") + ctx_text.count("Type=Table"),
                                                         used_tokens),
                       "reader_ctx_text": ctx_text,
                       "delivery_trace": delivery_trace}
                req["multi_modal_data"] = {"image": images}
                return req
        elif images and is_internvl:
            # InternVL: insert <image> placeholders in text, bypass content parts
            img_markers = "".join(f"<image>\n" for _ in images)
            internvl_prompt = img_markers + prompt_text
            messages = [{"role": "user", "content": internvl_prompt}]
        elif images:
            # Qwen / default: use OpenAI-style content parts
            content_parts = []
            for img in images:
                content_parts.append({"type": "image"})
            content_parts.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_parts}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        try:
            templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=not args.disable_thinking,
            )
        except TypeError:
            templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            templated = prompt_text
        reader_prompt_tokens = len(
            tokenizer.encode(templated, add_special_tokens=False)
        )
        if (
            not images
            and reader_prompt_tokens + args.max_new_tokens > args.max_model_len
        ):
            raise ValueError(
                f"rendered prompt plus generation allowance has "
                f"{reader_prompt_tokens}+{args.max_new_tokens} tokens, above "
                f"max_model_len={args.max_model_len}"
            )
        req = {"prompt": templated, "stat": (ctx_text.count("Type=Text"),
                                              ctx_text.count("Type=Figure") + ctx_text.count("Type=Table"),
                                              used_tokens),
               "reader_ctx_text": ctx_text,
               "delivery_trace": delivery_trace,
               "reader_prompt_tokens": reader_prompt_tokens}
        if images:
            req["multi_modal_data"] = {"image": images}
        return req

    pending: List[Dict[str, Any]] = []
    pending_meta: List[Dict[str, Any]] = []

    def flush(batch, batch_meta):
        nonlocal n_done
        prompts = []
        per_sampling = []
        for r, mr in zip(batch, batch_meta):
            mm = r.get("multi_modal_data")
            entry = {"prompt": r["prompt"]}
            if mm: entry["multi_modal_data"] = mm
            prompts.append(entry)
            per_sampling.append(_per_row_sampling(mr))
        # if every row uses the default sampling, pass one SamplingParams for speed;
        # else pass a list of SamplingParams to vLLM (per-request).
        try:
            if all(sp is sampling for sp in per_sampling):
                outputs = llm.generate(prompts, sampling)
            else:
                outputs = llm.generate(prompts, per_sampling)
        except (ValueError, RuntimeError) as e:
            if args.fail_on_generation_error:
                raise
            # Likely prompt-too-long error; emit empty answers for all items in this batch
            # (retrying one-at-a-time can hang vLLM if the prompt exceeds KV cache capacity)
            print(f"[warn] batch generate failed ({e}), emitting empty for batch of {len(prompts)}", flush=True)
            outputs = []
            for _ in prompts:
                class _DummyOutput:
                    def __init__(self):
                        self.text = ""
                        self.token_ids = []
                        self.logprobs = None
                class _DummyRequestOutput:
                    def __init__(self):
                        self.outputs = [_DummyOutput()]
                outputs.append(_DummyRequestOutput())
        # Order is preserved by vLLM
        for meta_r, out in zip(batch_meta, outputs):
            ans = out.outputs[0].text.strip() if out.outputs else ""
            stat = batch[outputs.index(out)]["stat"] if False else None  # not used
        # Simpler order-preserving emit
        for i, out in enumerate(outputs):
            meta_r = batch_meta[i]
            ans = out.outputs[0].text.strip() if out.outputs else ""
            (n_text, n_image, used) = batch[i]["stat"]
            row = {
                "id": meta_r.get("id"),
                "question": meta_r.get("question") or "",
                "generated_answer": ans,
                "dataset": args.dataset,
                "reader_model": args.model,
                "prompt_variant": args.prompt_variant,
                "token_budget": args.token_budget,
                "max_model_len": args.max_model_len,
                "max_images_total": args.max_images_total,
                "thinking_enabled": not args.disable_thinking,
                "max_new_tokens": args.max_new_tokens,
                "n_text_units_in_pack": n_text,
                "n_image_units_in_pack": n_image,
                "prompt_used_tokens": used,
                "text_source": args.text_source,
                "serialization_mode": args.serialization_mode,
                "hold_evidence_set": args.hold_evidence_set,
                "evidence_budget_serialization_mode": args.evidence_budget_serialization_mode,
            }
            ctx_text = batch[i]["reader_ctx_text"]
            rendered_prompt = batch[i]["prompt"]
            body_trace = [
                {
                    "unit_id": item.get("unit_id"),
                    "doc_id": item.get("doc_id"),
                    "body_sha256": hashlib.sha256(
                        str(item.get("body_text") or "").encode("utf-8")
                    ).hexdigest(),
                }
                for item in batch[i].get("delivery_trace", [])
                if item.get("body_text")
            ]
            body_trace_json = json.dumps(body_trace, ensure_ascii=False, sort_keys=True)
            row["reader_ctx_sha256"] = hashlib.sha256(ctx_text.encode("utf-8")).hexdigest()
            row["reader_prompt_sha256"] = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
            row["evidence_body_trace_sha256"] = hashlib.sha256(body_trace_json.encode("utf-8")).hexdigest()
            row["reader_prompt_tokens"] = batch[i].get("reader_prompt_tokens")
            if row["reader_prompt_tokens"] is None:
                row["reader_prompt_tokens"] = len(
                    tokenizer.encode(rendered_prompt, add_special_tokens=False)
                )
            if packet_f is not None:
                packet_f.write(json.dumps({
                    "id": meta_r.get("id"),
                    "reader_ctx_sha256": row["reader_ctx_sha256"],
                    "reader_prompt_sha256": row["reader_prompt_sha256"],
                    "evidence_body_trace_sha256": row["evidence_body_trace_sha256"],
                    "evidence_body_trace": body_trace,
                    "reader_ctx_text": ctx_text,
                    "reader_prompt_tokens": row["reader_prompt_tokens"],
                    "dataset": args.dataset,
                    "reader_model": args.model,
                    "prompt_variant": args.prompt_variant,
                    "text_source": args.text_source,
                    "serialization_mode": args.serialization_mode,
                    "tokenizer": args.model,
                    "token_budget": args.token_budget,
                    "max_model_len": args.max_model_len,
                    "max_images_total": args.max_images_total,
                    "thinking_enabled": not args.disable_thinking,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_used_tokens": used,
                    "budget_serialization_mode": args.evidence_budget_serialization_mode,
                    "truncation_policy": "common-evidence-selection@structured-wrapper-cost",
                }, ensure_ascii=False) + "\n")
            if args.logprobs > 0 and out.outputs and getattr(out.outputs[0], "logprobs", None) is not None:
                token_lps = []
                token_ids = list(out.outputs[0].token_ids) if out.outputs[0].token_ids else []
                lp_seq = out.outputs[0].logprobs or []
                for tok_id, pos in zip(token_ids, lp_seq):
                    if pos is None:
                        token_lps.append(None)
                        continue
                    entry = pos.get(tok_id) if isinstance(pos, dict) else None
                    if entry is None:
                        # take any value's logprob as fallback (vLLM versions differ)
                        try: any_v = next(iter(pos.values())) if isinstance(pos, dict) else None
                        except Exception: any_v = None
                        token_lps.append(getattr(any_v, "logprob", None) if any_v is not None else None)
                    else:
                        token_lps.append(getattr(entry, "logprob", entry if isinstance(entry, (int, float)) else None))
                row["token_logprobs"] = token_lps
                row["n_generated_tokens"] = len(token_lps)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
        out_f.flush()
        if packet_f is not None:
            packet_f.flush()

    for ri, r in enumerate(rrows):
        try:
            req = make_request(r)
        except Exception as e:
            if args.fail_on_generation_error:
                raise
            print(f"[warn] skip {r.get('id')}: {e}", flush=True)
            continue
        pending.append(req)
        pending_meta.append(r)
        if len(pending) >= args.batch_size:
            flush(pending, pending_meta)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(f"  done={n_done}/{n_total} rate={rate:.2f}q/s elapsed={elapsed:.1f}s", flush=True)
            pending = []; pending_meta = []
    if pending:
        flush(pending, pending_meta)
    out_f.close()
    if packet_f is not None:
        packet_f.close()
    elapsed = time.time() - t0
    print(f"[done] wrote {args.out}  total={n_done} elapsed={elapsed:.1f}s rate={n_done/elapsed:.2f}q/s", flush=True)


if __name__ == "__main__":
    main()
