#!/usr/bin/env python3
"""
HiKEY Stage-2 multimodal MaxSim retrieval (paper §3.2 — full version).

Inputs:
  - Pre-built text indexes (BM25 + Qwen3-Embedding) for both Doc_card and Sec_card,
    from the prior MultiDocGraphRAG pipeline.
  - Pre-built visual embeddings (this repo: encode_visual_regions.py) keyed by
    (chunk_id, region_id) using nvidia/MM-Embed.
  - chunks.jsonl with section_path + media.image_regions (text-or-visual unit info).
  - questions.json (val.json).

Pipeline per query q:
  Stage-1 (Doc routing): S_doc(d, q) = α · mm(BM25(x_d, q)) + (1-α) · mm(text-dense(x_d, q))
                          → keep top-K_doc candidate documents.

  Stage-2 (Section MaxSim): for each section s in candidate docs:
     for each unit c in s:
       if τ(c) == Text:   s(c,q) = β · BM25(text(c)) + (1-β) · text_dense(text(c), q)
       if τ(c) ∈ {Tab,Img}: s(c,q) = γ · vis(c, q)_mmembed
                                    + (1-γ) · text_hybrid(ctx(c), q)
     S_sec(s, q) = max_c s(c, q)
  S_final(s, q) = λ · S_doc(d(s), q) + (1-λ) · S_sec(s, q)

  Top-K_sec sections are emitted; doc ranking from Stage-1 is also emitted
  for downstream eval.

Score normalisation: per-query min-max on each component before fusing.

Output (JSONL, legacy-compatible with eval_chunk_retrieval_frames.py):
  {id, question, doc_candidates, sectioncard_candidates}

NOTE: This script intentionally reuses the prior repo's BM25/dense Doc_card +
Sec_card indexes — Stage-1 already matches the paper closely. The novel logic
here is the per-unit Type-Specific scoring with MM-Embed visual fusion.
"""
from __future__ import annotations
import argparse, json, os, pickle, re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

INDEX_ROOT = Path(os.environ.get("ADMIT_INDEX", os.environ.get("HIKEY_INDEX", str(Path(__file__).resolve().parents[1] / "data/index"))))


# --------- BM25 — compat with prior repo's pickled BM25Index dataclass ---------

from dataclasses import dataclass as _dc
from typing import Tuple as _Tuple


@_dc
class BM25Index:
    """Compat shim — must match prior MultiDocGraphRAG pickled schema."""
    doc_ids: List[str]
    doc_len: List[int]
    avgdl: float
    k1: float
    b: float
    doc_freq: Dict[str, int]
    idf: Dict[str, float]
    postings: Dict[str, List[_Tuple[int, int]]]
    tokenizer: str = ""

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        N = len(self.doc_ids)
        scores = np.zeros((N,), dtype=np.float32)
        avgdl = float(self.avgdl)
        k1 = float(self.k1); b = float(self.b)
        for term in query_tokens:
            if term not in self.postings: continue
            idf = float(self.idf.get(term, 0.0))
            for doc_idx, tf in self.postings[term]:
                dl = float(self.doc_len[doc_idx])
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                scores[doc_idx] += idf * (tf * (k1 + 1.0) / (denom + 1e-9))
        return scores


# Make BM25Index discoverable under __main__ for pickles that stored it there
import sys as _sys
_sys.modules["__main__"].BM25Index = BM25Index  # type: ignore


BM25_TOK = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[가-힣]+", re.UNICODE)
def tok(s): return [t.lower() for t in BM25_TOK.findall(s or "")]


def load_bm25(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Legacy variants: dict {"bm25": BM25Okapi, "doc_ids": [...]}
    if isinstance(obj, dict) and "bm25" in obj:
        return obj["bm25"]
    return obj


def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9: return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# --------- Qwen3-Embedding query encoder (text) ---------

class TextQueryEncoder:
    def __init__(self, model_name: str, device: str):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        # Monkey-patch legacy DynamicCache API used by gte-Qwen2 modeling.py
        # (doc index build worked because it happened before some transformers version gap)
        try:
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, "get_usable_length"):
                def get_usable_length(self, new_seq_length=None, layer_idx=0):
                    try:
                        return self.get_seq_length(layer_idx)
                    except Exception:
                        return 0
                DynamicCache.get_usable_length = get_usable_length
            # Also patch from_legacy_cache to handle None input
            if hasattr(DynamicCache, "from_legacy_cache"):
                orig_from_legacy = DynamicCache.from_legacy_cache
                def patched_from_legacy(past_key_values=None):
                    if past_key_values is None:
                        return DynamicCache()
                    return orig_from_legacy(past_key_values)
                DynamicCache.from_legacy_cache = staticmethod(patched_from_legacy)
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    def encode(self, queries: List[str]) -> np.ndarray:
        emb = self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)


# --------- SigLIP text-tower query encoder (replacement for MM-Embed visual side) ---------

class SigLIPQueryEncoder:
    """SigLIP-large text tower for query→visual cross-modal retrieval.
    Replaces MM-Embed's text query embedding when MM-Embed is unavailable
    (gated nvidia/NV-Embed-v1 dependency). Output is L2-normalized.
    """
    MODEL_NAME = "google/siglip-large-patch16-384"

    def __init__(self, device: str):
        import torch
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.float16
        ).to(device).eval()

    def encode(self, queries: List[str], instruction: str = "") -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = self.processor(text=queries, padding="max_length",
                                    truncation=True, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float32).numpy()


# Back-compat alias for any caller that still uses the old name
MMEmbedQueryEncoder = SigLIPQueryEncoder


# --------- Visual embeddings index (output of encode_visual_regions.py) ---------

@dataclass
class VisualIndex:
    embs: np.ndarray            # [N, D] float32, L2-normalized
    region_keys: List[Tuple[str, str]]  # (chunk_id, region_id_or_str(idx))
    by_chunk: Dict[str, List[int]]  # chunk_id -> list of row indices
    by_doc: Dict[str, List[int]]    # doc_id   -> list of row indices

    @classmethod
    def load(cls, dataset_dir: Path) -> "VisualIndex":
        npys = sorted(dataset_dir.glob("visual_embs_shard*.npy"))
        metas = sorted(dataset_dir.glob("visual_meta_shard*.jsonl"))
        if not npys: raise FileNotFoundError(f"no shards in {dataset_dir}")
        embs_list, keys, by_chunk, by_doc = [], [], defaultdict(list), defaultdict(list)
        offset = 0
        for npyp in npys:
            arr = np.load(npyp)
            shard_idx = re.search(r"shard(\d+)", npyp.name).group(1)
            metap = dataset_dir / f"visual_meta_shard{shard_idx}.jsonl"
            n_meta = 0
            with open(metap) as f:
                for line in f:
                    m = json.loads(line)
                    cid = m["chunk_id"]
                    rid = str(m.get("region_id") or n_meta)
                    keys.append((cid, rid))
                    by_chunk[cid].append(offset + n_meta)
                    by_doc[m["doc_id"]].append(offset + n_meta)
                    n_meta += 1
            assert n_meta == arr.shape[0], f"meta/embs mismatch in shard {shard_idx}: {n_meta} vs {arr.shape[0]}"
            embs_list.append(arr)
            offset += n_meta
        embs = np.concatenate(embs_list, axis=0).astype(np.float32)
        # L2 normalize for cosine
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs /= norms
        return cls(embs=embs, region_keys=keys, by_chunk=dict(by_chunk), by_doc=dict(by_doc))

    def vis_scores_for_chunks(self, q_vec: np.ndarray, chunk_ids: List[str]) -> Dict[str, float]:
        """Return per-chunk MaxSim visual score over its regions. Missing chunks get -inf."""
        out: Dict[str, float] = {}
        q = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        for cid in chunk_ids:
            rows = self.by_chunk.get(cid)
            if not rows:
                out[cid] = float("-inf")
                continue
            sims = self.embs[rows] @ q
            out[cid] = float(sims.max())
        return out


# --------- Doc / Sec card loaders (text indexes from prior repo) ---------

def load_jsonl(p): return [json.loads(l) for l in open(p)]


def load_doc_index(doc_dir: Path):
    docs = load_jsonl(doc_dir / "docs_index.jsonl")
    doc_ids = [d["doc_id"] for d in docs]
    doc_id_to_row = {d: i for i, d in enumerate(doc_ids)}
    embs = np.load(doc_dir / "doc_embs.npy", mmap_mode="r")
    bm25 = load_bm25(doc_dir / "bm25_index.pkl")
    bm25_text = {d["doc_id"]: d.get("bm25_text", d.get("summary_text","")) for d in docs}
    return docs, doc_ids, doc_id_to_row, embs, bm25, bm25_text


def load_sec_index(sec_dir: Path):
    secs = load_jsonl(sec_dir / "section_cards_index.jsonl")
    sec_ids = [s["card_id"] for s in secs]
    sec_id_to_row = {s: i for i, s in enumerate(sec_ids)}
    embs = np.load(sec_dir / "sectioncard_embs.npy", mmap_mode="r")
    bm25 = load_bm25(sec_dir / "sectioncard_bm25_index.pkl")
    return secs, sec_ids, sec_id_to_row, embs, bm25


# --------- Main retrieval ---------

def retrieve_query(
    q: str,
    qvec_text: np.ndarray,
    qvec_mmvis: Optional[np.ndarray],
    docs, doc_ids, doc_id_to_row, doc_embs, doc_bm25, doc_bm25_text,
    secs, sec_ids, sec_id_to_row, sec_embs, sec_bm25,
    chunks_by_card: Dict[str, List[Dict[str, Any]]],
    visual_idx: Optional[VisualIndex],
    alpha: float, beta: float, gamma: float, lam: float,
    top_k_doc: int, top_k_cards: int,
) -> Dict[str, Any]:
    # ---- Stage-1: Doc routing ----
    bm = np.asarray(doc_bm25.get_scores(tok(q)), dtype=np.float32)
    de = np.asarray(doc_embs @ qvec_text, dtype=np.float32)
    s_doc = alpha * minmax(bm) + (1 - alpha) * minmax(de)
    top_doc_idx = np.argsort(-s_doc)[:max(top_k_doc * 4, top_k_doc)]
    cand_doc_ids = {doc_ids[i] for i in top_doc_idx[:top_k_doc]}
    doc_score_map = {doc_ids[i]: float(s_doc[i]) for i in top_doc_idx}

    # ---- Stage-2: per-section MaxSim over units within candidate docs ----
    sec_bm = np.asarray(sec_bm25.get_scores(tok(q)), dtype=np.float32)
    sec_de = np.asarray(sec_embs @ qvec_text, dtype=np.float32)
    sec_text_score = beta * minmax(sec_bm) + (1 - beta) * minmax(sec_de)

    # restrict to candidate docs
    cand_sec_idx = [i for i, s in enumerate(secs) if s["doc_id"] in cand_doc_ids]
    if not cand_sec_idx:
        cand_sec_idx = list(range(len(secs)))

    s_sec = np.full(len(secs), -np.inf, dtype=np.float32)
    for i in cand_sec_idx:
        s_sec[i] = sec_text_score[i]

    # Visual fusion: for cards with at least one visual unit, replace s_sec[i]
    # with γ·vis_max + (1-γ)·s_text_card, only if visual is present.
    if visual_idx is not None and qvec_mmvis is not None:
        # collect chunk_ids per candidate card
        card_chunk_ids: Dict[str, List[str]] = {}
        for i in cand_sec_idx:
            cid_list = secs[i].get("chunk_ids") or []
            if not cid_list:
                # try paths[*].chunk_ids
                paths = secs[i].get("paths") or []
                cid_list = [cid for p in paths for cid in (p.get("chunk_ids") or [])]
            card_chunk_ids[secs[i]["card_id"]] = cid_list

        # Per-card visual MaxSim by aggregating MaxSim over each chunk's regions
        # (the chunk-level score is itself MaxSim over its image regions)
        vis_card_score: Dict[str, float] = {}
        for cid, chunk_list in card_chunk_ids.items():
            if not chunk_list: continue
            chunk_scores = visual_idx.vis_scores_for_chunks(qvec_mmvis, chunk_list)
            best = max((v for v in chunk_scores.values() if np.isfinite(v)), default=None)
            if best is not None:
                vis_card_score[cid] = best

        # Normalize visual scores per query (only over cards that have any)
        if vis_card_score:
            vals = np.asarray(list(vis_card_score.values()), dtype=np.float32)
            v_lo, v_hi = float(vals.min()), float(vals.max())
            for cid in vis_card_score:
                if v_hi - v_lo < 1e-9:
                    vis_card_score[cid] = 0.0
                else:
                    vis_card_score[cid] = (vis_card_score[cid] - v_lo) / (v_hi - v_lo)
            # apply fusion
            for i in cand_sec_idx:
                cid = secs[i]["card_id"]
                if cid in vis_card_score:
                    s_sec[i] = gamma * vis_card_score[cid] + (1 - gamma) * s_sec[i]

    # Final fuse
    for i in cand_sec_idx:
        d = secs[i]["doc_id"]
        s_sec[i] = lam * doc_score_map.get(d, 0.0) + (1 - lam) * s_sec[i]

    top_sec_idx = np.argsort(-s_sec)[:top_k_cards]
    sectioncard_candidates = []
    for i in top_sec_idx:
        if not np.isfinite(s_sec[i]): break
        s = secs[i]
        # Provide text for downstream Reader (best-effort: prefer embed_text/bm25_text)
        card_text = s.get("embed_text") or s.get("bm25_text") or s.get("text") or ""
        sectioncard_candidates.append({
            "card_id": s["card_id"],
            # legacy key for tools that read `chunk_id` as card_id
            "chunk_id": s["card_id"],
            "doc_id": s["doc_id"],
            "card_type": s.get("card_type"),
            "path_str": s.get("path_str"),
            "page_span": s.get("page_span"),
            "chunk_ids": s.get("chunk_ids"),
            "text": card_text,
            "final": float(s_sec[i]),
        })

    doc_candidates = [
        {"doc_id": doc_ids[i], "score": float(s_doc[i]), "rank": rank + 1}
        for rank, i in enumerate(top_doc_idx[:top_k_doc])
    ]

    # Algorithm 1 Phase-3 support: per candidate doc, top-M visual regions ranked
    # by query-visual cosine (SigLIP). Emitted as semantic_visual_by_doc so the
    # Reader can do ancestry-aware semantic expansion without loading SigLIP.
    semantic_visual_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    top_m_semantic = 8
    if visual_idx is not None and qvec_mmvis is not None:
        q = qvec_mmvis / (np.linalg.norm(qvec_mmvis) + 1e-9)
        for d_id in cand_doc_ids:
            rows = visual_idx.by_doc.get(d_id)
            if not rows: continue
            sims = visual_idx.embs[rows] @ q
            order = np.argsort(-sims)[:top_m_semantic]
            items = []
            for oi in order:
                row = rows[int(oi)]
                chunk_id, region_id = visual_idx.region_keys[row]
                items.append({
                    "chunk_id": chunk_id,
                    "region_id": region_id,
                    "score": float(sims[int(oi)]),
                })
            semantic_visual_by_doc[d_id] = items

    return {
        "doc_candidates": doc_candidates,
        "sectioncard_candidates": sectioncard_candidates,
        "semantic_visual_by_doc": semantic_visual_by_doc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["M3DocVQA", "FRAMES", "VisDoMSlide"])
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--top_k_doc", type=int, default=10)
    ap.add_argument("--top_k_cards", type=int, default=10)
    ap.add_argument("--text_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--no_visual", action="store_true",
                    help="Disable MM-Embed visual fusion (text-only ablation)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--doc_dir_override", default=None,
                    help="Override default doc_index dir (e.g. doc_index_md_gte7b)")
    ap.add_argument("--sec_dir_override", default=None,
                    help="Override default sec_index dir (e.g. sec_index_md_gte7b)")
    args = ap.parse_args()

    ds_root = INDEX_ROOT / args.dataset
    if args.dataset in ("M3DocVQA", "VisDoMSlide"):
        doc_dir = ds_root / "doc_index_md"
        sec_dir = ds_root / "sec_index_md"
    else:
        doc_dir = ds_root / "doc_index"
        sec_dir = ds_root / "sec_index"
    if args.doc_dir_override:
        doc_dir = ds_root / args.doc_dir_override
    if args.sec_dir_override:
        sec_dir = ds_root / args.sec_dir_override

    print("[load] doc index", doc_dir, flush=True)
    docs, doc_ids, doc_id_to_row, doc_embs, doc_bm25, doc_bm25_text = load_doc_index(doc_dir)
    print(f"  docs={len(doc_ids)} embs={doc_embs.shape}", flush=True)
    print("[load] sec index", sec_dir, flush=True)
    secs, sec_ids, sec_id_to_row, sec_embs, sec_bm25 = load_sec_index(sec_dir)
    print(f"  secs={len(sec_ids)} embs={sec_embs.shape}", flush=True)

    vis_dir = Path(os.environ.get("ADMIT_WORK", str(Path(__file__).resolve().parents[1] / "work"))) / "visual_embs" / args.dataset
    visual_idx = None
    if not args.no_visual and vis_dir.exists() and any(vis_dir.glob("visual_embs_shard*.npy")):
        print("[load] visual index", vis_dir, flush=True)
        visual_idx = VisualIndex.load(vis_dir)
        print(f"  visual_embs={visual_idx.embs.shape}", flush=True)

    chunks_by_card: Dict[str, List[Dict[str, Any]]] = {}  # not used in current variant

    print("[load] text query encoder", args.text_model, flush=True)
    tenc = TextQueryEncoder(args.text_model, args.device)

    venc: Optional[MMEmbedQueryEncoder] = None
    if visual_idx is not None:
        print("[load] MM-Embed query encoder", flush=True)
        venc = MMEmbedQueryEncoder(args.device)

    questions = json.load(open(args.questions))
    print(f"[run] {len(questions)} queries", flush=True)
    out_f = open(args.out, "w")
    from tqdm import tqdm
    for q in tqdm(questions, desc="queries"):
        qid = q.get("id")
        qtext = q.get("question") or q.get("query") or ""
        qvec_text = tenc.encode([qtext])[0]
        qvec_vis = venc.encode([qtext], instruction="Retrieve a passage or image that answers the question.")[0] if venc is not None else None
        out = retrieve_query(
            qtext, qvec_text, qvec_vis,
            docs, doc_ids, doc_id_to_row, doc_embs, doc_bm25, doc_bm25_text,
            secs, sec_ids, sec_id_to_row, sec_embs, sec_bm25,
            chunks_by_card, visual_idx,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=args.lam,
            top_k_doc=args.top_k_doc, top_k_cards=args.top_k_cards,
        )
        out_f.write(json.dumps({"id": qid, "question": qtext, **out}, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"[done] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
