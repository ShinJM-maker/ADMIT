#!/usr/bin/env python3
"""Freeze comparable text packets from eight DQA evidence pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


import os
ROOT = Path(os.environ.get("ADMIT_UPSTREAM_ROOT", Path(__file__).resolve().parents[2] / "work/upstreams"))  # frozen upstream outputs
EXP = Path(__file__).resolve().parent
OUT = EXP / "inputs"
N = 300

PATHS = {
    "bm25": ROOT / "projects/hikey/work/baselines_eval/packs/M3DocVQA_bm25_only.jsonl",
    "colpali_ocr": ROOT / "projects/hikey/work/baselines_eval/packs/M3DocVQA_colpali.jsonl",
    "raptor": ROOT / "projects/hikey/work/pilot_pevid/chunking_baselines/m3_stage2_raptor_real.jsonl",
    "late_chunking": ROOT / "projects/hikey/work/chunking_utility_late_chunking/M3DocVQA/M3DocVQA_latechunking512_stage2.jsonl",
    "multidocfusion": ROOT / (
        "projects/hikey/work/chunking_utility_missing/M3DocVQA/"
        "M3DocVQA_mdf_section_fusionmeta_stage2.jsonl"
    ),
    "m3docdep": ROOT / (
        "projects/hikey/work/chunking_utility_missing/M3DocVQA/"
        "M3DocVQA_m3docdep_stage2.jsonl"
    ),
    "hikey": ROOT / "projects/hikey/work/baselines_eval/packs/M3DocVQA_hikey_kdoc50.jsonl",
    "vega_meta": ROOT / "projects/PILAR/data/results/bottleneck/retrieval_metadata.json",
    "vega_chunks": ROOT / "projects/AnswerGainAudit/reproducibility_bundle_final/raw/agentic_backing/data/indices/m3docvqa_chunks.json",
}


def read_selected(path: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["id"])
            if qid in selected:
                rows[qid] = row
    if set(rows) != selected:
        raise RuntimeError(f"{path}: missing {len(selected - set(rows))} selected qids")
    return rows


def serialize_cards(cards: list[dict[str, Any]], limit: int = 10) -> tuple[str, list[str]]:
    units, unit_ids = [], []
    for index, card in enumerate(cards[:limit], 1):
        unit_id = str(card.get("card_id") or card.get("chunk_id") or f"unit-{index}")
        doc_id = str(card.get("doc_id") or card.get("source_id") or "unknown")
        path = str(card.get("path_str") or card.get("page_id") or "")
        text = str(card.get("text") or "").strip()
        if not text:
            continue
        units.append(
            f"[UNIT {index} | id={unit_id} | doc={doc_id} | path={path}]\n{text}"
        )
        unit_ids.append(unit_id)
    return "\n\n".join(units), unit_ids


def main() -> None:
    vega_rows = json.loads(PATHS["vega_meta"].read_text(encoding="utf-8"))
    if len(vega_rows) < N:
        raise RuntimeError("GraphRAG frozen sample has fewer than 300 rows")
    # The source artifact is already a fixed seed-42 sample. Preserve its first 300
    # records so that no outcome-dependent resampling is introduced here.
    selected_meta = vega_rows[:N]
    selected_order = [str(row["id"]) for row in selected_meta]
    selected = set(selected_order)
    meta_by_id = {str(row["id"]): row for row in selected_meta}

    upstream = {
        key: read_selected(path, selected)
        for key, path in PATHS.items()
        if key in {
            "bm25", "colpali_ocr", "raptor", "late_chunking",
            "multidocfusion", "m3docdep", "hikey",
        }
    }
    vega_chunks = json.loads(PATHS["vega_chunks"].read_text(encoding="utf-8"))

    rows = []
    for qid_index, qid in enumerate(selected_order):
        meta = meta_by_id[qid]
        gold = meta.get("gold") or []
        if not isinstance(gold, list):
            gold = [gold]
        common = {
            "qid": qid,
            "qid_index": qid_index,
            "question": str(meta.get("question") or ""),
            "gold": [str(value) for value in gold],
        }
        for pipeline in (
            "bm25", "colpali_ocr", "raptor", "late_chunking",
            "multidocfusion", "m3docdep", "hikey",
        ):
            cards = upstream[pipeline][qid].get("sectioncard_candidates") or []
            packet, unit_ids = serialize_cards(cards)
            if not packet:
                raise RuntimeError(f"empty packet for {pipeline}/{qid}")
            rows.append({
                **common,
                "pipeline": pipeline,
                "upstream_artifact": str(PATHS[pipeline]),
                "retrieval_unit_ids": unit_ids,
                "raw_packet_text": packet,
            })

        page_ids = [str(value) for value in meta.get("final_page_ids") or []]
        cards = []
        for page_id in page_ids:
            chunk = vega_chunks.get(page_id)
            if chunk is None:
                raise KeyError(f"GraphRAG page {page_id} is absent from frozen backing")
            cards.append({
                "card_id": page_id,
                "doc_id": chunk.get("doc_id"),
                "path_str": f"page_{chunk.get('page', '')}",
                "text": chunk.get("text", ""),
            })
        packet, unit_ids = serialize_cards(cards, limit=20)
        if not packet:
            raise RuntimeError(f"empty packet for vega_kg/{qid}")
        rows.append({
            **common,
            "pipeline": "vega_kg",
            "upstream_artifact": str(PATHS["vega_meta"]),
            "retrieval_unit_ids": unit_ids,
            "retrieval_roles": meta.get("final_roles") or {},
            "raw_packet_text": packet,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "frozen_packets.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "selected_qids.json").write_text(
        json.dumps(selected_order, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_qids": len(selected_order),
        "n_pipelines": 8,
        "n_rows": len(rows),
        "output": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
