"""Document-level guideline catalog used for metadata-aware RAG routing."""

from typing import Any


# One entry per logical clinical document/family.
# Chunks inherit these fields at ingest time; retrieval routes here first.
GUIDELINE_CATALOG: list[dict[str, Any]] = [
    {
        "doc_id": "nice_ng109_uti",
        "guideline": "NICE NG109",
        "condition": "lower urinary tract infection",
        "keywords": [
            "uti",
            "urinary",
            "urine",
            "dysuria",
            "cystitis",
            "bladder",
            "frequency",
            "burning",
            "cloudy",
            "flank",
            "pyelonephritis",
            "urethra",
        ],
        "filename_hints": ["ng109", "uti", "urinary"],
    },
    {
        "doc_id": "nice_ng136_hypertension",
        "guideline": "NICE NG136",
        "condition": "hypertension",
        "keywords": [
            "hypertension",
            "blood pressure",
            "bp",
            "systolic",
            "diastolic",
            "antihypertensive",
        ],
        "filename_hints": ["hypertension", "blood pressure"],
    },
    {
        "doc_id": "nice_ng28_diabetes",
        "guideline": "NICE NG28",
        "condition": "type 2 diabetes",
        "keywords": [
            "diabetes",
            "diabetic",
            "glucose",
            "hba1c",
            "hyperglycemia",
            "insulin",
            "metformin",
            "polyuria",
            "polydipsia",
        ],
        "filename_hints": ["diabetes"],
    },
]


def metadata_for_filename(filename: str) -> dict[str, str]:
    lowered = filename.lower()
    for entry in GUIDELINE_CATALOG:
        if any(hint in lowered for hint in entry["filename_hints"]):
            return {
                "doc_id": entry["doc_id"],
                "guideline": entry["guideline"],
                "condition": entry["condition"],
            }
    return {
        "doc_id": "general_guideline",
        "guideline": "clinical guideline",
        "condition": "general",
    }


def route_documents(summary: dict, entities: dict, top_n: int = 2) -> list[dict[str, Any]]:
    """
    Pick the most relevant guideline documents BEFORE chunk search.
    Uses keyword overlap against summary/entities (fast, deterministic).
    """
    summary = summary if isinstance(summary, dict) else {}
    entities = entities if isinstance(entities, dict) else {}

    text_parts = [
        str(summary.get("chief_complaint", "")),
        str(summary.get("assessment", "")),
        str(summary.get("history_of_present_illness", "")),
        str(summary.get("summary", "")),
        str(entities.get("duration", "")),
        " ".join(str(x) for x in (entities.get("symptoms") or [])),
        " ".join(str(x) for x in (entities.get("possible_conditions") or [])),
        " ".join(str(x) for x in (entities.get("warning_signs") or [])),
        " ".join(str(x) for x in (entities.get("medical_history") or [])),
    ]
    haystack = " ".join(text_parts).lower()

    scored: list[dict[str, Any]] = []
    for entry in GUIDELINE_CATALOG:
        hits = [kw for kw in entry["keywords"] if kw in haystack]
        if not hits:
            continue
        scored.append(
            {
                "doc_id": entry["doc_id"],
                "guideline": entry["guideline"],
                "condition": entry["condition"],
                "route_score": len(hits),
                "matched_keywords": hits,
            }
        )

    scored.sort(key=lambda item: item["route_score"], reverse=True)
    return scored[:top_n]
