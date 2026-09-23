"""Document-level guideline catalog used for metadata-aware RAG routing."""

from typing import Any


# One entry per logical clinical document/family.
# Chunks inherit these fields at ingest time; retrieval routes here first.
# filename_hints must match PDF names under Rag/data/guidelines/ (order = first match wins).
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
    {
        "doc_id": "nice_ng79_sinusitis",
        "guideline": "NICE NG79",
        "condition": "acute sinusitis",
        "keywords": [
            "sinusitis",
            "sinus",
            "nasal",
            "facial pain",
            "facial pressure",
            "runny nose",
            "rhinorrhea",
            "blocked nose",
            "congestion",
            "postnasal",
            "ent",
        ],
        # visual-summary-pdf-4656316717.pdf = NG79 sinusitis visual summary
        "filename_hints": ["4656316717", "sinusitis", "ng79"],
    },
    {
        "doc_id": "nice_ng91_otitis_media",
        "guideline": "NICE NG91",
        "condition": "acute otitis media",
        "keywords": [
            "otitis",
            "ear infection",
            "earache",
            "ear pain",
            "otalgia",
            "middle ear",
            "ear discharge",
            "eardrum",
            "hearing",
            "ent",
        ],
        # otitis-media-*.pdf + visual-summary-pdf-4787282702.pdf
        "filename_hints": ["otitis", "1837750121413", "4787282702", "ng91"],
    },
    {
        "doc_id": "nice_ng106_heart_failure",
        "guideline": "NICE NG106",
        "condition": "chronic heart failure",
        "keywords": [
            "heart failure",
            "cardiac failure",
            "cardiovascular",
            "dyspnoea",
            "dyspnea",
            "shortness of breath",
            "breathlessness",
            "oedema",
            "edema",
            "ankle swelling",
            "orthopnoea",
            "orthopnea",
            "ejection fraction",
            "nt-probnp",
            "bnp",
            "hfref",
            "hfpef",
        ],
        "filename_hints": [
            "heart-failure",
            "chronic-heart-failure",
            "6663137725",
            "ng106",
        ],
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
