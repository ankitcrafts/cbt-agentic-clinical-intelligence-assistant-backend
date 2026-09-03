import json
import re


def parse_llm_json(text: str):
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def sanitize_for_state(value):
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(value, dict):
        return {key: sanitize_for_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_state(item) for item in value]
    if np is not None and isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return float(value)
    return value
