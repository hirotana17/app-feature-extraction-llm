import json
import re
from typing import List


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def parse_feature_response(raw_text: str) -> List[str]:
    text = _strip_code_fence(raw_text)

    candidates = [text]
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        candidates.append(bracket_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                cleaned = [str(x).strip() for x in parsed if str(x).strip()]
                return _unique_keep_order(cleaned)
        except json.JSONDecodeError:
            continue

    # Strict mode: keep only valid JSON array outputs.
    return []
