from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_STIMULUS_FIELDS = {"id", "text"}
PROHIBITED_STIMULUS_TERMS = {
    "diagnosis",
    "diagnose",
    "clinical",
    "mental health",
    "mental-health",
    "surveillance",
    "hiring",
    "education scoring",
    "law enforcement",
    "medical decision",
    "medical decision-making",
}
CATEGORY_LEAKAGE_TERMS = {
    "neutral",
    "threat",
    "danger",
    "afraid",
    "fear",
    "sad",
    "sadness",
    "confused",
    "confusion",
    "relief",
    "relieved",
    "happy",
    "anxious",
    "depressed",
    "ordinary",
    "approach",
    "approaching",
    "loss",
    "absence",
    "absent",
    "contradict",
    "contradicts",
    "contradictory",
    "resolved",
    "resolve",
    "problem",
    "consistent",
    "inconsistent",
    "static",
    "missing",
    "present",
    "unresolved",
    "fixed",
}
COMPARISON_DISCLAIMER = (
    "This is predicted stimulus-response similarity, not emotion detection, diagnosis, "
    "or measurement of an individual person's mental state."
)
TEMPORAL_SIGNATURES = {
    "net_movement",
    "avg_slope_per_transition",
    "abs_total_movement",
    "late_minus_early",
}


@dataclass(frozen=True)
class DictionaryStimulus:
    dictionary_name: str
    category: str
    stimulus_id: str
    text: str
    pair_id: str | None = None


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return slug.lower() or "asne_dictionary"


def load_stimulus_dictionary(path: str | Path) -> dict[str, Any]:
    dictionary_path = Path(path)
    payload = json.loads(dictionary_path.read_text(encoding="utf-8"))
    validate_stimulus_dictionary(payload)
    if not payload.get("name"):
        payload["name"] = dictionary_path.stem
    return payload


def validate_stimulus_dictionary(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Dictionary payload must be a JSON object.")
    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Dictionary must include a non-empty 'categories' object.")
    for category, stimuli in categories.items():
        if not isinstance(category, str) or not category:
            raise ValueError("Dictionary category names must be non-empty strings.")
        if not isinstance(stimuli, list) or not stimuli:
            raise ValueError(f"Category {category!r} must contain a non-empty stimulus list.")
        for item in stimuli:
            if not isinstance(item, dict):
                raise ValueError(f"Stimulus in category {category!r} must be an object.")
            missing = REQUIRED_STIMULUS_FIELDS - set(item)
            if missing:
                raise ValueError(
                    f"Stimulus in category {category!r} is missing required fields: {sorted(missing)}"
                )
            if not str(item["id"]).strip() or not str(item["text"]).strip():
                raise ValueError(f"Stimulus id/text in category {category!r} cannot be empty.")


def validate_dictionary_expansion(
    dictionary_payload: dict[str, Any],
    expected_examples_per_category: int = 10,
    avoid_category_words: bool = True,
) -> list[str]:
    """Return validator warnings for larger dictionary construction."""

    validate_stimulus_dictionary(dictionary_payload)
    warnings = []
    seen_texts: set[str] = set()
    for category, stimuli in dictionary_payload["categories"].items():
        if len(stimuli) != expected_examples_per_category:
            warnings.append(
                f"Category {category!r} has {len(stimuli)} examples; expected {expected_examples_per_category}."
            )
        for item in stimuli:
            text = str(item["text"]).strip()
            normalized = re.sub(r"\s+", " ", text.lower())
            if normalized in seen_texts:
                warnings.append(f"Duplicate stimulus text found: {item['id']!r}.")
            seen_texts.add(normalized)
            if avoid_category_words:
                for term in sorted(CATEGORY_LEAKAGE_TERMS | {category.lower()}):
                    if re.search(rf"\b{re.escape(term)}\b", normalized):
                        warnings.append(f"Stimulus {item['id']!r} contains category leakage term: {term!r}.")
            for term in PROHIBITED_STIMULUS_TERMS:
                if term in normalized:
                    warnings.append(f"Stimulus {item['id']!r} contains prohibited wording: {term!r}.")
    return warnings


def validate_eval_expansion(
    eval_payload: dict[str, Any],
    dictionary_payload: dict[str, Any] | None = None,
    expected_examples_per_category: int = 5,
    avoid_category_words: bool = True,
) -> list[str]:
    """Return validator warnings for held-out dictionary evaluation sets."""

    from .evaluation import validate_eval_set

    validate_eval_set(eval_payload)
    warnings = []
    counts: dict[str, int] = {}
    dictionary_texts = set()
    if dictionary_payload:
        validate_stimulus_dictionary(dictionary_payload)
        for stimuli in dictionary_payload["categories"].values():
            for item in stimuli:
                dictionary_texts.add(re.sub(r"\s+", " ", str(item["text"]).strip().lower()))

    seen_texts: set[str] = set()
    for item in eval_payload["items"]:
        category = str(item["expected_category"])
        counts[category] = counts.get(category, 0) + 1
        normalized = re.sub(r"\s+", " ", str(item["text"]).strip().lower())
        if normalized in seen_texts:
            warnings.append(f"Duplicate eval text found: {item['id']!r}.")
        seen_texts.add(normalized)
        if normalized in dictionary_texts:
            warnings.append(f"Eval item {item['id']!r} duplicates a dictionary stimulus.")
        if avoid_category_words:
            for term in sorted(CATEGORY_LEAKAGE_TERMS | {category.lower()}):
                if re.search(rf"\b{re.escape(term)}\b", normalized):
                    warnings.append(f"Eval item {item['id']!r} contains category leakage term: {term!r}.")
        for term in PROHIBITED_STIMULUS_TERMS:
            if term in normalized:
                warnings.append(f"Eval item {item['id']!r} contains prohibited wording: {term!r}.")

    for category, count in counts.items():
        if count != expected_examples_per_category:
            warnings.append(
                f"Eval category {category!r} has {count} examples; expected {expected_examples_per_category}."
            )
    return warnings


def validate_paired_contrast_benchmark(
    dictionary_payload: dict[str, Any],
    eval_payload: dict[str, Any] | None = None,
    *,
    expected_dictionary_pairs: int = 10,
    expected_eval_pairs: int = 5,
    max_pair_length_delta_words: int = 6,
    banned_terms: set[str] | None = None,
) -> list[str]:
    """Validate paired binary contrast dictionaries and optional held-out eval sets."""

    warnings: list[str] = []
    banned = set(CATEGORY_LEAKAGE_TERMS | PROHIBITED_STIMULUS_TERMS)
    if banned_terms:
        banned.update(term.lower() for term in banned_terms)

    validate_stimulus_dictionary(dictionary_payload)
    warnings.extend(
        _validate_paired_items(
            items_by_category=dictionary_payload["categories"],
            source_name=str(dictionary_payload.get("name") or "dictionary"),
            expected_pairs=expected_dictionary_pairs,
            banned_terms=banned,
            max_pair_length_delta_words=max_pair_length_delta_words,
        )
    )

    dictionary_texts = {
        _normalize_text(str(item["text"]))
        for stimuli in dictionary_payload["categories"].values()
        for item in stimuli
    }
    if eval_payload is not None:
        from .evaluation import validate_eval_set

        validate_eval_set(eval_payload)
        eval_by_category: dict[str, list[dict[str, Any]]] = {}
        for item in eval_payload["items"]:
            eval_by_category.setdefault(str(item["expected_category"]), []).append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "pair_id": item.get("pair_id"),
                }
            )
            normalized = _normalize_text(str(item["text"]))
            if normalized in dictionary_texts:
                warnings.append(f"Eval item {item['id']!r} duplicates a dictionary stimulus.")
        warnings.extend(
            _validate_paired_items(
                items_by_category=eval_by_category,
                source_name=str(eval_payload.get("name") or "eval"),
                expected_pairs=expected_eval_pairs,
                banned_terms=banned,
                max_pair_length_delta_words=max_pair_length_delta_words,
            )
        )
    return warnings


def _validate_paired_items(
    *,
    items_by_category: dict[str, list[dict[str, Any]]],
    source_name: str,
    expected_pairs: int,
    banned_terms: set[str],
    max_pair_length_delta_words: int,
) -> list[str]:
    warnings: list[str] = []
    categories = list(items_by_category)
    if len(categories) != 2:
        warnings.append(f"{source_name}: expected exactly 2 categories, found {len(categories)}.")
        return warnings

    pair_to_items: dict[str, dict[str, dict[str, Any]]] = {}
    seen_texts: set[str] = set()
    for category, items in items_by_category.items():
        if len(items) != expected_pairs:
            warnings.append(f"{source_name}: category {category!r} has {len(items)} items; expected {expected_pairs}.")
        for item in items:
            pair_id = str(item.get("pair_id") or "").strip()
            if not pair_id:
                warnings.append(f"{source_name}: item {item.get('id')!r} is missing pair_id.")
                continue
            text = str(item.get("text") or "").strip()
            normalized = _normalize_text(text)
            if normalized in seen_texts:
                warnings.append(f"{source_name}: duplicate text found for item {item.get('id')!r}.")
            seen_texts.add(normalized)
            for term in sorted(banned_terms | {category.lower()}):
                if _contains_term(normalized, term):
                    warnings.append(f"{source_name}: item {item.get('id')!r} contains banned term {term!r}.")
            pair_to_items.setdefault(pair_id, {})[category] = item

    if len(pair_to_items) != expected_pairs:
        warnings.append(f"{source_name}: found {len(pair_to_items)} pairs; expected {expected_pairs}.")
    for pair_id, by_category in sorted(pair_to_items.items()):
        missing = [category for category in categories if category not in by_category]
        if missing:
            warnings.append(f"{source_name}: pair {pair_id!r} is missing categories: {missing}.")
            continue
        lengths = [
            len(str(by_category[category].get("text") or "").split())
            for category in categories
        ]
        if max(lengths) - min(lengths) > max_pair_length_delta_words:
            warnings.append(
                f"{source_name}: pair {pair_id!r} has length imbalance {lengths}; "
                f"max allowed delta is {max_pair_length_delta_words} words."
            )
    return warnings


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = re.sub(r"[_-]+", " ", term.lower()).strip()
    if not normalized_term:
        return False
    if " " in normalized_term:
        return normalized_term in normalized_text
    return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None


def iter_dictionary_stimuli(payload: dict[str, Any]) -> list[DictionaryStimulus]:
    validate_stimulus_dictionary(payload)
    dictionary_name = str(payload.get("name") or "asne_dictionary")
    stimuli: list[DictionaryStimulus] = []
    for category, items in payload["categories"].items():
        for item in items:
            stimuli.append(
                DictionaryStimulus(
                    dictionary_name=dictionary_name,
                    category=category,
                    stimulus_id=str(item["id"]),
                    text=str(item["text"]),
                    pair_id=str(item["pair_id"]) if item.get("pair_id") is not None else None,
                )
            )
    return stimuli


def aggregate_response_segments(predictions: Any, percentile: float = 95.0) -> dict[str, Any]:
    array = np.asarray(predictions, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape((-1, array.shape[-1]))

    abs_array = np.abs(array)
    return {
        "raw_shape": list(np.asarray(predictions).shape),
        "response_shape": [int(array.shape[-1])],
        "segments": int(array.shape[0]),
        "mean_response": np.mean(array, axis=0),
        "peak_abs_response": np.max(abs_array, axis=0),
        "percentile_abs_response": np.percentile(abs_array, percentile, axis=0),
        "percentile": float(percentile),
    }


def normalize_raw_segments(raw_segments: Any) -> np.ndarray:
    array = np.asarray(raw_segments, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape((-1, array.shape[-1]))
    return array


def compute_temporal_movement_signatures(raw_segments: Any) -> dict[str, Any]:
    array = normalize_raw_segments(raw_segments)
    segment_count = int(array.shape[0])
    transition_count = max(segment_count - 1, 0)
    start_response = array[0]
    end_response = array[-1]

    if segment_count == 1:
        zeros = np.zeros_like(start_response)
        return {
            "start_response": start_response,
            "end_response": end_response,
            "net_movement": zeros,
            "avg_slope_by_count": zeros,
            "avg_slope_per_transition": zeros,
            "step_deltas": np.empty((0, array.shape[-1]), dtype=float),
            "abs_total_movement": zeros,
            "avg_abs_step": zeros,
            "segment_count": segment_count,
            "transition_count": transition_count,
        }

    step_deltas = np.diff(array, axis=0)
    net_movement = end_response - start_response
    abs_total_movement = np.sum(np.abs(step_deltas), axis=0)
    return {
        "start_response": start_response,
        "end_response": end_response,
        "net_movement": net_movement,
        "avg_slope_by_count": net_movement / segment_count,
        "avg_slope_per_transition": net_movement / transition_count,
        "step_deltas": step_deltas,
        "abs_total_movement": abs_total_movement,
        "avg_abs_step": abs_total_movement / transition_count,
        "segment_count": segment_count,
        "transition_count": transition_count,
    }


def compute_late_minus_early(raw_segments: Any) -> np.ndarray:
    array = normalize_raw_segments(raw_segments)
    if array.shape[0] == 1:
        return np.zeros(array.shape[-1], dtype=float)
    split_index = max(array.shape[0] // 2, 1)
    early = array[:split_index]
    late = array[split_index:]
    if late.size == 0:
        return np.zeros(array.shape[-1], dtype=float)
    return np.mean(late, axis=0) - np.mean(early, axis=0)


def top_abs_indices(vector: Any, k: int = 10) -> list[dict[str, Any]]:
    array = np.asarray(vector, dtype=float)
    if array.ndim != 1 or array.size == 0:
        return []
    k = min(k, array.size)
    indices = np.argpartition(np.abs(array), -k)[-k:]
    indices = indices[np.argsort(np.abs(array[indices]))[::-1]]
    return [{"index": int(index), "value": float(array[index])} for index in indices]


def compute_neutral_baseline(records: list[dict[str, Any]], neutral_category: str = "neutral") -> np.ndarray:
    neutral_vectors = [
        np.asarray(record["mean_response"], dtype=float)
        for record in records
        if record.get("category") == neutral_category
    ]
    if not neutral_vectors:
        raise ValueError(f"Cannot compute neutral baseline: no {neutral_category!r} records found.")
    return np.mean(np.vstack(neutral_vectors), axis=0)


def count_neutral_records(records: list[dict[str, Any]], neutral_category: str = "neutral") -> int:
    return sum(1 for record in records if record.get("category") == neutral_category)


def add_delta_from_neutral(
    record: dict[str, Any],
    neutral_baseline: np.ndarray,
    neutral_category: str = "neutral",
) -> dict[str, Any]:
    response = np.asarray(record["mean_response"], dtype=float)
    delta = response - neutral_baseline
    updated = dict(record)
    updated["neutral_baseline_category"] = neutral_category
    updated["delta_from_neutral"] = delta.astype(float).tolist()
    updated["delta_from_neutral_summary"] = {
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }
    return updated


def cosine_similarity(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denom)


def weighted_cosine(a: Any, b: Any, weights: Any) -> float:
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if a_array.shape != b_array.shape or a_array.shape != weight_array.shape:
        raise ValueError("weighted_cosine inputs must have matching shapes.")
    weighted_a = a_array * weight_array
    weighted_b = b_array * weight_array
    denom = float(np.linalg.norm(weighted_a) * np.linalg.norm(weighted_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(weighted_a, weighted_b) / denom)


def pearson_correlation(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.size < 2 or right_array.size < 2:
        return 0.0
    if math.isclose(float(np.std(left_array)), 0.0) or math.isclose(float(np.std(right_array)), 0.0):
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


def signature_vector(
    record: dict[str, Any],
    signature: str = "mean_response",
    neutral_baseline: Any | None = None,
) -> np.ndarray:
    response = np.asarray(record["mean_response"], dtype=float)
    if signature == "mean_response":
        return response
    if signature == "delta_from_neutral":
        if neutral_baseline is None:
            raise ValueError("neutral_baseline is required for delta_from_neutral signatures.")
        return response - np.asarray(neutral_baseline, dtype=float)
    if signature in TEMPORAL_SIGNATURES:
        return temporal_signature_vector(record, signature)
    raise ValueError(f"Unsupported signature mode: {signature}")


def temporal_signature_vector(record: dict[str, Any], signature: str) -> np.ndarray:
    if signature == "late_minus_early":
        vector_path = record.get("late_minus_early_path")
        if vector_path:
            return np.load(vector_path)
        return compute_late_minus_early(_load_raw_segments_for_record(record))

    vector_path = record.get(f"{signature}_path")
    if vector_path:
        return np.load(vector_path)
    movement = compute_temporal_movement_signatures(_load_raw_segments_for_record(record))
    return np.asarray(movement[signature], dtype=float)


def _load_raw_segments_for_record(record: dict[str, Any]) -> np.ndarray:
    raw_path = record.get("raw_segment_prediction_path") or record.get("raw_segments_path")
    if not raw_path:
        raise ValueError(
            f"Record {record.get('stimulus_id', '<unknown>')!r} does not include raw segment predictions "
            "needed for temporal movement signatures."
        )
    return np.load(raw_path)


def rank_dictionary_signatures(
    query_vector: Any,
    records: list[dict[str, Any]],
    signature: str = "mean_response",
    neutral_baseline: Any | None = None,
    sort_metric: str = "cosine",
) -> list[dict[str, Any]]:
    if sort_metric not in {"cosine", "pearson"}:
        raise ValueError(f"Unsupported sort metric: {sort_metric}")

    ranked = []
    for record in records:
        vector = signature_vector(record, signature, neutral_baseline)
        ranked.append(
            {
                "category": record["category"],
                "stimulus_id": record["stimulus_id"],
                "cosine_similarity": cosine_similarity(query_vector, vector),
                "pearson_correlation": pearson_correlation(query_vector, vector),
            }
        )
    sort_field = "cosine_similarity" if sort_metric == "cosine" else "pearson_correlation"
    return sorted(ranked, key=lambda item: item[sort_field], reverse=True)


def vector_norm(vector: Any) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def select_top_k_dims(delta_vector: Any, k: int) -> np.ndarray:
    vector = np.asarray(delta_vector, dtype=float)
    if vector.ndim != 1:
        raise ValueError("select_top_k_dims expects a 1D vector.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    k = min(k, vector.size)
    indices = np.argpartition(np.abs(vector), -k)[-k:]
    return indices[np.argsort(np.abs(vector[indices]))[::-1]]


def build_signature_records(
    records: list[dict[str, Any]],
    signature: str = "mean_response",
    neutral_baseline: Any | None = None,
) -> list[dict[str, Any]]:
    signature_records = []
    for record in records:
        vector = signature_vector(record, signature, neutral_baseline)
        signature_records.append(
            {
                "category": record["category"],
                "stimulus_id": record["stimulus_id"],
                "vector": vector,
                "delta_norm": vector_norm(vector),
            }
        )
    return signature_records


def rank_signature_vectors(
    query_vector: Any,
    signature_records: list[dict[str, Any]],
    sort_metric: str = "cosine",
) -> list[dict[str, Any]]:
    if sort_metric not in {"cosine", "pearson"}:
        raise ValueError(f"Unsupported sort metric: {sort_metric}")

    ranked = []
    for record in signature_records:
        vector = record["vector"]
        ranked.append(
            {
                "category": record["category"],
                "stimulus_id": record["stimulus_id"],
                "cosine_similarity": cosine_similarity(query_vector, vector),
                "pearson_correlation": pearson_correlation(query_vector, vector),
                "delta_norm": record["delta_norm"],
            }
        )
    sort_field = "cosine_similarity" if sort_metric == "cosine" else "pearson_correlation"
    return sorted(ranked, key=lambda item: item[sort_field], reverse=True)


def aggregate_category_scores(
    ranked_records: list[dict[str, Any]],
    sort_metric: str = "cosine",
) -> list[dict[str, Any]]:
    if sort_metric not in {"cosine", "pearson"}:
        raise ValueError(f"Unsupported sort metric: {sort_metric}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in ranked_records:
        grouped.setdefault(item["category"], []).append(item)

    category_scores = []
    sort_field = "mean_cosine_similarity" if sort_metric == "cosine" else "mean_pearson_correlation"
    best_field = "cosine_similarity" if sort_metric == "cosine" else "pearson_correlation"
    for category, items in grouped.items():
        category_scores.append(
            {
                "category": category,
                "n_signatures": len(items),
                "mean_cosine_similarity": float(np.mean([item["cosine_similarity"] for item in items])),
                "mean_pearson_correlation": float(np.mean([item["pearson_correlation"] for item in items])),
                "best_stimulus_id": max(items, key=lambda item: item[best_field])["stimulus_id"],
            }
        )
    return sorted(category_scores, key=lambda item: item[sort_field], reverse=True)


def aggregate_category_best(
    ranked_records: list[dict[str, Any]],
    sort_metric: str = "cosine",
) -> list[dict[str, Any]]:
    if sort_metric not in {"cosine", "pearson"}:
        raise ValueError(f"Unsupported sort metric: {sort_metric}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in ranked_records:
        grouped.setdefault(item["category"], []).append(item)

    score_field = "cosine_similarity" if sort_metric == "cosine" else "pearson_correlation"
    category_scores = []
    for category, items in grouped.items():
        best = max(items, key=lambda item: item[score_field])
        category_scores.append(
            {
                "category": category,
                "n_signatures": len(items),
                "mean_cosine_similarity": float(np.mean([item["cosine_similarity"] for item in items])),
                "mean_pearson_correlation": float(np.mean([item["pearson_correlation"] for item in items])),
                "best_cosine_similarity": best["cosine_similarity"],
                "best_pearson_correlation": best["pearson_correlation"],
                "best_stimulus_id": best["stimulus_id"],
            }
        )
    sort_field = "best_cosine_similarity" if sort_metric == "cosine" else "best_pearson_correlation"
    return sorted(category_scores, key=lambda item: item[sort_field], reverse=True)


def compute_category_centroids(signature_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for record in signature_records:
        grouped.setdefault(record["category"], []).append(np.asarray(record["vector"], dtype=float))

    centroids = {}
    for category, vectors in grouped.items():
        centroid = np.mean(np.vstack(vectors), axis=0)
        centroids[category] = {
            "category": category,
            "vector": centroid,
            "centroid_norm": vector_norm(centroid),
            "n_signatures": len(vectors),
        }
    return centroids


def compute_binary_contrast_axis(
    category_a_centroid: Any,
    category_b_centroid: Any,
) -> dict[str, Any]:
    """Compute the midpoint and axis for a two-category response contrast."""

    category_a = np.asarray(category_a_centroid, dtype=float)
    category_b = np.asarray(category_b_centroid, dtype=float)
    if category_a.shape != category_b.shape:
        raise ValueError("Binary contrast centroids must have matching shapes.")
    midpoint = (category_a + category_b) / 2.0
    axis = category_b - category_a
    return {
        "midpoint": midpoint,
        "axis": axis,
        "axis_norm": vector_norm(axis),
        "midpoint_norm": vector_norm(midpoint),
    }


def score_binary_contrast_axis(
    query_response: Any,
    category_a_centroid: Any,
    category_b_centroid: Any,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Score a query along a binary contrast axis in raw response space."""

    query = np.asarray(query_response, dtype=float)
    axis_info = compute_binary_contrast_axis(category_a_centroid, category_b_centroid)
    axis = np.asarray(axis_info["axis"], dtype=float)
    midpoint = np.asarray(axis_info["midpoint"], dtype=float)
    if query.shape != axis.shape:
        raise ValueError("Query response and binary contrast axis must have matching shapes.")
    axis_norm = float(axis_info["axis_norm"])
    signed_score = float(np.dot(query - midpoint, axis) / (axis_norm + eps))
    predicted_side = "category_a" if signed_score < 0.0 else "category_b"
    return {
        "signed_score": signed_score,
        "axis_norm": axis_norm,
        "midpoint_norm": float(axis_info["midpoint_norm"]),
        "predicted_side": predicted_side,
    }


def _category_name_and_vector(category: Any, default_name: str) -> tuple[str, np.ndarray, int | None]:
    if isinstance(category, dict):
        name = str(category.get("category") or default_name)
        vector = np.asarray(category.get("vector", category.get("centroid")), dtype=float)
        n_signatures = category.get("n_signatures")
        return name, vector, int(n_signatures) if n_signatures is not None else None
    return default_name, np.asarray(category, dtype=float), None


def classify_binary_contrast_axis(
    query_response: Any,
    category_a: Any,
    category_b: Any,
) -> dict[str, Any]:
    """Classify a query against two raw category centroids without zeroing the baseline."""

    category_a_name, category_a_vector, category_a_n = _category_name_and_vector(category_a, "category_a")
    category_b_name, category_b_vector, category_b_n = _category_name_and_vector(category_b, "category_b")
    score = score_binary_contrast_axis(query_response, category_a_vector, category_b_vector)
    signed_score = float(score["signed_score"])
    category_a_score = -signed_score
    category_b_score = signed_score
    category_a_result = {
        "category": category_a_name,
        "score": category_a_score,
        "signed_score": signed_score,
        "axis_norm": score["axis_norm"],
        "midpoint_norm": score["midpoint_norm"],
        "predicted_side": score["predicted_side"],
        "category_side": "category_a",
        "raw_centroid_cosine_similarity": cosine_similarity(query_response, category_a_vector),
        "centroid_norm": vector_norm(category_a_vector),
        "n_signatures": category_a_n,
        "best_stimulus_id": None,
    }
    category_b_result = {
        "category": category_b_name,
        "score": category_b_score,
        "signed_score": signed_score,
        "axis_norm": score["axis_norm"],
        "midpoint_norm": score["midpoint_norm"],
        "predicted_side": score["predicted_side"],
        "category_side": "category_b",
        "raw_centroid_cosine_similarity": cosine_similarity(query_response, category_b_vector),
        "centroid_norm": vector_norm(category_b_vector),
        "n_signatures": category_b_n,
        "best_stimulus_id": None,
    }
    ranking = (
        [category_a_result, category_b_result]
        if signed_score < 0.0
        else [category_b_result, category_a_result]
    )
    return {
        **score,
        "category_a": category_a_name,
        "category_b": category_b_name,
        "ranking": ranking,
    }


def _infer_pair_id(record: dict[str, Any]) -> str | None:
    explicit = record.get("pair_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    stimulus_id = str(record.get("stimulus_id") or "")
    match = re.search(r"(?:^|_)(\d+)$", stimulus_id)
    if match:
        return match.group(1)
    return None


def build_paired_binary_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = []
    for record in records:
        category = str(record.get("category"))
        if category not in categories:
            categories.append(category)
    if len(categories) != 2:
        raise ValueError("paired_vote scoring requires a dictionary with exactly 2 categories.")

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        pair_id = _infer_pair_id(record)
        if pair_id is None:
            raise ValueError(
                "paired_vote scoring requires explicit pair_id metadata or safely inferable numeric stimulus IDs."
            )
        category = str(record["category"])
        if category in grouped.setdefault(pair_id, {}):
            raise ValueError(f"paired_vote found duplicate category {category!r} for pair {pair_id!r}.")
        grouped[pair_id][category] = record

    pairs = []
    for pair_id in sorted(grouped):
        items = grouped[pair_id]
        missing = [category for category in categories if category not in items]
        if missing:
            raise ValueError(f"paired_vote pair {pair_id!r} is missing categories: {missing}.")
        pairs.append(
            {
                "pair_id": pair_id,
                "category_a": items[categories[0]],
                "category_b": items[categories[1]],
            }
        )
    return {
        "category_a": categories[0],
        "category_b": categories[1],
        "pairs": pairs,
    }


def score_paired_vote(query_response: Any, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = np.asarray(query_response, dtype=float)
    paired = build_paired_binary_records(records)
    category_a = paired["category_a"]
    category_b = paired["category_b"]
    vote_counts = {category_a: 0, category_b: 0}
    margins = {category_a: [], category_b: []}
    winning_similarities = {category_a: [], category_b: []}
    votes = []

    for pair in paired["pairs"]:
        record_a = pair["category_a"]
        record_b = pair["category_b"]
        vector_a = np.asarray(record_a["mean_response"], dtype=float)
        vector_b = np.asarray(record_b["mean_response"], dtype=float)
        similarity_a = cosine_similarity(query, vector_a)
        similarity_b = cosine_similarity(query, vector_b)
        margin_b_minus_a = float(similarity_b - similarity_a)
        if margin_b_minus_a > 0.0:
            winner = category_b
            winning_similarity = similarity_b
            winning_margin = margin_b_minus_a
        else:
            winner = category_a
            winning_similarity = similarity_a
            winning_margin = float(similarity_a - similarity_b)
        vote_counts[winner] += 1
        margins[winner].append(winning_margin)
        winning_similarities[winner].append(float(winning_similarity))
        votes.append(
            {
                "pair_id": pair["pair_id"],
                "category_a": category_a,
                "category_b": category_b,
                "category_a_stimulus_id": record_a["stimulus_id"],
                "category_b_stimulus_id": record_b["stimulus_id"],
                "similarity_a": float(similarity_a),
                "similarity_b": float(similarity_b),
                "margin_b_minus_a": margin_b_minus_a,
                "winner": winner,
            }
        )

    category_scores = []
    for category in [category_a, category_b]:
        category_scores.append(
            {
                "category": category,
                "score": float(vote_counts[category]),
                "vote_count": vote_counts[category],
                "mean_margin": float(np.mean(margins[category])) if margins[category] else 0.0,
                "mean_similarity_to_winning_side": (
                    float(np.mean(winning_similarities[category]))
                    if winning_similarities[category]
                    else 0.0
                ),
                "n_pairs": len(paired["pairs"]),
                "best_stimulus_id": None,
            }
        )

    if vote_counts[category_a] == vote_counts[category_b]:
        category_scores = sorted(
            category_scores,
            key=lambda item: 0 if item["category"] == category_a else 1,
        )
        predicted_category = category_a
    else:
        category_scores = sorted(category_scores, key=lambda item: item["vote_count"], reverse=True)
        predicted_category = category_scores[0]["category"]

    return category_scores, {
        "category_a": category_a,
        "category_b": category_b,
        "votes": votes,
        "vote_counts": vote_counts,
        "tie_handling_rule": "If vote counts tie, choose category_a, the first category in dictionary order.",
        "predicted_category": predicted_category,
    }


def compute_category_deltas(
    records: list[dict[str, Any]],
    neutral_baseline: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for record in records:
        grouped.setdefault(record["category"], []).append(
            np.asarray(record["mean_response"], dtype=float)
        )

    baseline = np.asarray(neutral_baseline, dtype=float)
    deltas = {}
    for category, vectors in grouped.items():
        centroid = np.mean(np.vstack(vectors), axis=0)
        delta = centroid - baseline
        deltas[category] = {
            "category": category,
            "centroid": centroid,
            "delta": delta,
            "category_delta_norm": vector_norm(delta),
            "n_signatures": len(vectors),
        }
    return deltas


def score_category_deltas(
    query_delta: Any,
    category_deltas: dict[str, dict[str, Any]],
    scoring: str = "full",
    top_k: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = np.asarray(query_delta, dtype=float)
    category_scores = []
    diagnostics = {
        "selected_dim_count": {},
        "selected_dim_indices": {},
        "category_delta_norm": {},
    }
    for category, info in category_deltas.items():
        delta = np.asarray(info["delta"], dtype=float)
        diagnostics["category_delta_norm"][category] = info["category_delta_norm"]
        if scoring == "full":
            score = cosine_similarity(query, delta)
            selected = np.arange(delta.size)
        elif scoring == "topk":
            selected = select_top_k_dims(delta, top_k)
            score = cosine_similarity(query[selected], delta[selected])
        elif scoring == "weighted":
            selected = np.arange(delta.size)
            score = weighted_cosine(query, delta, np.abs(delta))
        else:
            raise ValueError(f"Unsupported scoring mode: {scoring}")
        diagnostics["selected_dim_count"][category] = int(selected.size)
        diagnostics["selected_dim_indices"][category] = selected
        category_scores.append(
            {
                "category": category,
                "score": score,
                "centroid_cosine_similarity": score,
                "category_delta_norm": info["category_delta_norm"],
                "selected_dim_count": int(selected.size),
                "n_signatures": info["n_signatures"],
                "best_stimulus_id": None,
            }
        )
    return sorted(category_scores, key=lambda item: item["score"], reverse=True), diagnostics


def aggregate_category_centroids(
    query_vector: Any,
    signature_records: list[dict[str, Any]],
    ranked_stimuli: list[dict[str, Any]],
    sort_metric: str = "cosine",
) -> list[dict[str, Any]]:
    if sort_metric not in {"cosine", "pearson"}:
        raise ValueError(f"Unsupported sort metric: {sort_metric}")

    best_by_category = {}
    score_field = "cosine_similarity" if sort_metric == "cosine" else "pearson_correlation"
    for item in ranked_stimuli:
        current = best_by_category.get(item["category"])
        if current is None or item[score_field] > current[score_field]:
            best_by_category[item["category"]] = item

    category_scores = []
    for category, centroid_info in compute_category_centroids(signature_records).items():
        centroid = centroid_info["vector"]
        best = best_by_category.get(category, {})
        category_scores.append(
            {
                "category": category,
                "n_signatures": centroid_info["n_signatures"],
                "centroid_cosine_similarity": cosine_similarity(query_vector, centroid),
                "centroid_pearson_correlation": pearson_correlation(query_vector, centroid),
                "centroid_norm": centroid_info["centroid_norm"],
                "best_stimulus_id": best.get("stimulus_id"),
            }
        )
    sort_field = "centroid_cosine_similarity" if sort_metric == "cosine" else "centroid_pearson_correlation"
    return sorted(category_scores, key=lambda item: item[sort_field], reverse=True)


def aggregate_categories(
    query_vector: Any,
    signature_records: list[dict[str, Any]],
    ranked_stimuli: list[dict[str, Any]],
    aggregation: str = "centroid",
    sort_metric: str = "cosine",
) -> list[dict[str, Any]]:
    if aggregation == "best":
        return aggregate_category_best(ranked_stimuli, sort_metric=sort_metric)
    if aggregation == "mean":
        return aggregate_category_scores(ranked_stimuli, sort_metric=sort_metric)
    if aggregation == "centroid":
        return aggregate_category_centroids(
            query_vector,
            signature_records,
            ranked_stimuli,
            sort_metric=sort_metric,
        )
    raise ValueError(f"Unsupported aggregation mode: {aggregation}")


def load_dictionary_records(index_path: str | Path) -> list[dict[str, Any]]:
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    records = []
    for item in index.get("stimuli", []):
        output_path = Path(item["output_path"])
        records.append(json.loads(output_path.read_text(encoding="utf-8")))
    return records
