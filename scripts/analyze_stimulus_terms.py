#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_TERMS = [
    "checks",
    "verifies",
    "looks again",
    "expects",
    "searches",
    "instruction",
    "schedule",
    "message",
    "sign",
    "room",
    "label",
    "map",
]


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b")


def count_terms(text: str, terms: list[str]) -> dict[str, int]:
    normalized = re.sub(r"\s+", " ", text.lower())
    return {term: len(_term_pattern(term).findall(normalized)) for term in terms}


def _iter_dictionary_items(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for category, items in payload.get("categories", {}).items():
        for item in items:
            rows.append((str(category), str(item["id"]), str(item["text"])))
    return rows


def _iter_eval_items(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for item in payload.get("items", []):
        rows.append((str(item["expected_category"]), str(item["id"]), str(item["text"])))
    return rows


def summarize_term_counts(path: str | Path, terms: list[str] | None = None) -> dict[str, Any]:
    terms = terms or DEFAULT_TERMS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = _iter_dictionary_items(payload) if "categories" in payload else _iter_eval_items(payload)

    categories: dict[str, dict[str, Any]] = {}
    for category, item_id, text in rows:
        counts = count_terms(text, terms)
        entry = categories.setdefault(
            category,
            {
                "n_items": 0,
                "term_counts": {term: 0 for term in terms},
                "items_with_any_term": 0,
                "item_counts": [],
            },
        )
        entry["n_items"] += 1
        has_any = any(counts.values())
        entry["items_with_any_term"] += int(has_any)
        for term, count in counts.items():
            entry["term_counts"][term] += count
        entry["item_counts"].append({"id": item_id, "counts": counts, "has_any_term": has_any})

    return {
        "path": str(path),
        "terms": terms,
        "categories": categories,
    }


def format_summary(summary: dict[str, Any]) -> str:
    terms = summary["terms"]
    lines = [f"Stimulus term diagnostics: {summary['path']}"]
    for category, values in summary["categories"].items():
        lines.append(
            f"{category}: n={values['n_items']} items_with_any_term={values['items_with_any_term']}"
        )
        active = {
            term: count
            for term, count in values["term_counts"].items()
            if count
        }
        lines.append(f"  active_terms: {active if active else {}}")
    lines.append(f"terms_checked: {', '.join(terms)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Count uncertainty/checking words in ASNE stimulus files.")
    parser.add_argument("paths", nargs="+", help="Dictionary or eval JSON paths.")
    parser.add_argument("--terms", nargs="+", default=DEFAULT_TERMS, help="Terms or phrases to count.")
    args = parser.parse_args()

    for index, path in enumerate(args.paths):
        if index:
            print()
        print(format_summary(summarize_term_counts(path, args.terms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
