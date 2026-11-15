"""
HireNex - experience calculation (shared)

Pure date / duration helpers with no ML dependencies. Used by retrival.py to
compute total experience for filtering, and by evaluation.py to score parsers
with exactly the same logic, so the benchmark measures what recruiters see.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def parse_month_year(
    text: str,
) -> tuple[int, int] | None:

    match = re.search(
        r"\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\.?\s+"
        r"((?:19|20)\d{2})\b",
        text,
        re.I,
    )

    if not match:
        return None

    month_text = match.group(1).lower()
    year = int(match.group(2))

    if month_text.startswith("sep"):
        month = 9
    else:
        month = MONTHS.get(
            month_text[:3],
            1,
        )

    return year, month


def parse_year(
    text: str,
) -> int | None:

    match = re.search(
        r"\b((?:19|20)\d{2})\b",
        text,
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def duration_to_months(
    duration: str,
) -> int:

    text = clean_text(
        duration
    ).lower()

    if not text:
        return 0

    # Explicit "2 years 3 months"
    year_match = re.search(
        r"(\d+(?:\.\d+)?)\s+years?",
        text,
        re.I,
    )

    month_match = re.search(
        r"(\d+)\s+months?",
        text,
        re.I,
    )

    if year_match:

        months = round(
            float(year_match.group(1))
            * 12
        )

        if month_match:
            months += int(
                month_match.group(1)
            )

        return months

    # Month/year range, e.g. Jan 2022 - Dec 2024
    dates = re.findall(
        r"\b(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\.?\s+(?:19|20)\d{2}\b",
        text,
        re.I,
    )

    if len(dates) >= 2:

        start = parse_month_year(
            dates[0]
        )

        end = parse_month_year(
            dates[1]
        )

        if start and end:

            months = (
                (end[0] - start[0]) * 12
                + (end[1] - start[1])
            )

            return max(
                months,
                0,
            )

    # Year-only range, e.g. 2022 - 2024
    year_range = re.search(
        r"\b((?:19|20)\d{2})\s*"
        r"(?:-|–|—|to)\s*"
        r"((?:19|20)\d{2}|present|current)\b",
        text,
        re.I,
    )

    if year_range:

        start_year = int(
            year_range.group(1)
        )

        end_value = year_range.group(2)

        if end_value.lower() in {
            "present",
            "current",
        }:
            end_year = datetime.now().year

        else:
            end_year = int(
                end_value
            )

        return max(
            (end_year - start_year) * 12,
            0,
        )

    return 0


def experience_item_to_months(
    experience: dict[str, Any],
) -> int:

    duration = clean_text(
        experience.get("duration", "")
    )

    months = duration_to_months(
        duration
    )

    if months:
        return months

    start = parse_year(
        clean_text(
            experience.get("start_year", "")
        )
    )

    end_text = clean_text(
        experience.get("end_year", "")
    )

    if start is None:
        return 0

    if end_text.lower() in {
        "present",
        "current",
        "now",
    }:

        end = datetime.now().year

    else:

        end = parse_year(
            end_text
        )

    if end is None:
        return 0

    return max(
        (end - start) * 12,
        0,
    )


def experience_item_interval(
    experience: dict[str, Any],
) -> tuple[int, int] | None:
    """Return an approximate inclusive month interval for one job."""
    duration = clean_text(experience.get("duration", ""))

    # Month-year interval from the LLM duration field.
    month_matches = re.findall(
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(?:19|20)\d{2}",
        duration,
        re.I,
    )

    if len(month_matches) >= 2:
        start = parse_month_year(month_matches[0])
        end = parse_month_year(month_matches[1])
        if start and end:
            return (
                start[0] * 12 + start[1],
                end[0] * 12 + end[1],
            )

    # Year-only interval from either parser.
    start_year = parse_year(
        clean_text(experience.get("start_year", ""))
    )
    end_text = clean_text(
        experience.get("end_year", "")
    )

    if start_year is None:
        return None

    if end_text.lower() in {"present", "current", "now"}:
        end_year = datetime.now().year
    else:
        end_year = parse_year(end_text)

    if end_year is None:
        return None

    return (
        start_year * 12 + 1,
        end_year * 12 + 12,
    )


def calculate_total_experience(
    experiences: list[dict[str, Any]],
) -> float:
    """Calculate approximate total experience without double-counting overlap."""
    intervals: list[tuple[int, int]] = []

    for experience in experiences:
        interval = experience_item_interval(experience)
        if interval:
            intervals.append(interval)

    if intervals:
        intervals.sort()
        merged: list[list[int]] = []

        for start, end in intervals:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        total_months = sum(
            end - start + 1
            for start, end in merged
        )

        return round(total_months / 12, 1)

    # If dates are not parseable, fall back to explicit durations.
    total_months = sum(
        duration_to_months(
            clean_text(experience.get("duration", ""))
        )
        for experience in experiences
    )

    return round(total_months / 12, 1)

