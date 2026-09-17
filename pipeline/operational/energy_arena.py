"""Energy Arena challenge discovery, payload conversion, and submission."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from pipeline.delivery_index import validate_delivery_index


@dataclass(frozen=True)
class ArenaChallenge:
    challenge_id: str
    target_start: datetime
    deadline: datetime
    objective: str
    area: str
    timezone: str
    quantiles: tuple[float, ...]
    precision_decimals: int
    detail: dict[str, Any]


@dataclass(frozen=True)
class ArenaTargets:
    point: ArenaChallenge
    quantile: ArenaChallenge

    @property
    def target_start(self) -> datetime:
        return self.point.target_start

    @property
    def target_date(self) -> date:
        return self.target_start.date()


def _headers(api_key: str = "") -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _json_get(url: str, api_key: str = "", timeout: int = 30) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=_headers(api_key), timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Energy Arena request failed for {url}: {exc}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Energy Arena returned an unexpected response for {url}.")
    return body


def _parse_datetime(raw: object, label: str) -> datetime:
    text = str(raw or "").strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(f"Energy Arena {label} is not valid ISO-8601: {raw!r}") from exc
    if value.tzinfo is None:
        raise RuntimeError(f"Energy Arena {label} lacks a timezone: {raw!r}")
    return value


def _canonical_target_start(target_date: date, timezone: str) -> datetime:
    return datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        tzinfo=ZoneInfo(timezone),
    )


def resolve_targets(
    *,
    api_base: str,
    point_challenge_id: str,
    quantile_challenge_id: str,
    api_key: str = "",
    requested_target_date: Optional[date] = None,
    target_date_offset_days: int = 0,
    timeout: int = 30,
) -> ArenaTargets:
    """Resolve and validate the paired DE-LU day-ahead price challenges."""
    if requested_target_date is not None and target_date_offset_days:
        raise ValueError(
            "requested_target_date and target_date_offset_days cannot be combined."
        )
    catalog = _json_get(f"{api_base}/api/v1/challenges/open", api_key, timeout)
    entries = {
        str(item.get("challenge_id")): item
        for item in catalog.get("active_challenges", [])
        if isinstance(item, dict)
    }

    def resolve(challenge_id: str, expected_objective: str) -> ArenaChallenge:
        if challenge_id not in entries:
            raise RuntimeError(f"Energy Arena challenge {challenge_id} is not currently open.")
        entry = entries[challenge_id]
        detail = _json_get(
            f"{api_base}/api/v1/challenges/{challenge_id}", api_key, timeout
        )
        objective = str(detail.get("forecast_objective") or "").lower()
        areas = [str(value) for value in detail.get("areas", [])]
        target_code = str(detail.get("target_code") or "")
        target_period = detail.get("target_period") or {}
        timezone = str(
            target_period.get("timezone")
            or detail.get("reference_timezone")
            or "Europe/Berlin"
        )
        if objective != expected_objective:
            raise RuntimeError(
                f"Challenge {challenge_id} is {objective!r}, expected {expected_objective!r}."
            )
        if target_code != "day_ahead_price" or "DE_LU" not in areas:
            raise RuntimeError(
                f"Challenge {challenge_id} is not the DE-LU day-ahead price challenge."
            )
        api_target = _parse_datetime(entry.get("next_target_start"), "target start")
        api_target_local = api_target.astimezone(ZoneInfo(timezone))
        if requested_target_date is not None:
            target_start = _canonical_target_start(requested_target_date, timezone)
        elif target_date_offset_days:
            target_start = _canonical_target_start(
                api_target_local.date() + timedelta(days=target_date_offset_days),
                timezone,
            )
        else:
            target_start = api_target_local
        deadline = _parse_datetime(entry.get("next_submission_deadline"), "deadline")
        probabilities = detail.get("probabilistic_forecast") or {}
        quantiles = tuple(float(value) for value in probabilities.get("quantiles", []))
        precision = int((detail.get("constraints") or {}).get("precision_decimals", 2))
        return ArenaChallenge(
            challenge_id=str(detail.get("code") or challenge_id).strip(),
            target_start=target_start,
            deadline=deadline,
            objective=objective,
            area="DE_LU",
            timezone=timezone,
            quantiles=quantiles,
            precision_decimals=precision,
            detail=detail,
        )

    point = resolve(point_challenge_id, "point")
    quantile = resolve(quantile_challenge_id, "quantile")
    if point.target_start != quantile.target_start:
        raise RuntimeError(
            "Point and quantile challenges do not resolve to the same target start: "
            f"{point.target_start} != {quantile.target_start}."
        )
    if not quantile.quantiles:
        raise RuntimeError("Quantile challenge does not publish its required quantiles.")
    return ArenaTargets(point=point, quantile=quantile)


def physical_mtu_numbers(target_date: date, timezone: str) -> np.ndarray:
    """Map the physical target grid to canonical one-based local-clock MTUs."""
    tz = ZoneInfo(timezone)
    start = pd.Timestamp(
        datetime(target_date.year, target_date.month, target_date.day, tzinfo=tz)
    )
    next_date = target_date + timedelta(days=1)
    end = pd.Timestamp(
        datetime(next_date.year, next_date.month, next_date.day, tzinfo=tz)
    )
    physical = pd.date_range(start, end, freq="15min", inclusive="left")
    return np.asarray(physical.hour * 4 + physical.minute // 15 + 1, dtype=int)


def build_point_payload(
    forecast: pd.DataFrame,
    challenge: ArenaChallenge,
) -> dict[str, Any]:
    validate_delivery_index(forecast.index, require_complete_days=True)
    day = pd.Timestamp(challenge.target_start.date())
    try:
        canonical = forecast.loc[day, "y_pred"].reindex(range(1, 97))
    except KeyError as exc:
        raise ValueError(f"Point forecast lacks target day {day.date()}.") from exc
    mtus = physical_mtu_numbers(day.date(), challenge.timezone)
    values = canonical.reindex(mtus).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Point payload contains NaN or infinite values.")
    return {
        "challenge_id": challenge.challenge_id,
        "target_start": challenge.target_start.isoformat(),
        "values": np.round(values, challenge.precision_decimals).tolist(),
    }


def _quantile_column(quantile: float) -> str:
    return f"q{quantile:.3f}"


def build_quantile_payload(
    forecast: pd.DataFrame,
    challenge: ArenaChallenge,
) -> dict[str, Any]:
    validate_delivery_index(forecast.index, require_complete_days=True)
    day = pd.Timestamp(challenge.target_start.date())
    columns = [_quantile_column(value) for value in challenge.quantiles]
    missing = [column for column in columns if column not in forecast.columns]
    if missing:
        raise ValueError(f"SQRA forecast lacks Arena quantiles: {missing}")
    try:
        canonical = forecast.loc[day, columns].reindex(range(1, 97))
    except KeyError as exc:
        raise ValueError(f"SQRA forecast lacks target day {day.date()}.") from exc
    mtus = physical_mtu_numbers(day.date(), challenge.timezone)
    values = canonical.reindex(mtus).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Quantile payload contains NaN or infinite values.")
    if np.any(np.diff(values, axis=1) < 0):
        raise ValueError("Quantile payload contains crossing quantiles.")
    return {
        "challenge_id": challenge.challenge_id,
        "target_start": challenge.target_start.isoformat(),
        "values": np.round(values, challenge.precision_decimals).tolist(),
    }


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def account_fingerprint(api_key: str, profile: str) -> str:
    if profile:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in profile)
    return f"key_{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:12]}"


def submit_payload(
    *,
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 30,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("An Energy Arena API key is required for submission.")
    url = f"{api_base}/api/v1/submissions"
    try:
        response = requests.post(
            url,
            headers={"X-API-Key": api_key},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = getattr(locals().get("response"), "text", "")
        raise RuntimeError(f"Energy Arena submission failed: {exc}. {detail[:500]}") from exc
    return body if isinstance(body, dict) else {"response": body}


def submission_record_path(
    output_root: Path,
    account: str,
    challenge_id: str,
    target_start: datetime,
) -> Path:
    # Keep ``target_start`` in the interface for callers that also construct a
    # dated payload path. Submission receipts intentionally retain only the
    # latest successful response for each account/challenge pair.
    del target_start
    return output_root / "submissions" / account / challenge_id / "latest.json"


def already_submitted(record_path: Path, payload: dict[str, Any]) -> bool:
    if not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "success" and record.get("payload_hash") == payload_hash(payload)


def submit_with_record(
    *,
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    record_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    if already_submitted(record_path, payload) and not force:
        return {"status": "already_submitted", "record": str(record_path)}
    response = submit_payload(api_base=api_base, api_key=api_key, payload=payload)
    record = {
        "status": "success",
        "submitted_at": datetime.now().astimezone().isoformat(),
        "payload_hash": payload_hash(payload),
        "payload": payload,
        "response": response,
    }
    atomic_write_json(record_path, record)
    return response
