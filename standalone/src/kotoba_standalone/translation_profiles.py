from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "translation-profiles.json"
DEFAULT_PROFILE_NAME = "없음"


@dataclass(frozen=True)
class TranslationTerm:
    source: str
    target: str
    note: str = ""


@dataclass(frozen=True)
class TextCorrection:
    source: str
    target: str
    note: str = ""


@dataclass(frozen=True)
class SubtitleFilter:
    pattern: str
    match: str = "exact"
    note: str = ""


@dataclass(frozen=True)
class TranslationProfile:
    name: str
    instruction: str = ""
    terms: tuple[TranslationTerm, ...] = ()
    corrections: tuple[TextCorrection, ...] = ()
    subtitle_filters: tuple[SubtitleFilter, ...] = ()


def default_translation_profiles() -> dict[str, TranslationProfile]:
    return {
        "없음": TranslationProfile(name="없음"),
        "AV": TranslationProfile(
            name="AV",
            instruction=(
                "성인 영상 자막의 맥락과 직접적인 표현을 임의로 순화하지 않는다. "
                "화자의 말투와 관계를 유지하되, 원문보다 더 노골적으로 만들지는 않는다."
            ),
        ),
        "애니메이션": TranslationProfile(
            name="애니메이션",
            instruction=(
                "캐릭터의 개성과 말투를 살려 번역한다. "
                "고유명사, 기술명, 작품 내 용어는 문맥상 일관되게 유지한다."
            ),
        ),
    }


def load_translation_profiles(path: Path = PROFILES_PATH) -> dict[str, TranslationProfile]:
    defaults = default_translation_profiles()
    if not path.exists():
        return defaults
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return defaults

    profiles: dict[str, TranslationProfile] = {}
    raw_profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if isinstance(raw_profiles, list):
        for item in raw_profiles:
            profile = _profile_from_json(item)
            if profile is not None:
                profiles[profile.name] = profile
    elif isinstance(raw_profiles, dict):
        for name, item in raw_profiles.items():
            profile = _profile_from_json(item, fallback_name=str(name))
            if profile is not None:
                profiles[profile.name] = profile

    if not profiles:
        return defaults
    for name, profile in defaults.items():
        profiles.setdefault(name, profile)
    return profiles


def save_translation_profiles(
    profiles: dict[str, TranslationProfile],
    path: Path = PROFILES_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profiles": [
            {
                "name": profile.name,
                "instruction": profile.instruction,
                "terms": [
                    {"source": term.source, "target": term.target, "note": term.note}
                    for term in profile.terms
                ],
                "corrections": [
                    {"source": rule.source, "target": rule.target, "note": rule.note}
                    for rule in profile.corrections
                ],
                "subtitle_filters": [
                    {"pattern": rule.pattern, "match": rule.match, "note": rule.note}
                    for rule in profile.subtitle_filters
                ],
            }
            for profile in profiles.values()
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_translation_profile(
    name: str,
    path: Path = PROFILES_PATH,
) -> TranslationProfile:
    profiles = load_translation_profiles(path)
    return profiles.get(name.strip(), profiles[DEFAULT_PROFILE_NAME])


def _profile_from_json(item: object, fallback_name: str = "") -> TranslationProfile | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or fallback_name).strip()
    if not name:
        return None
    instruction = str(item.get("instruction") or "").strip()
    terms: list[TranslationTerm] = []
    raw_terms = item.get("terms", [])
    if isinstance(raw_terms, list):
        for raw_term in raw_terms:
            if not isinstance(raw_term, dict):
                continue
            source = str(raw_term.get("source") or "").strip()
            target = str(raw_term.get("target") or "").strip()
            if source and target:
                terms.append(
                    TranslationTerm(
                        source=source,
                        target=target,
                        note=str(raw_term.get("note") or "").strip(),
                    )
                )
    corrections: list[TextCorrection] = []
    raw_corrections = item.get("corrections", [])
    if isinstance(raw_corrections, list):
        for raw_rule in raw_corrections:
            if not isinstance(raw_rule, dict):
                continue
            source = str(raw_rule.get("source") or "").strip()
            target = str(raw_rule.get("target") or "").strip()
            if source and target:
                corrections.append(
                    TextCorrection(
                        source=source,
                        target=target,
                        note=str(raw_rule.get("note") or "").strip(),
                    )
                )
    subtitle_filters: list[SubtitleFilter] = []
    raw_filters = item.get("subtitle_filters", [])
    if isinstance(raw_filters, list):
        for raw_rule in raw_filters:
            if not isinstance(raw_rule, dict):
                continue
            pattern = str(raw_rule.get("pattern") or "").strip()
            if not pattern:
                continue
            match = str(raw_rule.get("match") or "exact").strip().lower()
            if match not in {"exact", "contains"}:
                match = "exact"
            subtitle_filters.append(
                SubtitleFilter(
                    pattern=pattern,
                    match=match,
                    note=str(raw_rule.get("note") or "").strip(),
                )
            )
    return TranslationProfile(
        name=name,
        instruction=instruction,
        terms=tuple(terms),
        corrections=tuple(corrections),
        subtitle_filters=tuple(subtitle_filters),
    )
