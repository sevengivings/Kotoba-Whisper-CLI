from __future__ import annotations

from pathlib import Path

from kotoba_standalone.translation_profiles import (
    SubtitleFilter,
    TextCorrection,
    TranslationProfile,
    TranslationTerm,
    get_translation_profile,
    load_translation_profiles,
    save_translation_profiles,
)


def test_default_profiles_include_requested_categories(tmp_path: Path) -> None:
    profiles = load_translation_profiles(tmp_path / "missing.json")

    assert tuple(profiles)[:3] == ("없음", "AV", "애니메이션")


def test_translation_profiles_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "translation-profiles.json"
    profiles = {
        "없음": TranslationProfile(name="없음"),
        "작품A": TranslationProfile(
            name="작품A",
            instruction="고유명사는 일관되게 유지한다.",
            terms=(TranslationTerm("先生", "선생님", "호칭"),),
            corrections=(TextCorrection("選手", "先生"),),
            subtitle_filters=(SubtitleFilter("ご視聴ありがとうございました", "contains"),),
        ),
    }

    save_translation_profiles(profiles, path)

    loaded = load_translation_profiles(path)
    assert loaded["작품A"].instruction == "고유명사는 일관되게 유지한다."
    assert loaded["작품A"].terms == (TranslationTerm("先生", "선생님", "호칭"),)
    assert loaded["작품A"].corrections == (TextCorrection("選手", "先生"),)
    assert loaded["작품A"].subtitle_filters == (SubtitleFilter("ご視聴ありがとうございました", "contains"),)
    assert get_translation_profile("없는 프로필", path).name == "없음"
