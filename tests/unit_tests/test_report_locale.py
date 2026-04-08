"""Locale helpers — pure logic, no DB or network."""

from tools.finance.report_locale import (
    infer_locale_from_question,
    normalize_report_locale,
    resolve_report_locale,
    risk_flag_message,
)


def test_infer_locale_empty_defaults_zh() -> None:
    assert infer_locale_from_question("") == "zh"


def test_infer_locale_cjk() -> None:
    assert infer_locale_from_question("你好世界请问苹果") == "zh"


def test_infer_locale_english() -> None:
    assert infer_locale_from_question("What was liquidity in fiscal 2024?") == "en"


def test_normalize_report_locale() -> None:
    assert normalize_report_locale("ZH-CN") == "zh"
    assert normalize_report_locale("auto") == "auto"
    assert normalize_report_locale("nope") is None


def test_resolve_explicit_overrides_inference() -> None:
    assert resolve_report_locale("plain english question", "zh") == "zh"


def test_risk_flag_message_en() -> None:
    msg = risk_flag_message("no_citations", "en")
    assert "citation" in msg.lower()
