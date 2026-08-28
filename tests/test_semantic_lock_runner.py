from __future__ import annotations

from dubber.models import Segment, Transcript
from dubber.sync_timeline import merge_semantic_continuations


def _segment(start, end, source, target):
    return Segment(
        start=start,
        end=end,
        speaker="Speaker 1",
        source_text=source,
        target_text=target,
        emotion="neutral",
    )


def test_real_parent_lock_failure_chunk_17_and_18_are_merged():
    transcript = Transcript(
        detected_language="English",
        target_language="Persian (فارسی)",
        title="",
        segments=[
            _segment(
                304.79,
                317.07,
                (
                    "Tap Copy now and you land on two modes. Smart copy is the "
                    "simpler one: it places orders at the same fund ratio as the "
                    "trader you are following, so your position sizes scale to "
                    "your capital automatically. All you enter is the Follow "
                    "amount, anywhere from 10 to 200,000 USDT,"
                ),
                (
                    "اکنون روی کپی ضربه بزنید و وارد دو حالت می‌شوید. کپی هوشمند "
                    "ساده‌تر است و اندازه موقعیت‌ها را متناسب با سرمایه تنظیم "
                    "می‌کند. مقدار دنبال‌کردن از ۱۰ تا ۲۰۰٬۰۰۰ USDT است،"
                ),
            ),
            _segment(
                317.07,
                324.28,
                "drawn from your futures account.",
                "که از حساب فیوچرز شما برداشت می‌شود.",
            ),
        ],
    )

    result = merge_semantic_continuations(transcript)

    assert len(result.segments) == 1
    merged = result.segments[0]
    assert merged.start == 304.79
    assert merged.end == 324.28
    assert round(merged.duration, 2) == 19.49
    assert "200,000 USDT, drawn from your futures account." in merged.source_text


def test_complete_sentence_is_not_merged_even_with_zero_gap():
    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="",
        segments=[
            _segment(0.0, 2.0, "First sentence.", "جمله اول."),
            _segment(2.0, 4.0, "Second sentence.", "جمله دوم."),
        ],
    )

    result = merge_semantic_continuations(transcript)
    assert len(result.segments) == 2


def test_nonzero_real_gap_is_not_consumed_by_semantic_merge():
    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="",
        segments=[
            _segment(0.0, 2.0, "unfinished,", "ادامه دارد،"),
            _segment(2.20, 4.0, "but after a pause.", "اما پس از مکث."),
        ],
    )

    result = merge_semantic_continuations(transcript)
    assert len(result.segments) == 2
