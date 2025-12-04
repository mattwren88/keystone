from tldr_email.summarizer import EmailTLDR, format_tldr, summarize_email


def test_summarize_prioritizes_schedule_and_keywords():
    email = """
    Subject: Week 3 Rehearsals
    Hi team,

    Monday 6:30pm rehearsal in Brooks Hall.
    Bring your stand and music.
    Thursday 7:00pm sectional in Room 201.

    Please wear all black for Friday's performance at 8pm.
    """

    summary = summarize_email(email)

    assert summary.highlights[:3] == [
        "Monday 6:30pm rehearsal in Brooks Hall.",
        "Thursday 7:00pm sectional in Room 201.",
        "Bring your stand and music.",
    ]
    assert summary.schedule == [
        "Monday 6:30pm rehearsal in Brooks Hall.",
        "Thursday 7:00pm sectional in Room 201.",
    ]


def test_summarize_falls_back_to_first_lines():
    email = """
    Hello all,
    Just a quick reminder to hydrate and rest.
    Looking forward to seeing you.
    """

    summary = summarize_email(email, max_highlights=2)
    assert summary.highlights == ["Hello all,", "Just a quick reminder to hydrate and rest."]
    assert summary.schedule == []


def test_format_tldr_creates_readable_output():
    email = """
    Friday 5:00pm dress rehearsal in the auditorium.
    Equipment check is required.
    """

    formatted = format_tldr(email)

    assert "Highlights" in formatted
    assert "Schedule" in formatted
    assert "Friday 5:00pm dress rehearsal in the auditorium." in formatted


def test_dedupes_lines_while_preserving_order():
    email = """
    Monday 6pm rehearsal in Brooks Hall.
    Monday 6pm rehearsal in Brooks Hall.
    Bring your scores.
    Bring your scores.
    """

    summary = summarize_email(email)
    assert summary.highlights == [
        "Monday 6pm rehearsal in Brooks Hall.",
        "Bring your scores.",
    ]
    assert summary.schedule == ["Monday 6pm rehearsal in Brooks Hall."]
