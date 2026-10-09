"""cfb.kalshi_display: tables fit the width, and --track lists only the orders worth a look. No network."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from cfb import kalshi_display as display

NOW = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)  # Friday 4pm ET


def tracked(oid, status, filled, kick_days, placed_minutes=600, result=None):
    return {"order_id": oid, "game": "Texas at Oklahoma", "side": "home", "team": "Oklahoma", "status": status,
            "fill_count": filled, "initial_count": 4, "count": 4, "plan_price": 0.26,
            "kickoff": (NOW + timedelta(days=kick_days)).isoformat(),
            "placed_at": (NOW - timedelta(minutes=placed_minutes)).isoformat(), "result": result,
            "pnl": -1.05 if result == "no" else None, "mark_to_market": None if result else 0.0,
            "maker_fees": 0.01, "taker_fees": 0}


def test_track_shows_resting_just_placed_and_recently_filled_orders():
    latest = pd.DataFrame([
        tracked("resting", "resting", 0, 1),
        tracked("part-cancelled", "canceled", 2, 1),
        tracked("cancelled", "canceled", 0, 1),
        tracked("expired", "executed", 0, -2),
        tracked("just-placed", "not found", None, 1, placed_minutes=1),
        tracked("purged", "not found", None, 1),                         # an old cancel Kalshi no longer lists
        tracked("lost-last-week", "executed", 4, -3, result="no"),
        tracked("lost-long-ago", "executed", 4, -20, result="no"),
    ]).set_index("order_id")
    assert list(display.shown_orders(latest, NOW).index) == ["resting", "part-cancelled", "just-placed",
                                                              "lost-last-week"]
    text = display.track_table(latest, NOW, width=80)
    assert "lost" in text and "-$1.05" in text and "placed" in text
    assert "not shown: 3 cancelled or expired without a fill, 1 settled over 7 days ago" in " ".join(text.split())


def test_tables_fit_the_width_and_cut_only_text_columns():
    rows = [["Florida International", "@ Jacksonville State", "Sat 12:00pm", "43¢", "replace 28¢ x7"]]
    for width in (100, 60, 40):
        text = display.table(["Team", "Opponent", "Kickoff", "Price", "Action"], rows, "llLrL", width)
        lines = text.splitlines()
        narrowest = 8 + 8 + len("Sat 12:00pm") + len("Price") + len("replace 28¢ x7") + 4 * len(display.GAP)
        assert max(map(len, lines)) <= max(width, narrowest)
        assert "Sat 12:00pm" in lines[2] and "43¢" in lines[2] and "replace 28¢ x7" in lines[2]
    assert "…" not in display.table(["Team"], [["Iowa"]], "l", 40)
    cut = display.table(["Team", "Opponent"], [rows[0][:2]], "ll", 39).splitlines()
    assert max(map(len, cut)) == 39 and cut[2].startswith("Florida Internati…  @ Jacksonville Sta…")


def test_cells():
    assert display.cents(0.43) == "43¢" and display.cents(0.456, 1) == "45.6¢" and display.cents(None) == ""
    assert display.signed_cents(0.0212) == "+2.1¢" and display.pct(0.4825) == "48%"
    assert display.money(1.5) == "$1.50" and display.money(-0.02, signed=True) == "-$0.02"
    assert display.money(0.001, signed=True) == "$0.00" and display.money(0.02, signed=True) == "+$0.02"
    assert display.book(0.43, 0.44) == "43/44¢" and display.book(None, 0.44) == ""
    assert display.matchup("Iowa at Washington", "away") == ("Iowa", "@ Washington")
    assert display.matchup("Iowa at Washington", "home") == ("Washington", "vs Iowa")
    assert display.matchup("Iowa at Washington", None, "Iowa") == ("Iowa", "")
    assert display.kickoff("2026-10-10T19:30:00+00:00") == "Sat 3:30pm"
    assert display.kickoff("2026-10-10T16:00:00+00:00", date=True) == "Sat 10/10"
    assert display.kickoff(None) == "TBD"


def test_rows_are_listed_by_kickoff():
    rows = pd.DataFrame([{"game": "Texas at Oklahoma", "side": "home", "kickoff": "2026-10-10T19:30:00+00:00"},
                         {"game": "Ohio at Akron", "side": "away", "kickoff": None},
                         {"game": "Iowa at Washington", "side": "away", "kickoff": "2026-10-10T01:00:00+00:00"}],
                        index=[7, 3, 5])
    assert list(display.by_kickoff(rows).index) == [5, 7, 3]
