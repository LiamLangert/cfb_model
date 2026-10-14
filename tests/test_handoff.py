"""cfb.handoff: RAPPA's week as ../kalshi reads it, and when the week's predictions are built. No network."""

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from cfb import handoff
from kalshi import handoff as kalshi_handoff


def part(gid, home, away, wp, day="2026-10-10", time="15:30", score=None):
    return {"game_id": gid, "home_team": home, "away_team": away, "rappa_wp": wp, "gameday": day,
            "gametime": time, "home_score": score}


def week(tmp_path, rows, season=2026, number=6):
    out = tmp_path / "cfb"
    (out / str(season)).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / str(season) / f"parts_wk{number:02d}.csv", index=False)
    return out


def test_the_handoff_has_the_weeks_unplayed_games_and_all_its_days(tmp_path):
    out = week(tmp_path, [part(401, "Ohio State", "Penn State", 0.72, day="2026-10-10", time="15:30"),
                          part(402, "Georgia", "Alabama", 0.52, day="2026-10-10", time=None),        # time TBD
                          part(403, "Troy", "Southern Miss", 0.61, day="2026-10-06", time="19:00", score=24)])
    made = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    path = handoff.hand_over(2026, 6, out, publish=False, data=tmp_path / "kalshi", made_at=made)
    body = json.loads(path.read_text())
    assert (body["strategy"], body["model"], body["season"], body["week"]) == ("cfb", "RAPPA", 2026, 6)
    assert (body["first_day"], body["last_day"]) == ("2026-10-06", "2026-10-10")  # the played game's day counts
    assert body["made_at"] == "2026-10-08T12:00:00+00:00"
    osu, uga = body["games"]                                                         # the scored game is left out
    assert osu == {"game_id": "401", "day": "2026-10-10", "start": "2026-10-10T19:30:00+00:00",
                   "home": "Ohio State", "away": "Penn State", "p_home": 0.72, "home_names": ["Ohio State"],
                   "away_names": ["Penn State"], "home_abbr": None, "away_abbr": None}
    assert uga["start"] is None and uga["game_id"] == "402"

    meta, games = kalshi_handoff.read("cfb", made + timedelta(days=1), path=path)  # as ../kalshi reads it
    assert list(games.game_id) == ["401", "402"] and meta["week"] == 6


def test_a_probability_outside_0_and_1_is_refused(tmp_path):
    out = week(tmp_path, [part(401, "Ohio State", "Penn State", 1.0)])
    with pytest.raises(ValueError, match="not a probability"):
        handoff.hand_over(2026, 6, out, publish=False, data=tmp_path / "kalshi")


def write_games(out, kickoffs):
    pd.DataFrame([{"season": 2026, "week": w, "season_type": "regular", "start_date": k.isoformat(),
                   "completed": True, "home_classification": "fbs", "away_classification": "fbs"}
                  for w, k in kickoffs]).to_csv(out / "2026" / "games.csv", index=False)


def test_next_weeks_predictions_wait_for_the_pull(tmp_path, monkeypatch):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    built = []
    monkeypatch.setattr("cfb.week_parts.main", lambda argv: built.append(argv))
    future = datetime.now(timezone.utc) + timedelta(days=1)
    write_games(out, [(6, future - timedelta(days=3)), (6, future)])  # pulled before week 6 was over
    assert handoff.ensure_parts(2026, 7, out) == "waiting" and built == []
    assert handoff.main(["2026", "7", "--out", str(out), "--no-publish"]) == 0  # nothing handed over

    write_games(out, [(6, datetime.now(timezone.utc) - timedelta(days=2))])  # pulled after
    assert handoff.ensure_parts(2026, 7, out) == "built" and built == [["2026", "7", "--out", str(out)]]


def test_stale_predictions_are_rebuilt(tmp_path, monkeypatch):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    built = []
    monkeypatch.setattr("cfb.week_parts.main", lambda argv: built.append(argv))
    parts = out / "2026" / "parts_wk06.csv"
    parts.write_text("x\n")
    write_games(out, [(5, datetime(2026, 10, 3, tzinfo=timezone.utc))])
    os.utime(parts, (1, 1))  # older than games.csv
    assert handoff.ensure_parts(2026, 6, out) == "built"
    os.utime(parts, None)
    os.utime(out / "2026" / "games.csv", (1, 1))
    assert handoff.ensure_parts(2026, 6, out) == "ok" and len(built) == 1
