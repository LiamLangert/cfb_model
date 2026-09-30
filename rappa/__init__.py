"""The league-agnostic parts of RAPPA, used by both cfb and nfl.

    model       the ridge fit that turns play-by-play into team ratings, and optionally passer ratings
    validation  scoring ratings on holdout plays, one team-game at a time
    games       turning ratings into game margins and grading them against the spread
"""
