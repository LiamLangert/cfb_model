"""The league-agnostic parts of RAPPA, used by both cfb and nfl.

    model       the ridge fit that turns play-by-play into team ratings, and optionally passer ratings
    validation  scoring ratings on holdout plays, one team-game at a time
    games       turning ratings into game margins and grading them against the spread
    special_teams  punt and kickoff ratings in points a game: each kick valued by expected points, rated by ridge
    outside     special teams and penalties: every point outside the play ratings, per game, rated by team
    expected_points  expected points by down, distance and yards to goal, and each snap's value in them
"""
