"""Terminal tables for betting.orders and betting.daily: kalshi.display's, with RAPPA and kickoffs.

Every table fits the terminal, cutting text columns widest first; prices are in cents, probabilities in
percent, kickoffs in Eastern time (see kalshi.display).
"""

from __future__ import annotations

from functools import partial

from betting.orders import LEAGUE
from kalshi import display as shared
from kalshi.display import (DEFAULT_WIDTH, ET, GAP, JUST_PLACED, MIN_TEXT, SHOW_SETTLED, book, cents,  # noqa: F401
                            count, is_take, is_topup, matchup, money, pct, signed_cents, skipped_lines, table,
                            term_width, wrap)

kickoff = shared.when
side_prob = partial(shared.side_prob, league=LEAGUE)
by_kickoff = partial(shared.by_start, league=LEAGUE)
orders_table = partial(shared.orders_table, league=LEAGUE)
reviews_table = partial(shared.reviews_table, league=LEAGUE)
shown_orders = partial(shared.shown_orders, league=LEAGUE)
track_table = partial(shared.track_table, league=LEAGUE)
