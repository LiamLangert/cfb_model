"""Kalshi's college football team names made comparable with CFBD's ("Iowa St." and "Iowa State")."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# CFBD's short names for schools Kalshi spells out, after normalize's other steps.
ALIASES = {
    "nc state": "north carolina state",
    "app state": "appalachian state",
    "ul monroe": "louisiana monroe",
    "louisianamonroe": "louisiana monroe",
    "southern miss": "southern mississippi",
    "fiu": "florida international",
    "fau": "florida atlantic",
    "utsa": "texas san antonio",
    "utep": "texas el paso",
    "uconn": "connecticut",
    "umass": "massachusetts",
    "ucf": "central florida",
    "usf": "south florida",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "byu": "brigham young",
    "lsu": "louisiana state",
    "unlv": "nevada las vegas",
    "miami fl": "miami",
    "miami oh": "miami ohio",
    "hawaii": "hawaii",
}


def normalize(name):
    """A team name made comparable across Kalshi ("Iowa St.") and CFBD ("Iowa State")."""
    n = name.lower().replace("&", "and").replace("é", "e").replace("ʻ", "").replace("'", "")
    n = re.sub(r"\bst\.?$", "state", n)
    n = re.sub(r"\bst\.\s", "state ", n)
    n = n.replace("-", " ")
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return ALIASES.get(n, n)


def similarity(a, b):
    a, b = normalize(a), normalize(b)
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()
