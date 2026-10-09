"""Kalshi's trade API, for either league: signed requests, order books, fees and team names.

Public market data needs no key. Account calls (balance, orders, positions)
and orders are signed with the key in .env:

    KALSHI_ID           the API key's id
    KALSHI_PRIVATE_KEY  its private key, PEM. Kalshi documents RSA keys signed
                        with PSS; an Ed25519 key, signed as is, works too.

Each signed request carries KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP (ms)
and KALSHI-ACCESS-SIGNATURE, a signature of timestamp + method + path, the
path from /trade-api/v2 on and without its query string.

A Client is read-only unless made with read_only=False: until then post and
delete refuse to send anything, so nothing that only looks at the market can
place or cancel an order by mistake. Signing needs the cryptography package,
which the project does not depend on: run with `uv run --with cryptography`.

Requests are spaced at the Client's rate, shared by every thread using it.
Unsigned reads are held to about 4 a second (measured on /historical candles
in October 2026); signed ones ran at 15 a second with no 429s, so a bulk pull
should use Client.from_env with a higher rate.

A read Kalshi throttles (429), or that fails on the server (5xx) or the
network, is retried: every thread waits, RETRY_WAIT seconds doubling each try
up to RETRY_CAP, and a 429 also slows every later request (SLOW_DOWN, at most
MAX_SLOWDOWN times the set spacing) until successes bring it back (SPEED_UP).
Writes are never retried: one that timed out or was throttled may have landed.

Fees, as Kalshi charges them on game markets (fee type
quadratic_with_maker_fees, multiplier 1), per contract at price P in dollars:

    taker   0.07 * P * (1 - P)     confirmed by the worked example in Kalshi's
                                   docs (getting_started/fee_rounding)
    maker   0.0175 * P * (1 - P)   from third-party guides only; no API returns it

Each fill's fee is rounded up to a millionth of a dollar and the balance then
to a cent, with the excess rebated across the order's fills.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

PROD = "https://api.elections.kalshi.com"
DEMO = "https://demo-api.kalshi.co"
PREFIX = "/trade-api/v2"

TAKER_RATE = 0.07
MAKER_RATE = 0.0175

# Prices move in whole cents on game markets (price_level_structure linear_cent).
TICK = 0.01

# Retrying a read: the pause before the first retry, in seconds, doubling each
# try up to RETRY_CAP. After a 429 the spacing between requests grows by
# SLOW_DOWN, to at most MAX_SLOWDOWN times the Client's, and each success
# shrinks it by SPEED_UP back toward the Client's.
RETRY_WAIT = 1.0
RETRY_CAP = 30.0
SLOW_DOWN = 1.5
MAX_SLOWDOWN = 4.0
SPEED_UP = 0.99

MONTHS = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}


class KalshiError(RuntimeError):
    """A request Kalshi answered with an error, with its HTTP status and body."""

    def __init__(self, status, body, method, path):
        super().__init__(f"{method} {path}: HTTP {status}: {str(body)[:300]}")
        self.status, self.body = status, body


class ReadOnlyError(RuntimeError):
    """A write attempted on a read-only Client."""


def taker_fee(price):
    return TAKER_RATE * price * (1 - price)


def maker_fee(price):
    return MAKER_RATE * price * (1 - price)


def load_env(path=".env"):
    """KALSHI_ID and KALSHI_PRIVATE_KEY from a .env file, or None for each that is missing."""
    from dotenv import dotenv_values
    env = dotenv_values(path) if Path(path).exists() else {}
    key_id = (env.get("KALSHI_ID") or "").strip() or None
    pem = (env.get("KALSHI_PRIVATE_KEY") or "").strip().replace("\\n", "\n") or None
    return key_id, pem


def urllib_send(method, url, headers, body):
    """POST/GET/DELETE with the standard library; returns (status, parsed JSON or text)."""
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "rappa", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")


class Client:
    """Kalshi's REST API, rate-limited, with signing when a key is given.

    send is the transport, (method, url, headers, body) -> (status, payload);
    tests swap in a fake.
    """

    def __init__(self, key_id=None, private_key_pem=None, host=PROD, read_only=True, rate=8.0, send=urllib_send,
                 sleep=time.sleep, retries=6):
        self.key_id, self.host, self.read_only, self.retries = key_id, host, read_only, retries
        self._key = None if private_key_pem is None else _load_key(private_key_pem)
        self._send, self._sleep = send, sleep
        # _spacing is the set gap between requests; _interval the one in use, wider after a 429.
        self._spacing = 1 / rate
        self._interval, self._next, self._lock = self._spacing, 0.0, threading.Lock()

    @classmethod
    def from_env(cls, path=".env", **kwargs):
        key_id, pem = load_env(path)
        return cls(key_id, pem, **kwargs)

    @property
    def signed(self):
        return self._key is not None and self.key_id is not None

    def _headers(self, method, path):
        if not self.signed:
            return {}
        ts = str(int(time.time() * 1000))
        sig = sign(self._key, ts + method + PREFIX + path)
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": sig}

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            self._sleep(wait)

    def _back_off(self, attempt, throttled):
        """After a read that failed: every thread's next request waits, and after a 429 all go out more slowly."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + min(RETRY_WAIT * 2 ** attempt, RETRY_CAP))
            if throttled:
                self._interval = min(self._interval * SLOW_DOWN, self._spacing * MAX_SLOWDOWN)

    def _recovered(self):
        with self._lock:
            self._interval = max(self._spacing, self._interval * SPEED_UP)

    def request(self, method, path, params=None, body=None, retries=None):
        if method != "GET" and self.read_only:
            raise ReadOnlyError(f"{method} {path} refused: this client is read-only")
        url = self.host + PREFIX + path + ("?" + urllib.parse.urlencode(params) if params else "")
        retries = self.retries if retries is None else retries
        for attempt in range(retries):
            self._throttle()
            last = attempt == retries - 1
            # Only reads are retried: a write that timed out or was throttled may still have landed.
            try:
                status, payload = self._send(method, url, self._headers(method, path), body)
            except OSError:  # the network: urllib's URLError, a timeout
                if method != "GET" or last:
                    raise
                self._back_off(attempt, throttled=False)
                continue
            if method == "GET" and status in (429, 500, 502, 503, 504) and not last:
                self._back_off(attempt, throttled=status == 429)
                continue
            if status >= 400:
                raise KalshiError(status, payload, method, path)
            self._recovered()
            return payload
        raise KalshiError(status, payload, method, path)

    def get(self, path, **params):
        return self.request("GET", path, params or None)

    def post(self, path, body):
        return self.request("POST", path, body=body)

    def delete(self, path, **params):
        return self.request("DELETE", path, params or None)

    def pages(self, path, key, **params):
        """Every item under `key` across a cursor-paged listing."""
        items, cursor = [], None
        while True:
            page = self.get(path, **params, **({"cursor": cursor} if cursor else {}))
            items += page.get(key) or []
            cursor = page.get("cursor")
            if not cursor or not page.get(key):
                return items


def _load_key(pem):
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_private_key(pem.encode(), password=None)


def sign(key, message):
    """Base64 signature of message: RSA-PSS with SHA-256 as Kalshi documents, or Ed25519."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    data = message.encode()
    if isinstance(key, rsa.RSAPrivateKey):
        raw = key.sign(data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                       hashes.SHA256())
    elif isinstance(key, ed25519.Ed25519PrivateKey):
        raw = key.sign(data)
    else:
        raise TypeError(f"unsupported key type {type(key).__name__}")
    return base64.b64encode(raw).decode()


@dataclass(frozen=True)
class Book:
    """The top of one market's book, from the YES side, in dollars. None where a side is empty."""
    bid: float | None
    ask: float | None
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self):
        return None if self.bid is None or self.ask is None else (self.bid + self.ask) / 2

    @property
    def width(self):
        return None if self.bid is None or self.ask is None else self.ask - self.bid


def parse_book(payload):
    """A Book from GET /markets/{ticker}/orderbook.

    Kalshi lists only bids, YES and NO, each as [price, count] from worst to
    best. A NO bid at x is a YES ask at 1 - x.
    """
    book = (payload or {}).get("orderbook_fp") or {}
    yes = [(float(p), float(n)) for p, n in book.get("yes_dollars") or [] if float(n) > 0]
    no = [(float(p), float(n)) for p, n in book.get("no_dollars") or [] if float(n) > 0]
    bid = max(yes) if yes else None
    no_bid = max(no) if no else None
    return Book(bid=None if bid is None else round(bid[0], 4),
                ask=None if no_bid is None else round(1 - no_bid[0], 4),
                bid_size=0.0 if bid is None else bid[1], ask_size=0.0 if no_bid is None else no_bid[1])


def ticker_date(event_ticker):
    """The game's date in an event ticker such as KXNCAAFGAME-26OCT10OSUPSU."""
    m = re.match(r"^[A-Z0-9]+-(\d\d)([A-Z]{3})(\d\d)", event_ticker)
    if not m or m.group(2) not in MONTHS:
        return None
    return date(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


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
