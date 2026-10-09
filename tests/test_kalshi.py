"""rappa.kalshi: fees, books, names, signing, and the read-only guard. No network."""

import base64
import urllib.error

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from rappa import kalshi
from rappa.kalshi import Book, Client, KalshiError, ReadOnlyError, parse_book


class FakeSend:
    """Records every request; answers from a queue of (status, payload), or 200 {} when it runs out."""

    def __init__(self, *answers):
        self.calls, self.answers = [], list(answers)

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        return self.answers.pop(0) if self.answers else (200, {})


def pem(key):
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


def test_fees_match_kalshi_worked_example():
    # docs getting_started/fee_rounding: a 5.5c buy has a model fee of $0.00363825
    assert kalshi.taker_fee(0.055) == pytest.approx(0.00363825)
    assert kalshi.taker_fee(0.5) == pytest.approx(0.0175)
    assert kalshi.maker_fee(0.5) == pytest.approx(0.004375)
    assert kalshi.maker_fee(0.8) == pytest.approx(0.0175 * 0.16)


def test_parse_book_turns_no_bids_into_yes_asks():
    payload = {"orderbook_fp": {"yes_dollars": [["0.4900", "315.00"], ["0.5000", "100.00"], ["0.5200", "4.32"]],
                                "no_dollars": [["0.0600", "206.00"], ["0.4700", "273.25"]]}}
    b = parse_book(payload)
    assert (b.bid, b.ask, b.bid_size, b.ask_size) == (0.52, 0.53, 4.32, 273.25)
    assert b.mid == pytest.approx(0.525) and b.width == pytest.approx(0.01)


def test_parse_book_empty_sides():
    b = parse_book({"orderbook_fp": {"yes_dollars": [], "no_dollars": None}})
    assert b.bid is None and b.ask is None and b.mid is None and b.width is None
    assert parse_book({}).mid is None
    # zero-size levels are not quotes
    assert parse_book({"orderbook_fp": {"yes_dollars": [["0.50", "0.00"]], "no_dollars": []}}).bid is None


def test_ticker_date():
    assert str(kalshi.ticker_date("KXNCAAFGAME-26OCT10OSUPSU")) == "2026-10-10"
    assert str(kalshi.ticker_date("KXNFLGAME-25SEP07PITNYJ")) == "2025-09-07"
    assert kalshi.ticker_date("garbage") is None


@pytest.mark.parametrize("a, b", [("Iowa St.", "Iowa State"), ("San Jose St.", "San José State"),
                                  ("North Carolina St.", "NC State"), ("Appalachian St.", "App State"),
                                  ("Louisiana-Monroe", "UL Monroe"), ("Southern Miss", "Southern Miss"),
                                  ("Hawai'i", "Hawai'i"),
                                  ("Texas A&M", "Texas A&M"), ("Miami (OH)", "Miami (OH)")])
def test_names_that_must_match(a, b):
    assert kalshi.similarity(a, b) == 1.0


def test_names_that_must_not_match_better_than_the_right_one():
    assert kalshi.similarity("North Carolina St.", "NC State") > kalshi.similarity("North Carolina", "NC State")
    assert kalshi.similarity("Miami (FL)", "Miami") > kalshi.similarity("Miami (FL)", "Miami (OH)")


@pytest.mark.parametrize("make", [lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048),
                                  ed25519.Ed25519PrivateKey.generate])
def test_signed_headers_verify(make, monkeypatch):
    key = make()
    send = FakeSend((200, {"balance_dollars": "1.00"}))
    client = Client("key-id", pem(key), send=send, sleep=lambda s: None)
    monkeypatch.setattr(kalshi.time, "time", lambda: 1700000000.123)
    client.get("/portfolio/balance", limit=5)
    method, url, headers, body = send.calls[0]
    assert url == "https://api.elections.kalshi.com/trade-api/v2/portfolio/balance?limit=5"
    assert headers["KALSHI-ACCESS-KEY"] == "key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000123"
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    # the signed message is timestamp + method + path, without the query string
    message = b"1700000000123GET/trade-api/v2/portfolio/balance"
    pub = key.public_key()
    if isinstance(key, rsa.RSAPrivateKey):
        pub.verify(sig, message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                   hashes.SHA256())
    else:
        pub.verify(sig, message)


def test_unsigned_client_sends_no_auth_headers():
    send = FakeSend()
    Client(send=send, sleep=lambda s: None).get("/markets", series_ticker="X")
    assert send.calls[0][2] == {}


def test_read_only_client_refuses_writes_without_sending():
    send = FakeSend()
    client = Client("id", pem(ed25519.Ed25519PrivateKey.generate()), send=send, sleep=lambda s: None)
    with pytest.raises(ReadOnlyError):
        client.post("/portfolio/events/orders", {"ticker": "X"})
    with pytest.raises(ReadOnlyError):
        client.delete("/portfolio/events/orders/abc")
    assert send.calls == []


def test_reads_retry_on_429_but_writes_never_retry():
    sleeps = []
    send = FakeSend((429, "slow"), (429, "slow"), (200, {"ok": 1}))
    client = Client(send=send, sleep=sleeps.append)
    assert client.get("/markets") == {"ok": 1}
    assert len(send.calls) == 3

    send = FakeSend((503, "busy"), (200, {"order_id": "o1"}))
    client = Client("id", pem(ed25519.Ed25519PrivateKey.generate()), read_only=False, send=send, sleep=lambda s: None)
    with pytest.raises(KalshiError) as err:
        client.post("/portfolio/events/orders", {"ticker": "X"})
    assert err.value.status == 503
    assert len(send.calls) == 1  # a write that may have landed is not resent


def test_retry_waits_double_up_to_the_last_try():
    sleeps = []
    send = FakeSend(*[(429, "slow")] * 6)
    client = Client(send=send, sleep=sleeps.append)
    with pytest.raises(KalshiError) as err:
        client.get("/historical/markets/X/candlesticks")
    assert err.value.status == 429 and len(send.calls) == 6
    # the fake sleep does not move the clock: each wait is the pause, which outlasts the spacing queued before it
    assert sleeps == pytest.approx([1, 2, 4, 8, 16], abs=0.05)


def test_a_429_holds_every_later_request_and_slows_the_rate():
    sleeps = []
    client = Client(send=FakeSend((429, "slow")), rate=4.0, sleep=sleeps.append)
    client.get("/markets")
    assert sleeps == pytest.approx([1.0], abs=0.05)
    assert client._interval == pytest.approx(0.25 * kalshi.SLOW_DOWN * kalshi.SPEED_UP)
    # Any thread's next request queues behind the pause, at the slower spacing.
    client.get("/markets")
    assert sleeps[1] == pytest.approx(1.0 + 0.25 * kalshi.SLOW_DOWN, abs=0.05)


def test_the_rate_recovers_and_is_capped():
    client = Client(send=FakeSend(*[(429, "slow"), (200, {})] * 10), rate=4.0, sleep=lambda s: None, retries=2)
    for _ in range(10):
        client.get("/markets")
    assert client._interval <= 0.25 * kalshi.MAX_SLOWDOWN
    for _ in range(500):
        client.get("/markets")
    assert client._interval == 0.25


def test_server_errors_wait_without_slowing_the_rate():
    sleeps = []
    client = Client(send=FakeSend((503, "busy"), (200, {"ok": 1})), rate=4.0, sleep=sleeps.append)
    assert client.get("/markets") == {"ok": 1}
    assert sleeps == pytest.approx([1.0], abs=0.05) and client._interval == 0.25


class FlakySend(FakeSend):
    """Raises a network error for the first `drops` requests."""

    def __init__(self, drops, *answers):
        super().__init__(*answers)
        self.drops = drops

    def __call__(self, method, url, headers, body):
        if self.drops:
            self.drops -= 1
            self.calls.append((method, url, headers, body))
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))
        return super().__call__(method, url, headers, body)


def test_network_errors_retry_reads_but_never_writes():
    send = FlakySend(2, (200, {"ok": 1}))
    assert Client(send=send, sleep=lambda s: None).get("/markets") == {"ok": 1}
    assert len(send.calls) == 3

    send = FlakySend(1, (200, {"order_id": "o1"}))
    client = Client("id", pem(ed25519.Ed25519PrivateKey.generate()), read_only=False, send=send, sleep=lambda s: None)
    with pytest.raises(urllib.error.URLError):
        client.post("/portfolio/events/orders", {"ticker": "X"})
    assert len(send.calls) == 1  # it may have landed: not resent

    with pytest.raises(urllib.error.URLError):
        Client(send=FlakySend(3), sleep=lambda s: None, retries=3).get("/markets")


def test_errors_carry_status():
    client = Client(send=FakeSend((404, {"error": "nope"})), sleep=lambda s: None)
    with pytest.raises(KalshiError) as err:
        client.get("/markets/X/orderbook")
    assert err.value.status == 404


def test_pages_follows_cursor():
    send = FakeSend((200, {"markets": [1, 2], "cursor": "c1"}), (200, {"markets": [3], "cursor": ""}))
    client = Client(send=send, sleep=lambda s: None)
    assert client.pages("/markets", "markets", limit=2) == [1, 2, 3]
    assert "cursor=c1" in send.calls[1][1]


def test_rate_limit_spaces_requests():
    sleeps = []
    client = Client(send=FakeSend(), rate=4.0, sleep=sleeps.append)
    for _ in range(3):
        client.get("/markets")
    # the fake sleep does not move the clock, so each wait is a further 1/rate behind the last
    assert sleeps == pytest.approx([0.25, 0.5], abs=0.01)
