import json
import os
import threading

from electrum.lnrouter import RouteEdge
from electrum.lnutil import ShortChannelID
from electrum.util import bfh

from electrum.plugins.lnd_pathfinder.lnd_log import (
    route_edge_to_dict, serialize_route, select_cheapest,
    build_route_compare_record, build_payment_outcome_record, append_jsonl,
)

from . import ElectrumTestCase


def _scid(n: int) -> ShortChannelID:
    return ShortChannelID(bfh(format(n, '016x')))


def _node(c: str) -> bytes:
    return b'\x02' + c.encode() * 32


def _edge(scid, start, end, *, fee_base, fee_ppm, cltv=10) -> RouteEdge:
    return RouteEdge(
        start_node=start, end_node=end, short_channel_id=scid,
        fee_base_msat=fee_base, fee_proportional_millionths=fee_ppm,
        cltv_delta=cltv, node_features=0)


def _route(*, fee_base, fee_ppm, hops=2):
    """A simple chain s -> n1 -> ... -> t with `hops` edges."""
    chars = ['s'] + [f'm{i}' for i in range(hops - 1)] + ['t']
    route = []
    for i in range(hops):
        route.append(_edge(_scid(i + 1), _node(chars[i]), _node(chars[i + 1]),
                           fee_base=fee_base, fee_ppm=fee_ppm))
    return route


class Test_LndPathfinderLog(ElectrumTestCase):
    TESTNET = True

    def test_route_edge_dict_is_whitelisted(self):
        e = _edge(_scid(7), _node('s'), _node('t'), fee_base=100, fee_ppm=150)
        d = route_edge_to_dict(e)
        self.assertEqual({"scid", "start", "end", "fee_base_msat", "fee_ppm", "cltv_delta"}, set(d.keys()))
        self.assertEqual(_node('s').hex(), d["start"])
        self.assertEqual(_node('t').hex(), d["end"])

    def test_serialize_route_success_and_nopath(self):
        route = _route(fee_base=100, fee_ppm=0, hops=2)
        s = serialize_route(route, amount_msat_for_dest=1_000_000, elapsed_ms=1.234)
        self.assertEqual("success", s["outcome"])
        self.assertEqual(2, s["num_hops"])
        # only the non-first hop charges a fee (first hop is our own channel)
        self.assertEqual(100, s["total_fee_msat"])
        self.assertEqual(2, len(s["route"]))
        self.assertEqual(1.234, s["elapsed_ms"])

        none = serialize_route(None, amount_msat_for_dest=1_000_000, error="boom")
        self.assertEqual("no_path", none["outcome"])
        self.assertEqual("boom", none["error"])
        self.assertNotIn("route", none)

    def test_select_cheapest_matrix(self):
        cheap = _route(fee_base=100, fee_ppm=0, hops=2)     # fee 100
        pricey = _route(fee_base=5000, fee_ppm=0, hops=2)   # fee 5000
        short = _route(fee_base=100, fee_ppm=0, hops=2)     # fee 100, 2 hops
        long_ = _route(fee_base=100, fee_ppm=0, hops=4)     # fee 300, 4 hops
        amt = 1_000_000

        self.assertEqual(("none", None, "both_failed"), select_cheapest(None, None, amount_msat_for_dest=amt))

        name, route, reason = select_cheapest(cheap, None, amount_msat_for_dest=amt)
        self.assertEqual(("native", "only_route"), (name, reason))
        self.assertIs(cheap, route)

        name, route, reason = select_cheapest(None, cheap, amount_msat_for_dest=amt)
        self.assertEqual(("lnd", "only_route"), (name, reason))

        # native pricey, lnd cheap -> lnd wins on fee
        name, route, reason = select_cheapest(pricey, cheap, amount_msat_for_dest=amt)
        self.assertEqual(("lnd", "lower_fee"), (name, reason))

        # native cheap, lnd pricey -> native wins on fee
        name, route, reason = select_cheapest(cheap, pricey, amount_msat_for_dest=amt)
        self.assertEqual(("native", "lower_fee"), (name, reason))

        # equal fee, different hops -> fewer hops wins (native is short here)
        name, route, reason = select_cheapest(short, long_, amount_msat_for_dest=amt)
        # short fee=100, long fee=300 -> actually lower_fee for short
        self.assertEqual(("native", "lower_fee"), (name, reason))

        # genuine tie: identical fee and hops -> native
        name, route, reason = select_cheapest(
            _route(fee_base=100, fee_ppm=0, hops=2),
            _route(fee_base=100, fee_ppm=0, hops=2),
            amount_msat_for_dest=amt)
        self.assertEqual(("native", "tie_native"), (name, reason))

    def test_select_cheapest_tie_break_hops(self):
        # same total fee, different hop counts -> fewer hops wins
        amt = 1_000_000
        two_hop = _route(fee_base=150, fee_ppm=0, hops=2)   # fee 150
        three_hop = _route(fee_base=75, fee_ppm=0, hops=3)  # fee 150 (2 charged hops * 75)
        self.assertEqual(150, serialize_route(two_hop, amount_msat_for_dest=amt)["total_fee_msat"])
        self.assertEqual(150, serialize_route(three_hop, amount_msat_for_dest=amt)["total_fee_msat"])
        name, route, reason = select_cheapest(three_hop, two_hop, amount_msat_for_dest=amt)
        self.assertEqual(("lnd", "fewer_hops"), (name, reason))

    def test_append_jsonl_roundtrip_and_no_secrets(self):
        path = os.path.join(self.electrum_path, "lnd_pathfinder.jsonl")
        lock = threading.Lock()
        rec1 = build_route_compare_record(
            ts="2026-06-17T00:00:00+00:00", network="mainnet",
            payment_hash="ab" * 32, source=_node('s').hex(), destination=_node('t').hex(),
            amount_msat=1_000_000, num_route_hints=0, gossip={"num_channels": 5},
            fee_settings={"fee_max_millionths": 10000, "fee_cutoff_msat": 10000},
            native=serialize_route(_route(fee_base=100, fee_ppm=0), amount_msat_for_dest=1_000_000),
            lnd=serialize_route(None, amount_msat_for_dest=1_000_000),
            chosen="native", chosen_reason="only_route")
        rec2 = build_payment_outcome_record(
            ts="2026-06-17T00:00:01+00:00", payment_hash="ab" * 32, success=True)
        append_jsonl(path, rec1, lock)
        append_jsonl(path, rec2, lock)

        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(2, len(lines))
        parsed = [json.loads(line) for line in lines]  # each line must be valid JSON
        self.assertEqual("route_compare", parsed[0]["event"])
        self.assertEqual("payment_outcome", parsed[1]["event"])
        self.assertEqual("ab" * 32, parsed[0]["payment_hash"])
        # no secret-ish material anywhere in the serialized log
        blob = "\n".join(lines).lower()
        for forbidden in ("secret", "preimage", "privkey", "private_key", "seed", "xprv"):
            self.assertNotIn(forbidden, blob)
