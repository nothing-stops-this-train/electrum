import time

from electrum.lnutil import ShortChannelID
from electrum.util import bfh

from electrum.plugins.lnd_pathfinder.mission_control import (
    MissionControlStore, BimodalEstimator, MIN_PROBABILITY,
    DEFAULT_BIMODAL_DECAY_TIME,
)

from . import ElectrumTestCase


def _scid(n: int) -> ShortChannelID:
    return ShortChannelID(bfh(format(n, '016x')))


def _node(c: str) -> bytes:
    return b'\x02' + c.encode() * 32


class FakeChannelInfo:
    def __init__(self, capacity_sat):
        self.capacity_sat = capacity_sat


class FakeChannelDB:
    """Minimal stand-in exposing only what BimodalEstimator reads."""
    def __init__(self, capacity_sat):
        self._capacity_sat = capacity_sat

    def get_channel_info(self, scid):
        return FakeChannelInfo(self._capacity_sat)

    def get_policy_for_node(self, scid, node_from):
        return None


class Test_BimodalProbability(ElectrumTestCase):
    TESTNET = True

    def test_zero_amount_is_certain(self):
        est = BimodalEstimator(FakeChannelDB(5_000_000))
        self.assertEqual(1.0, est(_node('a'), _node('b'), _scid(1), 0))

    def test_amount_above_capacity_is_near_zero(self):
        est = BimodalEstimator(FakeChannelDB(1_000))  # 1000 sat capacity
        p = est(_node('a'), _node('b'), _scid(1), 5_000_000)  # way over capacity
        self.assertEqual(MIN_PROBABILITY, p)

    def test_probability_monotonic_decreasing_in_amount(self):
        est = BimodalEstimator(FakeChannelDB(5_000_000))
        a, b, s = _node('a'), _node('b'), _scid(1)
        amounts = [10_000, 100_000, 1_000_000, 2_500_000, 4_500_000]  # sat... in msat below
        probs = [est(a, b, s, amt * 1000) for amt in amounts]
        for earlier, later in zip(probs, probs[1:]):
            self.assertGreaterEqual(earlier, later)
        # small amount over a large channel should be quite likely
        self.assertGreater(probs[0], 0.5)
        # near-capacity should be unlikely
        self.assertLess(probs[-1], 0.5)

    def test_store_success_bound_makes_small_amounts_certain(self):
        store = MissionControlStore()
        est = BimodalEstimator(FakeChannelDB(5_000_000), store=store)
        a, b, s = _node('a'), _node('b'), _scid(1)
        # observe a success at 1,000,000 msat -> anything below that is certain
        store.report_success(s, a, b, 1_000_000, now=time.time())
        self.assertEqual(1.0, est(a, b, s, 500_000))
        # at the exact (decaying) bound, probability is ~1
        self.assertGreater(est(a, b, s, 1_000_000), 0.999)

    def test_store_failure_bound_makes_large_amounts_near_zero(self):
        store = MissionControlStore()
        est = BimodalEstimator(FakeChannelDB(5_000_000), store=store)
        a, b, s = _node('a'), _node('b'), _scid(1)
        # observe a failure at 1,000,000 msat -> anything >= that is ~impossible
        store.report_failure(s, a, b, 1_000_000, now=time.time())
        self.assertEqual(MIN_PROBABILITY, est(a, b, s, 1_000_000))
        self.assertEqual(MIN_PROBABILITY, est(a, b, s, 2_000_000))
        # but a smaller amount is still plausible
        self.assertGreater(est(a, b, s, 100_000), MIN_PROBABILITY)

    def test_failure_bound_decays_over_time(self):
        store = MissionControlStore()
        cap = 5_000_000_000.0  # msat
        # fresh failure at 1,000,000 msat
        store.report_failure(_scid(1), _node('a'), _node('b'), 1_000_000, now=1000.0)
        # immediately: fail bound ~ the failed amount
        succ0, fail0 = store.get_bounds(_scid(1), _node('a'), _node('b'), cap,
                                        now=1000.0, decay_time=DEFAULT_BIMODAL_DECAY_TIME)
        self.assertAlmostEqual(1_000_000, fail0, delta=1.0)
        # many half-lives later: fail bound relaxes back toward capacity
        later = 1000.0 + 100 * DEFAULT_BIMODAL_DECAY_TIME
        succ1, fail1 = store.get_bounds(_scid(1), _node('a'), _node('b'), cap,
                                        now=later, decay_time=DEFAULT_BIMODAL_DECAY_TIME)
        self.assertGreater(fail1, fail0)
        self.assertAlmostEqual(cap, fail1, delta=cap * 1e-3)

    def test_success_then_failure_updates_bounds(self):
        store = MissionControlStore()
        a, b, s = _node('a'), _node('b'), _scid(1)
        cap = 5_000_000_000.0
        store.report_success(s, a, b, 2_000_000, now=1000.0)
        store.report_failure(s, a, b, 1_000_000, now=1001.0)
        # failure below the earlier success amount must clear the stale success bound
        succ, fail = store.get_bounds(s, a, b, cap, now=1001.0, decay_time=DEFAULT_BIMODAL_DECAY_TIME)
        self.assertEqual(0.0, succ)
        self.assertAlmostEqual(1_000_000, fail, delta=1.0)
