#!/usr/bin/env python
#
# Electrum - lightweight Bitcoin client
# Copyright (C) 2026 The Electrum Developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""A faithful, standalone port of LND's *bimodal* MissionControl estimator.

LND estimates the probability that a channel of capacity ``c`` can forward an
amount ``a`` by modelling the channel's local balance ``x`` as a *bimodal*
distribution -- liquidity tends to pile up at one end of the channel:

    P(x) ∝ exp(-x / s) + exp((x - c) / s)        (s = scale parameter)

The success probability for an amount is then ``P(x >= a)``, optionally
*conditioned* on observed liquidity bounds: a prior success at amount ``S`` tells
us ``x >= S`` (a lower bound), and a prior failure at amount ``F`` tells us
``x < F`` (an upper bound). Those observations decay back toward "no information"
over ``BimodalDecayTime``.

See LND ``routing/probability_bimodal.go``. The observation history lives in
:class:`MissionControlStore`, which is *independent* of Electrum's own
``LiquidityHintMgr`` and is fed by a dedicated ``htlc_route_result`` core hook.
"""

import math
import threading
import time
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from electrum.channel_db import ChannelDB
    from electrum.lnutil import ShortChannelID


# --- LND bimodal defaults (routing/probability_bimodal.go) -----------------
DEFAULT_BIMODAL_SCALE_MSAT = 300_000_000          # 300k sat: liquidity concentration
DEFAULT_BIMODAL_DECAY_TIME = 7 * 24 * 60 * 60.0   # 1 week (seconds): observation decay
# capacity used when neither a funding capacity nor an htlc_maximum is known
DEFAULT_CAPACITY_MSAT = 5_000_000 * 1000          # 5M sat
# probability floor/uncertainty fallback
MIN_PROBABILITY = 1e-5
FALLBACK_PROBABILITY = 0.6


class PairResult:
    """Most recent observed liquidity bounds for one directed channel."""
    __slots__ = ('success_amt', 'success_time', 'fail_amt', 'fail_time')

    def __init__(self):
        self.success_amt = 0          # largest amount observed to succeed (lower bound on balance)
        self.success_time = 0.0
        self.fail_amt = None          # smallest amount observed to fail (upper bound); None = unknown
        self.fail_time = 0.0


class MissionControlStore:
    """Independent observation history, keyed by directed channel.

    Fed by the ``htlc_route_result`` hook (a real payment outcome): a success
    raises the lower bound, a failure lowers the upper bound. Kept entirely
    separate from Electrum's ``LiquidityHintMgr``.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._results = {}  # type: Dict[Tuple[bytes, bool], PairResult]

    @staticmethod
    def _key(scid, node_from: bytes, node_to: bytes) -> Tuple[bytes, bool]:
        # direction convention matches Electrum's liquidity hints: node_from < node_to
        return (bytes(scid), node_from < node_to)

    def report_success(self, scid, node_from: bytes, node_to: bytes, amount_msat: int, *, now: float):
        with self._lock:
            r = self._results.setdefault(self._key(scid, node_from, node_to), PairResult())
            if amount_msat >= r.success_amt:
                r.success_amt = amount_msat
            r.success_time = now
            # sent more than the old failure bound -> that bound is stale
            if r.fail_amt is not None and amount_msat >= r.fail_amt:
                r.fail_amt = None

    def report_failure(self, scid, node_from: bytes, node_to: bytes, amount_msat: int, *, now: float):
        with self._lock:
            r = self._results.setdefault(self._key(scid, node_from, node_to), PairResult())
            r.fail_amt = amount_msat if r.fail_amt is None else min(r.fail_amt, amount_msat)
            r.fail_time = now
            # cannot also "succeed" at or above the new failure bound
            if r.success_amt >= amount_msat:
                r.success_amt = 0

    def get_bounds(self, scid, node_from: bytes, node_to: bytes, capacity_msat: float, *,
                   now: float, decay_time: float) -> Tuple[float, float]:
        """Return time-decayed ``(success_amount, fail_amount)`` bounds.

        With no/forgotten history this is ``(0, capacity)``. The success
        (lower) bound relaxes toward 0 and the failure (upper) bound relaxes
        toward capacity as observations age, exactly like LND.
        """
        with self._lock:
            r = self._results.get(self._key(scid, node_from, node_to))
        success_amt = 0.0
        fail_amt = float(capacity_msat)
        if r is not None:
            if r.success_time and r.success_amt:
                dt = max(0.0, now - r.success_time)
                success_amt = r.success_amt * math.exp(-dt / decay_time)
            if r.fail_amt is not None and r.fail_time:
                dt = max(0.0, now - r.fail_time)
                fail_amt = capacity_msat - (capacity_msat - r.fail_amt) * math.exp(-dt / decay_time)
        success_amt = max(0.0, min(success_amt, capacity_msat))
        fail_amt = max(0.0, min(fail_amt, capacity_msat))
        if fail_amt <= success_amt:
            # inconsistent after decay -> drop the (older) information
            success_amt, fail_amt = 0.0, float(capacity_msat)
        return success_amt, fail_amt


class BimodalEstimator:
    """LND-style bimodal success-probability estimator.

    Usable as a ``probability_func`` for :class:`LndPathFinder`: it is callable
    as ``estimator(start_node, end_node, scid, amount_msat) -> float``.

    :param channel_db: graph, for per-channel capacity lookup.
    :param store: optional :class:`MissionControlStore` for observed bounds.
        Without one, probabilities are the pure capacity-based prior.
    """

    def __init__(
            self,
            channel_db: 'ChannelDB',
            *,
            store: Optional[MissionControlStore] = None,
            scale_msat: int = DEFAULT_BIMODAL_SCALE_MSAT,
            decay_time: float = DEFAULT_BIMODAL_DECAY_TIME,
            default_capacity_msat: int = DEFAULT_CAPACITY_MSAT,
    ):
        self.channel_db = channel_db
        self.store = store
        self.scale_msat = float(scale_msat)
        self.decay_time = float(decay_time)
        self.default_capacity_msat = default_capacity_msat

    def _capacity_msat(self, scid, node_from: bytes) -> float:
        ci = self.channel_db.get_channel_info(scid)
        if ci is not None and ci.capacity_sat:
            return float(ci.capacity_sat) * 1000
        # gossip often lacks funding capacity; use htlc_maximum as a proxy
        try:
            policy = self.channel_db.get_policy_for_node(scid, node_from)
        except Exception:
            policy = None
        if policy is not None and policy.htlc_maximum_msat:
            return float(policy.htlc_maximum_msat)
        return float(self.default_capacity_msat)

    def _primitive(self, c: float, x: float) -> float:
        # antiderivative of exp((x - c)/s) + exp(-x/s); both exponents <= 0 here
        s = self.scale_msat
        return s * (math.exp((x - c) / s) - math.exp(-x / s))

    def probability(self, node_from: bytes, node_to: bytes, scid, amount_msat: int) -> float:
        if amount_msat <= 0:
            return 1.0
        capacity = self._capacity_msat(scid, node_from)
        if capacity <= 0:
            return FALLBACK_PROBABILITY
        a = float(amount_msat)
        if a > capacity:
            return MIN_PROBABILITY
        now = time.time()
        if self.store is not None:
            success_amt, fail_amt = self.store.get_bounds(
                scid, node_from, node_to, capacity, now=now, decay_time=self.decay_time)
        else:
            success_amt, fail_amt = 0.0, capacity
        if a <= success_amt:
            return 1.0
        if a >= fail_amt:
            return MIN_PROBABILITY
        p_fail = self._primitive(capacity, fail_amt)
        denom = p_fail - self._primitive(capacity, success_amt)
        if denom <= 0:
            return FALLBACK_PROBABILITY
        prob = (p_fail - self._primitive(capacity, a)) / denom
        return min(1.0, max(MIN_PROBABILITY, prob))

    def __call__(self, start_node: bytes, end_node: bytes, scid, amount_msat: int) -> float:
        return self.probability(start_node, end_node, scid, amount_msat)
