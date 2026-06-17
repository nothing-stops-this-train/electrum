# Copyright (C) 2026 The Electrum developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""Offline benchmark for Electrum's Lightning pathfinding.

This module measures how well :class:`~electrum.lnrouter.LNPathFinder` charts
routes over a *static* gossip graph, *without sending any payment*. It exists to
reproduce and quantify the class of failures reported in
https://github.com/spesmilo/electrum/issues/10443 (payments above ~10k sat fail
with "insufficient fee", and routes are much longer than necessary).

The module is intentionally free of any network / asyncio dependencies: it
operates on an already-populated :class:`~electrum.channel_db.ChannelDB` and a
:class:`~electrum.lnrouter.LNPathFinder`. Loading a real gossip snapshot into a
``ChannelDB`` is done by the runner script in
``contrib/pathfinding_benchmark/``; the synthetic-graph unit tests construct a
``ChannelDB`` in memory. This separation keeps the measurement logic fully
deterministic and testable in CI.

Metrics reported per payment-amount bucket:
  * ``no_route_pct``    -- attempts where find_route() returned nothing
  * ``over_budget_pct`` -- a route was found but its fee exceeds the budget
  * ``fail_pct``        -- no_route + over_budget (i.e. payment could not proceed)
  * ``mean/median fee_rate`` -- routing fee as a fraction of the amount
  * ``mean num_hops`` and ``mean/median excess_hops`` -- how much longer the
    chosen route is than the fewest-hops *feasible* path (the "ideal route"),
    which is the symptom from the bug report.

The "ideal route" baseline here is the minimum number of feasible hops, computed
by an independent BFS over the same graph (see :func:`min_feasible_hops`). It
deliberately does *not* compute a minimum-*fee* route: defining a provably
correct min-fee oracle means reimplementing fee-compounding pathfinding, which
would just be a second (possibly buggy) router. Fee quality is instead captured
by the absolute fee-rate metrics and the over-budget flag. A min-fee oracle
could be added later if a fee-vs-ideal ratio is wanted.
"""

import random
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .lnutil import ShortChannelID, LnFeatures, PaymentFeeBudget, NUM_MAX_EDGES_IN_PAYMENT_PATH
from .lnrouter import is_route_within_budget, LNPaymentRoute, LiquidityHintMgr

if TYPE_CHECKING:
    from .channel_db import ChannelDB
    from .lnrouter import LNPathFinder
    from .simple_config import SimpleConfig


# cltv per-edge cap used by LNPathFinder._edge_cost; mirrored here so the
# feasibility check used for the ideal-route oracle matches the real router.
MAX_EDGE_CLTV_DELTA = 14 * 144

# Wall-clock budget after which the real wallet gives up on a gossip payment.
# Mirrors LNWallet.PAYMENT_TIMEOUT (electrum/lnworker.py): in the default
# non-trampoline path, pay_to_node retries find_route -> send -> await-failure
# until this many seconds elapse since the payment started, *not* until a fixed
# number of attempts (that count form is only used for trampoline / unit tests).
# The liquidity simulation below reproduces this time-boxed give-up.
DEFAULT_PAYMENT_TIMEOUT_SEC = 120.0


class NodeTier(Enum):
    """Connectivity tier of a node, by channel degree."""
    WELL = "well"
    FAIRLY = "fairly"
    POORLY = "poorly"


@dataclass
class BenchmarkConfig:
    # payment amounts to test, in sat. The default set spans the range from the
    # bug report: small payments work, large ones (>=10k) reportedly fail.
    amounts_sat: Sequence[int] = (100, 1_000, 10_000, 100_000)
    # number of (well-connected) public nodes used as the paying node
    num_sources: int = 3
    # number of destinations sampled per connectivity tier
    dests_per_tier: int = 50
    # fee budget, matching the bug reporter's "max 5%" setup. Expressed as
    # millionths (50_000 == 5%). A floor (fee_cutoff_msat) applies to tiny
    # payments, mirroring SimpleConfig.LIGHTNING_PAYMENT_FEE_CUTOFF_MSAT.
    max_fee_millionths: int = 50_000
    fee_cutoff_msat: int = 1_000
    # RNG seed; fixed for reproducible before/after comparisons.
    seed: int = 0
    # degree percentiles delimiting the tiers (computed over the snapshot, so
    # we don't hardcode magic absolute channel counts that age with the network)
    well_connected_percentile: float = 90.0
    poorly_connected_percentile: float = 50.0


@dataclass
class RouteResult:
    amount_sat: int
    source: bytes
    dest: bytes
    dest_tier: NodeTier
    found: bool
    within_budget: bool
    num_hops: Optional[int] = None
    ideal_hops: Optional[int] = None
    excess_hops: Optional[int] = None
    fee_msat: Optional[int] = None
    fee_rate: Optional[float] = None  # fee_msat / amount_msat

    @property
    def usable(self) -> bool:
        """True iff the payment could actually proceed: a route was found and
        it is within the fee budget."""
        return self.found and self.within_budget


@dataclass
class BucketSummary:
    amount_sat: int
    attempts: int
    no_route: int
    over_budget: int
    success: int
    no_route_pct: float
    over_budget_pct: float
    fail_pct: float
    success_pct: float
    mean_fee_rate: Optional[float]
    median_fee_rate: Optional[float]
    mean_num_hops: Optional[float]
    mean_excess_hops: Optional[float]
    median_excess_hops: Optional[float]


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def node_degrees(channel_db: 'ChannelDB') -> Dict[bytes, int]:
    """Map every node that is a channel endpoint to its channel count."""
    # _channels_for_node is the endpoint index built by ChannelDB.load_data();
    # it covers nodes even when we lack their node_announcement.
    return {node_id: len(scids) for node_id, scids in channel_db._channels_for_node.items() if scids}


def _percentile(sorted_values: Sequence[int], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def classify_nodes(
        channel_db: 'ChannelDB',
        cfg: BenchmarkConfig,
) -> Tuple[Dict[bytes, NodeTier], Dict[str, float]]:
    """Assign each node a :class:`NodeTier` based on degree percentiles.

    Tiering is by channel degree, not capacity: in a gossip-only snapshot
    Electrum does not SPV-verify channels, so ``capacity_sat`` is ``None`` for
    essentially every channel and cannot be used.
    """
    degrees = node_degrees(channel_db)
    sorted_degrees = sorted(degrees.values())
    well_threshold = _percentile(sorted_degrees, cfg.well_connected_percentile)
    poor_threshold = _percentile(sorted_degrees, cfg.poorly_connected_percentile)
    tiers: Dict[bytes, NodeTier] = {}
    for node_id, deg in degrees.items():
        if deg >= well_threshold:
            tiers[node_id] = NodeTier.WELL
        elif deg <= poor_threshold:
            tiers[node_id] = NodeTier.POORLY
        else:
            tiers[node_id] = NodeTier.FAIRLY
    thresholds = {"well_threshold": well_threshold, "poor_threshold": poor_threshold}
    return tiers, thresholds


def select_sources(
        channel_db: 'ChannelDB',
        cfg: BenchmarkConfig,
        tiers: Dict[bytes, NodeTier],
) -> List[bytes]:
    """Pick the ``num_sources`` best-connected public nodes as paying nodes.

    Deterministic: sorted by (degree desc, node_id) so re-runs are identical.
    """
    degrees = node_degrees(channel_db)
    well = [n for n, t in tiers.items() if t is NodeTier.WELL]
    well.sort(key=lambda n: (-degrees[n], n))
    return well[:cfg.num_sources]


def select_destinations(
        cfg: BenchmarkConfig,
        tiers: Dict[bytes, NodeTier],
        *,
        exclude: Sequence[bytes],
        rng,
) -> Dict[NodeTier, List[bytes]]:
    """Deterministically sample ``dests_per_tier`` nodes from each tier."""
    exclude_set = set(exclude)
    by_tier: Dict[NodeTier, List[bytes]] = {t: [] for t in NodeTier}
    for node_id, tier in tiers.items():
        if node_id not in exclude_set:
            by_tier[tier].append(node_id)
    chosen: Dict[NodeTier, List[bytes]] = {}
    for tier, candidates in by_tier.items():
        candidates.sort()  # stable order before sampling, for reproducibility
        k = min(cfg.dests_per_tier, len(candidates))
        chosen[tier] = rng.sample(candidates, k) if k else []
    return chosen


# ---------------------------------------------------------------------------
# ideal-route oracle: minimum number of *feasible* hops (independent of cost)
# ---------------------------------------------------------------------------

def edge_is_usable(
        channel_db: 'ChannelDB',
        *,
        short_channel_id: ShortChannelID,
        start_node: bytes,
        end_node: bytes,
        amount_msat: int,
        now: int,
) -> bool:
    """Feasibility of forwarding ``amount_msat`` from ``start_node`` to
    ``end_node`` over a channel.

    Mirrors the non-cost feasibility checks in
    :meth:`LNPathFinder._edge_cost` so the BFS oracle only considers edges the
    real router would also consider eligible. It excludes cost, blacklist and
    liquidity hints (which are dynamic / empty on a fresh sync).
    """
    channel_info = channel_db.get_channel_info(short_channel_id)
    if channel_info is None:
        return False
    policy = channel_db.get_policy_for_node(short_channel_id, start_node, now=now)
    if policy is None:
        return False
    # channels that did not publish both policies often fail with temporary errors
    policy_backwards = channel_db.get_policy_for_node(short_channel_id, end_node, now=now)
    if policy_backwards is None:
        return False
    if policy.is_disabled():
        return False
    if amount_msat < policy.htlc_minimum_msat:
        return False
    if channel_info.capacity_sat is not None and amount_msat // 1000 > channel_info.capacity_sat:
        return False
    if policy.htlc_maximum_msat is not None and amount_msat > policy.htlc_maximum_msat:
        return False
    if policy.cltv_delta > MAX_EDGE_CLTV_DELTA:
        return False
    node_info = channel_db.get_node_info_for_node_id(end_node)
    if node_info is not None:
        if not LnFeatures(node_info.features).supports(LnFeatures.VAR_ONION_OPT):
            return False
    return True


def min_feasible_hops(
        channel_db: 'ChannelDB',
        *,
        source: bytes,
        dest: bytes,
        amount_msat: int,
        now: int,
        max_hops: int = NUM_MAX_EDGES_IN_PAYMENT_PATH,
) -> Optional[int]:
    """Fewest feasible hops from ``source`` to ``dest`` (BFS), or None if no
    feasible path of length <= ``max_hops`` exists.

    This is the "ideal route" baseline: it answers "how short *could* a usable
    route be?" so the benchmark can report how much longer the router's choice
    is. It is not amount-aware beyond per-edge feasibility (no balances exist in
    a static snapshot)."""
    if source == dest:
        return 0
    frontier = [source]
    visited = {source}
    depth = 0
    while frontier and depth < max_hops:
        depth += 1
        next_frontier = []
        for node in frontier:
            for scid in channel_db.get_channels_for_node(node):
                ci = channel_db.get_channel_info(scid)
                if ci is None:
                    continue
                other = ci.node2_id if ci.node1_id == node else ci.node1_id
                if other in visited:
                    continue
                if not edge_is_usable(
                        channel_db, short_channel_id=scid, start_node=node,
                        end_node=other, amount_msat=amount_msat, now=now):
                    continue
                if other == dest:
                    return depth
                visited.add(other)
                next_frontier.append(other)
        frontier = next_frontier
    return None


# ---------------------------------------------------------------------------
# per-route measurement and aggregation
# ---------------------------------------------------------------------------

def route_fee_msat(route: LNPaymentRoute, *, amount_msat_for_dest: int) -> int:
    """Total routing fee of ``route`` for delivering ``amount_msat_for_dest``.

    Computed by reverse traversal so fees compound correctly, mirroring
    :func:`electrum.lnrouter.is_route_within_budget`. The first hop (our own
    channel) charges us nothing, hence ``route[1:]``."""
    amt = amount_msat_for_dest
    for route_edge in reversed(route[1:]):
        amt += route_edge.fee_for_edge(amt)
    return amt - amount_msat_for_dest


def measure_route(
        *,
        path_finder: 'LNPathFinder',
        channel_db: 'ChannelDB',
        source: bytes,
        dest: bytes,
        dest_tier: NodeTier,
        amount_sat: int,
        budget: PaymentFeeBudget,
        now: int,
) -> RouteResult:
    amount_msat = amount_sat * 1000
    route = path_finder.find_route(
        nodeA=source,
        nodeB=dest,
        invoice_amount_msat=amount_msat,
    )
    if not route:
        return RouteResult(
            amount_sat=amount_sat, source=source, dest=dest, dest_tier=dest_tier,
            found=False, within_budget=False,
            ideal_hops=min_feasible_hops(
                channel_db, source=source, dest=dest, amount_msat=amount_msat, now=now),
        )
    fee_msat = route_fee_msat(route, amount_msat_for_dest=amount_msat)
    within_budget = is_route_within_budget(
        route, budget=budget, amount_msat_for_dest=amount_msat, cltv_delta_for_dest=0)
    ideal_hops = min_feasible_hops(
        channel_db, source=source, dest=dest, amount_msat=amount_msat, now=now)
    num_hops = len(route)
    excess = (num_hops - ideal_hops) if ideal_hops is not None else None
    return RouteResult(
        amount_sat=amount_sat, source=source, dest=dest, dest_tier=dest_tier,
        found=True, within_budget=within_budget,
        num_hops=num_hops, ideal_hops=ideal_hops, excess_hops=excess,
        fee_msat=fee_msat, fee_rate=fee_msat / amount_msat,
    )


def _budget_for_amount(cfg: BenchmarkConfig, amount_sat: int, config: 'SimpleConfig') -> PaymentFeeBudget:
    amount_msat = amount_sat * 1000
    max_fee_msat = max(amount_msat * cfg.max_fee_millionths // 1_000_000, cfg.fee_cutoff_msat)
    return PaymentFeeBudget.from_invoice_amount(
        invoice_amount_msat=amount_msat, config=config, max_fee_msat=max_fee_msat)


def run_benchmark(
        *,
        channel_db: 'ChannelDB',
        path_finder: 'LNPathFinder',
        config: 'SimpleConfig',
        cfg: Optional[BenchmarkConfig] = None,
        now: Optional[int] = None,
        progress=None,
) -> Tuple[List[RouteResult], List[BucketSummary]]:
    """Run the full pathfinding benchmark over an already-loaded gossip graph.

    ``progress`` is an optional callable(done, total) for reporting.
    Returns (per-attempt results, per-amount summaries).
    """
    if cfg is None:
        cfg = BenchmarkConfig()
    if now is None:
        now = int(time.time())
    rng = random.Random(cfg.seed)

    tiers, _thresholds = classify_nodes(channel_db, cfg)
    sources = select_sources(channel_db, cfg, tiers)
    if not sources:
        raise RuntimeError("no well-connected source nodes found in snapshot")
    dests_by_tier = select_destinations(cfg, tiers, exclude=sources, rng=rng)

    # pre-build the (dest, tier) work list, then iterate amounts x sources x dests
    dest_items: List[Tuple[bytes, NodeTier]] = [
        (d, tier) for tier, ds in dests_by_tier.items() for d in ds
    ]
    total = len(cfg.amounts_sat) * len(sources) * len(dest_items)
    results: List[RouteResult] = []
    done = 0
    for amount_sat in cfg.amounts_sat:
        budget = _budget_for_amount(cfg, amount_sat, config)
        for source in sources:
            for dest, tier in dest_items:
                if dest == source:
                    done += 1
                    continue
                results.append(measure_route(
                    path_finder=path_finder, channel_db=channel_db,
                    source=source, dest=dest, dest_tier=tier,
                    amount_sat=amount_sat, budget=budget, now=now))
                done += 1
                if progress is not None:
                    progress(done, total)
    return results, summarize(results)


def summarize(results: Sequence[RouteResult]) -> List[BucketSummary]:
    """Aggregate per-attempt results into one :class:`BucketSummary` per amount."""
    amounts = sorted({r.amount_sat for r in results})
    summaries: List[BucketSummary] = []
    for amount_sat in amounts:
        bucket = [r for r in results if r.amount_sat == amount_sat]
        attempts = len(bucket)
        no_route = sum(1 for r in bucket if not r.found)
        over_budget = sum(1 for r in bucket if r.found and not r.within_budget)
        success = sum(1 for r in bucket if r.usable)
        fee_rates = [r.fee_rate for r in bucket if r.found and r.fee_rate is not None]
        hops = [r.num_hops for r in bucket if r.found and r.num_hops is not None]
        excess = [r.excess_hops for r in bucket if r.found and r.excess_hops is not None]

        def pct(n: int) -> float:
            return 100.0 * n / attempts if attempts else 0.0

        summaries.append(BucketSummary(
            amount_sat=amount_sat,
            attempts=attempts,
            no_route=no_route,
            over_budget=over_budget,
            success=success,
            no_route_pct=pct(no_route),
            over_budget_pct=pct(over_budget),
            fail_pct=pct(no_route + over_budget),
            success_pct=pct(success),
            mean_fee_rate=statistics.mean(fee_rates) if fee_rates else None,
            median_fee_rate=statistics.median(fee_rates) if fee_rates else None,
            mean_num_hops=statistics.mean(hops) if hops else None,
            mean_excess_hops=statistics.mean(excess) if excess else None,
            median_excess_hops=statistics.median(excess) if excess else None,
        ))
    return summaries


# ---------------------------------------------------------------------------
# output formatting
# ---------------------------------------------------------------------------

RESULT_CSV_HEADER = [
    "amount_sat", "source", "dest", "dest_tier", "found", "within_budget",
    "num_hops", "ideal_hops", "excess_hops", "fee_msat", "fee_rate",
]


def result_to_csv_row(r: RouteResult) -> List:
    return [
        r.amount_sat, r.source.hex(), r.dest.hex(), r.dest_tier.value,
        int(r.found), int(r.within_budget), r.num_hops, r.ideal_hops,
        r.excess_hops, r.fee_msat,
        f"{r.fee_rate:.6f}" if r.fee_rate is not None else "",
    ]


SUMMARY_CSV_HEADER = [
    "amount_sat", "attempts", "no_route_pct", "over_budget_pct", "fail_pct",
    "success_pct", "mean_fee_rate", "median_fee_rate", "mean_num_hops",
    "mean_excess_hops", "median_excess_hops",
]


def summary_to_csv_row(s: BucketSummary) -> List:
    def f(x):
        return f"{x:.6f}" if x is not None else ""
    return [
        s.amount_sat, s.attempts, f"{s.no_route_pct:.2f}", f"{s.over_budget_pct:.2f}",
        f"{s.fail_pct:.2f}", f"{s.success_pct:.2f}", f(s.mean_fee_rate),
        f(s.median_fee_rate), f(s.mean_num_hops), f(s.mean_excess_hops),
        f(s.median_excess_hops),
    ]


def format_summary_table(summaries: Sequence[BucketSummary]) -> str:
    """Human-readable summary table for stdout."""
    def cell(x, fmt="{:.2f}"):
        return fmt.format(x) if x is not None else "-"
    header = (f"{'amount_sat':>11} {'attempts':>9} {'no_route%':>10} "
              f"{'over_bdgt%':>10} {'fail%':>7} {'success%':>9} "
              f"{'mean_fee%':>10} {'med_fee%':>9} {'mean_hops':>10} {'mean_excess':>12}")
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.amount_sat:>11} {s.attempts:>9} {s.no_route_pct:>10.2f} "
            f"{s.over_budget_pct:>10.2f} {s.fail_pct:>7.2f} {s.success_pct:>9.2f} "
            f"{cell(s.mean_fee_rate * 100 if s.mean_fee_rate is not None else None):>10} "
            f"{cell(s.median_fee_rate * 100 if s.median_fee_rate is not None else None):>9} "
            f"{cell(s.mean_num_hops):>10} {cell(s.mean_excess_hops):>12}")
    return "\n".join(lines)


# ===========================================================================
# liquidity-aware benchmark
# ---------------------------------------------------------------------------
# The static benchmark above measures a single find_route() over a snapshot
# that has no channel balances. A real payment instead drives a retry loop:
# find_route -> attempt -> on a per-hop liquidity shortfall, record a
# cannot_send liquidity hint (LNPathFinder.update_liquidity_hints) -> find_route
# again (now steered around the failing channel) -> ... That loop is where the
# issue #10443 symptoms (insufficient-fee / over-budget on retries) can actually
# appear, so we model it here.
#
# Give-up: the loop terminates exactly as LNWallet.pay_to_node does in the
# default gossip case (electrum/lnworker.py ~line 2054):
#   * no route at all                -> find_route() empty (NoPathFound). On the
#                                       first try this is "no_route"; later it
#                                       means hints pruned every remaining route.
#   * a liquid route over budget     -> terminal "over_budget" (real Electrum
#                                       raises FeeBudgetExceeded and does not
#                                       retry, as re-running find_route unchanged
#                                       returns the same cheapest route).
#   * liquidity shortfall            -> record a hint and retry, BUT charge the
#                                       failed HTLC's round-trip against a
#                                       wall-clock-style time budget. When the
#                                       accumulated time exceeds payment_timeout_sec
#                                       (mirroring the 120s PAYMENT_TIMEOUT) the
#                                       payment "times out" -- this is the real
#                                       give-up, and it is why a fixed attempt
#                                       count (the old model, which let every
#                                       payment retry until it succeeded) is
#                                       replaced by a simulated time budget.
# Modelling time rather than a flat count matters for #10443: a longer route
# burns the budget faster (the HTLC must travel further before failing), so
# larger payments -- which take longer routes -- realistically get fewer retries
# and give up sooner, instead of "succeeding eventually" like the count model did.
#
# Liquidity is synthetic (a static gossip snapshot has none): each channel gets
# a capacity from its htlc_maximum_msat (the best on-graph proxy; a fixed
# fallback where unset), then node1's share of that capacity is drawn from a
# Beta(b, b) prior and node2 holds the remainder. A hop can forward an amount
# iff the *sender's* side holds at least that much. Draws are seeded per-channel
# so the assignment is deterministic and independent of iteration order.
#
# The shape parameter b (LiquidityConfig.balance_beta) controls how realistic
# the split is:
#   * b == 1.0  -> Uniform(0, capacity): every split equally likely. This is the
#                  original, over-generous model -- both sides usually hold ~half
#                  the capacity, so small payments almost never hit a dry hop.
#   * b <  1.0  -> U-shaped (bimodal): probability mass piles up near 0 and near
#                  capacity, i.e. most channels are *depleted toward one end*.
#                  This matches the real network, where channels drain in the
#                  direction of net flow, and it is what makes a multi-hop route
#                  realistically likely to hit an empty sender side. b ~ 0.5
#                  (the arcsine distribution) is a reasonable starting point; it
#                  is the knob to *calibrate against measured mainnet success
#                  rates* (lower b => harsher => lower success%).
# The mean share stays 0.5 for any symmetric b, so b changes the *variance* (how
# lopsided channels are), not the average balance. Tiny payments still mostly
# succeed even at low b -- which is what happens in reality.
# ===========================================================================

# payment outcomes under the liquidity model
OUTCOME_SUCCESS = "success"
OUTCOME_NO_ROUTE = "no_route"                      # find_route() empty on the first try
OUTCOME_OVER_BUDGET = "over_budget"                # a liquid route exists but its fee exceeds budget
OUTCOME_LIQUIDITY_EXHAUSTED = "liquidity_exhausted"  # liquidity hints pruned every remaining route (real NoPathFound after retries)
OUTCOME_TIMED_OUT = "timed_out"                    # the payment_timeout_sec budget elapsed while still retrying (real PAYMENT_TIMEOUT give-up)
LIQUIDITY_OUTCOMES = (
    OUTCOME_SUCCESS, OUTCOME_NO_ROUTE, OUTCOME_OVER_BUDGET,
    OUTCOME_LIQUIDITY_EXHAUSTED, OUTCOME_TIMED_OUT)


@dataclass
class LiquidityConfig:
    # synthetic channel capacity (sat) used when htlc_maximum_msat is unavailable
    default_capacity_sat: int = 5_000_000
    # wall-clock budget per (source, dest, amount) before giving up, mirroring
    # LNWallet.PAYMENT_TIMEOUT. The retry loop accumulates a simulated round-trip
    # latency for each failed attempt and stops once this is exceeded.
    payment_timeout_sec: float = DEFAULT_PAYMENT_TIMEOUT_SEC
    # simulated one-way per-hop HTLC latency (seconds). A failed attempt costs
    # ~2 * hop_latency_sec * (hops to the failing hop), i.e. the time for the HTLC
    # to reach the erring hop and the failure to travel back. This is the SECOND
    # calibration knob (alongside balance_beta): raise it to model a slower
    # network / fewer retries-per-timeout, lower it for more. Must be > 0.
    hop_latency_sec: float = 1.0
    # hard backstop on the number of find_route attempts. Not the real give-up
    # gate (payment_timeout_sec is); only guards against non-termination if
    # hop_latency_sec is misconfigured to ~0. With the defaults the time budget
    # always trips long before this.
    max_attempts_backstop: int = 1_000
    # RNG seed for the hidden balance assignment; independent of BenchmarkConfig.seed
    # so the topology sampling and the liquidity draw can be varied separately.
    balance_seed: int = 0
    # shape of the Beta(b, b) prior for each channel's directional balance split.
    # 1.0 == Uniform (original, over-generous); < 1.0 == U-shaped/bimodal, i.e.
    # channels depleted toward one end (realistic). Lower => harsher => lower
    # success%. This is the knob to calibrate against measured mainnet success.
    balance_beta: float = 0.5


class LiquidityModel:
    """Hidden, directional per-channel balances + a feasibility check for a hop.

    See the module section header for the model. Balances are assigned lazily
    (only for channels actually probed) and cached; each channel's draw is seeded
    from (balance_seed, scid), so it is reproducible and order-independent.
    """

    def __init__(self, channel_db: 'ChannelDB', *, lcfg: LiquidityConfig, now: int):
        self._channel_db = channel_db
        self._lcfg = lcfg
        self._now = now
        self._cap_msat: Dict[ShortChannelID, int] = {}
        self._bal_node1_msat: Dict[ShortChannelID, int] = {}

    def _capacity_msat(self, scid: ShortChannelID, channel_info) -> int:
        """Capacity proxy = max htlc_maximum_msat over the channel's two policies,
        falling back to the configured default where neither policy sets it."""
        caps = []
        for node_id in (channel_info.node1_id, channel_info.node2_id):
            policy = self._channel_db.get_policy_for_node(scid, node_id, now=self._now)
            # ignore non-positive htlc_max (some policies publish 0): treating it
            # as the capacity would fabricate a permanently dead channel.
            if policy is not None and policy.htlc_maximum_msat:
                caps.append(policy.htlc_maximum_msat)
        if caps:
            return max(caps)
        return self._lcfg.default_capacity_sat * 1000

    def _ensure(self, scid: ShortChannelID, channel_info) -> None:
        if scid in self._cap_msat:
            return
        cap = self._capacity_msat(scid, channel_info)
        # stable per-channel seed, independent of probe order
        scid_int = int.from_bytes(bytes(scid), "big")
        rng = random.Random((self._lcfg.balance_seed * 1_000_003) ^ scid_int)
        # node1's share of capacity ~ Beta(b, b): b<1 is U-shaped (channels
        # depleted toward one end, realistic); b==1 reduces to Uniform.
        b = self._lcfg.balance_beta
        share = rng.betavariate(b, b) if b != 1.0 else rng.random()
        self._cap_msat[scid] = cap
        self._bal_node1_msat[scid] = int(cap * share)

    def available_msat(self, *, short_channel_id: ShortChannelID, start_node: bytes) -> Optional[int]:
        """Liquidity (msat) that ``start_node`` can forward over the channel, or
        None if the channel is unknown."""
        channel_info = self._channel_db.get_channel_info(short_channel_id)
        if channel_info is None:
            return None
        self._ensure(short_channel_id, channel_info)
        cap = self._cap_msat[short_channel_id]
        bal_node1 = self._bal_node1_msat[short_channel_id]
        return bal_node1 if start_node == channel_info.node1_id else cap - bal_node1

    def can_forward(self, *, short_channel_id: ShortChannelID, start_node: bytes, amount_msat: int) -> bool:
        avail = self.available_msat(short_channel_id=short_channel_id, start_node=start_node)
        if avail is None:
            return True  # unknown channel: don't fabricate a failure
        return amount_msat <= avail


@dataclass
class LiquidityRouteResult:
    amount_sat: int
    source: bytes
    dest: bytes
    dest_tier: NodeTier
    outcome: str
    attempts_used: int
    time_used_sec: float = 0.0  # simulated wall-clock spent retrying (see LiquidityConfig)
    ideal_hops: Optional[int] = None
    num_hops: Optional[int] = None
    excess_hops: Optional[int] = None
    fee_msat: Optional[int] = None
    fee_rate: Optional[float] = None

    @property
    def success(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS


@dataclass
class LiquidityBucketSummary:
    amount_sat: int
    attempts: int
    success: int
    no_route: int
    over_budget: int
    liquidity_exhausted: int
    timed_out: int
    success_pct: float
    no_route_pct: float
    over_budget_pct: float
    liquidity_exhausted_pct: float
    timed_out_pct: float
    mean_attempts: Optional[float]       # over successful payments
    median_attempts: Optional[float]
    mean_fee_rate: Optional[float]       # over successful payments
    mean_num_hops: Optional[float]
    mean_excess_hops: Optional[float]


def route_carried_amounts(route: LNPaymentRoute, amount_msat: int) -> List[int]:
    """Amount (msat) carried over each edge of ``route`` to deliver
    ``amount_msat`` to the destination, accounting for compounding downstream
    fees. ``carried[i]`` is what crosses ``route[i]``."""
    n = len(route)
    carried = [0] * n
    running = amount_msat
    for i in range(n - 1, -1, -1):
        carried[i] = running
        running += route[i].fee_for_edge(running)
    return carried


def _attempt_latency_sec(failing_idx: int, lcfg: LiquidityConfig) -> float:
    """Simulated wall-clock cost of one failed attempt.

    A liquidity shortfall surfaces in the real wallet as a failed HTLC: it had to
    travel ``failing_idx`` channels in to reach the erring hop, and the failure
    then travels back, i.e. a round-trip of ~``2 * failing_idx`` per-hop latencies.
    Accumulating this across retries is what reproduces the 120s PAYMENT_TIMEOUT
    give-up that a fixed attempt count cannot (longer routes cost more per try)."""
    return 2.0 * lcfg.hop_latency_sec * failing_idx


def simulate_payment_with_liquidity(
        *,
        path_finder: 'LNPathFinder',
        channel_db: 'ChannelDB',
        model: LiquidityModel,
        source: bytes,
        dest: bytes,
        dest_tier: NodeTier,
        amount_sat: int,
        budget: PaymentFeeBudget,
        lcfg: LiquidityConfig,
        now: int,
) -> LiquidityRouteResult:
    """Run the real find_route -> fail-on-liquidity -> update-hints -> retry loop
    against the hidden balances in ``model`` and report the outcome.

    The give-up condition mirrors LNWallet.pay_to_node in the default gossip case:
    each failed (liquidity-short) attempt charges a simulated round-trip latency
    against ``lcfg.payment_timeout_sec``; once that budget is exhausted the payment
    times out, exactly as the wallet gives up after PAYMENT_TIMEOUT seconds. A
    no-route or over-budget result is terminal (the wallet raises NoPathFound /
    FeeBudgetExceeded and does not keep retrying).

    The path_finder's liquidity hints and blacklist are reset first so each
    payment is measured independently (the real wallet keeps hints for an hour,
    but carrying them across unrelated payments here would make results depend on
    iteration order)."""
    amount_msat = amount_sat * 1000
    path_finder.liquidity_hints = LiquidityHintMgr()
    path_finder.clear_blacklist()
    ideal_hops = min_feasible_hops(
        channel_db, source=source, dest=dest, amount_msat=amount_msat, now=now)

    def _result(outcome: str, attempts: int, route: Optional[LNPaymentRoute], time_used_sec: float) -> LiquidityRouteResult:
        num_hops = len(route) if route else None
        excess = (num_hops - ideal_hops) if (route and ideal_hops is not None) else None
        fee_msat = route_fee_msat(route, amount_msat_for_dest=amount_msat) if route else None
        fee_rate = (fee_msat / amount_msat) if fee_msat is not None else None
        return LiquidityRouteResult(
            amount_sat=amount_sat, source=source, dest=dest, dest_tier=dest_tier,
            outcome=outcome, attempts_used=attempts, time_used_sec=time_used_sec,
            ideal_hops=ideal_hops, num_hops=num_hops, excess_hops=excess,
            fee_msat=fee_msat, fee_rate=fee_rate)

    attempts = 0
    sim_time_sec = 0.0
    while attempts < lcfg.max_attempts_backstop:
        route = path_finder.find_route(nodeA=source, nodeB=dest, invoice_amount_msat=amount_msat)
        attempts += 1
        if not route:
            # first try empty -> genuinely unreachable; later -> liquidity hints
            # have pruned every remaining route (the real NoPathFound).
            outcome = OUTCOME_NO_ROUTE if attempts == 1 else OUTCOME_LIQUIDITY_EXHAUSTED
            return _result(outcome, attempts, None, sim_time_sec)
        carried = route_carried_amounts(route, amount_msat)
        failing_idx = None
        # Skip the first hop: it leaves the source's own node, and LNPathFinder
        # treats that edge as cost-free (ignore_costs in _edge_cost), so a
        # liquidity hint on it is never consulted and the loop could not reroute
        # around it. The real wallet handles its own outgoing channel via its
        # known balance, not gossip hints; here we assume the source funds it.
        for i in range(1, len(route)):
            if not model.can_forward(
                    short_channel_id=route[i].short_channel_id,
                    start_node=route[i].start_node, amount_msat=carried[i]):
                failing_idx = i
                break
        if failing_idx is None:
            # liquidity OK along the whole route; budget is the remaining gate.
            within_budget = is_route_within_budget(
                route, budget=budget, amount_msat_for_dest=amount_msat, cltv_delta_for_dest=0)
            outcome = OUTCOME_SUCCESS if within_budget else OUTCOME_OVER_BUDGET
            return _result(outcome, attempts, route, sim_time_sec)
        # record the liquidity failure so the next find_route routes around it,
        # and charge the failed HTLC's round-trip against the time budget.
        path_finder.update_liquidity_hints(
            route, carried[failing_idx],
            failing_channel=ShortChannelID(route[failing_idx].short_channel_id))
        sim_time_sec += _attempt_latency_sec(failing_idx, lcfg)
        if sim_time_sec > lcfg.payment_timeout_sec:
            return _result(OUTCOME_TIMED_OUT, attempts, None, sim_time_sec)
    # backstop only (hop_latency_sec ~ 0); treated as a timeout-style give-up.
    return _result(OUTCOME_TIMED_OUT, attempts, None, sim_time_sec)


def run_benchmark_with_liquidity(
        *,
        channel_db: 'ChannelDB',
        path_finder: 'LNPathFinder',
        config: 'SimpleConfig',
        cfg: Optional[BenchmarkConfig] = None,
        lcfg: Optional[LiquidityConfig] = None,
        now: Optional[int] = None,
        progress=None,
) -> Tuple[List[LiquidityRouteResult], List[LiquidityBucketSummary]]:
    """Liquidity-aware counterpart to :func:`run_benchmark`: same topology
    sampling, but each (source, dest, amount) runs the full retry loop against
    synthetic hidden balances. Returns (per-attempt results, per-amount summaries)."""
    if cfg is None:
        cfg = BenchmarkConfig()
    if lcfg is None:
        lcfg = LiquidityConfig()
    if now is None:
        now = int(time.time())
    rng = random.Random(cfg.seed)

    model = LiquidityModel(channel_db, lcfg=lcfg, now=now)

    tiers, _thresholds = classify_nodes(channel_db, cfg)
    sources = select_sources(channel_db, cfg, tiers)
    if not sources:
        raise RuntimeError("no well-connected source nodes found in snapshot")
    dests_by_tier = select_destinations(cfg, tiers, exclude=sources, rng=rng)
    dest_items: List[Tuple[bytes, NodeTier]] = [
        (d, tier) for tier, ds in dests_by_tier.items() for d in ds
    ]
    total = len(cfg.amounts_sat) * len(sources) * len(dest_items)
    results: List[LiquidityRouteResult] = []
    done = 0
    for amount_sat in cfg.amounts_sat:
        budget = _budget_for_amount(cfg, amount_sat, config)
        for source in sources:
            for dest, tier in dest_items:
                if dest == source:
                    done += 1
                    continue
                results.append(simulate_payment_with_liquidity(
                    path_finder=path_finder, channel_db=channel_db, model=model,
                    source=source, dest=dest, dest_tier=tier,
                    amount_sat=amount_sat, budget=budget, lcfg=lcfg, now=now))
                done += 1
                if progress is not None:
                    progress(done, total)
    return results, summarize_liquidity(results)


def summarize_liquidity(results: Sequence[LiquidityRouteResult]) -> List[LiquidityBucketSummary]:
    """Aggregate liquidity results into one :class:`LiquidityBucketSummary` per amount."""
    amounts = sorted({r.amount_sat for r in results})
    summaries: List[LiquidityBucketSummary] = []
    for amount_sat in amounts:
        bucket = [r for r in results if r.amount_sat == amount_sat]
        attempts = len(bucket)
        success = sum(1 for r in bucket if r.outcome == OUTCOME_SUCCESS)
        no_route = sum(1 for r in bucket if r.outcome == OUTCOME_NO_ROUTE)
        over_budget = sum(1 for r in bucket if r.outcome == OUTCOME_OVER_BUDGET)
        liq_exhausted = sum(1 for r in bucket if r.outcome == OUTCOME_LIQUIDITY_EXHAUSTED)
        timed_out = sum(1 for r in bucket if r.outcome == OUTCOME_TIMED_OUT)
        succ = [r for r in bucket if r.outcome == OUTCOME_SUCCESS]
        attempt_counts = [r.attempts_used for r in succ]
        fee_rates = [r.fee_rate for r in succ if r.fee_rate is not None]
        hops = [r.num_hops for r in succ if r.num_hops is not None]
        excess = [r.excess_hops for r in succ if r.excess_hops is not None]

        def pct(n: int) -> float:
            return 100.0 * n / attempts if attempts else 0.0

        summaries.append(LiquidityBucketSummary(
            amount_sat=amount_sat,
            attempts=attempts,
            success=success,
            no_route=no_route,
            over_budget=over_budget,
            liquidity_exhausted=liq_exhausted,
            timed_out=timed_out,
            success_pct=pct(success),
            no_route_pct=pct(no_route),
            over_budget_pct=pct(over_budget),
            liquidity_exhausted_pct=pct(liq_exhausted),
            timed_out_pct=pct(timed_out),
            mean_attempts=statistics.mean(attempt_counts) if attempt_counts else None,
            median_attempts=statistics.median(attempt_counts) if attempt_counts else None,
            mean_fee_rate=statistics.mean(fee_rates) if fee_rates else None,
            mean_num_hops=statistics.mean(hops) if hops else None,
            mean_excess_hops=statistics.mean(excess) if excess else None,
        ))
    return summaries


LIQUIDITY_RESULT_CSV_HEADER = [
    "amount_sat", "source", "dest", "dest_tier", "outcome", "attempts_used",
    "time_used_sec", "num_hops", "ideal_hops", "excess_hops", "fee_msat", "fee_rate",
]


def liquidity_result_to_csv_row(r: LiquidityRouteResult) -> List:
    return [
        r.amount_sat, r.source.hex(), r.dest.hex(), r.dest_tier.value, r.outcome,
        r.attempts_used, f"{r.time_used_sec:.3f}", r.num_hops, r.ideal_hops,
        r.excess_hops, r.fee_msat,
        f"{r.fee_rate:.6f}" if r.fee_rate is not None else "",
    ]


LIQUIDITY_SUMMARY_CSV_HEADER = [
    "amount_sat", "attempts", "success_pct", "no_route_pct", "over_budget_pct",
    "liquidity_exhausted_pct", "timed_out_pct", "mean_attempts", "median_attempts",
    "mean_fee_rate", "mean_num_hops", "mean_excess_hops",
]


def liquidity_summary_to_csv_row(s: LiquidityBucketSummary) -> List:
    def f(x):
        return f"{x:.6f}" if x is not None else ""
    return [
        s.amount_sat, s.attempts, f"{s.success_pct:.2f}", f"{s.no_route_pct:.2f}",
        f"{s.over_budget_pct:.2f}", f"{s.liquidity_exhausted_pct:.2f}",
        f"{s.timed_out_pct:.2f}", f(s.mean_attempts), f(s.median_attempts),
        f(s.mean_fee_rate), f(s.mean_num_hops), f(s.mean_excess_hops),
    ]


def format_liquidity_summary_table(summaries: Sequence[LiquidityBucketSummary]) -> str:
    """Human-readable summary table for the liquidity-aware benchmark."""
    def cell(x, fmt="{:.2f}"):
        return fmt.format(x) if x is not None else "-"
    header = (f"{'amount_sat':>11} {'attempts':>9} {'success%':>9} {'no_route%':>10} "
              f"{'over_bdgt%':>10} {'liq_exh%':>9} {'timed_out%':>11} {'mean_try':>9} "
              f"{'mean_fee%':>10} {'mean_hops':>10} {'mean_excess':>12}")
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.amount_sat:>11} {s.attempts:>9} {s.success_pct:>9.2f} "
            f"{s.no_route_pct:>10.2f} {s.over_budget_pct:>10.2f} "
            f"{s.liquidity_exhausted_pct:>9.2f} {s.timed_out_pct:>11.2f} "
            f"{cell(s.mean_attempts):>9} "
            f"{cell(s.mean_fee_rate * 100 if s.mean_fee_rate is not None else None):>10} "
            f"{cell(s.mean_num_hops):>10} {cell(s.mean_excess_hops):>12}")
    return "\n".join(lines)
