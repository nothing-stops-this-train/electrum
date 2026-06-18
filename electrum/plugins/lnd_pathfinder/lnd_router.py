#!/usr/bin/env python
#
# Electrum - lightweight Bitcoin client
# Copyright (C) 2026 The Electrum Developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""A standalone, LND-style Lightning pathfinder.

This is intentionally *separate* from :class:`electrum.lnrouter.LNPathFinder`.
It reads the same gossip graph (``ChannelDB``) and produces a byte-compatible
``LNPaymentRoute`` (so the route it returns is a drop-in for Electrum's payment
engine), but it runs its own Dijkstra and, crucially, its own cost function.

The only behavioural difference from Electrum's router is the per-edge cost.
Electrum's ``_edge_cost`` is ``fee + cltv_cost + liquidity_penalty`` with *no
per-hop constant*; LND adds an absolute ``AttemptCost`` per hop (and divides the
running cost by an estimated success probability). That missing per-hop term is
the suspected cause of the circuitous / over-budget routes in
spesmilo/electrum#10443, so this module exists to A/B the two route choices.

LND references (routing/pathfind.go): ``AttemptCost`` (default 100 sat) +
``AttemptCostPPM`` (default 1000 ppm), ``RiskFactorBillionths`` (default 15,
which Electrum already uses for ``cltv_cost``), and probability-weighted edge
distance ``weight + penalty/probability``.

All pruning (htlc min/max, capacity, disabled, missing/var-onion features,
2-week cltv cap, blacklist) is kept identical to Electrum's router so that the
*only* variable under study is the cost function.
"""

import queue
import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

from electrum.lnrouter import (
    PathEdge, RouteEdge, LNPaymentPath, LNPaymentRoute,
    NoChannelPolicy, LNPathInconsistent,
)
from electrum.lnutil import ShortChannelID, LnFeatures
from electrum.logging import Logger

if TYPE_CHECKING:
    from electrum.channel_db import ChannelDB
    from electrum.lnchannel import Channel


# --- LND cost-model constants (routing/pathfind.go defaults) ---------------
ATTEMPT_COST_MSAT = 100_000        # absolute virtual cost per hop (100 sat)
ATTEMPT_COST_PPM = 1_000           # plus this fraction of the amount (0.1%)
RISK_FACTOR_BILLIONTHS = 15        # timelock risk factor (same as Electrum)
DEFAULT_APRIORI_PROBABILITY = 0.6  # LND's apriori per-hop success probability

# cltv cap, mirroring lnrouter._edge_cost (2 weeks)
MAX_CLTV_DELTA = 14 * 144

# small positive cost charged for the sender's own first hop (no real fee);
# kept >0 so equal-length routes don't all tie at exactly zero.
OWN_HOP_COST = 1.0


ProbabilityFunc = Callable[[bytes, bytes, ShortChannelID, int], float]
BlacklistFunc = Callable[..., bool]


class LndPathFinder(Logger):
    """LND-style route finder over an Electrum ``ChannelDB``.

    :param channel_db: the live gossip graph to search.
    :param blacklist_filter: optional ``f(scid, *, now) -> bool`` returning True
        for channels to skip (e.g. ``LNPathFinder._is_edge_blacklisted``), so a
        shadow comparison can honour the same recently-failed-channel set.
    :param probability_func: optional ``f(start, end, scid, amount) -> float`` in
        (0, 1]; the LND mission-control hook. Defaults to a constant apriori.
    """

    def __init__(
            self,
            channel_db: 'ChannelDB',
            *,
            blacklist_filter: Optional[BlacklistFunc] = None,
            probability_func: Optional[ProbabilityFunc] = None,
    ):
        Logger.__init__(self)
        self.channel_db = channel_db
        self._blacklist_filter = blacklist_filter
        self._probability_func = probability_func

    # -- cost ---------------------------------------------------------------

    def _edge_probability(self, start_node: bytes, end_node: bytes,
                          short_channel_id: ShortChannelID, amount_msat: int) -> float:
        if self._probability_func is not None:
            try:
                p = self._probability_func(start_node, end_node, short_channel_id, amount_msat)
            except Exception:
                p = DEFAULT_APRIORI_PROBABILITY
        else:
            p = DEFAULT_APRIORI_PROBABILITY
        # clamp to a sane, non-zero range to avoid divide-by-zero / negatives
        if not (p and p > 0):
            return 0.01
        return min(p, 1.0)

    def _is_blacklisted(self, short_channel_id: ShortChannelID, *, now: int) -> bool:
        if self._blacklist_filter is None:
            return False
        try:
            return bool(self._blacklist_filter(short_channel_id, now=now))
        except TypeError:
            # tolerate a positional-only filter
            return bool(self._blacklist_filter(short_channel_id))

    def _edge_cost(
            self,
            *,
            short_channel_id: ShortChannelID,
            start_node: bytes,
            end_node: bytes,
            payment_amt_msat: int,
            ignore_costs: bool = False,
            ignore_amount_constraints: bool = False,
            is_mine: bool = False,
            my_channels: Dict[ShortChannelID, 'Channel'] = None,
            private_route_edges: Dict[ShortChannelID, RouteEdge] = None,
            now: int,
    ) -> Tuple[float, int]:
        """Heuristic LND cost of going through a channel.

        Returns ``(heuristic_cost, fee_for_edge_msat)``. Pruning is identical to
        ``electrum.lnrouter.LNPathFinder._edge_cost``; only the final cost
        arithmetic differs (per-hop AttemptCost / probability).
        """
        if private_route_edges is None:
            private_route_edges = {}
        if self._is_blacklisted(short_channel_id, now=now):
            return float('inf'), 0
        channel_info = self.channel_db.get_channel_info(
            short_channel_id, my_channels=my_channels, private_route_edges=private_route_edges)
        if channel_info is None:
            return float('inf'), 0
        channel_policy = self.channel_db.get_policy_for_node(
            short_channel_id, start_node, my_channels=my_channels,
            private_route_edges=private_route_edges, now=now)
        if channel_policy is None:
            return float('inf'), 0
        # channels that did not publish both policies often return temporary channel failure
        channel_policy_backwards = self.channel_db.get_policy_for_node(
            short_channel_id, end_node, my_channels=my_channels,
            private_route_edges=private_route_edges, now=now)
        if (channel_policy_backwards is None
                and not is_mine
                and short_channel_id not in private_route_edges):
            return float('inf'), 0
        if channel_policy.is_disabled():
            return float('inf'), 0
        if not ignore_amount_constraints:
            if payment_amt_msat < channel_policy.htlc_minimum_msat:
                return float('inf'), 0  # payment amount too little
            if channel_info.capacity_sat is not None and \
                    payment_amt_msat // 1000 > channel_info.capacity_sat:
                return float('inf'), 0  # payment amount too large
            if channel_policy.htlc_maximum_msat is not None and \
                    payment_amt_msat > channel_policy.htlc_maximum_msat:
                return float('inf'), 0  # payment amount too large
        route_edge = private_route_edges.get(short_channel_id, None)
        if route_edge is None:
            node_info = self.channel_db.get_node_info_for_node_id(node_id=end_node)
            if node_info:
                # if we have the node_announcement, enforce var_onion_optin
                node_features = LnFeatures(node_info.features)
                if not node_features.supports(LnFeatures.VAR_ONION_OPT):
                    return float('inf'), 0
            route_edge = RouteEdge.from_channel_policy(
                channel_policy=channel_policy,
                short_channel_id=short_channel_id,
                start_node=start_node,
                end_node=end_node,
                node_info=node_info)
        if route_edge.cltv_delta > MAX_CLTV_DELTA:
            return float('inf'), 0
        fee_msat = route_edge.fee_for_edge(payment_amt_msat)
        if ignore_costs or ignore_amount_constraints:
            # sender's own first hop (or onion-message search): no real fee/cost
            return OWN_HOP_COST, 0
        # --- LND cost (this is the whole point of this module) ---
        timelock_penalty = route_edge.cltv_delta * payment_amt_msat * RISK_FACTOR_BILLIONTHS / 1_000_000_000
        attempt_cost = ATTEMPT_COST_MSAT + (payment_amt_msat * ATTEMPT_COST_PPM) // 1_000_000
        probability = self._edge_probability(start_node, end_node, short_channel_id, payment_amt_msat)
        overall_cost = fee_msat + timelock_penalty + attempt_cost / probability
        return overall_cost, fee_msat

    # -- search -------------------------------------------------------------

    def get_shortest_path_hops(
            self,
            *,
            nodeA: bytes,
            nodeB: bytes,
            invoice_amount_msat: Optional[int],
            my_sending_channels: Dict[ShortChannelID, 'Channel'] = None,
            private_route_edges: Dict[ShortChannelID, RouteEdge] = None,
    ) -> Dict[bytes, PathEdge]:
        """Backward Dijkstra (nodeB -> nodeA), mirroring Electrum's traversal so
        the only difference vs. native is :meth:`_edge_cost`."""
        if my_sending_channels is None:
            my_sending_channels = {}
        if private_route_edges is None:
            private_route_edges = {}
        ignore_amount_constraints = invoice_amount_msat is None
        distance_from_start = defaultdict(lambda: float('inf'))
        distance_from_start[nodeB] = 0
        previous_hops = {}  # type: Dict[bytes, PathEdge]
        nodes_to_explore = queue.PriorityQueue()
        nodes_to_explore.put((0, invoice_amount_msat or 0, nodeB))
        now = int(time.time())

        while nodes_to_explore.qsize() > 0:
            dist_to_edge_endnode, amount_msat, edge_endnode = nodes_to_explore.get()
            if edge_endnode == nodeA and previous_hops:
                break
            if dist_to_edge_endnode != distance_from_start[edge_endnode]:
                # stale duplicate (no decrease-key in PriorityQueue)
                continue
            channels_for_endnode = self.channel_db.get_channels_for_node(
                edge_endnode, my_channels=my_sending_channels, private_route_edges=private_route_edges)
            for edge_channel_id in channels_for_endnode:
                assert isinstance(edge_channel_id, bytes)
                if self._is_blacklisted(edge_channel_id, now=now):
                    continue
                channel_info = self.channel_db.get_channel_info(
                    edge_channel_id, my_channels=my_sending_channels, private_route_edges=private_route_edges)
                if channel_info is None:
                    continue
                edge_startnode = channel_info.node2_id if channel_info.node1_id == edge_endnode else channel_info.node1_id
                is_mine = edge_channel_id in my_sending_channels
                if edge_startnode == nodeA and my_sending_channels:  # payment outgoing, on our channel
                    if edge_channel_id not in my_sending_channels:
                        continue
                    if not ignore_amount_constraints \
                            and not my_sending_channels[edge_channel_id].can_pay(amount_msat, check_frozen=True):
                        continue
                edge_cost, fee_for_edge_msat = self._edge_cost(
                    short_channel_id=edge_channel_id,
                    start_node=edge_startnode,
                    end_node=edge_endnode,
                    payment_amt_msat=amount_msat,
                    ignore_costs=(edge_startnode == nodeA),
                    ignore_amount_constraints=ignore_amount_constraints,
                    is_mine=is_mine,
                    my_channels=my_sending_channels,
                    private_route_edges=private_route_edges,
                    now=now,
                )
                alt_dist_to_neighbour = distance_from_start[edge_endnode] + edge_cost
                if alt_dist_to_neighbour < distance_from_start[edge_startnode]:
                    distance_from_start[edge_startnode] = alt_dist_to_neighbour
                    previous_hops[edge_startnode] = PathEdge(
                        start_node=edge_startnode,
                        end_node=edge_endnode,
                        short_channel_id=ShortChannelID(edge_channel_id))
                    amount_to_forward_msat = amount_msat + fee_for_edge_msat
                    nodes_to_explore.put((alt_dist_to_neighbour, amount_to_forward_msat, edge_startnode))
        return previous_hops

    def find_path_for_payment(
            self,
            *,
            nodeA: bytes,
            nodeB: bytes,
            invoice_amount_msat: Optional[int],
            my_sending_channels: Dict[ShortChannelID, 'Channel'] = None,
            private_route_edges: Dict[ShortChannelID, RouteEdge] = None,
    ) -> Optional[LNPaymentPath]:
        assert type(nodeA) is bytes
        assert type(nodeB) is bytes
        assert type(invoice_amount_msat) is int or invoice_amount_msat is None
        if my_sending_channels is None:
            my_sending_channels = {}
        previous_hops = self.get_shortest_path_hops(
            nodeA=nodeA, nodeB=nodeB, invoice_amount_msat=invoice_amount_msat,
            my_sending_channels=my_sending_channels, private_route_edges=private_route_edges)
        if nodeA not in previous_hops:
            return None  # no path found
        edge_startnode = nodeA
        path = []
        while edge_startnode != nodeB or not path:
            edge = previous_hops[edge_startnode]
            path += [edge]
            edge_startnode = edge.node_id
        return path

    def create_route_from_path(
            self,
            path: Optional[LNPaymentPath],
            *,
            my_channels: Dict[ShortChannelID, 'Channel'] = None,
            private_route_edges: Dict[ShortChannelID, RouteEdge] = None,
    ) -> LNPaymentRoute:
        if path is None:
            raise Exception('cannot create route from None path')
        if private_route_edges is None:
            private_route_edges = {}
        route = []
        prev_end_node = path[0].start_node
        for path_edge in path:
            short_channel_id = path_edge.short_channel_id
            _endnodes = self.channel_db.get_endnodes_for_chan(short_channel_id, my_channels=my_channels)
            if _endnodes and sorted(_endnodes) != sorted([path_edge.start_node, path_edge.end_node]):
                raise LNPathInconsistent("endpoints of edge inconsistent with short_channel_id")
            if path_edge.start_node != prev_end_node:
                raise LNPathInconsistent("edges do not chain together")
            route_edge = private_route_edges.get(short_channel_id, None)
            if route_edge is None:
                channel_policy = self.channel_db.get_policy_for_node(
                    short_channel_id=short_channel_id,
                    node_id=path_edge.start_node,
                    my_channels=my_channels)
                if channel_policy is None:
                    raise NoChannelPolicy(short_channel_id)
                node_info = self.channel_db.get_node_info_for_node_id(node_id=path_edge.end_node)
                route_edge = RouteEdge.from_channel_policy(
                    channel_policy=channel_policy,
                    short_channel_id=short_channel_id,
                    start_node=path_edge.start_node,
                    end_node=path_edge.end_node,
                    node_info=node_info)
            route.append(route_edge)
            prev_end_node = path_edge.end_node
        return route

    def find_route(
            self,
            *,
            nodeA: bytes,
            nodeB: bytes,
            invoice_amount_msat: int,
            path: Optional[Sequence[PathEdge]] = None,
            my_sending_channels: Dict[ShortChannelID, 'Channel'] = None,
            private_route_edges: Dict[ShortChannelID, RouteEdge] = None,
    ) -> Optional[LNPaymentRoute]:
        """Drop-in replacement for ``LNPathFinder.find_route``."""
        if path is None:
            path = self.find_path_for_payment(
                nodeA=nodeA, nodeB=nodeB, invoice_amount_msat=invoice_amount_msat,
                my_sending_channels=my_sending_channels, private_route_edges=private_route_edges)
        if path is None:
            return None
        return self.create_route_from_path(
            path, my_channels=my_sending_channels, private_route_edges=private_route_edges)
