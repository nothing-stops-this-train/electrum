#!/usr/bin/env python
#
# Electrum - lightweight Bitcoin client
# Copyright (C) 2026 The Electrum Developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""Plugin that, on every Lightning payment, runs an LND-style pathfinder
alongside Electrum's native one, auto-selects the cheaper route, comprehensively
logs both (JSONL, in the wallet directory), and surfaces the choice to the user.

Built to gather real-world data for spesmilo/electrum#10443. The LND finder only
*returns* a path; Electrum's payment engine still does all the spending. Any
error in this plugin defers to the native pathfinder, so it can never break a
payment.
"""

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from electrum import constants, util
from electrum.plugin import BasePlugin, hook
from electrum.pathfinding_benchmark import route_fee_msat

from .lnd_router import LndPathFinder
from .lnd_log import (
    serialize_route, select_cheapest, build_route_compare_record,
    build_payment_outcome_record, append_jsonl,
)
from .mission_control import MissionControlStore, BimodalEstimator

if TYPE_CHECKING:
    from electrum.lnworker import LNWallet
    from electrum.wallet import Abstract_Wallet

LOG_FILENAME = "lnd_pathfinder.jsonl"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _network_name() -> str:
    try:
        return constants.net.NET_NAME
    except Exception:
        return "unknown"


class LndPathfinderPlugin(BasePlugin):

    def __init__(self, parent, config, name):
        BasePlugin.__init__(self, parent, config, name)
        self._contexts = {}  # id(channel_db) -> {"finder", "store", "estimator"}
        self._wallets = {}   # wallet -> {"log_path": str, "lock": Lock}
        self._setup_lock = threading.Lock()
        self._callback_registered = False

    # -- lifecycle ----------------------------------------------------------

    @hook
    def load_wallet(self, wallet: 'Abstract_Wallet', window=None):
        try:
            wallet_dir = os.path.dirname(wallet.storage.path)
            log_path = os.path.join(wallet_dir, LOG_FILENAME)
        except Exception:
            self.logger.exception("could not determine wallet log path; disabling for this wallet")
            return
        self._wallets[wallet] = {"log_path": log_path, "lock": threading.Lock()}
        if not self._callback_registered:
            util.register_callback(self._on_payment, ['payment_succeeded', 'payment_failed'])
            self._callback_registered = True
        self.logger.info(f"lnd_pathfinder active for {wallet.basename()}; logging to {log_path}")

    def on_close(self):
        if self._callback_registered:
            util.unregister_callback(self._on_payment)
            self._callback_registered = False

    def _wallet_ctx(self, wallet) -> Optional[dict]:
        return self._wallets.get(wallet)

    def _get_context(self, lnworker: 'LNWallet') -> dict:
        """Per-network LND finder + its bimodal MissionControl store/estimator."""
        channel_db = lnworker.channel_db
        key = id(channel_db)
        with self._setup_lock:
            ctx = self._contexts.get(key)
            if ctx is None:
                native = lnworker.network.path_finder
                blacklist = getattr(native, '_is_edge_blacklisted', None)
                store = MissionControlStore()
                estimator = BimodalEstimator(channel_db, store=store)
                finder = LndPathFinder(channel_db, blacklist_filter=blacklist,
                                       probability_func=estimator)
                ctx = {"finder": finder, "store": store, "estimator": estimator}
                self._contexts[key] = ctx
        return ctx

    # -- the route-selection hook ------------------------------------------

    @hook
    def get_route_for_payment(self, lnworker, payment_hash, invoice_pubkey, amount_msat,
                              full_path, my_sending_channels, private_route_edges):
        # Never break a payment: any failure here -> return None -> native path.
        try:
            return self._compare_and_choose(
                lnworker, payment_hash, invoice_pubkey, amount_msat,
                full_path, my_sending_channels, private_route_edges)
        except Exception as e:
            self.logger.exception(f"get_route_for_payment failed; deferring to native router: {e!r}")
            return None

    def _compare_and_choose(self, lnworker, payment_hash, invoice_pubkey, amount_msat,
                            full_path, my_sending_channels, private_route_edges):
        if full_path is not None:
            return None  # user-pinned path: nothing to compare, let native handle it
        nodeA = lnworker.node_keypair.pubkey
        native = lnworker.network.path_finder
        lnd = self._get_context(lnworker)["finder"]

        native_route, native_err, native_ms = self._run_finder(
            native, nodeA, invoice_pubkey, amount_msat, my_sending_channels, private_route_edges)
        lnd_route, lnd_err, lnd_ms = self._run_finder(
            lnd, nodeA, invoice_pubkey, amount_msat, my_sending_channels, private_route_edges)

        chosen_name, chosen_route, reason = select_cheapest(
            native_route, lnd_route, amount_msat_for_dest=amount_msat)

        self._log_compare(
            lnworker, payment_hash, nodeA, invoice_pubkey, amount_msat, private_route_edges,
            native_route, native_err, native_ms, lnd_route, lnd_err, lnd_ms, chosen_name, reason)
        self._surface(lnworker, chosen_name, chosen_route, amount_msat)
        # chosen_route may be None (both finders failed): native path will re-run
        # and raise NoPathFound, preserving native exception semantics.
        return chosen_route

    @staticmethod
    def _run_finder(finder, nodeA, nodeB, amount_msat, my_sending_channels, private_route_edges):
        t0 = time.perf_counter()
        try:
            route = finder.find_route(
                nodeA=nodeA, nodeB=nodeB, invoice_amount_msat=amount_msat,
                my_sending_channels=my_sending_channels, private_route_edges=private_route_edges)
            err = None
        except Exception as e:
            route, err = None, repr(e)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return route, err, elapsed_ms

    # -- logging / surfacing ------------------------------------------------

    def _gossip_stats(self, lnworker) -> dict:
        stats = {}
        try:
            cdb = lnworker.channel_db
            stats["num_nodes"] = cdb.num_nodes
            stats["num_channels"] = cdb.num_channels
            stats["num_policies"] = cdb.num_policies
        except Exception:
            pass
        try:
            stats["sync_progress"] = lnworker.network.lngossip.get_sync_progress_estimate()
        except Exception:
            stats["sync_progress"] = None
        return stats

    @staticmethod
    def _fee_settings(config) -> dict:
        try:
            return {
                "fee_max_millionths": config.LIGHTNING_PAYMENT_FEE_MAX_MILLIONTHS,
                "fee_cutoff_msat": config.LIGHTNING_PAYMENT_FEE_CUTOFF_MSAT,
            }
        except Exception:
            return {}

    def _log_compare(self, lnworker, payment_hash, nodeA, dest, amount_msat, private_route_edges,
                     native_route, native_err, native_ms, lnd_route, lnd_err, lnd_ms,
                     chosen_name, reason):
        ctx = self._wallet_ctx(lnworker.wallet)
        if not ctx:
            return
        record = build_route_compare_record(
            ts=_utcnow_iso(),
            network=_network_name(),
            payment_hash=payment_hash.hex() if payment_hash else None,
            source=nodeA.hex(),
            destination=dest.hex(),
            amount_msat=amount_msat,
            num_route_hints=len(private_route_edges or {}),
            gossip=self._gossip_stats(lnworker),
            fee_settings=self._fee_settings(lnworker.config),
            native=serialize_route(native_route, amount_msat_for_dest=amount_msat,
                                   elapsed_ms=native_ms, error=native_err),
            lnd=serialize_route(lnd_route, amount_msat_for_dest=amount_msat,
                                elapsed_ms=lnd_ms, error=lnd_err),
            chosen=chosen_name,
            chosen_reason=reason,
        )
        try:
            append_jsonl(ctx["log_path"], record, ctx["lock"])
        except Exception:
            self.logger.exception("failed to append route_compare record")

    def _surface(self, lnworker, chosen_name, chosen_route, amount_msat):
        if not chosen_route:
            self.logger.info("both native and LND pathfinders failed to find a route")
            return
        fee = route_fee_msat(chosen_route, amount_msat_for_dest=amount_msat)
        note = f"Route via {chosen_name} pathfinder ({fee} msat fee, {len(chosen_route)} hops) — auto-selected cheapest"
        self.logger.info(note)
        try:
            util.trigger_callback('payment_route_selected', lnworker.wallet, note)
        except Exception:
            pass

    @hook
    def htlc_route_result(self, lnworker, route, amount_msat, failing_channel):
        """Feed a real payment outcome into the LND finder's MissionControl store.

        ``failing_channel=None`` means the whole route forwarded successfully.
        Otherwise every hop up to (but excluding) the failing channel is a
        success and the failing channel is a failure, mirroring how Electrum
        updates its own liquidity hints. Side-effect only: returns None so it
        never overrides another hook's value."""
        try:
            store = self._get_context(lnworker)["store"]
            now = time.time()
            for edge in route:
                if failing_channel is not None and edge.short_channel_id == failing_channel:
                    store.report_failure(edge.short_channel_id, edge.start_node,
                                         edge.end_node, amount_msat, now=now)
                    break
                store.report_success(edge.short_channel_id, edge.start_node,
                                     edge.end_node, amount_msat, now=now)
        except Exception:
            self.logger.exception("error recording htlc route result into mission control")

    def _on_payment(self, event, *args):
        try:
            wallet = args[0]
            key = args[1] if len(args) > 1 else None  # payment_hash hex (see lnworker.pay_invoice)
            success = (event == 'payment_succeeded')
            reason = args[2] if (not success and len(args) > 2) else ""
            ctx = self._wallet_ctx(wallet)
            if not ctx:
                return
            record = build_payment_outcome_record(
                ts=_utcnow_iso(),
                payment_hash=key if isinstance(key, str) else None,
                success=success,
                reason=reason,
            )
            append_jsonl(ctx["log_path"], record, ctx["lock"])
        except Exception:
            self.logger.exception("error logging payment outcome")
