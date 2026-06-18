#!/usr/bin/env python
#
# Electrum - lightweight Bitcoin client
# Copyright (C) 2026 The Electrum Developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""Pure helpers for the lnd_pathfinder plugin: route serialization, cheapest
selection, divergence-record building, and JSONL appending.

These are deliberately free of any wallet/network/GUI dependencies so they can
be unit-tested directly. NOTHING in here ever touches a private key, payment
secret, or preimage: records are built field-by-field from public route data
(node pubkeys and short_channel_ids are public network identifiers).
"""

import json
import os
import threading
from typing import Optional, Sequence, Tuple

from electrum.lnrouter import LNPaymentRoute
from electrum.pathfinding_benchmark import route_fee_msat


def route_edge_to_dict(edge) -> dict:
    """Serialize a single RouteEdge to a JSON-safe dict. Whitelisted public
    fields only."""
    return {
        "scid": str(edge.short_channel_id),
        "start": edge.start_node.hex(),
        "end": edge.end_node.hex(),
        "fee_base_msat": int(edge.fee_base_msat),
        "fee_ppm": int(edge.fee_proportional_millionths),
        "cltv_delta": int(edge.cltv_delta),
    }


def serialize_route(
        route: Optional[LNPaymentRoute],
        *,
        amount_msat_for_dest: int,
        elapsed_ms: Optional[float] = None,
        error: Optional[str] = None,
) -> dict:
    """Serialize a finder's result (route or None) into a JSON-safe summary."""
    out = {}
    if elapsed_ms is not None:
        out["elapsed_ms"] = round(float(elapsed_ms), 3)
    if not route:
        out["outcome"] = "no_path"
        if error:
            out["error"] = str(error)
        return out
    out["outcome"] = "success"
    out["num_hops"] = len(route)
    out["total_fee_msat"] = route_fee_msat(route, amount_msat_for_dest=amount_msat_for_dest)
    out["route"] = [route_edge_to_dict(e) for e in route]
    return out


def select_cheapest(
        native_route: Optional[LNPaymentRoute],
        lnd_route: Optional[LNPaymentRoute],
        *,
        amount_msat_for_dest: int,
) -> Tuple[str, Optional[LNPaymentRoute], str]:
    """Pick the cheaper of the two routes.

    Rule: lowest total routing fee; tie-break fewer hops; final tie -> native.
    Returns ``(chosen_name, chosen_route, reason)`` where chosen_name is one of
    ``"native" | "lnd" | "none"``.
    """
    if not native_route and not lnd_route:
        return "none", None, "both_failed"
    if native_route and not lnd_route:
        return "native", native_route, "only_route"
    if lnd_route and not native_route:
        return "lnd", lnd_route, "only_route"
    native_fee = route_fee_msat(native_route, amount_msat_for_dest=amount_msat_for_dest)
    lnd_fee = route_fee_msat(lnd_route, amount_msat_for_dest=amount_msat_for_dest)
    if lnd_fee < native_fee:
        return "lnd", lnd_route, "lower_fee"
    if native_fee < lnd_fee:
        return "native", native_route, "lower_fee"
    # equal fee -> prefer fewer hops
    if len(lnd_route) < len(native_route):
        return "lnd", lnd_route, "fewer_hops"
    if len(native_route) < len(lnd_route):
        return "native", native_route, "fewer_hops"
    return "native", native_route, "tie_native"


def build_route_compare_record(
        *,
        ts: str,
        network: str,
        payment_hash: Optional[str],
        source: str,
        destination: str,
        amount_msat: int,
        num_route_hints: int,
        gossip: dict,
        fee_settings: dict,
        native: dict,
        lnd: dict,
        chosen: str,
        chosen_reason: str,
) -> dict:
    """Assemble a ``route_compare`` JSONL record from already-serialized parts.

    Every field is supplied explicitly by the caller; this function performs no
    reflection over wallet objects, so secrets cannot leak through it."""
    return {
        "event": "route_compare",
        "ts": ts,
        "network": network,
        "payment_hash": payment_hash,
        "source": source,
        "destination": destination,
        "amount_msat": int(amount_msat),
        "has_route_hints": bool(num_route_hints),
        "num_route_hints": int(num_route_hints),
        "gossip": gossip,
        "fee_settings": fee_settings,
        "native": native,
        "lnd": lnd,
        "chosen": chosen,
        "chosen_reason": chosen_reason,
    }


def build_payment_outcome_record(
        *,
        ts: str,
        payment_hash: Optional[str],
        success: bool,
        reason: str = "",
) -> dict:
    return {
        "event": "payment_outcome",
        "ts": ts,
        "payment_hash": payment_hash,
        "success": bool(success),
        "reason": str(reason or ""),
    }


def append_jsonl(path: str, record: dict, lock: Optional[threading.Lock] = None) -> None:
    """Append one JSON object as a line to ``path`` (created if needed)."""
    line = json.dumps(record, separators=(",", ":"), sort_keys=True)
    if lock is None:
        lock = threading.Lock()
    with lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
