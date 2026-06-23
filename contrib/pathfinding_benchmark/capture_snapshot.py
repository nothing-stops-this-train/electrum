#!/usr/bin/env python3
# Copyright (C) 2026 The Electrum developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""Capture a snapshot of Electrum's Lightning gossip database.

Electrum stores public gossip (channels, policies, node announcements) in an
sqlite file named ``gossip_db`` inside its data directory. To get a usable
snapshot you must first let a running Electrum sync gossip -- this takes several
minutes (and the graph keeps filling in for a while after that). Once synced,
this script copies the live ``gossip_db`` to a standalone snapshot file that
``run_benchmark.py`` can load offline and repeatedly, so before/after benchmark
runs use the exact same graph.

The snapshot is NOT committed to the repo (see .gitignore); regenerate it when
you want a fresher graph.

Usage:
    # let Electrum sync gossip first (LIGHTNING_USE_GOSSIP / non-trampoline), then:
    python3 capture_snapshot.py --gossip-db ~/.electrum/gossip_db --out ./snapshots/mainnet.gossip_db

If --gossip-db is omitted, common default locations are tried.
"""

import argparse
import os
import shutil
import sqlite3
import sys


DEFAULT_GOSSIP_DB_LOCATIONS = [
    "~/.electrum/gossip_db",
    "~/.electrum/testnet/gossip_db",
    "~/Library/Application Support/Electrum/gossip_db",
    os.path.expandvars("%APPDATA%/Electrum/gossip_db"),
]


def _find_gossip_db() -> str:
    for loc in DEFAULT_GOSSIP_DB_LOCATIONS:
        p = os.path.expanduser(loc)
        if os.path.isfile(p):
            return p
    raise SystemExit(
        "Could not find a gossip_db. Pass --gossip-db explicitly. "
        "Make sure Electrum has run with gossip enabled long enough to sync.")


def _row_counts(path: str) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out = {}
        for table in ("channel_info", "policy", "node_info", "address"):
            try:
                out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                out[table] = "n/a"
        return out
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gossip-db", help="path to the live Electrum gossip_db file")
    parser.add_argument("--out", required=True, help="destination snapshot path")
    args = parser.parse_args(argv)

    src = os.path.expanduser(args.gossip_db) if args.gossip_db else _find_gossip_db()
    if not os.path.isfile(src):
        raise SystemExit(f"gossip_db not found: {src}")

    counts = _row_counts(src)
    if counts.get("channel_info") in (0, "n/a"):
        print(f"WARNING: snapshot looks empty/unsynced: {counts}", file=sys.stderr)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # copy with the sqlite backup API so we get a consistent file even if
    # Electrum is still running and writing to it.
    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(out)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()

    size_mb = os.path.getsize(out) / 1e6
    print(f"captured snapshot -> {out} ({size_mb:.1f} MB)")
    print(f"  channels={counts['channel_info']} policies={counts['policy']} "
          f"nodes={counts['node_info']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
