#!/usr/bin/env python3
# Copyright (C) 2026 The Electrum developers
# Distributed under the MIT software license, see the accompanying
# file LICENCE or http://www.opensource.org/licenses/mit-license.php

"""Run the offline Lightning pathfinding benchmark against a gossip snapshot.

This loads a captured ``gossip_db`` snapshot (see capture_snapshot.py) into an
Electrum :class:`~electrum.channel_db.ChannelDB`, then measures how
:class:`~electrum.lnrouter.LNPathFinder` charts routes for a range of payment
amounts -- without sending any payment. See electrum/pathfinding_benchmark.py
for the metrics and methodology.

Typical use (benchmark current code, then re-run after changing the router):
    python3 run_benchmark.py --snapshot ./snapshots/mainnet.gossip_db --out-dir ./results/before
    # ...edit electrum/lnrouter.py ...
    python3 run_benchmark.py --snapshot ./snapshots/mainnet.gossip_db --out-dir ./results/after

Note: the run is single-threaded Dijkstra over a real graph; with the default
sample sizes expect it to take a while (minutes). The amount, source and
destination sampling is seeded, so the two runs above are directly comparable.
"""

import argparse
import asyncio
import csv
import os
import shutil
import sys
import tempfile

# allow running directly from the source tree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import electrum  # noqa: E402
electrum.util.AS_LIB_USER_I_WANT_TO_MANAGE_MY_OWN_ASYNCIO_LOOP = True

from electrum import constants, lnrouter, util  # noqa: E402
from electrum.channel_db import ChannelDB  # noqa: E402
from electrum.simple_config import SimpleConfig  # noqa: E402
from electrum import pathfinding_benchmark as pb  # noqa: E402


async def _load_channel_db(config: SimpleConfig) -> ChannelDB:
    """Load a gossip snapshot into a ChannelDB.

    ChannelDB.load_data is @sql-decorated: calling it returns a future and the
    actual sqlite reads run on the DB worker thread (sqlite connections are
    thread-bound). This is exactly how Network.start_gossip loads gossip at
    runtime."""
    loop = asyncio.get_running_loop()

    class FakeNetwork:
        pass
    fake = FakeNetwork()
    fake.config = config
    fake.asyncio_loop = loop
    fake.trigger_callback = lambda *args, **kwargs: None
    fake.register_callback = lambda *args, **kwargs: None
    fake.interface = None

    cdb = ChannelDB(fake)
    await cdb.load_data()
    await cdb.data_loaded.wait()
    return cdb


def _write_csv(path: str, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


async def _amain(args) -> int:
    # Electrum's util.trigger_callback (fired from the DB worker thread during
    # load_data) resolves the global event loop; register ours so it's findable
    # from any thread, like Electrum does at startup.
    util._asyncio_event_loop = asyncio.get_running_loop()

    if args.network == "mainnet":
        constants.BitcoinMainnet.set_as_network()
    elif args.network == "testnet":
        constants.BitcoinTestnet.set_as_network()
    else:
        raise SystemExit(f"unsupported network: {args.network}")

    snapshot = os.path.expanduser(args.snapshot)
    if not os.path.isfile(snapshot):
        raise SystemExit(f"snapshot not found: {snapshot}")

    cfg = pb.BenchmarkConfig(
        amounts_sat=tuple(args.amounts),
        num_sources=args.num_sources,
        dests_per_tier=args.dests_per_tier,
        max_fee_millionths=args.max_fee_millionths,
        seed=args.seed,
    )
    lcfg = pb.LiquidityConfig(
        default_capacity_sat=args.default_capacity_sat,
        payment_timeout_sec=args.payment_timeout_sec,
        hop_latency_sec=args.hop_latency_sec,
        balance_seed=args.balance_seed,
        balance_beta=args.balance_beta,
    ) if args.liquidity else None

    # ChannelDB expects its file at <config.path>/gossip_db; copy the snapshot
    # into a scratch dir so we never touch the original.
    workdir = tempfile.mkdtemp(prefix="lnbench-")
    try:
        config = SimpleConfig({"electrum_path": workdir})
        shutil.copyfile(snapshot, os.path.join(workdir, "gossip_db"))

        print(f"loading snapshot {snapshot} ...")
        cdb = await _load_channel_db(config)
        print(f"loaded: {cdb.num_channels} channels, {cdb.num_policies} policies, "
              f"{cdb.num_nodes} nodes")
        try:
            path_finder = lnrouter.LNPathFinder(cdb)

            def progress(done, total):
                if done % 50 == 0 or done == total:
                    print(f"\r  {done}/{total} attempts", end="", flush=True)

            if args.compare_lnd:
                from electrum.plugins.lnd_pathfinder.lnd_router import LndPathFinder
                from electrum.plugins.lnd_pathfinder.mission_control import BimodalEstimator
                # offline: no observation store, so the bimodal estimator uses the
                # pure capacity-based prior (htlc_max / funding capacity per channel).
                estimator = BimodalEstimator(cdb)
                lnd_finder = LndPathFinder(
                    cdb, blacklist_filter=path_finder._is_edge_blacklisted,
                    probability_func=estimator)
                print("compare mode: native LNPathFinder vs. standalone LndPathFinder "
                      "(bimodal mission-control probability)")
                compare_rows = pb.run_compare_lnd(
                    channel_db=cdb, native_finder=path_finder, alt_finder=lnd_finder,
                    config=config, cfg=cfg, progress=progress)
                print()
            elif lcfg is not None:
                print(f"liquidity mode: capacity<-htlc_max (fallback {lcfg.default_capacity_sat} sat), "
                      f"payment_timeout_sec={lcfg.payment_timeout_sec}, "
                      f"hop_latency_sec={lcfg.hop_latency_sec}, balance_seed={lcfg.balance_seed}, "
                      f"balance_beta={lcfg.balance_beta}")
                results, summaries = pb.run_benchmark_with_liquidity(
                    channel_db=cdb, path_finder=path_finder, config=config, cfg=cfg,
                    lcfg=lcfg, progress=progress)
            else:
                results, summaries = pb.run_benchmark(
                    channel_db=cdb, path_finder=path_finder, config=config, cfg=cfg,
                    progress=progress)
            print()
        finally:
            cdb.stop()
            await cdb.stopped_event.wait()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.compare_lnd:
        _write_csv(os.path.join(out_dir, "compare.csv"), pb.COMPARE_CSV_HEADER,
                   [pb.compare_row_to_csv_row(r) for r in compare_rows])
        table = pb.format_compare_summary(compare_rows)
        alt_only = sum(1 for r in compare_rows if r.label == "alt_only")
        native_only = sum(1 for r in compare_rows if r.label == "native_only")
        print("\n" + table)
        print(f"\nalt_only (native fails, LND succeeds): {alt_only}   "
              f"native_only (LND fails, native succeeds): {native_only}")
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(table + "\n")
        print(f"\nwrote results to {out_dir}/ (compare.csv, summary.txt)")
        return 0
    if lcfg is not None:
        _write_csv(os.path.join(out_dir, "attempts.csv"), pb.LIQUIDITY_RESULT_CSV_HEADER,
                   [pb.liquidity_result_to_csv_row(r) for r in results])
        _write_csv(os.path.join(out_dir, "summary.csv"), pb.LIQUIDITY_SUMMARY_CSV_HEADER,
                   [pb.liquidity_summary_to_csv_row(s) for s in summaries])
        table = pb.format_liquidity_summary_table(summaries)
    else:
        _write_csv(os.path.join(out_dir, "attempts.csv"), pb.RESULT_CSV_HEADER,
                   [pb.result_to_csv_row(r) for r in results])
        _write_csv(os.path.join(out_dir, "summary.csv"), pb.SUMMARY_CSV_HEADER,
                   [pb.summary_to_csv_row(s) for s in summaries])
        table = pb.format_summary_table(summaries)
    print("\n" + table)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(table + "\n")
    print(f"\nwrote results to {out_dir}/ (attempts.csv, summary.csv, summary.txt)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", required=True, help="path to a captured gossip_db snapshot")
    parser.add_argument("--out-dir", required=True, help="directory to write results into")
    parser.add_argument("--network", default="mainnet", choices=["mainnet", "testnet"])
    parser.add_argument("--amounts", type=int, nargs="+", default=[100, 1000, 10000, 100000],
                        help="payment amounts to test, in sat")
    parser.add_argument("--num-sources", type=int, default=3,
                        help="number of well-connected nodes used as paying node")
    parser.add_argument("--dests-per-tier", type=int, default=50,
                        help="destinations sampled per connectivity tier")
    parser.add_argument("--max-fee-millionths", type=int, default=50_000,
                        help="fee budget in millionths (50000 = 5%%, matching the bug report)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    parser.add_argument("--liquidity", action="store_true",
                        help="model hidden channel balances and run the find_route->fail->retry "
                             "loop (exercises the dynamic liquidity path, not just static find_route)")
    parser.add_argument("--compare-lnd", action="store_true",
                        help="differential oracle: run the native finder and the standalone "
                             "LndPathFinder over the same samples and write compare.csv "
                             "(per-sample native vs LND outcome). Ignores --liquidity.")
    parser.add_argument("--payment-timeout-sec", type=float, default=120.0,
                        help="[liquidity] wall-clock budget per payment before giving up "
                             "(mirrors LNWallet.PAYMENT_TIMEOUT)")
    parser.add_argument("--hop-latency-sec", type=float, default=1.0,
                        help="[liquidity] simulated one-way per-hop HTLC latency; a failed "
                             "attempt costs ~2*this*(hops to failure). Calibration knob: "
                             "higher => fewer retries per timeout")
    parser.add_argument("--default-capacity-sat", type=int, default=5_000_000,
                        help="[liquidity] capacity assumed when htlc_maximum_msat is unset")
    parser.add_argument("--balance-seed", type=int, default=0,
                        help="[liquidity] RNG seed for the hidden per-channel balance assignment")
    parser.add_argument("--balance-beta", type=float, default=0.5,
                        help="[liquidity] Beta(b,b) shape for the per-channel balance split: "
                             "1.0=Uniform (over-generous), <1.0=bimodal/depleted (realistic; "
                             "lower=harsher). Calibrate against measured mainnet success rates.")
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
