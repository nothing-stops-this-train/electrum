import statistics
from typing import Optional

from electrum import util, lnrouter, constants
from electrum.channel_db import ChannelDB
from electrum.simple_config import SimpleConfig
from electrum.constants import BitcoinTestnet
from electrum.util import bfh
from electrum.lnutil import ShortChannelID, LnFeatures
from electrum.lnmsg import encode_msg, decode_msg
from electrum import pathfinding_benchmark as pb

from . import ElectrumTestCase


def channel(number: int) -> ShortChannelID:
    return ShortChannelID(bfh(format(number, '016x')))


def node(character: str) -> bytes:
    return b'\x02' + f'{character}'.encode() * 32


def alias(character: str) -> bytes:
    return (character * 8).encode('utf-8')


def node_features(extra: LnFeatures = None) -> bytes:
    lnf = LnFeatures(0) | LnFeatures.VAR_ONION_OPT
    if extra:
        lnf |= extra
    return lnf.to_bytes(8, 'big')


class Test_PathfindingBenchmark(ElectrumTestCase):
    TESTNET = True

    cdb = None  # type: Optional[ChannelDB]

    def setUp(self):
        super().setUp()
        self.config = SimpleConfig({'electrum_path': self.electrum_path})
        self.assertIsNone(self.cdb)

    async def asyncTearDown(self):
        if self.cdb:
            self.cdb.stop()
            await self.cdb.stopped_event.wait()
        await super().asyncTearDown()

    def prepare_graph(self):
        """Same topology as tests/test_lnrouter.py:
                3
            A  ---  B
            |    2/ |
          6 |   E   | 1
            | /5 \\7 |
            D  ---  C
                4
        Shortest A->E is 2 hops (A-3->B-2->E or A-6->D-5->E).
        """
        class fake_network:
            config = self.config
            asyncio_loop = util.get_asyncio_loop()
            trigger_callback = lambda *args: None
            register_callback = lambda *args: None
            interface = None

        fake_network.channel_db = ChannelDB(fake_network())
        fake_network.channel_db.data_loaded.set()
        self.cdb = fake_network.channel_db
        self.path_finder = lnrouter.LNPathFinder(self.cdb)

        chans = [
            (1, 'b', 'c'), (2, 'b', 'e'), (3, 'a', 'b'), (4, 'c', 'd'),
            (5, 'd', 'e'), (6, 'a', 'd'), (7, 'c', 'e'),
        ]
        for num, n1, n2 in chans:
            self.cdb.add_channel_announcements({
                'node_id_1': node(n1), 'node_id_2': node(n2),
                'bitcoin_key_1': node(n1), 'bitcoin_key_2': node(n2),
                'short_channel_id': channel(num),
                'chain_hash': BitcoinTestnet.rev_genesis_bytes(),
                'len': 0, 'features': b'',
            }, trusted=True)
        for char in 'abcde':
            self.cdb.add_node_announcements({
                'node_id': node(char), 'alias': alias(char), 'addresses': [],
                'features': node_features(), 'timestamp': 0,
            })

        def add_upd(num, channel_flags, **kw):
            payload = {
                'short_channel_id': channel(num), 'message_flags': b'\x00',
                'channel_flags': channel_flags, 'cltv_expiry_delta': 10,
                'htlc_minimum_msat': 250, 'fee_base_msat': 100,
                'fee_proportional_millionths': 150,
                'chain_hash': BitcoinTestnet.rev_genesis_bytes(), 'timestamp': 0,
            }
            payload.update(kw)
            self.cdb.add_channel_update(payload, verify=False)

        # both directions for every channel so edges are usable
        for num in range(1, 8):
            add_upd(num, b'\x00')
            add_upd(num, b'\x01')

    # -- pure helpers (no graph needed) ------------------------------------

    def test_percentile(self):
        self.assertEqual(2.0, pb._percentile([2, 3, 3, 3, 3], 0))
        self.assertEqual(3.0, pb._percentile([2, 3, 3, 3, 3], 100))
        self.assertEqual(1.5, pb._percentile([1, 2], 50))
        self.assertEqual(0.0, pb._percentile([], 50))

    # -- graph-dependent ----------------------------------------------------

    async def test_node_degrees_and_classification(self):
        self.prepare_graph()
        degrees = pb.node_degrees(self.cdb)
        self.assertEqual(2, degrees[node('a')])  # channels 3, 6
        self.assertEqual(3, degrees[node('b')])  # channels 1, 2, 3
        self.assertEqual(5, len(degrees))

        cfg = pb.BenchmarkConfig()
        tiers, thresholds = pb.classify_nodes(self.cdb, cfg)
        self.assertEqual(5, len(tiers))
        # the least-connected node must not be ranked above a better-connected one
        self.assertNotEqual(pb.NodeTier.WELL, tiers[node('a')])
        self.assertEqual(pb.NodeTier.WELL, tiers[node('b')])

    async def test_select_sources_picks_best_connected(self):
        self.prepare_graph()
        cfg = pb.BenchmarkConfig(num_sources=2)
        tiers, _ = pb.classify_nodes(self.cdb, cfg)
        sources = pb.select_sources(self.cdb, cfg, tiers)
        self.assertEqual(2, len(sources))
        self.assertNotIn(node('a'), sources)  # degree 2, weakest

    async def test_min_feasible_hops(self):
        self.prepare_graph()
        amt = 1_000_000
        now = 0
        self.assertEqual(0, pb.min_feasible_hops(self.cdb, source=node('a'), dest=node('a'), amount_msat=amt, now=now))
        self.assertEqual(1, pb.min_feasible_hops(self.cdb, source=node('a'), dest=node('b'), amount_msat=amt, now=now))
        self.assertEqual(2, pb.min_feasible_hops(self.cdb, source=node('a'), dest=node('e'), amount_msat=amt, now=now))
        self.assertEqual(2, pb.min_feasible_hops(self.cdb, source=node('a'), dest=node('c'), amount_msat=amt, now=now))
        # unknown node is unreachable
        self.assertIsNone(pb.min_feasible_hops(self.cdb, source=node('a'), dest=node('z'), amount_msat=amt, now=now))

    async def test_min_feasible_hops_respects_htlc_minimum(self):
        self.prepare_graph()
        # all edges require htlc_minimum_msat=250; below that, no edge is usable
        self.assertIsNone(pb.min_feasible_hops(self.cdb, source=node('a'), dest=node('e'), amount_msat=100, now=0))

    async def test_route_fee_msat(self):
        self.prepare_graph()
        amount_msat = 100_000
        route = self.path_finder.find_route(nodeA=node('a'), nodeB=node('e'), invoice_amount_msat=amount_msat)
        self.assertIsNotNone(route)
        self.assertEqual(2, len(route))  # A -> B -> E
        # only the B->E hop charges us: base 100 + 150ppm * 100_000/1e6 = 115 msat
        self.assertEqual(115, pb.route_fee_msat(route, amount_msat_for_dest=amount_msat))

    async def test_measure_route_optimal_on_clean_graph(self):
        self.prepare_graph()
        cfg = pb.BenchmarkConfig()
        budget = pb._budget_for_amount(cfg, 100, self.config)
        r = pb.measure_route(
            path_finder=self.path_finder, channel_db=self.cdb,
            source=node('a'), dest=node('e'), dest_tier=pb.NodeTier.WELL,
            amount_sat=100, budget=budget, now=0)
        self.assertTrue(r.found)
        self.assertTrue(r.within_budget)
        self.assertEqual(2, r.num_hops)
        self.assertEqual(2, r.ideal_hops)
        self.assertEqual(0, r.excess_hops)  # router is optimal here
        self.assertTrue(r.usable)

    async def test_measure_route_no_route(self):
        self.prepare_graph()
        cfg = pb.BenchmarkConfig()
        budget = pb._budget_for_amount(cfg, 100, self.config)
        r = pb.measure_route(
            path_finder=self.path_finder, channel_db=self.cdb,
            source=node('a'), dest=node('z'), dest_tier=pb.NodeTier.POORLY,
            amount_sat=100, budget=budget, now=0)
        self.assertFalse(r.found)
        self.assertFalse(r.usable)
        self.assertIsNone(r.num_hops)
        self.assertIsNone(r.ideal_hops)

    # -- snapshot persist + reload (exercises the runner's load path) -------

    def _chan_ann_payload(self, num, n1, n2):
        ch = constants.net.rev_genesis_bytes()
        a, b = sorted([node(n1), node(n2)])
        raw = encode_msg("channel_announcement", len=0, features=b'', chain_hash=ch,
                         short_channel_id=channel(num), node_id_1=a, node_id_2=b,
                         bitcoin_key_1=a, bitcoin_key_2=b)
        _, payload = decode_msg(raw)
        payload['raw'] = raw
        return payload

    def _chan_upd_payload(self, num, channel_flags):
        ch = constants.net.rev_genesis_bytes()
        raw = encode_msg("channel_update", short_channel_id=channel(num),
                         channel_flags=channel_flags, message_flags=b'\x01',
                         cltv_expiry_delta=10, htlc_minimum_msat=250, htlc_maximum_msat=10**9,
                         fee_base_msat=100, fee_proportional_millionths=150,
                         chain_hash=ch, timestamp=1)
        _, payload = decode_msg(raw)
        payload['raw'] = raw
        return payload

    async def test_persist_and_reload_snapshot(self):
        """Persist a gossip_db, then reload it via ChannelDB.load_data() exactly
        as run_benchmark.py does, and benchmark the reloaded graph."""
        # 1) build & persist a snapshot (real encoded messages so raw is stored)
        class fake_network:
            config = self.config
            asyncio_loop = util.get_asyncio_loop()
            trigger_callback = lambda *args: None
            register_callback = lambda *args: None
            interface = None

        writer = ChannelDB(fake_network())
        writer.data_loaded.set()
        chans = [(1, 'b', 'c'), (2, 'b', 'e'), (3, 'a', 'b'), (4, 'c', 'd'),
                 (5, 'd', 'e'), (6, 'a', 'd'), (7, 'c', 'e')]
        for num, n1, n2 in chans:
            writer.add_channel_announcements(self._chan_ann_payload(num, n1, n2), trusted=True)
        for num in range(1, 8):
            writer.add_channel_update(self._chan_upd_payload(num, b'\x00'), verify=False)
            writer.add_channel_update(self._chan_upd_payload(num, b'\x01'), verify=False)
        await writer._db_save_node_addresses([])  # FIFO drain so @sql writes land
        writer.stop()
        await writer.stopped_event.wait()

        # 2) reload from disk on the DB worker thread, like the runner does
        reader = ChannelDB(fake_network())
        self.cdb = reader  # let tearDown stop it
        await reader.load_data()
        await reader.data_loaded.wait()
        self.assertEqual(7, reader.num_channels)
        self.assertEqual(14, reader.num_policies)

        # 3) benchmark the reloaded graph
        path_finder = lnrouter.LNPathFinder(reader)
        cfg = pb.BenchmarkConfig(amounts_sat=(10_000,), num_sources=1, dests_per_tier=10, seed=0)
        results, summaries = pb.run_benchmark(
            channel_db=reader, path_finder=path_finder, config=self.config, cfg=cfg, now=0)
        self.assertTrue(results)
        self.assertEqual(100.0, summaries[0].success_pct)

    async def test_run_benchmark_end_to_end(self):
        self.prepare_graph()
        cfg = pb.BenchmarkConfig(amounts_sat=(1_000, 10_000), num_sources=1, dests_per_tier=10, seed=0)
        results, summaries = pb.run_benchmark(
            channel_db=self.cdb, path_finder=self.path_finder, config=self.config, cfg=cfg, now=0)
        self.assertTrue(results)
        self.assertEqual(2, len(summaries))
        for s in summaries:
            # fully-connected feasible graph: every payment should route within budget
            self.assertEqual(0.0, s.no_route_pct)
            self.assertEqual(100.0, s.success_pct)
            self.assertEqual(0.0, s.fail_pct)
            self.assertEqual(0.0, s.mean_excess_hops)  # router is optimal here
            self.assertLess(s.mean_fee_rate, 0.05)
        # smoke-test the formatters
        self.assertIn("amount_sat", pb.format_summary_table(summaries))
        self.assertEqual(len(pb.RESULT_CSV_HEADER), len(pb.result_to_csv_row(results[0])))
        self.assertEqual(len(pb.SUMMARY_CSV_HEADER), len(pb.summary_to_csv_row(summaries[0])))

    # -- liquidity model ----------------------------------------------------

    def _add_htlc_max(self, num, htlc_maximum_msat):
        """Re-publish both directions of channel `num` with an htlc_maximum_msat
        so the LiquidityModel can derive a capacity from it. timestamp must be
        >60s past the original (0) or add_channel_update rejects it as a stale
        re-broadcast (channel_db.py: `timestamp <= old_policy.timestamp + 60`)."""
        for channel_flags in (b'\x00', b'\x01'):
            self.cdb.add_channel_update({
                'short_channel_id': channel(num), 'message_flags': b'\x01',
                'channel_flags': channel_flags, 'cltv_expiry_delta': 10,
                'htlc_minimum_msat': 250, 'htlc_maximum_msat': htlc_maximum_msat,
                'fee_base_msat': 100, 'fee_proportional_millionths': 150,
                'chain_hash': BitcoinTestnet.rev_genesis_bytes(), 'timestamp': 1000,
            }, verify=False)

    async def test_liquidity_model_capacity_and_direction(self):
        self.prepare_graph()
        # channel 3 is A(node 'a')<->B(node 'b'); give it a known htlc_max => capacity
        self._add_htlc_max(3, 2_000_000)
        lcfg = pb.LiquidityConfig(default_capacity_sat=5_000_000, balance_seed=1)
        model = pb.LiquidityModel(self.cdb, lcfg=lcfg, now=0)
        ci = self.cdb.get_channel_info(channel(3))
        fwd = model.available_msat(short_channel_id=channel(3), start_node=ci.node1_id)
        bwd = model.available_msat(short_channel_id=channel(3), start_node=ci.node2_id)
        # the two directional balances of a channel sum to its capacity (htlc_max)
        self.assertEqual(2_000_000, fwd + bwd)
        self.assertTrue(0 <= fwd <= 2_000_000)
        # a channel with no htlc_max uses the configured default capacity
        ci1 = self.cdb.get_channel_info(channel(1))
        f1 = model.available_msat(short_channel_id=channel(1), start_node=ci1.node1_id)
        b1 = model.available_msat(short_channel_id=channel(1), start_node=ci1.node2_id)
        self.assertEqual(5_000_000 * 1000, f1 + b1)

    async def test_liquidity_model_is_deterministic(self):
        self.prepare_graph()
        self._add_htlc_max(3, 2_000_000)
        a = pb.LiquidityModel(self.cdb, lcfg=pb.LiquidityConfig(balance_seed=7), now=0)
        b = pb.LiquidityModel(self.cdb, lcfg=pb.LiquidityConfig(balance_seed=7), now=0)
        ci = self.cdb.get_channel_info(channel(3))
        self.assertEqual(
            a.available_msat(short_channel_id=channel(3), start_node=ci.node1_id),
            b.available_msat(short_channel_id=channel(3), start_node=ci.node1_id))
        # different seed should (almost surely) give a different split
        c = pb.LiquidityModel(self.cdb, lcfg=pb.LiquidityConfig(balance_seed=8), now=0)
        self.assertNotEqual(
            a.available_msat(short_channel_id=channel(3), start_node=ci.node1_id),
            c.available_msat(short_channel_id=channel(3), start_node=ci.node1_id))

    async def test_liquidity_balance_beta_is_bimodal(self):
        """A Beta(b,b) prior with b<1 should pile balance mass toward the channel
        ends (depleted channels), while b==1 stays uniform. Sample channel 3's
        node1 share across many balance_seeds and compare tail mass."""
        self.prepare_graph()
        cap = 2_000_000
        self._add_htlc_max(3, cap)
        ci = self.cdb.get_channel_info(channel(3))

        def shares(beta):
            out = []
            for s in range(300):
                m = pb.LiquidityModel(
                    self.cdb, lcfg=pb.LiquidityConfig(balance_seed=s, balance_beta=beta), now=0)
                fwd = m.available_msat(short_channel_id=channel(3), start_node=ci.node1_id)
                out.append(fwd / cap)
            return out

        def tail_frac(xs):  # fraction landing in the outer 20% (near-empty/near-full)
            return sum(1 for x in xs if x < 0.1 or x > 0.9) / len(xs)

        bimodal = shares(0.3)
        uniform = shares(1.0)
        # bimodal should put markedly more mass in the tails than uniform (~0.2)
        self.assertGreater(tail_frac(bimodal), tail_frac(uniform))
        self.assertGreater(tail_frac(bimodal), 0.4)
        # mean share stays ~0.5 for the symmetric prior (it changes variance, not mean)
        self.assertAlmostEqual(0.5, statistics.mean(bimodal), delta=0.1)

    async def test_route_carried_amounts(self):
        self.prepare_graph()
        amount_msat = 100_000
        route = self.path_finder.find_route(nodeA=node('a'), nodeB=node('e'), invoice_amount_msat=amount_msat)
        self.assertEqual(2, len(route))  # A -> B -> E
        carried = pb.route_carried_amounts(route, amount_msat)
        # last hop delivers the invoice amount; first hop carries that plus the B->E fee (115)
        self.assertEqual(amount_msat, carried[1])
        self.assertEqual(amount_msat + 115, carried[0])

    class _FullLiquidity:
        """Stand-in model where every channel can forward any amount."""
        def can_forward(self, **kw):
            return True

    class _NoLiquidity:
        """Stand-in model where no channel can forward anything."""
        def can_forward(self, **kw):
            return False

    async def test_simulate_success_first_attempt_when_liquid(self):
        self.prepare_graph()
        cfg = pb.BenchmarkConfig()
        budget = pb._budget_for_amount(cfg, 100, self.config)
        r = pb.simulate_payment_with_liquidity(
            path_finder=self.path_finder, channel_db=self.cdb, model=self._FullLiquidity(),
            source=node('a'), dest=node('e'), dest_tier=pb.NodeTier.WELL,
            amount_sat=100, budget=budget, lcfg=pb.LiquidityConfig(), now=0)
        self.assertEqual(pb.OUTCOME_SUCCESS, r.outcome)
        self.assertEqual(1, r.attempts_used)
        self.assertEqual(2, r.num_hops)
        self.assertTrue(r.success)

    async def test_simulate_liquidity_exhausted_when_dry(self):
        self.prepare_graph()
        cfg = pb.BenchmarkConfig()
        budget = pb._budget_for_amount(cfg, 100, self.config)
        # Generous time budget so the small graph's routes are *exhausted* (every
        # hop is dry, so each find_route gets pruned away) before the clock runs
        # out -> the real NoPathFound-after-retries outcome.
        r = pb.simulate_payment_with_liquidity(
            path_finder=self.path_finder, channel_db=self.cdb, model=self._NoLiquidity(),
            source=node('a'), dest=node('e'), dest_tier=pb.NodeTier.WELL,
            amount_sat=100, budget=budget,
            lcfg=pb.LiquidityConfig(payment_timeout_sec=1e9, hop_latency_sec=1.0), now=0)
        self.assertEqual(pb.OUTCOME_LIQUIDITY_EXHAUSTED, r.outcome)
        # at least one retry happened before every route was pruned away
        self.assertGreaterEqual(r.attempts_used, 2)
        self.assertGreater(r.time_used_sec, 0.0)
        self.assertFalse(r.success)

    async def test_simulate_times_out_when_budget_too_small(self):
        """If the per-hop latency makes even the first failed attempt exceed the
        time budget, the payment times out instead of rerouting -- the real
        PAYMENT_TIMEOUT give-up. Uses a model that fails a downstream hop so a
        retry is triggered, but a tiny budget so there is no time to retry."""
        self.prepare_graph()
        cfg = pb.BenchmarkConfig()
        budget = pb._budget_for_amount(cfg, 100, self.config)
        r = pb.simulate_payment_with_liquidity(
            path_finder=self.path_finder, channel_db=self.cdb, model=self._NoLiquidity(),
            source=node('a'), dest=node('e'), dest_tier=pb.NodeTier.WELL,
            amount_sat=100, budget=budget,
            lcfg=pb.LiquidityConfig(payment_timeout_sec=1.0, hop_latency_sec=100.0), now=0)
        self.assertEqual(pb.OUTCOME_TIMED_OUT, r.outcome)
        self.assertEqual(1, r.attempts_used)  # gave up after the first failure
        self.assertGreater(r.time_used_sec, 1.0)
        self.assertFalse(r.success)

    async def test_simulate_reroutes_around_failing_channel(self):
        """A->E has two 2-hop paths (A-B-E via ch2, or A-D-E). Starve the second
        (non-first) hop the router picks; the retry loop records a cannot_send
        hint and the next find_route takes the alternative and succeeds. The
        first hop is exempt (the router ignores its liquidity), so we deliberately
        starve a downstream hop."""
        self.prepare_graph()
        cfg = pb.BenchmarkConfig()
        budget = pb._budget_for_amount(cfg, 100, self.config)
        amount_msat = 100 * 1000

        # find the second hop the router chooses, then starve exactly that
        # channel+direction and nothing else.
        route = self.path_finder.find_route(nodeA=node('a'), nodeB=node('e'), invoice_amount_msat=amount_msat)
        self.assertEqual(2, len(route))
        starved_scid = route[1].short_channel_id
        starved_start = route[1].start_node

        class _StarveOne:
            def can_forward(self, *, short_channel_id, start_node, amount_msat):
                return not (short_channel_id == starved_scid and start_node == starved_start)

        r = pb.simulate_payment_with_liquidity(
            path_finder=self.path_finder, channel_db=self.cdb, model=_StarveOne(),
            source=node('a'), dest=node('e'), dest_tier=pb.NodeTier.WELL,
            amount_sat=100, budget=budget, lcfg=pb.LiquidityConfig(), now=0)
        # outcome==SUCCESS with >=2 attempts proves the loop rerouted around the
        # starved channel (attempt 1 hit it, a later attempt avoided it). The
        # default time budget (120s) leaves ample room for the one reroute.
        self.assertEqual(pb.OUTCOME_SUCCESS, r.outcome)
        self.assertGreaterEqual(r.attempts_used, 2)

    async def test_run_benchmark_with_liquidity_end_to_end(self):
        self.prepare_graph()
        # generous capacities everywhere so most payments succeed
        for num in range(1, 8):
            self._add_htlc_max(num, 50_000_000_000)
        cfg = pb.BenchmarkConfig(amounts_sat=(1_000, 10_000), num_sources=1, dests_per_tier=10, seed=0)
        lcfg = pb.LiquidityConfig(balance_seed=0)
        results, summaries = pb.run_benchmark_with_liquidity(
            channel_db=self.cdb, path_finder=self.path_finder, config=self.config,
            cfg=cfg, lcfg=lcfg, now=0)
        self.assertTrue(results)
        self.assertEqual(2, len(summaries))
        for s in summaries:
            # every result falls into exactly one outcome bucket
            self.assertEqual(
                s.attempts,
                s.success + s.no_route + s.over_budget + s.liquidity_exhausted + s.timed_out)
            self.assertAlmostEqual(
                100.0,
                (s.success_pct + s.no_route_pct + s.over_budget_pct
                 + s.liquidity_exhausted_pct + s.timed_out_pct),
                places=5)
        for r in results:
            self.assertIn(r.outcome, pb.LIQUIDITY_OUTCOMES)
        # smoke-test the formatters
        self.assertIn("success%", pb.format_liquidity_summary_table(summaries))
        self.assertEqual(len(pb.LIQUIDITY_RESULT_CSV_HEADER), len(pb.liquidity_result_to_csv_row(results[0])))
        self.assertEqual(len(pb.LIQUIDITY_SUMMARY_CSV_HEADER), len(pb.liquidity_summary_to_csv_row(summaries[0])))
