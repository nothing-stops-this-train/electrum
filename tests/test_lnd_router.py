from typing import Optional

from electrum import util, lnrouter
from electrum.channel_db import ChannelDB
from electrum.simple_config import SimpleConfig
from electrum.constants import BitcoinTestnet
from electrum.util import bfh
from electrum.lnutil import ShortChannelID, LnFeatures

from electrum.logging import Logger
from electrum.plugins.lnd_pathfinder.lnd_router import LndPathFinder
from electrum.plugins.lnd_pathfinder.lnd_pathfinder import LndPathfinderPlugin
from electrum.plugins.lnd_pathfinder.mission_control import MissionControlStore, BimodalEstimator

from . import ElectrumTestCase


def channel(number: int) -> ShortChannelID:
    return ShortChannelID(bfh(format(number, '016x')))


def node(character: str) -> bytes:
    return b'\x02' + f'{character}'.encode() * 32


def alias(character: str) -> bytes:
    return (character * 8).encode('utf-8')


def node_features() -> bytes:
    return (LnFeatures(0) | LnFeatures.VAR_ONION_OPT).to_bytes(8, 'big')


class Test_LndRouter(ElectrumTestCase):
    TESTNET = True

    cdb = None  # type: Optional[ChannelDB]

    def setUp(self):
        super().setUp()
        self.config = SimpleConfig({'electrum_path': self.electrum_path})

    async def asyncTearDown(self):
        if self.cdb:
            self.cdb.stop()
            await self.cdb.stopped_event.wait()
        await super().asyncTearDown()

    def _new_cdb(self):
        class fake_network:
            config = self.config
            asyncio_loop = util.get_asyncio_loop()
            trigger_callback = lambda *args: None
            register_callback = lambda *args: None
            interface = None
        fake_network.channel_db = ChannelDB(fake_network())
        fake_network.channel_db.data_loaded.set()
        self.cdb = fake_network.channel_db
        return self.cdb

    def _add_channel(self, num, na, nb):
        n1, n2 = sorted([node(na), node(nb)])  # channel_db requires node_id_1 < node_id_2
        self.cdb.add_channel_announcements({
            'node_id_1': n1, 'node_id_2': n2,
            'bitcoin_key_1': n1, 'bitcoin_key_2': n2,
            'short_channel_id': channel(num),
            'chain_hash': BitcoinTestnet.rev_genesis_bytes(),
            'len': 0, 'features': b'',
        }, trusted=True)

    def _add_nodes(self, chars):
        for char in chars:
            self.cdb.add_node_announcements({
                'node_id': node(char), 'alias': alias(char), 'addresses': [],
                'features': node_features(), 'timestamp': 0,
            })

    def _add_update(self, num, channel_flags, *, fee_base_msat=100, fee_ppm=150,
                    htlc_minimum_msat=250, cltv=10):
        payload = {
            'short_channel_id': channel(num), 'message_flags': b'\x00',
            'channel_flags': channel_flags, 'cltv_expiry_delta': cltv,
            'htlc_minimum_msat': htlc_minimum_msat, 'fee_base_msat': fee_base_msat,
            'fee_proportional_millionths': fee_ppm,
            'chain_hash': BitcoinTestnet.rev_genesis_bytes(), 'timestamp': 0,
        }
        self.cdb.add_channel_update(payload, verify=False)

    def _both_dirs(self, num, **kw):
        self._add_update(num, b'\x00', **kw)
        self._add_update(num, b'\x01', **kw)

    # ----------------------------------------------------------------------

    def _diamond(self):
        """A-B-C-D-E diamond (same as test_lnrouter)."""
        self._new_cdb()
        for num, n1, n2 in [(1, 'b', 'c'), (2, 'b', 'e'), (3, 'a', 'b'),
                            (4, 'c', 'd'), (5, 'd', 'e'), (6, 'a', 'd'), (7, 'c', 'e')]:
            self._add_channel(num, n1, n2)
        self._add_nodes('abcde')
        for num in range(1, 8):
            self._both_dirs(num)

    async def test_finds_valid_route(self):
        self._diamond()
        finder = LndPathFinder(self.cdb)
        route = finder.find_route(nodeA=node('a'), nodeB=node('e'), invoice_amount_msat=100_000)
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 0)
        # route chains together and ends at the destination
        self.assertEqual(node('a'), route[0].start_node)
        self.assertEqual(node('e'), route[-1].end_node)
        prev_end = route[0].start_node
        for edge in route:
            self.assertEqual(prev_end, edge.start_node)
            prev_end = edge.end_node
        # A->E shortest is 2 hops on this graph
        self.assertEqual(2, len(route))

    async def test_attempt_cost_prefers_fewer_hops(self):
        """Cheap-but-long path vs slightly-pricier short path.

        Native (no per-hop cost) takes the long cheap path; the LND finder's
        per-hop AttemptCost makes it take the short path instead. This directly
        encodes the issue #10443 hypothesis.
        """
        self._new_cdb()
        # short path: S -X- T   (one expensive intermediate edge X->T)
        # long path:  S -P- Q -R- T  (three cheap intermediate edges)
        self._add_channel(1, 's', 'x')
        self._add_channel(2, 'x', 't')
        self._add_channel(3, 's', 'p')
        self._add_channel(4, 'p', 'q')
        self._add_channel(5, 'q', 'r')
        self._add_channel(6, 'r', 't')
        self._add_nodes('sxtpqr')
        # S's own edges are free (first hop); make X->T pricey, long path cheap.
        self._both_dirs(1, fee_base_msat=100, fee_ppm=0)
        self._both_dirs(2, fee_base_msat=50_000, fee_ppm=0)   # expensive short hop
        self._both_dirs(3, fee_base_msat=100, fee_ppm=0)
        self._both_dirs(4, fee_base_msat=100, fee_ppm=0)
        self._both_dirs(5, fee_base_msat=100, fee_ppm=0)
        self._both_dirs(6, fee_base_msat=100, fee_ppm=0)

        amount = 1_000_000
        native = lnrouter.LNPathFinder(self.cdb)
        lnd = LndPathFinder(self.cdb)
        native_route = native.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=amount)
        lnd_route = lnd.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=amount)

        self.assertIsNotNone(native_route)
        self.assertIsNotNone(lnd_route)
        # native minimizes fee -> takes the 4-edge cheap path
        self.assertEqual(4, len(native_route))
        # LND's AttemptCost -> takes the 2-edge short path despite higher fee
        self.assertEqual(2, len(lnd_route))
        self.assertEqual(node('t'), lnd_route[-1].end_node)

    async def test_prunes_below_htlc_minimum(self):
        self._new_cdb()
        self._add_channel(1, 's', 't')
        self._add_nodes('st')
        self._both_dirs(1, htlc_minimum_msat=1_000_000)
        finder = LndPathFinder(self.cdb)
        # amount under htlc_minimum -> no usable edge -> no route
        route = finder.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=500_000)
        self.assertIsNone(route)
        # amount at/above htlc_minimum -> route found
        route = finder.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=2_000_000)
        self.assertIsNotNone(route)

    async def test_prunes_disabled_channel(self):
        self._new_cdb()
        # only path S->T is via chan 1; disable it -> no route
        self._add_channel(1, 's', 't')
        self._add_nodes('st')
        # channel_flags bit 0x02 = FLAG_DISABLE
        self._add_update(1, b'\x02')
        self._add_update(1, b'\x03')
        finder = LndPathFinder(self.cdb)
        route = finder.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=100_000)
        self.assertIsNone(route)

    async def test_mission_control_reroutes_after_failure(self):
        """A recorded failure on the otherwise-preferred channel raises its cost
        (low probability -> attempt_cost/probability blows up), so the bimodal
        MissionControl-driven finder reroutes around it."""
        import time as _time
        self._new_cdb()
        # two parallel 2-hop paths S->A->T (chan 1,2) and S->B->T (chan 3,4)
        self._add_channel(1, 's', 'a')
        self._add_channel(2, 'a', 't')
        self._add_channel(3, 's', 'b')
        self._add_channel(4, 'b', 't')
        self._add_nodes('satb')
        # make the A path cheaper so it's preferred absent any liquidity info
        self._both_dirs(1, fee_base_msat=10, fee_ppm=0)
        self._both_dirs(2, fee_base_msat=10, fee_ppm=0)
        self._both_dirs(3, fee_base_msat=100, fee_ppm=0)
        self._both_dirs(4, fee_base_msat=100, fee_ppm=0)

        amount = 1_000_000
        store = MissionControlStore()
        est = BimodalEstimator(self.cdb, store=store)
        finder = LndPathFinder(self.cdb, probability_func=est)

        # without history: the cheaper A path (via node 'a') is chosen
        route = finder.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=amount)
        self.assertEqual([node('a')], [e.end_node for e in route][:1])

        # record a failure forwarding `amount` over the a->t channel (chan 2)
        store.report_failure(channel(2), node('a'), node('t'), amount, now=_time.time())
        # now the A path is improbable -> finder reroutes via node 'b'
        route2 = finder.find_route(nodeA=node('s'), nodeB=node('t'), invoice_amount_msat=amount)
        self.assertEqual([node('b')], [e.end_node for e in route2][:1])
        self.assertEqual(node('t'), route2[-1].end_node)

    async def test_plugin_compare_and_logs(self):
        """End-to-end of the plugin's get_route_for_payment: runs both finders,
        returns a usable route, and appends a route_compare JSONL record."""
        import json
        import os
        import threading

        self._diamond()
        native = lnrouter.LNPathFinder(self.cdb)
        log_path = os.path.join(self.electrum_path, "lnd_pathfinder.jsonl")

        cdb = self.cdb
        config = self.config

        class FakeStorage:
            path = os.path.join(self.electrum_path, "mywallet")

        class FakeWallet:
            storage = FakeStorage()
            def basename(self):
                return "mywallet"

        class FakeLngossip:
            def get_sync_progress_estimate(self):
                return 1.0

        class FakeKeypair:
            pubkey = node('a')

        class FakeNetwork:
            path_finder = native
            lngossip = FakeLngossip()

        fake_wallet = FakeWallet()

        class FakeLnworker:
            node_keypair = FakeKeypair()
            network = FakeNetwork()
            channel_db = cdb
            wallet = fake_wallet
        FakeLnworker.config = config
        lnworker = FakeLnworker()

        # build the plugin without the plugin-manager machinery
        plugin = LndPathfinderPlugin.__new__(LndPathfinderPlugin)
        Logger.__init__(plugin)
        plugin._contexts = {}
        plugin._setup_lock = threading.Lock()
        plugin._callback_registered = False
        plugin._wallets = {fake_wallet: {"log_path": log_path, "lock": threading.Lock()}}

        route = plugin.get_route_for_payment(
            lnworker, b'\xab' * 32, node('e'), 100_000, None, {}, {})

        self.assertIsNotNone(route)
        self.assertEqual(node('e'), route[-1].end_node)

        with open(log_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(1, len(lines))
        rec = json.loads(lines[0])
        self.assertEqual("route_compare", rec["event"])
        self.assertEqual("ab" * 32, rec["payment_hash"])
        self.assertEqual("success", rec["native"]["outcome"])
        self.assertEqual("success", rec["lnd"]["outcome"])
        self.assertIn(rec["chosen"], ("native", "lnd"))
