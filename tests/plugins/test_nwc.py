import asyncio
import json
import time
from unittest.mock import MagicMock

from electrum_aionostr.event import Event as nEvent
from electrum_aionostr.key import PrivateKey

from electrum.invoices import PR_PAID, PR_UNPAID
from electrum.simple_config import SimpleConfig
from electrum.util import OldTaskGroup

import electrum.plugins.nwc  # noqa: F401  # registers the NWC_RELAY config var
from electrum.plugins.nwc.nwcserver import NWCServerPlugin, NWCServer

from .. import ElectrumTestCase


class TestNWCServer(ElectrumTestCase):

    def setUp(self):
        super().setUp()
        self.config = SimpleConfig({'electrum_path': self.electrum_path})
        self.plugin = NWCServerPlugin(MagicMock(), self.config, 'nwc')
        self.plugin.connections = {}
        self.wallet = MagicMock()
        self.server = NWCServer(self.config, self.wallet, self.plugin.connections)
        self.responses = []  # type: list[tuple[str, dict, str]]

        async def record_response(to_pubkey_hex, content, response_event_id, *, add_tags=None):
            self.responses.append((to_pubkey_hex, json.loads(content), response_event_id))
        self.server.send_encrypted_response = record_response

    def tearDown(self):
        self.server.unregister_callbacks()
        self.plugin.close()
        super().tearDown()

    def _create_connection(self, name: str, **kwargs) -> str:
        """Creates a connection and returns the client pubkey. kwargs as in create_connection."""
        kwargs.setdefault('daily_limit_sat', None)
        kwargs.setdefault('valid_for_sec', None)
        connection_string = self.plugin.create_connection(name=name, **kwargs)
        client_pubkey = self.plugin.get_client_pubkey_from_connection_string(connection_string)
        self._client_secrets = getattr(self, '_client_secrets', {})
        import urllib.parse
        query = urllib.parse.urlparse(connection_string).query
        secret_hex = urllib.parse.parse_qs(query)['secret'][0]
        self._client_secrets[client_pubkey] = PrivateKey(raw_secret=bytes.fromhex(secret_hex))
        return client_pubkey

    def _make_request_event(self, client_pubkey: str, method: str, params=None) -> nEvent:
        """Crafts an encrypted NIP-47 request event as the client would send it."""
        client_secret = self._client_secrets[client_pubkey]
        our_secret_hex = self.plugin.connections[client_pubkey]['our_secret']
        our_pubkey = PrivateKey(raw_secret=bytes.fromhex(our_secret_hex)).public_key.hex()
        content = json.dumps({'method': method, 'params': params or {}})
        encrypted_content = client_secret.encrypt_message(content, our_pubkey)
        event = nEvent(
            pubkey=client_pubkey,
            content=encrypted_content,
            created_at=int(time.time()),
            kind=NWCServer.REQUEST_EVENT_KIND,
            tags=[['p', our_pubkey]],
        )
        return event.sign(client_secret.hex())

    async def _request(self, client_pubkey: str, method: str, params=None) -> None:
        event = self._make_request_event(client_pubkey, method, params)
        await self.server._handle_single_request(event)

    def _last_response(self) -> dict:
        assert self.responses, "no response was sent"
        return self.responses[-1][1]

    def test_create_connection_receive_only(self):
        client_pubkey = self._create_connection('pos', receive_only=True)
        conn = self.plugin.connections[client_pubkey]
        self.assertTrue(conn['receive_only'])
        self.assertTrue(self.server.is_restricted(client_pubkey))
        self.assertTrue(self.server.is_receive_only(client_pubkey))
        connections = self.plugin.list_connections()
        self.assertTrue(connections['pos']['receive_only'])
        # a normal connection has no flag set
        client_pubkey2 = self._create_connection('normal')
        self.assertNotIn('receive_only', self.plugin.connections[client_pubkey2])
        self.assertFalse(self.server.is_restricted(client_pubkey2))
        self.assertFalse(connections.get('normal', {}).get('receive_only', False))
        # a receive-only connection cannot have a spending budget
        with self.assertRaises(ValueError):
            self.plugin.create_connection(
                name='pos2', daily_limit_sat=1000, valid_for_sec=None, receive_only=True)

    def test_get_supported_methods(self):
        restricted = self._create_connection('pos', receive_only=True)
        budget_zero = self._create_connection('viewer', daily_limit_sat=0)
        normal = self._create_connection('normal')
        self.assertEqual(NWCServer.RECEIVE_ONLY_METHODS, self.server.get_supported_methods(restricted))
        self.assertEqual(
            NWCServer.SUPPORTED_METHODS - NWCServer.SUPPORTED_SPENDING_METHODS,
            self.server.get_supported_methods(budget_zero))
        self.assertEqual(NWCServer.SUPPORTED_METHODS, self.server.get_supported_methods(normal))
        self.assertEqual(["payment_received"], self.server.get_supported_notifications(restricted))
        self.assertEqual(NWCServer.SUPPORTED_NOTIFICATIONS, self.server.get_supported_notifications(normal))

    async def test_restricted_connection_methods_rejected(self):
        client_pubkey = self._create_connection('pos', receive_only=True)
        self.server.handle_pay_invoice = MagicMock()
        self.server.handle_get_balance = MagicMock()
        self.server.handle_list_transactions = MagicMock()
        for method in ('pay_invoice', 'get_balance', 'list_transactions'):
            await self._request(client_pubkey, method)
            response = self._last_response()
            self.assertEqual("RESTRICTED", response['error']['code'], method)
            self.assertEqual(method, response['result_type'])
        self.server.handle_pay_invoice.assert_not_called()
        self.server.handle_get_balance.assert_not_called()
        self.server.handle_list_transactions.assert_not_called()

    async def test_restricted_connection_methods_allowed(self):
        client_pubkey = self._create_connection('pos', receive_only=True)
        for method in ('make_invoice', 'lookup_invoice', 'get_info'):
            handler_mock = MagicMock(return_value=None)
            setattr(self.server, f"handle_{method}", handler_mock)
            await self._request(client_pubkey, method)
            handler_mock.assert_called_once()

    async def test_normal_connection_can_pay(self):
        client_pubkey = self._create_connection('normal')
        self.server.handle_pay_invoice = MagicMock(return_value=None)
        await self._request(client_pubkey, 'pay_invoice', {'invoice': 'lnbc1...'})
        self.server.handle_pay_invoice.assert_called_once()

    async def test_budget_zero_connection(self):
        # regression: a connection with a budget of 0 sat cannot pay but can still use other methods
        client_pubkey = self._create_connection('viewer', daily_limit_sat=0)
        self.server.handle_pay_invoice = MagicMock()
        await self._request(client_pubkey, 'pay_invoice', {'invoice': 'lnbc1...'})
        self.assertEqual("RESTRICTED", self._last_response()['error']['code'])
        self.server.handle_pay_invoice.assert_not_called()
        self.server.handle_get_balance = MagicMock(return_value=None)
        await self._request(client_pubkey, 'get_balance')
        self.server.handle_get_balance.assert_called_once()

    async def test_unknown_method_not_implemented(self):
        client_pubkey = self._create_connection('normal')
        await self._request(client_pubkey, 'signMessage')
        self.assertEqual("NOT_IMPLEMENTED", self._last_response()['error']['code'])

    async def test_expired_connection(self):
        client_pubkey = self._create_connection('short', valid_for_sec=1)
        self.plugin.connections[client_pubkey]['valid_until'] = int(time.time()) - 1
        await self._request(client_pubkey, 'get_info')
        self.assertEqual("UNAUTHORIZED", self._last_response()['error']['code'])
        self.assertNotIn(client_pubkey, self.plugin.connections)

    async def test_make_invoice(self):
        client_pubkey = self._create_connection('pos', receive_only=True)
        payment_hash = bytes.fromhex('11' * 32)
        self.wallet.create_request.return_value = payment_hash.hex()
        request_mock = MagicMock()
        request_mock.payment_hash = payment_hash
        request_mock.get_expiration_date.return_value = 5000
        self.wallet.get_request.return_value = request_mock
        lnaddr_mock = MagicMock()
        lnaddr_mock.paymenthash = payment_hash
        lnaddr_mock.date = 4000
        self.wallet.lnworker.get_bolt11_invoice.return_value = (lnaddr_mock, 'lnbc1testinvoice')
        event = self._make_request_event(client_pubkey, 'make_invoice')
        await self.server.handle_make_invoice(event, {'amount': 21000, 'description': 'coffee'})
        response = self._last_response()
        self.assertEqual('make_invoice', response['result_type'])
        self.assertEqual('lnbc1testinvoice', response['result']['invoice'])
        self.assertEqual(21000, response['result']['amount'])
        self.assertEqual(payment_hash.hex(), response['result']['payment_hash'])

    def _setup_incoming_request_mock(self, rhash: str):
        request_mock = MagicMock()
        request_mock.is_lightning.return_value = True
        request_mock.rhash = rhash
        request_mock.message = 'incoming test'
        request_mock.get_amount_msat.return_value = 5000
        request_mock.time = 1000
        request_mock.get_expiration_date.return_value = 2000
        self.wallet.get_invoice.return_value = None
        self.wallet.get_request.return_value = request_mock
        self.wallet.get_invoice_status.return_value = PR_UNPAID
        self.wallet.lnworker.get_bolt11_invoice.return_value = (None, 'lnbc1incoming')
        self.wallet.lnworker.get_payments.return_value = {}

    def _setup_outgoing_invoice_mock(self, rhash: str):
        invoice_mock = MagicMock()
        invoice_mock.is_lightning.return_value = True
        invoice_mock.rhash = rhash
        invoice_mock.message = 'outgoing test'
        invoice_mock.get_amount_msat.return_value = 7000
        invoice_mock.time = 1000
        invoice_mock.get_expiration_date.return_value = 2000
        invoice_mock.lightning_invoice = 'lnbc1outgoing'
        self.wallet.get_invoice.return_value = invoice_mock
        self.wallet.get_request.return_value = None
        self.wallet.get_invoice_status.return_value = PR_PAID
        self.wallet.lnworker.get_preimage_hex.return_value = '44' * 32
        self.wallet.lnworker.get_payments.return_value = {}

    async def test_lookup_invoice_restricted_incoming(self):
        client_pubkey = self._create_connection('pos', receive_only=True)
        rhash = '22' * 32
        self._setup_incoming_request_mock(rhash)
        event = self._make_request_event(client_pubkey, 'lookup_invoice')
        await self.server.handle_lookup_invoice(event, {'payment_hash': rhash})
        response = self._last_response()
        self.assertEqual('lookup_invoice', response['result_type'])
        self.assertEqual(rhash, response['result']['payment_hash'])
        self.assertEqual('incoming', response['result']['type'])
        self.assertEqual('pending', response['result']['state'])

    async def test_lookup_invoice_restricted_outgoing_not_found(self):
        client_pubkey = self._create_connection('pos', receive_only=True)
        rhash = '33' * 32
        self._setup_outgoing_invoice_mock(rhash)
        event = self._make_request_event(client_pubkey, 'lookup_invoice')
        await self.server.handle_lookup_invoice(event, {'payment_hash': rhash})
        response = self._last_response()
        self.assertEqual('NOT_FOUND', response['error']['code'])

    async def test_lookup_invoice_normal_outgoing(self):
        # regression: a normal connection can still look up outgoing payments
        client_pubkey = self._create_connection('normal')
        rhash = '33' * 32
        self._setup_outgoing_invoice_mock(rhash)
        event = self._make_request_event(client_pubkey, 'lookup_invoice')
        await self.server.handle_lookup_invoice(event, {'payment_hash': rhash})
        response = self._last_response()
        self.assertEqual('outgoing', response['result']['type'])
        self.assertEqual('44' * 32, response['result']['preimage'])

    async def test_get_info(self):
        restricted = self._create_connection('pos', receive_only=True)
        normal = self._create_connection('normal')
        self.wallet.lnworker.network.blockchain().height.return_value = 100
        self.wallet.lnworker.network.blockchain().get_hash.return_value = '00' * 32
        self.wallet.lnworker.node_keypair.pubkey.hex.return_value = '02' * 33
        for client_pubkey, expected_methods, expected_notifications in (
            (restricted, NWCServer.RECEIVE_ONLY_METHODS, ["payment_received"]),
            (normal, NWCServer.SUPPORTED_METHODS, NWCServer.SUPPORTED_NOTIFICATIONS),
        ):
            event = self._make_request_event(client_pubkey, 'get_info')
            await self.server.handle_get_info(event)
            response = self._last_response()
            self.assertEqual(expected_methods, set(response['result']['methods']))
            self.assertEqual(expected_notifications, response['result']['notifications'])

    def test_info_event_content(self):
        restricted = self._create_connection('pos', receive_only=True)
        normal = self._create_connection('normal')
        tags, content = self.server.get_info_event_content(restricted)
        self.assertEqual(NWCServer.RECEIVE_ONLY_METHODS, set(content.split(' ')))
        self.assertIn(['notifications', 'payment_received'], tags)
        tags, content = self.server.get_info_event_content(normal)
        self.assertEqual(NWCServer.SUPPORTED_METHODS, set(content.split(' ')))
        self.assertIn(['notifications', ' '.join(NWCServer.SUPPORTED_NOTIFICATIONS)], tags)

    async def test_notification_filtering(self):
        restricted = self._create_connection('pos', receive_only=True)
        normal = self._create_connection('normal')
        self.server.taskgroup = OldTaskGroup()
        self.server.manager = MagicMock()
        published = []  # (client_pubkey from p tag)

        async def fake_add_event(manager, *, kind, tags, content, private_key):
            published.append(tags[0][1])

        import electrum.plugins.nwc.nwcserver as nwcserver_module
        orig_add_event = nwcserver_module.aionostr._add_event
        nwcserver_module.aionostr._add_event = fake_add_event
        try:
            self.server.publish_notification_event({"notification_type": "payment_sent", "notification": {}})
            await asyncio.sleep(0.1)
            self.assertEqual([normal], published)
            published.clear()
            self.server.publish_notification_event({"notification_type": "payment_received", "notification": {}})
            await asyncio.sleep(0.1)
            self.assertEqual({restricted, normal}, set(published))
        finally:
            nwcserver_module.aionostr._add_event = orig_add_event
            await self.server.taskgroup.cancel_remaining()
