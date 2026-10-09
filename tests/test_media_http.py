import http.client
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from media_operations.adapters.http import (
    PublicHTTPClient, PublicNetworkPolicy, _PinnedHTTPConnection, _PinnedHTTPSConnection,
)
from media_operations.adapters.web_extract import WebExtractor
from media_operations.research_models import PageQuery
from tooling.errors import ClassifiedToolError
from tooling.registry import UnsafeRequestError
from tooling.result import ErrorCode


class Response:
    def __init__(self, body=b"hello", status=200, headers=None):
        self.status = status
        self.headers = headers or {"Content-Type": "text/plain; charset=utf-8"}
        self.stream = io.BytesIO(body)
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=Mock()))
        self.closed = False

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, size):
        return self.stream.read(size)

    def close(self):
        self.closed = True
        self.stream.close()


class MediaHTTPTests(unittest.TestCase):
    def client(self, responses, **options):
        client = PublicHTTPClient(resolver=Mock(return_value=["8.8.8.8"]), **options)
        connections = [Mock(sock=Mock(), getresponse=Mock(return_value=response)) for response in responses]
        client._connection = Mock(side_effect=connections)
        return client, connections

    def test_url_policy_rejects_local_credentials_and_unsupported_transports(self):
        policy = PublicNetworkPolicy()
        for url in ("file:///etc/passwd", "ftp://example.org/a", "https://u:p@example.org/a",
                    "http://localhost/a", "http://x.local/a", "http://127.0.0.1/a",
                    "http://10.0.0.1/a", "http://169.254.169.254/a", "http://[::1]/a",
                    "http://[::ffff:127.0.0.1]/a", "https://example.org:8000/a",
                    "https://example.org/a?access_token=secret", "https://example.org\\a",
                    "https://example.org/\nheader", "https://example.org:bad/a"):
            with self.subTest(url=url), self.assertRaises(UnsafeRequestError):
                policy.validate_url(url)

    def test_domain_allowlist_matches_whole_domain_boundaries(self):
        policy = PublicNetworkPolicy(["example.org"])
        for url in ("https://example.org/", "https://docs.example.org/a"):
            policy.validate_url(url)
        for url in ("https://notexample.org/", "https://example.org.attacker.com/"):
            with self.assertRaises(UnsafeRequestError):
                policy.validate_url(url)

    def test_all_dns_answers_must_be_public_and_no_connection_precedes_validation(self):
        client, _ = self.client([])
        for addresses in ([], ["8.8.8.8", "192.168.0.2"], ["::1"]):
            client.resolver.return_value = addresses
            with self.assertRaises(UnsafeRequestError):
                client.request("GET", "https://example.org/a")
        client._connection.assert_not_called()

    def test_validated_address_is_passed_to_connection_and_url_path_is_preserved(self):
        response = Response()
        client, connections = self.client([response])
        document = client.request("GET", "https://example.org/a?x=1", headers={"Accept": "text/plain"})
        self.assertEqual(client._connection.call_args.args[1], "8.8.8.8")
        self.assertEqual(client._connection.call_args.args[0].hostname, "example.org")
        connections[0].request.assert_called_once_with("GET", "/a?x=1", body=None, headers={
            "User-Agent": "AgentStudy-Media/1.0", "Accept-Encoding": "identity", "Accept": "text/plain",
        })
        self.assertEqual((document.body, document.charset), (b"hello", "utf-8"))
        self.assertTrue(response.closed)
        connections[0].close.assert_called_once()

    def test_pinned_socket_uses_validated_ip_and_https_uses_original_hostname(self):
        raw_socket, context = Mock(), Mock()
        with patch("media_operations.adapters.http.socket.create_connection", return_value=raw_socket) as connect:
            with patch("media_operations.adapters.http.ssl.create_default_context", return_value=context):
                connection = _PinnedHTTPSConnection("example.org", 443, "8.8.8.8", 5)
                connection.connect()
        connect.assert_called_once_with(("8.8.8.8", 443), 5)
        context.wrap_socket.assert_called_once_with(raw_socket, server_hostname="example.org")
        with patch("media_operations.adapters.http.socket.create_connection", return_value=raw_socket) as connect:
            _PinnedHTTPConnection("example.org", 80, "8.8.8.8", 5).connect()
        connect.assert_called_once_with(("8.8.8.8", 80), 5)

    def test_tls_handshake_failure_closes_raw_socket(self):
        raw_socket = Mock()
        with patch("media_operations.adapters.http.socket.create_connection", return_value=raw_socket):
            with patch("media_operations.adapters.http.ssl.create_default_context") as context:
                context.return_value.wrap_socket.side_effect = OSError("handshake")
                with self.assertRaises(OSError):
                    _PinnedHTTPSConnection("example.org", 443, "8.8.8.8", 5).connect()
        raw_socket.close.assert_called_once()

    def test_redirects_revalidate_dns_and_strip_authentication(self):
        responses = [Response(status=302, headers={"Location": "https://other.org/final"}), Response()]
        client, connections = self.client(responses)
        document = client.request("GET", "https://example.org/start", headers={
            "Authorization": "Bearer secret", "Cookie": "session=secret", "Accept": "text/plain",
        })
        self.assertEqual(document.url, "https://other.org/final")
        self.assertEqual([call.args[0] for call in client.resolver.call_args_list], ["example.org", "other.org"])
        headers = connections[1].request.call_args.kwargs["headers"]
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Cookie", headers)
        self.assertEqual(headers["Accept"], "text/plain")
        self.assertTrue(all(response.closed for response in responses))

    def test_private_redirect_and_dns_rebinding_are_rejected_before_second_connection(self):
        for target, answers in (("http://127.0.0.1/a", [["8.8.8.8"]]),
                                ("https://other.org/a", [["8.8.8.8"], ["10.0.0.2"]])):
            response = Response(status=302, headers={"Location": target})
            client, _ = self.client([response])
            client.resolver.side_effect = answers
            with self.assertRaises(UnsafeRequestError):
                client.request("GET", "https://example.org/a")
            self.assertEqual(client._connection.call_count, 1)
            self.assertTrue(response.closed)

    def test_post_redirect_and_redirect_limit_are_rejected(self):
        for method, options in (("POST", {}), ("GET", {"max_redirects": 0})):
            response = Response(status=307, headers={"Location": "https://other.org/a"})
            client, _ = self.client([response], **options)
            with self.assertRaises(UnsafeRequestError):
                client.request(method, "https://example.org/a", headers={"Authorization": "Bearer secret"})
            self.assertEqual(client._connection.call_count, 1)

    def test_read_size_is_bounded_and_truncation_is_explicit(self):
        for body, truncated in ((b"12345", False), (b"123456789", True)):
            response = Response(body)
            client, _ = self.client([response], max_bytes=5)
            document = client.request("GET", "https://example.org/a")
            self.assertEqual((document.body, document.truncated), (b"12345", truncated))

    def test_detached_response_socket_gets_timeout_and_is_closed(self):
        response = Response()
        client, connections = self.client([response])
        connections[0].sock = None
        client.request("GET", "https://example.org/a")
        response.fp.raw._sock.settimeout.assert_called()
        self.assertTrue(response.closed)

    def test_http_errors_and_compression_have_stable_codes_and_close_connections(self):
        cases = [(401, {}, ErrorCode.AUTHENTICATION_ERROR), (403, {}, ErrorCode.AUTHENTICATION_ERROR),
                 (429, {}, ErrorCode.RATE_LIMITED), (503, {}, ErrorCode.SERVER_ERROR),
                 (404, {}, ErrorCode.REMOTE_ERROR), (200, {"Content-Encoding": "gzip"}, ErrorCode.PROTOCOL_ERROR)]
        for status, headers, code in cases:
            response = Response(status=status, headers=headers)
            client, connections = self.client([response])
            with self.assertRaises(ClassifiedToolError) as raised:
                client.request("GET", "https://example.org/a")
            self.assertEqual(raised.exception.error_code, code)
            self.assertTrue(response.closed)
            connections[0].close.assert_called_once()

    def test_timeout_and_malformed_http_close_connections(self):
        for error in (TimeoutError("timeout"), http.client.BadStatusLine("malformed")):
            client, connections = self.client([Response()])
            connections[0].getresponse.side_effect = error
            expected = TimeoutError if isinstance(error, TimeoutError) else ClassifiedToolError
            with self.assertRaises(expected):
                client.request("GET", "https://example.org/a")
            connections[0].close.assert_called_once()

    def test_unsupported_binary_page_is_not_treated_as_text(self):
        client, _ = self.client([Response(headers={"Content-Type": "application/pdf"})])
        with self.assertRaises(ClassifiedToolError) as raised:
            WebExtractor(client).extract(PageQuery(url="https://example.org/a"))
        self.assertEqual(raised.exception.error_code, ErrorCode.REMOTE_ERROR)


if __name__ == "__main__":
    unittest.main()
