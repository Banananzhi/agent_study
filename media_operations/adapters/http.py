"""有限 HTTP 读取；公共 IP 校验后连接固定地址，TLS 仍校验原主机名。"""

import http.client
import ipaddress
import queue
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from email.message import Message
from urllib.parse import parse_qsl, urljoin, urlsplit

from tooling.errors import ClassifiedToolError
from tooling.registry import UnsafeRequestError
from tooling.result import ErrorCode


def resolve_host(host, port, timeout):
    result = queue.Queue(maxsize=1)

    def lookup():
        try:
            result.put([item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)])
        except Exception as error:
            result.put(error)

    # 系统 DNS 不支持 Python 层取消；超时返回，后台仅完成只读解析。
    threading.Thread(target=lookup, daemon=True).start()
    try:
        value = result.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError("DNS 查询超时") from error
    if isinstance(value, Exception):
        raise value
    return value


class PublicNetworkPolicy:
    def __init__(self, allowed_domains=()):
        self.allowed_domains = tuple(domain.lower().rstrip(".") for domain in allowed_domains)

    def validate_url(self, url):
        if len(url) > 2048 or any(ord(char) < 32 for char in url) or "\\" in url:
            raise UnsafeRequestError("URL 长度或字符不符合访问策略")
        try:
            parsed = urlsplit(url)
            host, port = parsed.hostname, parsed.port
        except ValueError as error:
            raise UnsafeRequestError("URL 格式无效") from error
        if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
            raise UnsafeRequestError("仅允许不含凭据的 HTTP/HTTPS URL")
        host = host.lower().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local")):
            raise UnsafeRequestError("禁止本地地址")
        if port not in {None, 80 if parsed.scheme == "http" else 443}:
            raise UnsafeRequestError("仅允许 HTTP/HTTPS 标准端口")
        if any(key.lower() in {"api_key", "access_token", "password", "secret", "token"} for key, _ in parse_qsl(parsed.query)):
            raise UnsafeRequestError("不允许带凭据参数的资料 URL")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise UnsafeRequestError("禁止本机或内网地址")
        if self.allowed_domains and not any(host == domain or host.endswith("." + domain) for domain in self.allowed_domains):
            raise UnsafeRequestError("域名不在允许列表中")
        return parsed


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, port, address, timeout):
        super().__init__(host, port=port, timeout=timeout)
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    def connect(self):
        super().connect()
        raw_socket = self.sock
        try:
            self.sock = ssl.create_default_context().wrap_socket(raw_socket, server_hostname=self.host)
        except Exception:
            raw_socket.close()
            raise


@dataclass(frozen=True)
class HTTPDocument:
    url: str
    body: bytes
    content_type: str
    charset: str
    truncated: bool = False


class PublicHTTPClient:
    def __init__(self, *, timeout=15, max_bytes=500000, max_redirects=5, policy=None, resolver=None):
        if timeout <= 0 or max_bytes < 1 or max_redirects < 0:
            raise ValueError("HTTP 限制无效")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.policy = policy or PublicNetworkPolicy()
        self.resolver = resolver or resolve_host

    def _connection(self, parsed, address, timeout):
        cls = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
        return cls(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), address, timeout)

    def request(self, method, url, *, body=None, headers=None):
        deadline = time.monotonic() + self.timeout
        for redirect in range(self.max_redirects + 1):
            parsed = self.policy.validate_url(url)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HTTP 总时限已到")
            addresses = self.resolver(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), min(5, remaining))
            if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
                raise UnsafeRequestError("DNS 指向非公共地址")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HTTP 总时限已到")
            connection = self._connection(parsed, addresses[0], remaining)
            response = None
            try:
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                connection.request(method, path, body=body, headers={
                    "User-Agent": "AgentStudy-Media/1.0", "Accept-Encoding": "identity", **(headers or {}),
                })
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("HTTP 总时限已到")
                if connection.sock:
                    connection.sock.settimeout(remaining)
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if method != "GET" or not location or redirect == self.max_redirects:
                        raise UnsafeRequestError("不允许此重定向或重定向次数已达上限")
                    url = urljoin(url, location)
                    headers = {key: value for key, value in (headers or {}).items()
                               if key.lower() not in {"authorization", "cookie", "proxy-authorization"}}
                    continue  # 下次请求前复验域名和 DNS；不转发 POST 的认证头。
                if not 200 <= response.status < 300:
                    status = response.status
                    code = (ErrorCode.AUTHENTICATION_ERROR if status in {401, 403}
                            else ErrorCode.RATE_LIMITED if status == 429
                            else ErrorCode.SERVER_ERROR if status >= 500 else ErrorCode.REMOTE_ERROR)
                    raise ClassifiedToolError(code, f"资料服务返回 HTTP {status}")
                if response.getheader("Content-Encoding", "identity").lower() not in {"identity", ""}:
                    raise ClassifiedToolError(ErrorCode.PROTOCOL_ERROR, "资料服务返回不支持的压缩内容")
                pieces, size = [], 0
                while size <= self.max_bytes:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("HTTP 总时限已到")
                    # Connection: close 时 HTTPConnection 已移交 socket 给响应流。
                    response_socket = getattr(getattr(response.fp, "raw", None), "_sock", None)
                    active_socket = connection.sock or response_socket
                    if active_socket:
                        active_socket.settimeout(remaining)
                    piece = response.read1(min(16384, self.max_bytes + 1 - size))
                    if not piece:
                        break
                    pieces.append(piece)
                    size += len(piece)
                metadata = Message()
                if time.monotonic() > deadline:
                    raise TimeoutError("HTTP 总时限已到")
                metadata["Content-Type"] = response.getheader("Content-Type", "application/octet-stream")
                return HTTPDocument(
                    url=url, body=b"".join(pieces)[:self.max_bytes],
                    content_type=metadata.get_content_type(), charset=metadata.get_content_charset() or "utf-8",
                    truncated=size > self.max_bytes,
                )
            except (ssl.SSLError, http.client.HTTPException) as error:
                raise ClassifiedToolError(ErrorCode.PROTOCOL_ERROR, "HTTP/TLS 响应协议验证失败") from error
            finally:
                if response is not None:
                    response.close()
                connection.close()
        raise UnsafeRequestError("重定向次数已达上限")
