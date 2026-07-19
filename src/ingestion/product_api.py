"""Mock REST API backed by the two RetailRocket item-property files."""

from __future__ import annotations

import json
import itertools
import logging
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.ingestion.item_property_csv import read_item_properties

logger = logging.getLogger(__name__)


def property_rows(raw_dir: Path):
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        for row in read_item_properties(raw_dir / filename):
            yield row


def handler_for(raw_dir: Path):
    # The ingestion client requests monotonically increasing offsets. Retaining
    # this cursor means the 20M-row source is scanned once, rather than once per
    # HTTP page. A non-sequential request safely resets and seeks the stream.
    lock = threading.Lock()
    stream = property_rows(raw_dir)
    position = 0

    def page(offset: int, limit: int):
        nonlocal stream, position
        with lock:
            if offset != position:
                logger.info("Product API cursor reset: requested offset %s", f"{offset:,}")
                stream = property_rows(raw_dir)
                position = 0
                for _ in itertools.islice(stream, offset):
                    position += 1
            items = list(itertools.islice(stream, limit))
            position += len(items)
            return items

    class ProductApiHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self._json({"status": "ok"})
                return
            if parsed.path != "/item-properties":
                self.send_error(404)
                return
            query = urllib.parse.parse_qs(parsed.query)
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(10_000, max(1, int(query.get("limit", ["1000"])[0])))
            except ValueError:
                self.send_error(400, "offset and limit must be integers")
                return
            items = page(offset, limit)
            self._json({
                "items": items,
                "offset": offset,
                "next_offset": offset + len(items),
                "has_more": len(items) == limit,
            })

        def _json(self, payload: object):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ProductApiHandler


def make_server(host: str, port: int, raw_dir: Path) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_for(raw_dir))
